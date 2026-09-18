"""MotionCanvas Pipeline: ODE-based inference with clean imputation.

The trained conditioning protocol fixes known coordinates to clean evidence:
``x_t[known] = x1[known]``. Inference mirrors it by replacing known
coordinates with ``clean_motion`` at each ODE step.

Replacement guidance modes
--------------------------
- ``"none"``: Diagnostic ablation without per-step replacement.
- ``"all"``: At every ODE step, replace known regions with ``clean_motion``.
- ``"skip_last"``: Same as ``"all"`` but skip replacement on the final step.

When ``replacement_guidance != 'none'``, the batch **must** contain a
``clean_motion`` key: the full normalized motion ``(B, T, D)`` **without**
masked regions zeroed.  The initial noise ``y0`` is also set to
``clean_motion`` in known regions (matching training where ``x_t[known] = x1``).

Position constraint support
---------------------------
Position cues are provided through the 198-D canvas. Optional additional IK
projection and automatic repair remain available in the full Motius integration,
not in this inference-only package.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from animatecanvas.base import BasePipeline



def normalize_repair_seed(seed: int) -> tuple[int, int]:
    """Return deterministic Torch/NumPy seeds for one repair sample.

    Torch accepts a wider integer range than NumPy's legacy global RNG. Keep
    the caller-visible seed intact for Torch while mapping it into NumPy's
    uint32 domain. Reject negative seeds so ``base + sample_index`` cannot
    silently acquire backend-specific wrap semantics.
    """
    torch_seed = int(seed)
    if torch_seed < 0:
        raise ValueError(f"repair_seed must be non-negative, got {torch_seed}")
    return torch_seed, torch_seed % (2**32)


def reset_repair_rng(seed: int) -> None:
    """Reset every RNG used by the repair pass at the Step-E boundary."""
    torch_seed, numpy_seed = normalize_repair_seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def _sdedit_integration_times(
    base_times: Tensor,
    t_init: Optional[float],
    *,
    exact_start: bool,
) -> Tensor:
    """Return the active ODE grid for a partial-noise SDEdit start."""
    if base_times.ndim != 1 or len(base_times) < 2:
        raise ValueError("base_times must be a one-dimensional ODE grid")
    if t_init is None:
        return base_times
    start = float(t_init)
    if not 0.0 <= start < 1.0:
        raise ValueError(f"SDEdit t_init must lie in [0,1), got {start}")
    if exact_start:
        start_t = torch.as_tensor(
            start, device=base_times.device, dtype=base_times.dtype
        ).reshape(1)
        future_t = base_times[base_times > start_t[0] + 1e-7]
        active = torch.cat([start_t, future_t], dim=0)
    else:
        values = base_times.detach().cpu().tolist()
        start_index = next(
            (index for index, value in enumerate(values) if value + 1e-6 >= start),
            0,
        )
        active = base_times[start_index:]
    if len(active) < 1 or (
        len(active) > 1 and not bool(torch.all(active[1:] > active[:-1]))
    ):
        raise RuntimeError("SDEdit integration schedule is invalid")
    if exact_start and len(active) < 2:
        raise RuntimeError("exact SDEdit schedule has no positive-time step")
    return active


def _base_grid_step_index(base_times: Tensor, active_time: Tensor) -> int:
    """Map an active SDEdit step back to the legacy base-grid phase.

    A snapped partial start is a suffix of ``base_times``. Its first active
    loop index is zero, but position-IK cadence historically used the original
    global base-grid index. Exact inserted detector substeps map to the interval
    immediately below them; detector probes currently carry no position IK.
    """
    if base_times.ndim != 1 or len(base_times) < 2:
        raise ValueError("base_times must be a one-dimensional ODE grid")
    value = torch.as_tensor(
        active_time, device=base_times.device, dtype=base_times.dtype
    ).reshape(())
    index = int(torch.searchsorted(base_times, value, right=True).item() - 1)
    if index < 0 or index >= len(base_times) - 1:
        raise ValueError(
            f"active ODE time must lie in [base_times[0], base_times[-1]): "
            f"got {float(value)}"
        )
    return index


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _batch_sources(batch: Dict[str, Any], batch_size: int) -> Optional[List[str]]:
    sources = batch.get('data_src', batch.get('source'))
    if sources is None:
        return None
    if isinstance(sources, str):
        return [sources] * batch_size
    if isinstance(sources, Tensor):
        return None
    if isinstance(sources, (list, tuple)):
        if len(sources) != batch_size:
            return None
        return [str(s) for s in sources]
    return None


def _length_list(value: Any, batch_size: int, fallback: int) -> List[int]:
    if value is None:
        return [fallback] * batch_size
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, int):
        return [value] * batch_size
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and batch_size > 1:
            return [int(value[0])] * batch_size
        return [int(v) for v in value]
    return [fallback] * batch_size


def _gaussian_temporal_smooth(
    x: Tensor,
    sigma: float,
    protect_mask: Optional[Tensor] = None,
) -> Tensor:
    """1-D Gaussian temporal smoothing along axis 1 of a (B, T, D) tensor.

    Ported from ``scripts/eval/eval_m2m_v2_all_tasks._gaussian_temporal_smooth``.
    Used to pre-smooth the LQ ``clean_motion`` (the *kept* region the model
    conditions on, and which is copied back into the output) before masked
    imputation. Because partial regeneration keeps the jittery LQ on unmasked
    cells, smoothing there both lowers the residual jitter in the output and
    gives the regenerated region a smooth boundary to blend against.

    Where ``protect_mask > 0.5`` (the *generate* region) values pass through
    unchanged -- smoothing there is pointless (overwritten by the model) and
    would bleed bad values across defect boundaries.
    """
    if sigma <= 0.0:
        return x
    T = x.shape[1]
    radius = max(1, int(round(3.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    B, _, D = x.shape
    x_flat = x.permute(0, 2, 1).reshape(B * D, 1, T)
    w = kernel.view(1, 1, -1)
    x_pad = F.pad(x_flat, (radius, radius), mode='replicate')
    y_flat = F.conv1d(x_pad, w)
    y = y_flat.reshape(B, D, T).permute(0, 2, 1).contiguous()
    if protect_mask is not None:
        y = torch.where(protect_mask > 0.5, x, y)
    return y


class MotionCanvasPipeline(BasePipeline):
    """Inference pipeline for MotionCanvas.

    Uses ODE integration to solve the flow matching ODE from noise to clean
    motion, conditioned on imputed motion evidence, edit context, and optional text.

    Parameters
    ----------
    bundle : MotionCanvasBundle
        The model bundle.
    num_steps : int
        Number of ODE integration steps.
    text_guidance_scale : float
        Classifier-free guidance scale for text conditioning.
    replacement_guidance : str
        Controls per-step replacement of unmasked (known) regions during
        ODE integration. This implements train-consistent imputation: during training,
        ``x_t[known] = x1`` (clean), so replacing known regions with clean
        motion at each step matches the training distribution.

        - ``"none"``: Diagnostic ablation without replacement.
        - ``"all"``: At every ODE step, replace known regions with
          ``clean_motion`` from the batch.
        - ``"skip_last"``: Same as ``"all"`` but skip replacement on the
          final step.

        When not ``"none"``, the batch must contain ``clean_motion``:
        the full normalized motion (B, T, D) without masked-region zeroing.
    """

    VALID_REPLACEMENT_MODES = ('none', 'all', 'skip_last', 'flow_interp')
    BUNDLE_CLS = 'animatecanvas.bundle.MotionCanvasBundle'

    def __init__(
        self,
        bundle,
        num_steps: int = 50,
        text_guidance_scale: float = 1.0,
        replacement_guidance: str = 'all',
        position_constraint_interval: int = 5,
        max_text_len: int = 128,
        sdedit_tau: float = 0.0,
        official_t2m_frames: int = 360,
    ):
        super().__init__(bundle)
        if replacement_guidance not in self.VALID_REPLACEMENT_MODES:
            raise ValueError(
                f'replacement_guidance must be one of '
                f'{self.VALID_REPLACEMENT_MODES}, got {replacement_guidance!r}'
            )
        self.bundle = bundle
        self.num_steps = num_steps
        self.text_guidance_scale = text_guidance_scale
        self.replacement_guidance = replacement_guidance
        self.position_constraint_interval = position_constraint_interval
        # IMPORTANT: must match the trainer's max_text_len (default 128) so
        # the context attention mask and positional statistics at inference
        # match what the model saw during training. Using the raw per-sample
        # token length (12-20) instead of padding to 128 was a bug that made
        # captioned inference produce distorted outputs (2026-04-20).
        self.max_text_len = max_text_len
        self.official_t2m_frames = int(official_t2m_frames)
        # SDEdit-style partial-noise start for E9 motion repair. In
        # flow-matching convention (t=0 -> pure noise, t=1 -> clean data),
        # the default inpainting path starts from t=0 on the masked region
        # (full regeneration). SDEdit τ lets us start from t = 1 - τ instead
        # — i.e. the masked region is initialized as `(1-τ)*x_clean + τ*z`
        # and the ODE runs from t=1-τ up to t=1. Smaller τ → more LQ retained,
        # closer to a "cleanup" of defects; larger τ (→1) → full regeneration.
        # Only applied when replacement_guidance != 'none' (requires
        # `clean_motion` in the batch to know what the masked region's LQ is).
        if not (0.0 <= sdedit_tau <= 1.0):
            raise ValueError(
                f'sdedit_tau must be in [0, 1], got {sdedit_tau!r}'
            )
        self.sdedit_tau = float(sdedit_tau)

    @staticmethod
    def _as_motion_batch(motion: Tensor, name: str) -> Tensor:
        motion = torch.as_tensor(motion)
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if motion.ndim != 3 or motion.shape[-1] != 198:
            raise ValueError(
                f'{name} must have shape (T, 198) or (B, T, 198), '
                f'got {tuple(motion.shape)}'
            )
        return motion.float()

    @staticmethod
    def _as_generation_mask(mask: Tensor, motion: Tensor) -> Tensor:
        mask = torch.as_tensor(mask, dtype=motion.dtype, device=motion.device)
        if mask.ndim == 1:
            mask = mask[None, :, None]
        elif mask.ndim == 2:
            if mask.shape == motion.shape[:2]:
                mask = mask[..., None]
            elif motion.shape[0] == 1 and mask.shape[0] == motion.shape[1]:
                mask = mask[None]
        if mask.ndim != 3:
            raise ValueError(
                'generation_mask must be frame-level (T)/(B,T) or '
                'channel-level (T,198)/(B,T,198)'
            )
        if mask.shape[0] == 1 and motion.shape[0] > 1:
            mask = mask.expand(motion.shape[0], -1, -1)
        if mask.shape[:2] != motion.shape[:2]:
            raise ValueError(
                f'generation_mask leading shape {tuple(mask.shape[:2])} '
                f'does not match motion {tuple(motion.shape[:2])}'
            )
        if mask.shape[-1] == 1:
            mask = mask.expand_as(motion)
        if mask.shape[-1] != motion.shape[-1]:
            raise ValueError(
                f'generation_mask must end in 1 or 198, got {mask.shape[-1]}'
            )
        return mask.clamp(0.0, 1.0)

    @torch.no_grad()
    def infer_m2m(
        self,
        source_motion: Optional[Tensor] = None,
        generation_mask: Optional[Tensor] = None,
        *,
        captions: Optional[List[str] | str] = None,
        num_frames: Optional[List[int] | int] = None,
        edit_mode: bool = False,
        edit_source_motion: Optional[Tensor] = None,
        input_is_normalized: bool = False,
        seed: Optional[int] = None,
        **batch_overrides,
    ) -> Dict[str, Any]:
        """Run the unified MotionCanvas generation/editing contract.

        ``generation_mask`` follows the training convention: 1 regenerates a
        coordinate and 0 preserves clean source evidence. The native motion is
        the 198-D tensor ``[root translation, local rotation-6D, FK joints]``.
        """
        device = next(self.bundle.motion_transformer.parameters()).device
        dtype = next(self.bundle.motion_transformer.parameters()).dtype

        if source_motion is None:
            caption_count = (
                1
                if captions is None or isinstance(captions, str)
                else len(captions)
            )
            if num_frames is None:
                num_frames = 180
            lengths = (
                [int(num_frames)] * caption_count
                if isinstance(num_frames, int)
                else [int(value) for value in num_frames]
            )
            source = torch.zeros(
                len(lengths),
                max(lengths),
                198,
                device=device,
                dtype=dtype,
            )
            clean_motion = source.clone()
            mask = torch.ones_like(source)
        else:
            source_raw = self._as_motion_batch(source_motion, 'source_motion').to(
                device=device,
                dtype=dtype,
            )
            lengths = (
                [source_raw.shape[1]] * source_raw.shape[0]
                if num_frames is None
                else (
                    [int(num_frames)] * source_raw.shape[0]
                    if isinstance(num_frames, int)
                    else [int(value) for value in num_frames]
                )
            )
            if len(lengths) != source_raw.shape[0]:
                raise ValueError('num_frames must contain one value per motion')
            clean_motion = (
                source_raw
                if input_is_normalized
                else self.bundle.normalize_motion(source_raw)
            )
            mask = (
                torch.ones_like(clean_motion)
                if generation_mask is None
                else self._as_generation_mask(generation_mask, clean_motion)
            )
            if edit_source_motion is not None:
                edit_source = self._as_motion_batch(
                    edit_source_motion,
                    'edit_source_motion',
                ).to(device=device, dtype=dtype)
                if edit_source.shape != source_raw.shape:
                    raise ValueError('edit_source_motion must match source_motion')
                edit_source = (
                    edit_source
                    if input_is_normalized
                    else self.bundle.normalize_motion(edit_source)
                )
            else:
                edit_source = clean_motion
            source = (
                clean_motion * (1.0 - mask)
                if not edit_mode
                else clean_motion * (1.0 - mask) + edit_source * mask
            )

        batch = {
            'src_motion': source,
            'src_mask': mask,
            'clean_motion': clean_motion,
            'src_length': lengths,
            'tgt_length': lengths,
            **batch_overrides,
        }
        if captions is not None:
            caption_list = [captions] if isinstance(captions, str) else list(captions)
            if len(caption_list) == 1 and len(lengths) > 1:
                caption_list *= len(lengths)
            if len(caption_list) != len(lengths):
                raise ValueError('captions must contain one string per motion')
            batch.update(self.bundle.encode_text(caption_list))
            batch['caption'] = caption_list

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        result = self(batch)
        max_length = max(lengths)
        for key in (
            'latent_denorm',
            'keypoints3d',
            'rot6d',
            'transl',
            'root_rotations_mat',
            'latent',
        ):
            value = result.get(key)
            if isinstance(value, Tensor) and value.ndim >= 2:
                result[key] = value[:, :max_length]
        result['motion_198'] = result['latent_denorm']
        result['lengths'] = lengths
        return result

    def infer_text_to_motion(
        self,
        captions: List[str] | str,
        num_frames: List[int] | int = 180,
        **kwargs,
    ) -> Dict[str, Any]:
        return self.infer_m2m(
            captions=captions,
            num_frames=num_frames,
            **kwargs,
        )

    infer_t2m = infer_text_to_motion

    def infer_temporal_motion_completion(self, source_motion, generation_mask, **kwargs):
        return self.infer_m2m(source_motion, generation_mask, edit_mode=False, **kwargs)

    def infer_motion_inbetweening(self, source_motion, generation_mask, **kwargs):
        return self.infer_m2m(source_motion, generation_mask, edit_mode=False, **kwargs)

    def infer_keyframe_motion_control(self, source_motion, generation_mask, **kwargs):
        return self.infer_m2m(source_motion, generation_mask, edit_mode=False, **kwargs)

    def infer_kinematic_motion_control(
        self,
        source_motion,
        generation_mask,
        *,
        position_constraints=None,
        **kwargs,
    ):
        if position_constraints is not None:
            kwargs['position_constraints'] = position_constraints
        return self.infer_m2m(source_motion, generation_mask, edit_mode=False, **kwargs)

    def infer_motion_editing(self, source_motion, generation_mask, **kwargs):
        return self.infer_m2m(source_motion, generation_mask, edit_mode=True, **kwargs)


    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a batch.

        Uses midpoint ODE solver for numerical stability (euler diverges).
        """
        return self._inference(batch)

    def _inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Actual inference logic.

        Returns:
            Dict with keys: rot6d, transl, keypoints3d (optional), latent.
        """
        device = next(self.bundle.motion_transformer.parameters()).device

        src_motion = batch['src_motion'].to(device)
        B, T, D = src_motion.shape

        src_mask = batch.get('src_mask')
        if src_mask is not None:
            src_mask = src_mask.to(device)

        src_length = batch.get('src_length')
        src_length = _length_list(src_length, B, T)

        tgt_length = batch.get('tgt_length', src_length)
        tgt_length = _length_list(tgt_length, B, max(src_length) if src_length else T)

        # Every official single-segment MotionCanvas call runs the ODE on the
        # same 360-frame canvas used by training, then lets the caller crop the
        # decoded result to ``tgt_length``.  This is not specific to text-only
        # full regeneration: conditional completion/editing on a short raw
        # canvas changes the transformer's temporal statistics and invalidates
        # cross-setting comparisons, including checkpoint-matched ablations.
        #
        # Keep the valid lengths unchanged so ``tgt_padding_mask`` marks only
        # the requested frames.  Pad normalized motion, condition mask, and
        # clean imputation target with training-parity zeros.  Padding mask
        # entries must be zero (not "generate"); valid-frame filtering below
        # prevents them from activating replacement guidance.
        use_official_canvas = (
            T < self.official_t2m_frames
            and max(tgt_length) <= self.official_t2m_frames
        )
        if use_official_canvas:
            pad_t = self.official_t2m_frames - T
            src_motion = F.pad(src_motion, (0, 0, 0, pad_t))
            if src_mask is not None:
                src_mask = F.pad(src_mask, (0, 0, 0, pad_t), value=0.0)
            if 'clean_motion' in batch and isinstance(batch['clean_motion'], Tensor):
                batch = dict(batch)
                batch['clean_motion'] = F.pad(
                    batch['clean_motion'].to(device),
                    (0, 0, 0, pad_t),
                )
            if (
                'completion_unanchored_mask' in batch
                and isinstance(batch['completion_unanchored_mask'], Tensor)
            ):
                batch = dict(batch)
                batch['completion_unanchored_mask'] = F.pad(
                    batch['completion_unanchored_mask'].to(device),
                    (0, 0, 0, pad_t),
                    value=False,
                )
            T = self.official_t2m_frames

        ref_pose = batch.get('ref_pose')
        if ref_pose is not None and not isinstance(ref_pose, Tensor):
            ref_pose = None
        if ref_pose is not None:
            ref_pose = ref_pose.to(device)

        tgt_padding_mask = _length_to_mask(
            torch.tensor(tgt_length, dtype=torch.long, device=device), T
        )

        # Prepare text
        # CRITICAL: must match training-time convention (see
        # MotionCanvasTrainer._prepare_and_forward):
        #   1. ctxt_input is always padded to max_text_len=128, regardless of
        #      the per-sample caption length.
        #   2. ctxt_mask_temporal marks valid tokens (True = valid, False = pad).
        #   3. Null-caption samples get the *learned* null_ctxt_input broadcast
        #      to the full (1, 128, 4096) shape — NOT a zero tensor with one
        #      active token. Attention masks are all-False for null samples
        #      (training convention in `_length_to_mask(ctxt_length=0, 128)`).
        #   4. All tensors must match the transformer's parameter dtype so
        #      attention math happens in the same precision as training.
        # Earlier code used raw caption seq_len (12-20) for ctxt and
        # zeros-with-first-token-null for CFG, which led to distorted
        # captioned outputs because the model never saw those distributions
        # during training. (2026-04-20)
        pad_len = self.max_text_len
        model_dtype = next(self.bundle.motion_transformer.parameters()).dtype

        def _pad_ctxt(ctxt: Tensor, length_is_valid: bool) -> Tensor:
            """Pad / truncate ctxt to (B, pad_len, D)."""
            if ctxt.shape[1] == pad_len:
                return ctxt
            if ctxt.shape[1] < pad_len:
                return F.pad(ctxt, (0, 0, 0, pad_len - ctxt.shape[1]))
            return ctxt[:, :pad_len]

        if 'text_vec_raw' in batch:
            vtxt_input = batch['text_vec_raw'].to(device=device, dtype=model_dtype)
            ctxt_raw = batch['text_ctxt_raw'].to(device=device, dtype=model_dtype)
            ctxt_input = _pad_ctxt(ctxt_raw, True)
            ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)
        else:
            # Unconditioned inference: MUST match training convention in
            # MotionCanvasTrainer._prepare_and_forward lines 212-215:
            #   ctxt_input = null_ctxt_input.expand(B, 1, -1)   ← 1 token
            #   ctxt_length = 1
            #   ctxt_mask = all-True over length 1
            # Earlier code used pad_len (128) here for symmetry with the
            # captioned branch, but the uncond-trained model never saw a
            # 128-token context during training — it always saw a single
            # null token. Feeding 128 repeated null tokens + all-False
            # attention mask is a severe OOD shift that produces catastrophic
            # jitter in output (found 2026-04-21).
            vtxt_input = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(B, 1, -1).contiguous()
            ctxt_length = torch.ones(B, dtype=torch.long, device=device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1)

        # Prepare edit context and target mask.
        condition_context = self.bundle.prepare_condition_context(
            src_motion=src_motion,
            ref_pose=ref_pose,
            src_mask=src_mask,
        )

        do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode

        # CFG null-branch construction.  The "silent" CFG branch nulls BOTH
        # sentence-level vtxt AND token-level ctxt to match training-time
        # mask_text_cond behavior (which masks both vtxt and ctxt).
        # Previously only vtxt was nulled while ctxt was kept intact, making
        # CFG guidance depend solely on the 768-dim vtxt difference — far too
        # weak for effective caption guidance.  Fixed 2026-05-15.
        if do_cfg:
            null_vtxt = self.bundle.null_vtxt_feat.to(dtype=model_dtype).expand_as(vtxt_input)
            # Expand null_ctxt to match ctxt_input's sequence length so
            # torch.cat along batch dim works correctly.
            null_ctxt = self.bundle.null_ctxt_input.to(dtype=model_dtype).expand(
                ctxt_input.shape[0], ctxt_input.shape[1], -1
            ).contiguous()
            # Match HYMotion T2M: the null branch uses learned null embeddings
            # expanded to the caption length and reuses the caption attention
            # mask.  Collapsing to a single token changes the CFG distribution
            # for HYMotion-Lite warm starts.
            null_ctxt_mask = ctxt_mask_temporal

        # ODE function
        ode_cfg = dict(self.bundle._noise_scheduler_cfg)
        ode_cfg.pop('method', None)  # odeint uses it positionally
        sources = _batch_sources(batch, B)
        sources_cfg = ([''] * B + sources) if (do_cfg and sources is not None) else sources

        def fn(t: Tensor, x: Tensor) -> Tensor:
            x_input = torch.cat([x, condition_context], dim=-1)
            if do_cfg:
                x_input = torch.cat([x_input, x_input], dim=0)
            x_pred = self.bundle.predict_flow(
                x_input=x_input,
                ctxt_input=(
                    ctxt_input if not do_cfg
                    else torch.cat([null_ctxt, ctxt_input], dim=0)
                ),
                vtxt_input=(
                    vtxt_input if not do_cfg
                    else torch.cat([null_vtxt, vtxt_input], dim=0)
                ),
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=(
                    tgt_padding_mask if not do_cfg
                    else tgt_padding_mask.repeat(2, 1)
                ),
                ctxt_mask_temporal=(
                    ctxt_mask_temporal if not do_cfg
                    else torch.cat([null_ctxt_mask, ctxt_mask_temporal], dim=0)
                ),
                sources=sources_cfg,
                trigger_sources={'Taobao', 'Game'},
                special_game_prob=1.0,
            )

            if self.bundle.pred_type == 'x1':
                t_eps = 0.05
                if do_cfg:
                    x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t).clamp_min(t_eps)
                else:
                    x_pred = (x_pred - x) / (1.0 - t).clamp_min(t_eps)

            if do_cfg:
                pred_basic, pred_text = x_pred.chunk(2, dim=0)
                x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)
            return x_pred

        # -----------------------------------------------------------------
        # Initial y0 and replacement guidance setup
        # -----------------------------------------------------------------
        z = torch.randn(B, T, D, device=device, dtype=src_motion.dtype)
        t = torch.linspace(0, 1, self.num_steps + 1, device=device, dtype=src_motion.dtype)

        rep_mode = self.replacement_guidance
        valid_frame_mask = tgt_padding_mask.unsqueeze(-1)  # (B, T, 1)
        valid_generate = (
            (src_mask > 0.5) & valid_frame_mask
            if src_mask is not None
            else None
        )
        valid_known = (
            (src_mask < 0.5) & valid_frame_mask
            if src_mask is not None
            else None
        )
        use_replacement = (
            rep_mode != 'none'
            and src_mask is not None
            and bool(valid_generate.any().item())
            and bool(valid_known.any().item())
        )

        if use_replacement:
            # keep_mask: (B, T, D), True = known region (mask=0).
            #
            # ⚠️ 2026-04-24 bug fix ("E13 每段尾帧静止"): exclude PAD frames
            # from the known region even if src_mask=0 there. Rationale: under
            # training distribution, pad frames (idx >= tgt_length) carry
            # src_mask=0 AND src_motion=0 AND attention is masked out by
            # tgt_padding_mask AND loss is masked out. The model never
            # "sees" pad frames during training. If we leave keep_mask=True
            # on pad frames at inference time, the replacement loop below
            # pins x[pad] ← x_clean[pad] = replicate(normalize(synthetic-zero
            # last frame)) every ODE step — i.e. it anchors the entire pad
            # region to the training-set MEAN pose. For cases where the
            # "synthetic zero" is just the training mean (E13 where src_raw
            # is zeros outside the prefix, or any short-clip inference where
            # we pad by replicating a valid end frame), this mean-pose
            # anchor leaks into the valid region via shared LayerNorms /
            # residual paths (pad is key-masked but still flows as a
            # query/value through per-token feedforwards) and pulls the
            # TAIL of the valid region visibly toward static "mean pose".
            # Users reported this as "每段尾帧几乎不动、静止" on E13.
            #
            # Fix: combine src_mask (per-frame-per-dim "is this a known
            # sample?") with tgt_padding_mask ("is this a valid frame in
            # the model's view?"). Pad frames get keep_mask=False → neither
            # y0 init (below) nor per-step replacement touches them. They
            # become ordinary ODE free-evolve tokens — consistent with
            # training where the model is simply not asked about them.
            keep_mask = (src_mask < 0.5) & valid_frame_mask
            # clean_motion: full normalized motion WITHOUT masked-region
            # zeroing. Required for clean imputation.
            assert 'clean_motion' in batch, (
                'replacement_guidance requires "clean_motion" in batch '
                '(full normalized motion without zeroing masked regions)'
            )
            x_clean = batch['clean_motion'].to(device)

            if self.sdedit_tau > 0.0:
                # SDEdit-style partial-noise start on masked region.
                # Flow-matching convention (in this pipeline): t=0 → noise,
                # t=1 → clean, so x_t = (1-t)*z + t*clean. To start from τ
                # noise fraction we set t_init = 1 - τ. The loop below will
                # honor this by skipping ODE steps with t_curr < t_init.
                tau = self.sdedit_tau
                t_init = 1.0 - tau
                completion_unanchored = batch.get(
                    'completion_unanchored_mask'
                )
                promote_completion_to_full_regeneration = False
                if completion_unanchored is not None:
                    completion_unanchored = completion_unanchored.to(
                        device=device, dtype=torch.bool
                    )
                    if completion_unanchored.shape != x_clean.shape:
                        raise ValueError(
                            'completion_unanchored_mask shape differs from '
                            f'latent: {tuple(completion_unanchored.shape)} != '
                            f'{tuple(x_clean.shape)}'
                        )
                    promote_completion_to_full_regeneration = bool(
                        completion_unanchored.any().item()
                    )
                if promote_completion_to_full_regeneration:
                    # Full-span generated channels have no clean-blind SDEdit
                    # anchor. Starting them at t=0 noise while integrating
                    # only the final tau fraction left essentially raw noise
                    # in the output. Promote the sample to ordinary completion:
                    # all generated cells receive the complete ODE trajectory,
                    # while known cells remain hard-imputed throughout.
                    y0 = torch.where(keep_mask, x_clean, z)
                    sdedit_t_init = None
                else:
                    x_partial_noised = (
                        (1.0 - t_init) * z + t_init * x_clean
                    )
                    y0 = torch.where(keep_mask, x_clean, x_partial_noised)
                    sdedit_t_init = t_init
            else:
                # Training fixes x_t[known] = x1 (clean). Match it at t=0.
                y0 = torch.where(keep_mask, x_clean, z)
                sdedit_t_init = None
        else:
            y0 = z
            sdedit_t_init = None

        # -----------------------------------------------------------------
        # Position constraint setup
        # -----------------------------------------------------------------
        position_constraints = batch.get('position_constraints')
        use_pos_constraint = position_constraints is not None and len(position_constraints) > 0
        pos_solver = None
        pos_affected_dims = None

        if use_pos_constraint:
            raise NotImplementedError('Optional IK projection is provided by Motius; use canvas cues here')

        # -----------------------------------------------------------------
        # ODE integration
        # -----------------------------------------------------------------
        exact_sdedit_start = bool(batch.get("integration_t_start_exact", False))
        if use_replacement or use_pos_constraint:
            # Manual Euler with per-step imputation and/or position constraint.
            # ``y0`` is sampled at the exact SDEdit time. Snapping its first
            # vector-field evaluation to the next coarse grid point creates a
            # deterministic residual on every channel. Detector probes insert
            # the exact start; legacy Step-E repair retains its frozen grid.
            integration_t = _sdedit_integration_times(
                t,
                sdedit_t_init,
                exact_start=exact_sdedit_start,
            )
            # Store initial noise for flow_interp mode
            z0 = y0.clone() if rep_mode == 'flow_interp' else None
            x = y0
            active_steps = len(integration_t) - 1
            for active_i in range(active_steps):
                t_curr = integration_t[active_i]
                dt = integration_t[active_i + 1] - integration_t[active_i]
                is_last_step = active_i == active_steps - 1

                v = fn(t_curr, x)
                x = x + v * dt

                # Imputation: force known regions back to expected values.
                if use_replacement:
                    if rep_mode == 'flow_interp' and not is_last_step:
                        t_next = integration_t[active_i + 1]
                        x_interp = (1 - t_next) * z0 + t_next * x_clean
                        x = torch.where(keep_mask, x_interp, x)
                    elif rep_mode == 'all' or (rep_mode == 'skip_last' and not is_last_step):
                        x = torch.where(keep_mask, x_clean, x)

            # Final hard replacement guarantees exact evidence preservation for
            # skip_last and flow_interp. Training now targets zero velocity on
            # clean-imputed coordinates, but an exact projection still avoids
            # accumulating solver and finite-precision error on known values.
            if use_replacement and rep_mode in ('skip_last', 'flow_interp'):
                x = torch.where(keep_mask, x_clean, x)

            sampled = x
        else:
            # Standard path: use torchdiffeq if available, else manual Euler.
            try:
                from torchdiffeq import odeint
                method = self.bundle._noise_scheduler_cfg.get('method', 'euler')
                trajectory = odeint(fn, y0, t, method=method)
            except ImportError:
                trajectory = [y0]
                dt = 1.0 / self.num_steps
                x = y0
                for i in range(self.num_steps):
                    t_val = torch.tensor(i * dt, device=device, dtype=src_motion.dtype)
                    v = fn(t_val, x)
                    x = x + v * dt
                    trajectory.append(x)
                trajectory = torch.stack(trajectory, dim=0)

            sampled = trajectory[-1]

        # Decode to motion
        result = self.bundle.decode_motion_from_latent(sampled)
        result['latent'] = sampled
        result['rotation_space'] = getattr(self.bundle, 'rotation_space', 'local')
        return result

    # ------------------------------------------------------------------ #
    # Motion repair (E9): defect detection + masked regeneration.        #
    # ------------------------------------------------------------------ #
    T_PAD_REPAIR = 360  # the context length the model was trained with


    @staticmethod
    def _completion_safe_sdedit_anchor(
        clean_motion: Tensor,
        generate_mask: Tensor,
        *,
        valid_len: int,
    ) -> tuple[Tensor, Tensor]:
        """Remove masked LQ values from a completion-mode SDEdit anchor.

        Masked runs are linearly interpolated from their nearest known values
        in the same channel.  One-sided runs use the nearest known value.
        Channels masked over the complete valid interval cannot be inferred
        from condition support; their returned ``unanchored`` mask instructs
        :meth:`_inference` to start those cells from pure noise.
        """
        clean = clean_motion.clone()
        mask = generate_mask >= 0.5
        if clean.ndim != 3 or mask.shape != clean.shape:
            raise ValueError(
                "completion SDEdit anchor expects equal [B,T,D] tensors, got "
                f"{tuple(clean.shape)} and {tuple(mask.shape)}"
            )
        length = int(valid_len)
        if length < 1 or length > clean.shape[1]:
            raise ValueError(
                f"invalid completion anchor valid_len={length} for T={clean.shape[1]}"
            )
        unanchored = torch.zeros_like(mask)
        frame_indices = torch.arange(
            length, device=clean.device, dtype=clean.dtype
        )
        for batch_index in range(clean.shape[0]):
            for channel in range(clean.shape[2]):
                channel_mask = mask[batch_index, :length, channel]
                if not bool(channel_mask.any().item()):
                    continue
                known_indices = torch.nonzero(
                    ~channel_mask, as_tuple=False
                ).flatten()
                if known_indices.numel() == 0:
                    clean[batch_index, :length, channel] = 0.0
                    unanchored[batch_index, :length, channel] = channel_mask
                    continue

                insertion = torch.searchsorted(known_indices, frame_indices)
                left_slot = torch.clamp(insertion - 1, min=0)
                right_slot = torch.clamp(
                    insertion, max=known_indices.numel() - 1
                )
                left_index = known_indices[left_slot]
                right_index = known_indices[right_slot]
                left_value = clean[batch_index, left_index, channel]
                right_value = clean[batch_index, right_index, channel]
                interval = (right_index - left_index).to(clean.dtype)
                alpha = torch.where(
                    interval > 0,
                    (frame_indices - left_index.to(clean.dtype))
                    / interval.clamp_min(1.0),
                    torch.zeros_like(frame_indices),
                )
                interpolated = left_value + alpha * (right_value - left_value)
                clean[batch_index, :length, channel] = torch.where(
                    channel_mask,
                    interpolated,
                    clean[batch_index, :length, channel],
                )
        return clean, unanchored



