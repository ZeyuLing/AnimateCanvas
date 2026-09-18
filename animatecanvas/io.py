"""Validated physical-space canvas I/O; no body assets or checkpoints required."""
from pathlib import Path
import numpy as np


def load_cues(path):
    with np.load(path, allow_pickle=False) as data:
        if not {"motion", "generation_mask"}.issubset(data.files):
            raise ValueError("NPZ must contain motion and generation_mask")
        motion = np.asarray(data["motion"], dtype=np.float32)
        mask = np.asarray(data["generation_mask"], dtype=np.float32)
    if motion.ndim not in (2, 3) or motion.shape[-1] != 198 or motion.shape != mask.shape:
        raise ValueError("Expected matching (T,198) or (B,T,198) motion/mask arrays")
    if not 1 <= motion.shape[-2] <= 360 or (motion.ndim == 3 and motion.shape[0] < 1):
        raise ValueError("Use a nonempty batch of 1..360 frames")
    if not np.isfinite(motion).all() or not np.isin(mask, [0.0, 1.0]).all():
        raise ValueError("Motion must be finite; generation_mask must be binary")
    rotations = mask[..., 3:135].reshape(*mask.shape[:-1], 22, 6)
    if not (rotations == rotations[..., :1]).all():
        raise ValueError("A rotation cue must preserve all six rotation channels together")
    if motion.ndim == 2:
        motion, mask = motion[None], mask[None]
    return motion, mask


def save_result(path, result):
    arrays = {}
    for key, value in result.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, (np.ndarray, list, tuple, int, float)):
            array = np.asarray(value)
            if array.dtype != object:
                arrays[key] = array
    if "motion_198" not in arrays:
        raise ValueError("The pipeline did not return motion_198")
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Output filename must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
