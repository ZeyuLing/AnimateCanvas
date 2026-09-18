<p align="center">
  <a href="https://zeyuling.github.io/AnimateCanvas/#demo"><img src="assets/cover.png" width="100%" alt="AnimateCanvas: compose kinematic cues, generate coherent full-body motion"></a>
</p>

<h2 align="center">Learning Implicit Motion Planning<br>from Composable Kinematic Cues</h2>

<p align="center">
  Zeyu Ling · Di Kang · Qing Shuai · Yuxin Wen · Jing Li<br>
  Zhanke Wang · Heng Li · Chunchao Guo · Changqing Zou* · Linchao Bao
</p>
<p align="center"><sub>Zhejiang University · Tencent · Peking University · Sun Yat-sen University · Zhejiang Lab<br>* Corresponding author</sub></p>

<p align="center">
  <a href="assets/AnimateCanvas.pdf"><img src="https://img.shields.io/badge/Paper-PDF-B64B3A?style=flat-square" alt="Read the paper PDF"></a>
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionCanvas-0.46B"><img src="https://img.shields.io/badge/Hugging_Face-Model-E3AF35?style=flat-square" alt="Hugging Face model; availability noted below"></a>
  <a href="https://zeyuling.github.io/AnimateCanvas/#demo"><img src="https://img.shields.io/badge/Watch-Demo-1F6F77?style=flat-square" alt="Watch the full demo"></a>
  <a href="https://zeyuling.github.io/AnimateCanvas/"><img src="https://img.shields.io/badge/Project-Homepage-274252?style=flat-square" alt="Project homepage"></a>
  <a href="https://github.com/ZeyuLing/Motius"><img src="https://img.shields.io/badge/Integrated_in-Motius-426D58?style=flat-square" alt="Full Motius integration"></a>
</p>

**One motion canvas. Composable control.** AnimateCanvas generates full-body motion
from position and rotation cues specified across joints and time, with optional
language and input motion. One model supports temporal completion, spatial control,
sequential generation, language-guided editing, and repair while retaining
text-to-motion generation.

**Fully integrated in [Motius](https://github.com/ZeyuLing/Motius).** The maintained
network, training pipeline, cue sampler, inference, and evaluation interfaces live
in Motius. This focused repository provides the paper, demos, documentation, and
small runnable entry points using that same implementation, not a divergent model fork.

> **Model availability:** the linked checkpoint is currently private; public access
> is being enabled. Until then, inference requires authorized access or a compatible
> local checkpoint. The legacy model identifier is retained for compatibility.

## See it in motion

**[Official credited demo · 1440p / Chinese-captioned 1080p](https://github.com/ZeyuLing/AnimateCanvas/releases/tag/animatecanvas-official-demo-20260918)**

The formal release includes author credits, affiliations, and project links.

<table>
<tr>
<td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/route-v56.mp4"><img src="assets/route.webp" alt="Dense trajectory and local wrist cues" width="100%"></a><br><b>Trajectories + local cues</b></td>
<td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/footsteps-animatecanvas.mp4"><img src="assets/footsteps.webp" alt="Sparse foot positions and root heading" width="100%"></a><br><b>Sparse positions + heading</b></td>
<td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/jump-v56.mp4"><img src="assets/jump.webp" alt="Key poses and trajectory cues for a jump" width="100%"></a><br><b>Keyframes + trajectories</b></td>
</tr>
<tr>
<td align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/editing-v56.mp4"><img src="assets/editing.webp" alt="Language-guided motion editing" width="100%"></a><br><b>Language-guided editing</b></td>
<td align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/boxing-v56.mp4"><img src="assets/boxing.webp" alt="Composed local controls for a strike" width="100%"></a><br><b>Composed local controls</b></td>
<td align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/basketball-animatecanvas.mp4"><img src="assets/basketball.webp" alt="A timed spatial cue for a dunk" width="100%"></a><br><b>Timed spatial targets</b></td>
</tr>
</table>

[Full narrated demo](https://zeyuling.github.io/AnimateCanvas/#demo) ·
[Benchmark gallery](https://zeyuling.github.io/AnimateCanvas/#benchmarks) ·
[Media provenance](https://github.com/ZeyuLing/AnimateCanvas/blob/main/MEDIA.md)

## Quick start

Use Python 3.10+ and a CUDA-capable PyTorch installation for practical inference.
From this repository directory:

```bash
python -m pip install -e .
python examples/generate.py --checkpoint /path/to/checkpoint --local-files-only \
  --task text --text "A person walks forward and waves." --frames 180 \
  --output outputs/walking.npz
```

Once you have access to the Hub artifact, `--checkpoint` can also take
`ZeyuLing/Motius-MotionCanvas-0.46B`; omit `--local-files-only` to permit downloading.
The full artifact includes text encoders and is approximately 20 GB. Use
`--cache-dir /path/to/large/cache` when necessary. Model access and licenses remain
separate from installing this repository.

```python
from animatecanvas import load_pipeline

pipe = load_pipeline("/path/to/checkpoint", local_files_only=True)
result = pipe.infer_text_to_motion(
    "A person turns left and continues walking", num_frames=180, seed=42
)
motion = result["motion_198"]
```

### Control, completion, and editing

Create an NPZ with `motion` and `generation_mask`, each shaped `(T, 198)` or
`(B, T, 198)`. Use physical-space motion in **meters**, **30 fps**, at most
**360 frames**. Mask value **0 preserves a cue**, and **1 generates a value**.

```bash
python examples/generate.py --checkpoint /path/to/checkpoint --local-files-only \
  --task control --cues cues.npz --text "A person walks along the path." \
  --output outputs/controlled.npz

python examples/generate.py --checkpoint /path/to/checkpoint --local-files-only \
  --task edit --cues input_motion_and_mask.npz --text "Wave with the right hand." \
  --output outputs/edited.npz
```

Use `--task completion` for temporal completion. Editing additionally uses the
input motion as context in the regions being regenerated. The CLI does not
silently apply IK, resample motion, or retarget skeletons.

[Canvas format and cue preparation](docs/canvas.md) ·
[Motius integration and training](docs/motius.md)

## How it works

<img src="assets/pipeline.png" width="100%" alt="Shared motion canvas, flow-matching model, and cue imputation">

- **Compose:** place heterogeneous kinematic cues on one canvas across frames,
  joints, and position or rotation channels.
- **Generate:** use a shared flow-matching model with optional language and input motion.
- **Preserve:** impute specified canvas values during training and sampling.

Root translation and local rotation cues are directly preserved. Non-root
positions are decoded through forward kinematics, so exact canvas preservation
does not imply exact decoded joint positions. Optional IK is a separate refinement.

## Motius integration

| Capability | Maintained entry point |
| --- | --- |
| Text-to-motion | `infer_text_to_motion` |
| Temporal completion | `infer_temporal_motion_completion` |
| Spatial and composed control | `infer_kinematic_motion_control` |
| Language-guided editing | `infer_motion_editing` |
| Repair | `infer_motion_repair` |
| Training | [Trainer and configuration](docs/motius.md) |
| Sequential generation | [Motius model card and task documentation](https://github.com/ZeyuLing/Motius/blob/main/docs/model_zoo/motioncanvas.md) |

Existing `MotionCanvas` / `motioncanvas` API identifiers and the checkpoint ID
are retained for compatibility; the paper and public project name is **AnimateCanvas**.

## Citation

```bibtex
@misc{ling2026animatecanvas,
  title={AnimateCanvas: Learning Implicit Motion Planning from Composable Kinematic Cues},
  author={Zeyu Ling and Di Kang and Qing Shuai and Yuxin Wen and Jing Li and Zhanke Wang and Heng Li and Chunchao Guo and Changqing Zou and Linchao Bao},
  year={2026}
}
```

The arXiv identifier will be added once the submission has been announced.

## Testing and licenses

```bash
python -m unittest discover -s tests -v
```

These tests check canvas input validation and NPZ output without model downloads.
They are not an end-to-end inference benchmark. See [validation](docs/validation.md)
for the scope of these checks.

The original Python adapter, examples, and tests in this repository are released
under the [MIT license](LICENSE). See [NOTICE](NOTICE.md) for its scope.
Motius and its dependencies retain their applicable terms.
Checkpoints, training data, body models, and character meshes are not included.
Rendered media does not grant redistribution rights to the underlying assets.
