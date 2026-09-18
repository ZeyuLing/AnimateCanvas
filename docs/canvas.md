# Preparing a motion canvas

Use native physical-space arrays, not a rendered mesh or HumanML3D feature vector.
The checkpoint's representation is 198-dimensional at 30 fps:

| Channels | Values |
| --- | --- |
| `0:3` | Root translation XYZ |
| `3:135` | 22 local joint rotations, six channels each |
| `135:198` | 21 non-root positions relative to the pelvis, XYZ |

Native positions and root translation are in meters. Paper errors in millimeters
do not change input units. The pelvis-relative position axes retain world-axis
orientation rather than being inverse-rotated by the root heading. World-space
position requirements therefore need consistent root translation as well.

Rotation-6D uses the first two matrix columns flattened in row-major order;
identity is `[1, 0, 0, 1, 0, 0]`. Keep all six channels of a rotation cue together.
Position cues can select individual axes. Follow the checkpoint's SMPL-22 joint
order, root offset, normalization and skeleton conventions.

For example, to preserve only root XZ at selected frames of a compatible motion:

```python
import numpy as np

with np.load("reference_native_198.npz", allow_pickle=False) as data:
    motion = data["motion"].astype(np.float32)  # (T, 198)
mask = np.ones_like(motion)
frames = np.array([0, 30, 90])
mask[frames[:, None], np.array([0, 2])[None, :]] = 0
np.savez_compressed("cues.npz", motion=motion, generation_mask=mask)
```

The sequence must contain these frames. For authored waypoints, replace the
corresponding root XZ values before saving; keep coordinates consistent with the
rest of the supplied cues. This example does not invent an identity skeleton or
fake checkpoint normalization. The pipeline normalizes raw motion internally.

For a complete pose keyframe, set `mask[frame, :] = 0`. For editing, retain the
original input motion throughout the array and zero only the mask entries that
should remain fixed. The generated regions receive editing context, not hard cues.

The CLI saves tensor outputs as NPZ, including `motion_198` and valid lengths.
FBX/mesh export and retargeting are provided by Motius rather than this small CLI.
