# Standalone inference and Motius integration

AnimateCanvas contains its own inference implementation and has **no Motius
installation dependency**. Core sources were extracted from public Motius revision
`6d259de4672ff33c43a44d948fe8182f2c6eafb2`, preserving transformer parameter names and
the published checkpoint format.

## Included here

- `animatecanvas/network/`: MMDiT backbone, attention, embeddings and text encoding.
- `animatecanvas/bundle.py`: Hub/local loading, normalization and motion decoding.
- `animatecanvas/pipeline.py`: sampling, classifier-free guidance and cue imputation.
- `animatecanvas/kinematics/`: rotation conversions and skeleton forward kinematics.
- `examples/`: text generation, completion, control, editing and keyframe preparation.

The registry, trainer, dataset framework, unrelated models and demo production tools
are not included. Training-only loss and freezing settings in existing checkpoint
metadata are accepted for compatibility but do not execute a training framework.
Skeleton decoding uses the checkpoint's own `bone_offsets_22.pt`; proprietary mesh
or body assets are not required.

## Full Motius implementation

For automatic repair, extra IK projection, training, evaluation, sequential workflows,
visualization and character export, use [Motius](https://github.com/ZeyuLing/Motius):

- [Model documentation](https://github.com/ZeyuLing/Motius/blob/main/docs/model_zoo/motioncanvas.md)
- [Trainer](https://github.com/ZeyuLing/Motius/tree/6d259de4672ff33c43a44d948fe8182f2c6eafb2/motius/trainers/motioncanvas)
- [Training configuration](https://github.com/ZeyuLing/Motius/blob/6d259de4672ff33c43a44d948fe8182f2c6eafb2/configs/motioncanvas/train_motioncanvas_0p46b.py)

The public Hub ID and internal `MotionCanvas` class names are retained for checkpoint
compatibility. Neither repository distributes the training corpus or third-party
character/body assets.
