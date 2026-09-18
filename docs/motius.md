# Maintained implementation in Motius

AnimateCanvas is fully integrated in [Motius](https://github.com/ZeyuLing/Motius).
This repository's loader calls the existing bundle and pipeline directly. It
does not wrap a remote inference service or download an executable model script.
Use only trusted checkpoints.

The installation pins the public Motius commit
`6d259de4672ff33c43a44d948fe8182f2c6eafb2` for a reproducible integration snapshot.
The following links expose the actual maintained source, not copied excerpts:

- [Model and bundle](https://github.com/ZeyuLing/Motius/tree/6d259de4672ff33c43a44d948fe8182f2c6eafb2/motius/models/motioncanvas)
- [Inference pipeline](https://github.com/ZeyuLing/Motius/tree/6d259de4672ff33c43a44d948fe8182f2c6eafb2/motius/pipelines/motioncanvas)
- [Training implementation](https://github.com/ZeyuLing/Motius/tree/6d259de4672ff33c43a44d948fe8182f2c6eafb2/motius/trainers/motioncanvas)
- [Compositional cue sampler](https://github.com/ZeyuLing/Motius/blob/6d259de4672ff33c43a44d948fe8182f2c6eafb2/motius/datasets/motion/motionhub/transforms/condition_sampler.py)
- [Training configuration](https://github.com/ZeyuLing/Motius/blob/6d259de4672ff33c43a44d948fe8182f2c6eafb2/configs/motioncanvas/train_motioncanvas_0p46b.py)
- [Model card and advanced usage](https://github.com/ZeyuLing/Motius/blob/main/docs/model_zoo/motioncanvas.md)

For full training, evaluation, motion repair, sequential generation, visualization,
and character export, use the Motius checkout and its task documentation. Dataset
access, pretrained text encoders, body-model assets, and machine-specific paths
must be configured separately. A training configuration is not a redistribution
of the training corpus.

The legacy `motioncanvas` package names and artifact identifiers intentionally
remain unchanged. Renaming files inside a downloaded checkpoint can break loading.
