# Validation scope

The standalone package no longer imports or installs Motius. On 2026-09-18,
16 tests passed with PyTorch 2.6.0 CPU in an isolated dependency directory:

- real tiny-transformer sampling with fixed-seed repeatability;
- preservation of full-pose, root-axis, local-rotation and joint-position canvas cues;
- completion, spatial control and editing code paths;
- forward-kinematics skeleton output;
- safetensors checkpoint save/reload and NPZ serialization;
- keyframe preparation and completion CLI execution in separate processes;
- caption/CFG execution with synthetic text embeddings;
- explicit rejection of optional IK projection (available in Motius);
- input validation, loader argument forwarding and no-Motius dependency checks.

A wheel was built and installed into a separate package directory, and the same
16 tests passed outside the source checkout. The imported package path was verified
to be the installed wheel, and `motius` was not importable in that test environment.

The numerical tests use a small randomly initialized network and no pretrained text
encoder. The loader unit test additionally uses mocks to check argument forwarding;
it is not the only inference test. These tests verify execution contracts, **not
pretrained motion quality or full-checkpoint GPU performance**.

The public checkpoint and configuration are anonymously accessible. Its safetensors
header was compared against the standalone model instantiated from the published
configuration: all 306 parameter keys and shapes match, with no missing or extra
tensors. This structural check did not download or execute the tensor weights.

Full generation
with its 0.46B motion model plus pretrained text encoders has not been run in this
release environment. No GPU-memory or runtime benchmark is claimed.
