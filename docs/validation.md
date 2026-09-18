# Validation scope

This release entry point is a thin adapter to the public Motius implementation.
Input shape, finite values, binary masks, rotation mask grouping, and safe NPZ
serialization are covered by CPU tests requiring NumPy only.

On 2026-09-18, all nine canvas-I/O and mocked-loader tests passed with
`python -m unittest discover -s tests -v`. README preview checks at 1280 px and
390 px found no page overflow or broken images. This was a local GitHub-like
Markdown rendering, not a check of a live GitHub repository.

No model weights are bundled. Full checkpoint inference and CUDA performance have
not been validated in this release-preparation environment. Do not interpret unit
test success as a new motion-quality benchmark or a tested hardware requirement.

Anonymous Hub API access was verified on 2026-09-18: the model repository is public
and non-gated. This access check is separate from end-to-end inference validation.
