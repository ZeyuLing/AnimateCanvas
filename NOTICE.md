# License scope

The MIT license covers the original entry points (`animatecanvas/__init__.py`,
`animatecanvas/io.py`, `animatecanvas/base.py`), examples and tests.

Inference implementation in `animatecanvas/bundle.py`, `animatecanvas/pipeline.py`,
`animatecanvas/network/` and `animatecanvas/kinematics/` is adapted from the author's
public Motius revision `6d259de4672ff33c43a44d948fe8182f2c6eafb2` (2026-09-18 extraction).
Upstream names and notices are retained for attribution and checkpoint compatibility.
Motius is not an installation dependency. This MIT grant does not independently
relicense upstream implementation code, model checkpoints, third-party
libraries, datasets, SMPL-family body models, character meshes, or motion assets.
Consult their respective distributions before use or redistribution.

The paper, rendered videos, figures, and website media are not included in this
code-license grant. Media provenance is documented in `MEDIA.md`. Displaying a
rendered character does not grant rights to redistribute its underlying assets.
