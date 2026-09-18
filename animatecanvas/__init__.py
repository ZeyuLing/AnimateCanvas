"""Standalone AnimateCanvas inference; no Motius installation required."""
DEFAULT_CHECKPOINT = "ZeyuLing/Motius-MotionCanvas-0.46B"


def load_pipeline(checkpoint=DEFAULT_CHECKPOINT, *, device="cuda", text_dtype="bf16",
                  cache_dir=None, revision=None, local_files_only=False,
                  num_steps=50, guidance_scale=2.0):
    """Load a trusted local bundle or a Hub checkpoint accessible to your account.

    Legacy MotionCanvas module identifiers are preserved for checkpoint compatibility.
    Network, sampler, imputation, text encoding, and skeleton decoding are
    included in this package. The legacy Hub artifact remains compatible.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    from .bundle import MotionCanvasBundle
    from .pipeline import MotionCanvasPipeline
    bundle = MotionCanvasBundle.from_pretrained(
        str(checkpoint), device=device, text_dtype=text_dtype, cache_dir=cache_dir,
        revision=revision, local_files_only=local_files_only,
    )
    bundle.eval()
    return MotionCanvasPipeline(bundle, num_steps=num_steps,
                                text_guidance_scale=guidance_scale,
                                replacement_guidance="all")
