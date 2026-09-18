"""Checkpoint-compatible AnimateCanvas motion transformer."""
from .hymotion_mmdit import HunyuanMotionMMDiT

class MotionCanvasMMDiT(HunyuanMotionMMDiT):
    """Keep the published architecture and parameter names unchanged."""
