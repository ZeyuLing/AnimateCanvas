"""Text generation or canvas-conditioned completion, control, and editing."""
import argparse
from animatecanvas import DEFAULT_CHECKPOINT, load_pipeline
from animatecanvas.io import load_cues, save_result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--task", choices=["text", "completion", "control", "edit"], default="text")
    p.add_argument("--text")
    p.add_argument("--cues", help="Physical-space NPZ with motion and generation_mask")
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir")
    p.add_argument("--revision")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--output", default="outputs/motion.npz")
    a = p.parse_args()
    if not 1 <= a.frames <= 360: p.error("--frames must be within 1..360")
    if a.steps < 1 or a.seed < 0: p.error("steps must be positive; seed must be nonnegative")
    if a.task == "text" and (not a.text or a.cues): p.error("Text generation needs --text and no --cues")
    if a.task != "text" and not a.cues: p.error("This task requires --cues")
    if a.task == "edit" and not a.text: p.error("Editing requires a language instruction in --text")
    cues = load_cues(a.cues) if a.cues else None
    import torch
    pipe = load_pipeline(a.checkpoint, device=a.device, num_steps=a.steps,
                         cache_dir=a.cache_dir, revision=a.revision, local_files_only=a.local_files_only)
    with torch.inference_mode():
        if a.task == "text":
            result = pipe.infer_text_to_motion(a.text, num_frames=a.frames, seed=a.seed)
        else:
            motion, mask = [torch.from_numpy(x).to(a.device) for x in cues]
            call = {"completion": pipe.infer_temporal_motion_completion,
                    "control": pipe.infer_kinematic_motion_control,
                    "edit": pipe.infer_motion_editing}[a.task]
            result = call(motion, mask, captions=a.text, seed=a.seed)
    save_result(a.output, result)
    print(f"Saved {a.output}")


if __name__ == "__main__": main()
