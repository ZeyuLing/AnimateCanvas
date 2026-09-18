"""Keep selected full-pose frames from a saved motion as completion cues."""
import argparse
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--frames',type=int,nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with np.load(args.source,allow_pickle=False) as data:
        if 'motion_198' not in data:
            parser.error('Source must contain motion_198, as produced by generate.py')
        motion=np.asarray(data['motion_198'],dtype=np.float32)
    if motion.ndim not in (2,3) or motion.shape[-1]!=198 or not np.isfinite(motion).all():
        parser.error('Source must be finite (T,198) or (B,T,198) motion')
    if not 1 <= motion.shape[-2] <= 360:
        parser.error('Source must contain 1..360 frames')
    if any(i<0 or i>=motion.shape[-2] for i in args.frames):
        parser.error('Keyframe index outside source motion; indices start at zero')
    mask=np.ones_like(motion); mask[...,args.frames,:]=0
    cues=np.where(mask==0,motion,0)
    output=Path(args.output)
    if output.suffix.lower()!='.npz': parser.error('Output must end in .npz')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,motion=cues,generation_mask=mask)
    print(f'Saved cues at frames {sorted(set(args.frames))}: {output}')

if __name__=='__main__': main()
