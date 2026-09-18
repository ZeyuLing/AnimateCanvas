import tempfile, unittest
from pathlib import Path
import numpy as np
from animatecanvas.io import load_cues, save_result


class CanvasIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"cues.npz"
    def tearDown(self): self.tmp.cleanup()
    def sample(self, **changes):
        data = dict(motion=np.zeros((60,198)), generation_mask=np.ones((60,198)))
        data.update(changes); np.savez(self.path, **data)
        return load_cues(self.path)
    def test_batch_added(self):
        motion,mask=self.sample(); self.assertEqual(motion.shape,(1,60,198)); self.assertEqual(mask.dtype,np.float32)
    def test_axis_only_positions(self):
        mask=np.ones((60,198));mask[[10,30],0]=0
        self.assertEqual(self.sample(generation_mask=mask)[1][0,10,0],0)
    def test_atomic_rotation(self):
        mask=np.ones((60,198));mask[5,3:9]=0;self.sample(generation_mask=mask)
        mask[5,4]=1
        with self.assertRaises(ValueError):self.sample(generation_mask=mask)
    def test_nonfinite(self):
        x=np.zeros((60,198));x[0,0]=np.nan
        with self.assertRaises(ValueError):self.sample(motion=x)
    def test_nonbinary(self):
        with self.assertRaises(ValueError):self.sample(generation_mask=np.full((60,198),0.5))
    def test_size(self):
        with self.assertRaises(ValueError):self.sample(motion=np.zeros((361,198)),generation_mask=np.ones((361,198)))
    def test_save(self):
        out=Path(self.tmp.name)/"result.npz"
        save_result(out,{"motion_198":np.zeros((1,60,198)),"lengths":[60]})
        with np.load(out,allow_pickle=False) as d:self.assertEqual(d['lengths'][0],60)


if __name__ == "__main__": unittest.main()
