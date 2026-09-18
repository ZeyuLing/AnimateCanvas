"""CPU inference smoke tests with real, tiny randomly initialized networks.

No public checkpoint or pretrained language model is needed for these tests.
They validate execution and cue preservation, not generation quality.
"""
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from animatecanvas import load_pipeline
from animatecanvas.bundle import MotionCanvasBundle
from animatecanvas.pipeline import MotionCanvasPipeline
from animatecanvas.io import save_result


def config():
    return dict(
        motion_transformer=dict(type='MotionCanvasMMDiT', input_dim=594,
            output_dim=198, feat_dim=32, num_heads=4, num_layers=3,
            ctxt_input_dim=16, vtxt_input_dim=8, text_refiner_cfg={'num_layers': 1},
            mask_mode='narrowband', apply_rope_to_single_branch=False,
            insert_start_token=False, time_factor=1000.),
        ctxt_input_dim=16, vtxt_input_dim=8, uncondition_mode=False,
        noise_scheduler_cfg={'method':'euler'}, rotation_space='local')


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.bundle = MotionCanvasBundle(**config())
        self.bundle.mean = torch.zeros(198)
        self.bundle.std = torch.ones(198)
        offsets = torch.zeros(22,3); offsets[1:,1] = .1
        self.bundle.set_bone_offsets_override(offsets)
        self.pipe = MotionCanvasPipeline(self.bundle, num_steps=2,
            official_t2m_frames=8, max_text_len=4, text_guidance_scale=2)

    def test_real_network_generation_and_fk(self):
        result = self.pipe.infer_m2m(num_frames=8, seed=42)
        self.assertEqual(result['motion_198'].shape, (1,8,198))
        self.assertEqual(result['keypoints3d'].shape, (1,8,22,3))
        self.assertTrue(torch.isfinite(result['keypoints3d']).all())
        repeat = self.pipe.infer_m2m(num_frames=8, seed=42)
        torch.testing.assert_close(result['motion_198'], repeat['motion_198'], rtol=0, atol=0)

    def test_mixed_cues_are_preserved(self):
        motion = torch.randn(1,8,198); mask = torch.ones_like(motion)
        mask[:,0] = 0  # full pose
        mask[:,:,0] = 0  # root X only
        mask[:,4,3+20*6:3+21*6] = 0  # local wrist rotation
        mask[:,2,135+19*3+1] = 0  # wrist Y only
        for call in (self.pipe.infer_kinematic_motion_control,
                     self.pipe.infer_temporal_motion_completion,
                     self.pipe.infer_motion_editing):
            result = call(motion, mask, seed=42)
            torch.testing.assert_close(result['motion_198'][mask==0], motion[mask==0], rtol=0, atol=0)

    def test_caption_and_cfg_path(self):
        # Synthetic embeddings exercise the actual text-conditioned network/CFG
        # path, without downloading a pretrained language encoder.
        self.bundle.encode_text = lambda captions: dict(
            text_vec_raw=torch.ones(len(captions),1,8),
            text_ctxt_raw=torch.ones(len(captions),4,16),
            text_ctxt_raw_length=torch.full((len(captions),),4,dtype=torch.long))
        result=self.pipe.infer_text_to_motion('A person walks.',num_frames=8,seed=42)
        self.assertTrue(torch.isfinite(result['motion_198']).all())

    def test_checkpoint_roundtrip_and_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'motioncanvas_config.json').write_text(json.dumps({'config':config()}))
            np.save(root/'Mean.npy',np.zeros(198,dtype=np.float32))
            np.save(root/'Std.npy',np.ones(198,dtype=np.float32))
            torch.save(self.bundle.get_bone_offsets(),root/'bone_offsets_22.pt')
            state = {k:v.contiguous() for k,v in self.bundle.state_dict().items() if k not in ('mean','std')}
            save_file(state,str(root/'motion_transformer.safetensors'))
            loaded = load_pipeline(root,device='cpu',text_dtype='fp32',local_files_only=True,num_steps=2)
            loaded.official_t2m_frames=8; loaded.max_text_len=4
            actual = loaded.infer_m2m(num_frames=8,seed=42)
            expected = self.pipe.infer_m2m(num_frames=8,seed=42)
            torch.testing.assert_close(actual['motion_198'],expected['motion_198'],rtol=0,atol=0)
            save_result(root/'out.npz', actual)
            with np.load(root/'out.npz',allow_pickle=False) as output:
                self.assertEqual(output['rot6d'].shape,(1,8,22,6))
            examples=Path(__file__).resolve().parents[1]/'examples'
            subprocess.run([sys.executable,str(examples/'make_keyframe_cues.py'),
                '--source',str(root/'out.npz'),'--frames','0','7',
                '--output',str(root/'cues.npz')],check=True,capture_output=True)
            subprocess.run([sys.executable,str(examples/'generate.py'),
                '--checkpoint',str(root),'--local-files-only','--device','cpu',
                '--task','completion','--cues',str(root/'cues.npz'),
                '--steps','2','--output',str(root/'cli.npz')],check=True,capture_output=True)
            with np.load(root/'cli.npz',allow_pickle=False) as output:
                self.assertEqual(output['motion_198'].shape,(1,8,198))
                np.testing.assert_array_equal(output['motion_198'][:,[0,7]],actual['motion_198'].numpy()[:,[0,7]])

    def test_optional_ik_is_explicitly_unsupported(self):
        with self.assertRaises(NotImplementedError):
            self.pipe.infer_m2m(num_frames=8,position_constraints=[object()])


if __name__ == '__main__':
    unittest.main()
