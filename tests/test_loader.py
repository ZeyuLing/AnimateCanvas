import sys, types, unittest
from unittest.mock import Mock, patch
from animatecanvas import load_pipeline


class LoaderTests(unittest.TestCase):
    def test_load_delegates_without_changing_checkpoint(self):
        model = types.ModuleType("motius.models.motioncanvas")
        pipes = types.ModuleType("motius.pipelines.motioncanvas")
        bundle = Mock(); model.MotionCanvasBundle = Mock()
        model.MotionCanvasBundle.from_pretrained.return_value = bundle
        pipes.MotionCanvasPipeline = Mock()
        with patch.dict(sys.modules,{model.__name__:model,pipes.__name__:pipes}):
            result = load_pipeline("/model", local_files_only=True, num_steps=25)
        model.MotionCanvasBundle.from_pretrained.assert_called_once_with(
            "/model", device="cuda", text_dtype="bf16", cache_dir=None,
            revision=None, local_files_only=True)
        bundle.eval.assert_called_once()
        pipes.MotionCanvasPipeline.assert_called_once_with(
            bundle,num_steps=25,text_guidance_scale=2.0,replacement_guidance="all")
        self.assertIs(result,pipes.MotionCanvasPipeline.return_value)

    def test_invalid_steps_fail_before_import(self):
        with self.assertRaises(ValueError): load_pipeline(num_steps=0)


if __name__ == "__main__": unittest.main()
