import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional neural extra not installed')
class NeuralSmoke(unittest.TestCase):
    def test_corner_training_and_inference_contract(self):
        import torch
        from zip2cad.release_neural import train_corners,corner_inference
        from zip2cad.research_roof_keypoints import RoofKeypointHead
        from zip2cad.roof_branch_geometry import state_digest
        root=Path(__file__).resolve().parents[1];base=root/'outputs/unit_test_temp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            p=Path(tmp);rng=np.random.default_rng(1);points=rng.random((128,3));rgb=rng.random((128,3)).astype(np.float32)
            np.savez_compressed(p/'input.npz',points_CAD_m=points,rgb=rgb,frame_refs=np.arange(128)%4)
            truth=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,1,1]],float)
            np.savez_compressed(p/'targets.npz',target_world_m=truth,weights=np.ones(6))
            torch.manual_seed(20);before=state_digest(RoofKeypointHead())
            train_corners(p/'input.npz',p/'targets.npz',p/'train',steps=1)
            checkpoint=torch.load(p/'train/checkpoint.pt',weights_only=True);model=RoofKeypointHead();model.load_state_dict(checkpoint['model'])
            self.assertNotEqual(state_digest(model),before);self.assertTrue(checkpoint['smoke_only'])
            self.assertNotIn('target_world_m',checkpoint)
            result=corner_inference(p/'input.npz',p/'train/checkpoint.pt',p/'prediction.json',p/'train/input_contract.json')
            self.assertFalse(result['reference_read']);self.assertEqual(np.asarray(result['predictions_world_m']).shape,(6,3))

    def test_joint_input_rejects_reference_contract(self):
        from zip2cad.release_neural import joint_inference
        root=Path(__file__).resolve().parents[1];base=root/'outputs/unit_test_temp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            p=Path(tmp);(p/'input_contract.json').write_text(json.dumps(dict(reference_geometry_in_inputs=True,files=[dict(file='all.npz',sha256='bad')])),encoding='utf8')
            (p/'all.npz').write_bytes(b'not an input')
            with self.assertRaises(ValueError):joint_inference(p,p/'nonexistent.pt')


if __name__=='__main__':unittest.main()
