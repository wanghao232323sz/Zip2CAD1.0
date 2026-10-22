"""Verify public task weights and exercise inference on synthetic inputs only."""
import hashlib
import json
import tempfile
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from zip2cad.release_neural import corner_inference
    from zip2cad.release_semantic import semantics

    manifest = json.loads((ROOT / 'models/pretrained/manifest.json').read_text(encoding='utf8'))
    for row in manifest['models'].values():
        assert hashlib.sha256((ROOT / row['file']).read_bytes()).hexdigest() == row['sha256']
    rng = np.random.default_rng(21)
    points = rng.random((128, 3)).astype(np.float32)
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp)
        np.savez_compressed(p / 'points.npz', points_CAD_m=points,
                            rgb=points, frame_refs=np.arange(128) % 4)
        corner = corner_inference(p / 'points.npz', ROOT / manifest['models']['corners']['file'],
                                  p / 'corners.json')
        assert np.isfinite(corner['predictions_world_m']).all()
        state = torch.load(ROOT / manifest['models']['semantic']['file'], weights_only=True)
        features = np.broadcast_to(state['feature_mean'].numpy(), (128, 1088)).copy()
        probability = semantics(features, points, ROOT / manifest['models']['semantic']['file'])
        assert probability.shape == (128, 9) and np.isfinite(probability).all()
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
        # A synthetic frame verifies decoding, without publishing campus CAD geometry.
        frame = dict(origin_xy=[0., 0.], basis_xy=[[1., 0.], [0., 1.]],
                     span_m=[20., 20.], base_z_m=0.,
                     outline_normalized=[[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        from zip2cad.release_neural import build_joint_inputs
        build_joint_inputs(p / 'points.npz', probability, frame, p / 'joint')
        from zip2cad.roof_branch_geometry import BranchSeparatedGeometryNet, state_digest
        from zip2cad.roof_joint_geometry import metric_parameters
        state = torch.load(ROOT / manifest['models']['geometry']['file'], weights_only=True)
        model = BranchSeparatedGeometryNet(state['freeze_room']).eval()
        model.load_state_dict(state['model'])
        assert state_digest(model.room_model) == state['initial_room_state_digest']
        with np.load(p / 'joint/all.npz') as data, torch.inference_mode():
            vector = model(torch.from_numpy(data['features'][None]))[0].numpy()
        assert vector.shape == (25,) and np.isfinite(vector).all()
        assert len(metric_parameters(vector, frame)['ducts']) == 2
    print(json.dumps(dict(status='passed', models=3, reference_cad_read=False,
                          inputs='synthetic software smoke only; not accuracy validation'), indent=2))


if __name__ == '__main__':
    main()
