import json,subprocess,sys,unittest
from pathlib import Path
import numpy as np
import trimesh
from zip2cad.release_pipeline import STAGES,joint_frame
from zip2cad.release_io import similarity
from zip2cad.delivery_metrics import compare_surfaces,surface_sample


class PortableCore(unittest.TestCase):
    def test_fourteen_steps(self):
        self.assertEqual(len(STAGES),14);self.assertEqual(len(set(STAGES)),14)

    def test_similarity_and_rotation_guard(self):
        t=dict(scale=2,rotation=np.eye(3).tolist(),translation=[1,2,3])
        np.testing.assert_allclose(similarity([[1,2,3]],t),[[3,6,9]])
        t['rotation']=(np.eye(3)*2).tolist()
        with self.assertRaises(ValueError):similarity([[1,2,3]],t)

    def test_frame_roundtrip(self):
        outline=np.array([[10,20],[20,20],[20,30],[10,30]])
        f=joint_frame(outline,5);b=np.array(f['basis_xy']);span=np.array(f['span_m']);o=np.array(f['origin_xy'])
        np.testing.assert_allclose(np.array(f['outline_normalized'])*span@b.T+o,outline)
        self.assertFalse(f['reference_roof_used'])

    def test_identical_surface(self):
        m=trimesh.creation.box();v,a,b=compare_surfaces(m,m,count=200)
        self.assertLess(max(a.max(),b.max()),1e-12);self.assertEqual(v['thresholds'][1]['F1'],1)
        np.testing.assert_array_equal(surface_sample(m,100),surface_sample(m,100))

    def test_known_plane_separation(self):
        a=trimesh.Trimesh([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],[[0,1,2],[0,2,3]],process=False);b=a.copy();b.apply_translation([0,0,.4])
        score,x,y=compare_surfaces(a,b,count=200);np.testing.assert_allclose(x,.4);np.testing.assert_allclose(y,.4)
        self.assertEqual(score['thresholds'][1]['F1'],0)

    def test_missing_surface_recall(self):
        a=trimesh.creation.box();b=a.copy();b.apply_translation([5,0,0]);v,_,_=compare_surfaces(a,trimesh.util.concatenate([a,b]),count=2000)
        self.assertEqual(v['thresholds'][0]['precision'],1);self.assertTrue(.4<v['thresholds'][0]['recall']<.6)


if __name__=='__main__':unittest.main()
