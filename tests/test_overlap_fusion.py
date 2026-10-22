import json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from zip2cad.release_io import sha
from zip2cad.release_pipeline import run


class OverlapFusion(unittest.TestCase):
    def test_two_blocks_align_and_count_physical_frames_once(self):
        base=Path(__file__).resolve().parents[1]/'outputs/unit_test_temp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            p=Path(tmp);video=p/'six_frames.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),1,(64,48))
            yy,xx=np.indices((48,64))
            for i in range(6):
                gray=(((xx+i*3)//5+yy//5)%2*90+70).astype(np.uint8);writer.write(np.stack([gray,gray,gray],axis=-1))
            writer.release();(p/'original.skp').write_bytes(b'CAD placeholder; test ends before SDK import')
            points=np.stack([np.stack([xx*.1,yy*.1,np.sin(xx*.1)*.3+i*.02],-1) for i in range(6)])
            theta=.23;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]);s=.8;t=np.array([2,1,-.5])
            for name,refs,cloud in [('a',[0,1,2,3],points[:4]),('b',[2,3,4,5],((points[2:]-t)@r)/s)]:
                np.savez_compressed(p/(name+'.npz'),world_points=cloud.astype(np.float32),world_points_conf=np.ones((4,48,64),np.float32),images=np.full((4,3,48,64),.5,np.float32))
            cfg=dict(schema='zip2cad_pipeline_v1',project_root='.',video=video.name,source_cad='original.skp',
                 video_preprocessing=dict(sample_fps=1,min_sharpness=0,min_frame_difference=0,block_size=4,block_overlap=2),
                 reconstruction=dict(mode='cached',source_video_sha256=sha(video),blocks=[dict(path='a.npz',frame_refs=[0,1,2,3]),dict(path='b.npz',frame_refs=[2,3,4,5])]),
                 fusion=dict(confidence_percentile=0,pixel_stride=2),neural=dict(enabled=False))
            (p/'project.json').write_text(json.dumps(cfg),encoding='utf8');out=run(p/'project.json','run',to_stage=4)
            metadata=json.loads((out/'04_fusion.json').read_text(encoding='utf8'));self.assertEqual(metadata['physical_views'],6)
            with np.load(out/'04_fused_arbitrary.npz') as z:
                for i in range(6):np.testing.assert_allclose(z['points'][z['frame_refs']==i],points[i,::2,::2].reshape(-1,3),atol=1e-5)


if __name__=='__main__':unittest.main()
