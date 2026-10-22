"""Create a clearly synthetic cached-prediction fixture for all 14 IO stages.

This does not run or simulate the accuracy of ZipMap. Use the official mode
for actual reconstruction. A local SketchUp SDK is needed to create the CAD.
"""
import argparse,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np,trimesh
from zip2cad.release_io import save,sha
from zip2cad.sketchup_sdk import SketchUpSDK


def create(output,sdk=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    cad=output/'original.skp';mesh=trimesh.creation.box(extents=[10,8,6]);mesh.apply_translation([40,-12,3])
    with SketchUpSDK(sdk) as api:
        model=api.ref('SUModelCreate')
        try:
            api.add_solid(model,api.ref('SUModelGetEntities',model),dict(component_id='synthetic_existing_body',vertices_world_CAD_m=mesh.vertices.tolist(),triangles=mesh.faces.tolist(),top_triangle_mask=[False]*len(mesh.faces),top_color=[180,180,180],side_color=[210,210,210]))
            api.call('SUModelSaveToFile',model,str(cad).encode('utf-8'))
        finally:api.release(model)
    video=output/'synthetic.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),2,(160,120))
    images=[]
    for i in range(8):
        yy,xx=np.indices((120,160));checker=((xx//10+yy//10+i)%2)*110+60
        image=np.stack([checker,np.roll(checker,i*5,axis=1),np.full_like(checker,70+i*15)],axis=-1).astype(np.uint8)
        images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1)/255);writer.write(image)
    writer.release()
    x,y=np.meshgrid(np.linspace(35.3,44.7,160),np.linspace(-15.7,-8.3,120))
    point=np.stack([x,y,9-abs(x-40)*.5],-1);rng=np.random.default_rng(5)
    points=np.stack([point+rng.normal(0,.006,point.shape) for _ in range(8)]).astype(np.float32)
    prediction=output/'synthetic_cached_predictions.npz'
    np.savez_compressed(prediction,world_points=points,world_points_conf=np.ones((8,120,160),np.float32),images=np.array(images,dtype=np.float32))
    identity=dict(scale=1.,rotation=np.eye(3).tolist(),translation=[0,0,0])
    cfg=dict(schema='zip2cad_pipeline_v1',project_root='.',video=video.name,source_cad=cad.name,
       video_preprocessing=dict(sample_fps=2,min_sharpness=0,min_frame_difference=0,block_size=8,block_overlap=2),
       reconstruction=dict(mode='cached',source_video_sha256=sha(video),blocks=[dict(path=prediction.name,frame_refs=list(range(8)),to_common=identity)]),
       fusion=dict(confidence_percentile=0,pixel_stride=1),registration=dict(mode='manual_similarity',transform=identity),
       neural=dict(enabled=False),roof_context=dict(voxel_m=.15,xy_margin_m=1),synthetic_fixture=True)
    save(output/'project.json',cfg);save(output/'SYNTHETIC_NOTICE.json',dict(synthetic=True,official_ZipMap_inference_executed=False,purpose='14-stage software integration only'))
    return output/'project.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(create(a.output))
