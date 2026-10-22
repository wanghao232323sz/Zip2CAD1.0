"""Synthetic, openly reproducible wall/roof example; not ZipMap evidence."""
from pathlib import Path
import numpy as np,trimesh
from .sketchup_sdk import SketchUpSDK
from .cloud_cli import complete_cloud
from .model_preview import preview_models
from .skp_exchange import canonical_json

def demo(output,sdk_path=None):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh demo output required')
    output.mkdir(parents=True);source=output/'synthetic_building.skp'
    box=trimesh.creation.box(extents=[10,8,6]);box.apply_translation([40,-12,3])
    with SketchUpSDK(sdk_path) as sdk:
        model=sdk.ref('SUModelCreate')
        try:
            sdk.add_solid(model,sdk.ref('SUModelGetEntities',model),{'component_id':'synthetic_existing_body','vertices_world_CAD_m':box.vertices.tolist(),'triangles':box.faces.tolist(),
                          'top_triangle_mask':[False]*len(box.faces),'top_color':[210,210,205],'side_color':[231,231,226]})
            sdk.call('SUModelSaveToFile',model,str(source).encode())
        finally:sdk.release(model)
    points=[];refs=[];rng=np.random.default_rng(30)
    for ref in [10,20,30,40]:
        for sign in [-1,1]:
            x,y=np.meshgrid(np.arange(.3,4.8,.15)*sign+40,np.arange(-15.7,-8.2,.15));p=np.c_[x.ravel(),y.ravel(),9-abs(x.ravel()-40)*.5]
            points.append(p+rng.normal(0,.012,p.shape));refs.extend([ref]*len(p))
        y,z=np.meshgrid(np.arange(-15.5,-8.5,.15),np.arange(.5,5.5,.15));p=np.c_[np.full(y.size,46.5),y.ravel(),z.ravel()]
        points.append(p+rng.normal(0,.012,p.shape));refs.extend([ref]*len(p))
    cloud=output/'synthetic_observations.npz';np.savez_compressed(cloud,points_CAD_m=np.concatenate(points),frame_refs=refs)
    result=complete_cloud(source,cloud,output/'completion',[10,20,30],[40],sdk_path=sdk_path)
    import json
    receipt=json.loads((output/'completion/native/sdk_native_receipt.json').read_text())
    preview_models(source,result['output_model'],[],receipt['generated_assembly_pid'],output/'preview',sdk_path)
    (output/'DEMO_NOTICE.json').write_text(canonical_json({'synthetic_data':True,'actual_ZipMap_inference_used':False,'purpose':'Reproducible arbitrary-direction plane extraction and standalone SKP writeback','component_count':result['approved_component_count']}),encoding='utf-8')
    return result
