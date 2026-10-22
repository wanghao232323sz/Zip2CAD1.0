"""General finite wall and roof surfaces from a source-attributed metric cloud."""
import json,shutil
from pathlib import Path
import numpy as np,trimesh
from .sketchup_sdk import SketchUpSDK
from .structural_surfaces import SurfacePolicy,extract_structural_surfaces
from .native_skp import write_native
from .skp_exchange import canonical_json,sha256_file
from .stage_ledger import inventory

def complete_cloud(cad,cloud,output,training,heldout,scope=None,sdk_path=None,transform=None,thickness=.2):
    cad,cloud,output=map(lambda p:Path(p).resolve(),(cad,cloud,output))
    if output.exists() or cad.is_relative_to(output) or cloud.is_relative_to(output):raise ValueError('Fresh output separate from inputs required')
    if not cad.is_file() or not cloud.is_file():raise ValueError('Source CAD and attributed cloud files required')
    with np.load(cloud) as z:
        points=z['points_CAD_m'].copy() if 'points_CAD_m' in z else z['points'].copy();refs=z['frame_refs'].copy()
        if 'points_CAD_m' not in z and transform is None:raise ValueError('Arbitrary-scale cloud requires --transform; points_CAD_m is an explicit metric declaration')
    if transform:
        t=json.loads(Path(transform).read_text(encoding='utf-8'));r=np.asarray(t['rotation'],float);s=float(t['scale']);translation=np.asarray(t['translation'],float)
        if r.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(translation).all() or not np.isfinite(s) or s<=0 or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or np.linalg.det(r)<.999:raise ValueError('A positive proper similarity transform is required')
        points=s*(points@r.T)+translation
    with SketchUpSDK(sdk_path) as sdk:
        model=sdk.load(cad)
        try:data=sdk.geometry(model,scope=scope)
        finally:sdk.release(model)
    mesh=trimesh.Trimesh(data['vertices_world_m'],data['faces'],process=False);mesh.update_faces(mesh.nondegenerate_faces())
    if not len(mesh.faces):raise ValueError('Source CAD scope contains no surfaces')
    report=extract_structural_surfaces(points,refs,mesh,training,heldout,SurfacePolicy(thickness_prior_m=thickness));output.mkdir(parents=True)
    report.update(source_cloud_sha256=sha256_file(cloud),source_model_sha256=sha256_file(cad),producer_sha256=sha256_file(Path(__file__)),training_views=training,heldout_views=heldout,
                  component_count=len(report['components']),coordinate_system='CAD_metres',source_frame_attribution_is_input=True)
    (output/'structural_surface_report.json').write_text(canonical_json(report),encoding='utf-8')
    if report['components']:
        evidence={'components':[dict(r,native_edit_approved=r['accepted']) for r in report['candidate_rows']]}
        evidence_path=output/'pointcloud_completion_report.json';evidence_path.write_text(canonical_json(evidence),encoding='utf-8')
        plan={'schema':'zip2cad_native_completion_plan_v2','source_model':str(cad),'source_model_sha256':sha256_file(cad),
              'components':report['components'],'textures_allowed':False,'hide_existing_pid_paths':[],'view_scope_pid_path':scope or [],'evidence_report_sha256':sha256_file(evidence_path)}
        plan_path=output/'native_pointcloud_plan.json';plan_path.write_text(canonical_json(plan),encoding='utf-8');receipt=write_native(plan_path,output/'native',sdk_path)
        target=Path(receipt['updated_model']);status='completed_native_engineering_preview'
    else:
        target=output/'unchanged.skp';shutil.copy2(cad,target);status='no_supported_edit'
    result={'status':status,'output_model':str(target),'output_model_sha256':sha256_file(target),'approved_component_count':len(report['components']),
            'native_UI_used':False,'Ruby_used':False,'all_missing_building_parts_completed':False,'whole_building_survey_certified':False}
    (output/'result.json').write_text(canonical_json(result),encoding='utf-8');(output/'artifact_manifest.json').write_text(canonical_json({'files':inventory(output)}),encoding='utf-8');print(canonical_json(result));return result
