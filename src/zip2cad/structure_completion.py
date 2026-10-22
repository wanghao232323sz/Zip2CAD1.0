"""Extend a reproducible completion run with finite small roof structures."""
import json,shutil
from pathlib import Path
import numpy as np
from .roof_structures import extract_structures,StructurePolicy
from .completion_pipeline import load_target_mesh
from .native_skp import write_native
from .model_preview import preview_models
from .skp_exchange import sha256_file,canonical_json
from .stage_ledger import inventory

def complete(config_path,output,sdk_path=None):
    config_path=Path(config_path).resolve();output=Path(output).resolve();settings=json.loads(config_path.read_text(encoding='utf-8'))
    parent=(config_path.parent/settings['source_run']).resolve()
    if output.exists():raise ValueError('Fresh structure output required')
    c=json.loads((parent/'resolved_completion_config.json').read_text(encoding='utf-8'))
    plan=json.loads((parent/'completion/native_pointcloud_plan.json').read_text(encoding='utf-8'))
    proof=json.loads((parent/'completion/pointcloud_completion_report.json').read_text(encoding='utf-8'))
    if plan['evidence_report_sha256']!=sha256_file(parent/'completion/pointcloud_completion_report.json'):raise ValueError('Parent evidence changed')
    if plan['source_model_sha256']!=sha256_file(Path(plan['source_model'])):raise ValueError('Original CAD changed')
    cloud_path=parent/'completion'/c.get('attributed_cloud_file_name','refined_ZipMap_cloud.npz')
    with np.load(cloud_path) as z:points=z['points_CAD_m'];refs=z['frame_refs'];pixels=z['source_pixel_index']
    with np.load(c['prediction']) as z:images=z['images'][c['prediction_frame_indices']].transpose(0,2,3,1)
    rgb=np.empty((len(points),3))
    for i,ref in enumerate(c['frame_refs']):
        use=refs==ref;rgb[use]=images[i].reshape(-1,3)[pixels[use]]
    use=points[:,2]>c['CAD_attachment_z_m']+settings.get('minimum_roof_height_m',.18)
    appearance=settings.get('structure_appearance_filter')
    if appearance:
        # Optional explicit material assumption, not a universal semantic
        # classifier. Omit this filter for coloured structural materials.
        use&=(rgb.mean(1)>=appearance['minimum_luminance'])&(np.ptp(rgb,axis=1)<=appearance['maximum_channel_range'])
    mesh,_=load_target_mesh(Path(c['CAD']),c['CAD_scope_prefix']);policy=StructurePolicy(**settings.get('policy',{}))
    result=extract_structures(points[use],refs[use],mesh,c['training_views'],c['heldout_views'],policy,plan['components'])
    from .linear_structures import extract_linear_structures
    linear=extract_linear_structures(points[use],refs[use],mesh,c['training_views'],c['heldout_views'],c['CAD_target_polygon_xy'],plan['components']+result['components'],**settings.get('linear_policy',{}))
    result['components']+=linear['components'];result['candidates']+=linear['candidates'];result['linear_part_count']=len(linear['components']);result['linear_producer_sha256']=sha256_file(Path(__file__).with_name('linear_structures.py'))
    output.mkdir(parents=True);evidence=output/'completion';evidence.mkdir()
    result.update(source_cloud_sha256=sha256_file(cloud_path),source_run=str(parent),point_count=len(points),selected_structure_points=int(use.sum()),appearance_filter=appearance,reconstruction_backend=c.get('reconstruction_backend','ZipMap'),producer_sha256=sha256_file(Path(__file__).with_name('roof_structures.py')))
    structure_path=evidence/'structure_report.json';structure_path.write_text(canonical_json(result),encoding='utf-8')
    proof['components']+=result['candidates'];plan['components']+=result['components'];proof['native_component_count']=len(plan['components']);proof['structure_report_sha256']=sha256_file(structure_path);proof['new_triangle_count']=sum(len(x['triangles']) for x in plan['components']);proof['full_library_roof_completed']=False
    proof['limitations']+=['Additional finite patches are observed surfaces, not a verified exhaustive semantic component inventory.','Single-sided patches use an explicit thickness prior; opposing observed patches may determine thickness.','Optional bright neutral material filter is a declared case configuration.']
    proof_path=evidence/'pointcloud_completion_report.json';proof_path.write_text(canonical_json(proof),encoding='utf-8');plan['evidence_report_sha256']=sha256_file(proof_path)
    if plan.get('appearance_report_sha256'):shutil.copy2(parent/'completion/roof_appearance_report.json',evidence/'roof_appearance_report.json')
    plan_path=evidence/'native_pointcloud_plan.json';plan_path.write_text(canonical_json(plan),encoding='utf-8')
    receipt=write_native(plan_path,output/'native',sdk_path)
    preview_models(receipt['source_model'],receipt['updated_model'],plan['view_scope_pid_path'],receipt['generated_assembly_pid'],output/'preview',sdk_path)
    summary={'schema':'zip2cad_structure_completion_result_v1','reconstruction_backend':result['reconstruction_backend'],'source_model_sha256':plan['source_model_sha256'],'output_model':receipt['updated_model'],'output_model_sha256':sha256_file(Path(receipt['updated_model'])),'original_component_count':len(plan['components'])-len(result['components']),'additional_structure_count':len(result['components']),'total_native_components':len(plan['components']),'new_triangle_count':proof['new_triangle_count'],'source_cloud_sha256':sha256_file(cloud_path),'input_config_sha256':sha256_file(config_path),'producer_sha256':sha256_file(Path(__file__)),'source_run':str(parent),'native_UI_used':False,'hand_drawn_geometry':False,'all_missing_building_parts_completed':False,'independent_survey_accuracy_claimed':False}
    (output/'result.json').write_text(canonical_json(summary),encoding='utf-8');(output/'artifact_manifest.json').write_text(canonical_json({'files':inventory(output)}),encoding='utf-8');print(canonical_json(summary));return summary
