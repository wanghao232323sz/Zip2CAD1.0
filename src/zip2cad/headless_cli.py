"""Portable command entry for video/ZipMap/CAD completion and native C SDK IO."""
import argparse,json,os,sys,subprocess
from pathlib import Path
import numpy as np,trimesh,cv2
from scipy.spatial import ConvexHull
from .sketchup_sdk import SketchUpSDK,discover_sdk,SDKError
from .native_skp import write_native
from .skp_exchange import canonical_json,sha256_file
from .stage_ledger import inventory

PROFILES={
    'conservative':{'voxel_m':.5,'minimum_area_m2':15.,'plane_tolerance_m':.30,'maximum_components':6,'minimum_training_views':2,'minimum_heldout_fraction':.7,'minimum_change_m':.5,'maximum_boundary_vertices':12,'maximum_roof_slope_degrees':20.},
    'engineering':{'voxel_m':.4,'minimum_area_m2':4.,'plane_tolerance_m':.4,'maximum_components':16,'minimum_training_views':2,'minimum_heldout_fraction':.5,'minimum_change_m':.25,'maximum_boundary_vertices':12,'maximum_roof_slope_degrees':55.}}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def relative_input(value,base):
    p=Path(value).expanduser();return (p if p.is_absolute() else base/p).resolve()

def export_cad(source,output,scope=None,namespace='building',sdk_path=None):
    with SketchUpSDK(sdk_path) as sdk:
        model=sdk.load(source)
        try:
            data=sdk.geometry(model,namespace,include_appearance=True);np.savez_compressed(output,**{k:v for k,v in data.items() if k!='rows'})
            selected=np.array([r['pid_path'][:len(scope)]==list(scope) for r in data['rows']]) if scope else np.ones(len(data['rows']),bool)
            target=trimesh.Trimesh(data['vertices_world_m'],data['faces'][selected[data['face_owner_indices']]],process=False)
            if len(target.faces)<4:raise ValueError('CAD target scope has no geometry')
            bounds=target.bounds;centres=target.triangles_center;area=target.area_faces
            near=(abs(target.face_normals[:,2])>.9)&(centres[:,2]>bounds[1,2]-max(2.,np.ptp(bounds[:,2])*.15))
            if not near.any():raise ValueError('No upper CAD support plane; supply a target polygon and support height')
            levels=np.round(centres[near,2]/.1).astype(int);values=np.unique(levels);level=max(values,key=lambda x:area[near][levels==x].sum())
            support=near&(abs(centres[:,2]-.1*level)<.06);base=float(np.average(centres[support,2],weights=area[support]))
            xy=target.triangles[support,:,:2].reshape(-1,2);outline=xy[ConvexHull(xy).vertices]
            # A simplified CAD's support slab can protrude past the actual
            # upper roof frame. Prefer the upper edge envelope when it spans
            # the principal roof; this avoids classifying coping as an interior
            # addition solely because the lower slab has a curved cantilever.
            upper=(abs(target.face_normals[:,2])>.9)&(centres[:,2]>=base+.25)&(centres[:,2]<=base+1.2)
            outline_source='CAD_support_plane_envelope'
            if upper.sum()>=4:
                edge_xy=target.triangles[upper,:,:2].reshape(-1,2);edge_outline=edge_xy[ConvexHull(edge_xy).vertices]
                ratio=cv2.contourArea(edge_outline.astype(np.float32))/cv2.contourArea(outline.astype(np.float32))
                if .5<=ratio<=1.25:outline=edge_outline;outline_source='CAD_upper_roof_edge_envelope'
            return {'CAD_attachment_z_m':base,'CAD_target_polygon_xy':outline.tolist(),'source_model_sha256':sha256_file(source),
                'scope_pid_path':scope or [],'outline_source':outline_source,'auto_support_is_existing_CAD_prior':True,'convex_target_envelope_used':True,'target_triangle_count':len(target.faces)}
        finally:sdk.release(model)

def complete(config_path,output,sdk_path=None):
    from .completion_pipeline import run
    config_path=Path(config_path).resolve();config=read(config_path);base=config_path.parent
    if config.get('schema')!='zip2cad_headless_project_v1':raise ValueError('Expected zip2cad_headless_project_v1 project configuration')
    output=Path(output).resolve()
    if output.exists():raise ValueError('Output folder must be fresh')
    input_names=['source_model','video']+(['camera_inputs'] if config.get('camera_inputs') else [])+(['prediction'] if not config.get('prediction_blocks') else [])
    inputs={k:relative_input(config[k],base) for k in input_names}
    if any(output==p or p.is_relative_to(output) for p in inputs.values()):raise ValueError('Output overlaps input resources')
    for name,p in inputs.items():
        if not p.is_file():raise ValueError('Missing '+name+': '+str(p))
    # The original video is examined as input; cached predictions are bound to
    # frame timestamps by the configured manifest, when supplied.
    cap=cv2.VideoCapture(str(inputs['video']))
    if not cap.isOpened():raise ValueError('Cannot read source video')
    video_metadata={'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':float(cap.get(cv2.CAP_PROP_FPS))};cap.release()
    output.mkdir(parents=True)
    scope=config.get('scope_pid_path',[]);namespace=config.get('project_id','building')
    cad=output/'source_cad.npz';context=export_cad(inputs['source_model'],cad,scope,namespace,sdk_path)
    if 'camera_inputs' in inputs:camera=read(inputs['camera_inputs'])
    else:
        from .auto_registration import initialize_cameras
        first=config['prediction_blocks'][0] if config.get('prediction_blocks') else {'path':str(inputs['prediction'])}
        prediction=relative_input(first['path'],base)
        with np.load(cad) as z:
            prefix='skp:'+namespace+':'+('.'.join(map(str,scope))+'.' if scope else '');owner=np.array([str(s).startswith(prefix) for s in z['entity_ids']]);mesh=trimesh.Trimesh(z['vertices_world_m'],z['faces'][owner[z['face_owner_indices']]],process=False)
        mesh.update_faces(mesh.nondegenerate_faces())
        with np.load(prediction) as z:
            n=len(z['images']);choose=np.linspace(0,n-1,min(8,n),dtype=int);mapping=first.get('frame_refs',config.get('frame_refs',list(range(n))))
            camera=initialize_cameras(z,choose,[mapping[i] for i in choose],mesh,[video_metadata['width'],video_metadata['height']])
        inputs['camera_inputs']=output/'automatic_camera_inputs.json';inputs['camera_inputs'].write_text(canonical_json(camera),encoding='utf-8')
        if not config.get('prediction_blocks'):config['prediction_frame_indices']=choose.tolist()
    refs=camera['frame_refs'];indices=config.get('prediction_frame_indices',list(range(len(refs))))
    automatic_initialization=bool(camera.get('initialization_report'))
    if len(refs)!=len(indices) or len(refs)<5:raise ValueError('At least five source-bound calibrated views are required')
    if set(camera['training_views'])&set(camera['heldout_views']):raise ValueError('Training and heldout frames overlap')
    coverage=None
    if config.get('prediction_blocks'):
        from .full_video_completion import prepare_views
        prepared=output/'video_views';prepared.mkdir()
        inputs['prediction'],camera,coverage=prepare_views(config,base,inputs['video'],camera,context['CAD_target_polygon_xy'],context['CAD_attachment_z_m'],prepared)
        refs=camera['frame_refs'];indices=list(range(len(refs)))
    profile=config.get('profile','engineering')
    if profile not in PROFILES:raise ValueError('Profile must be engineering or conservative')
    policy={**PROFILES[profile],**config.get('policy',{})}
    camera_report={'calibration':{k:camera[k] for k in ['intrinsics','rotations','centers']}}
    (output/'camera_inputs.json').write_text(canonical_json(camera_report),encoding='utf-8')
    pipeline={'schema':'zip2cad_pointcloud_completion_config_v1','prediction':str(inputs['prediction']),
        'camera_calibration_report':str(output/'camera_inputs.json'),'camera_calibration_key':'calibration',
        'camera_native_image_size':camera['native_image_size'],'frame_refs':refs,'prediction_frame_indices':indices,
        'registration_training_views':camera.get('registration_training_views',camera['training_views']),
        'training_views':camera['training_views'],'heldout_views':camera['heldout_views'],'CAD':str(cad),
        'CAD_scope_prefix':'skp:'+namespace+':'+('.'.join(map(str,scope))+'.' if scope else ''),
        'CAD_target_polygon_xy':config.get('CAD_target_polygon_xy',context['CAD_target_polygon_xy']),
        'CAD_attachment_z_m':config.get('CAD_attachment_z_m',context['CAD_attachment_z_m']),
        'CAD_attachment_is_existing_model_prior':True,'user_accepts_existing_exterior_for_metric_calibration':config.get('trusted_existing_exterior',False),
        'source_model':str(inputs['source_model']),'view_scope_pid_path':scope,'hide_existing_pid_paths':[],
        'policy':policy,'appearance':config.get('appearance',{'top_color':[160,160,156],'side_color':[231,231,226],'photographic_textures':False}),
        'protect_existing_opaque_roof_geometry':True,'write_partial_cloud_patches':config.get('write_partial_cloud_patches',False),
        'per_frame_stable_shell_depth_bias':bool(config.get('prediction_blocks')),'maximum_depth_reference_views':3,'maximum_boundary_training_views':3,
        'limitations':['CAD and camera calibration inputs set engineering scale.','Profile supports finite near-horizontal and inclined roof patches; hidden structure is not measured.',
            'Independent survey certification and arbitrary-building complete success are not asserted.']}
    resolved=output/'resolved_completion_config.json';resolved.write_text(canonical_json(pipeline),encoding='utf-8')
    evidence=output/'completion';report=run(resolved,evidence)
    appearance=None
    if config.get('roof_finish_correction',True):
        from .roof_appearance import propose_roof_finishes
        plan=read(evidence/'native_pointcloud_plan.json');hides=plan['hide_existing_pid_paths']
        with np.load(cad) as z:cad_data={k:z[k] for k in z.files}
        with np.load(inputs['prediction']) as z:images=z['images'][indices].transpose(0,2,3,1)
        scale=np.diag([images.shape[2]/camera['native_image_size'][0],images.shape[1]/camera['native_image_size'][1],1])
        appearance=propose_roof_finishes(cad_data,images,scale@np.array(camera['intrinsics']),np.array(camera['rotations']),np.array(camera['centers']),
            refs,camera['training_views'],camera['heldout_views'],pipeline['CAD_scope_prefix'],pipeline['CAD_attachment_z_m'],pipeline['CAD_target_polygon_xy'],hides,
            pipeline['appearance']['top_color'],pipeline['appearance']['side_color'])
        appearance_path=evidence/'roof_appearance_report.json';appearance_path.write_text(canonical_json(appearance),encoding='utf-8')
        plan.update(schema='zip2cad_native_completion_plan_v2',face_material_updates=appearance['updates'],appearance_report_sha256=sha256_file(appearance_path))
        (evidence/'native_pointcloud_plan.json').write_text(canonical_json(plan),encoding='utf-8')
        (evidence/'artifact_manifest.json').write_text(canonical_json({'files':[r for r in inventory(evidence) if r['path']!='artifact_manifest.json']}),encoding='utf-8')
    if report['native_component_count'] or (appearance and appearance['accepted_face_count']):
        native=write_native(evidence/'native_pointcloud_plan.json',output/'native',sdk_path);status='completed_native_engineering_preview'
    else:
        import shutil
        unchanged=output/'unchanged.skp';shutil.copy2(inputs['source_model'],unchanged)
        native={'updated_model':str(unchanged),'updated_model_sha256':sha256_file(unchanged)};status='no_supported_edit'
    summary={'schema':'zip2cad_headless_project_result_v1','status':status,'profile':profile,
        'source_model_sha256':sha256_file(inputs['source_model']),'source_video_sha256':sha256_file(inputs['video']),
        'source_prediction_sha256':sha256_file(inputs['prediction']),'source_camera_input_sha256':sha256_file(inputs['camera_inputs']),
        'input_config_sha256':sha256_file(config_path),'producer_sha256':sha256_file(Path(__file__)),'video_metadata':video_metadata,
        'source_CAD_context':context,'camera_view_count':len(refs),'corrected_point_count':sum(r['retained_points'] for r in report['refinement_rows']),
        'candidate_count':len(report['components']),'approved_component_count':report['native_component_count'],'output_model':native['updated_model'],
        'output_model_sha256':native['updated_model_sha256'],'native_UI_used':False,'Ruby_used':False,'whole_building_survey_certified':False,
        'all_missing_building_parts_completed':False,'camera_calibration_is_input':not automatic_initialization,'automatic_camera_initialization_experimental':automatic_initialization,'source_resources_may_be_cached':True}
    if coverage:summary['full_video_coverage']={k:coverage[k] for k in ['unique_video_frames_examined','prediction_block_count','accepted_background_camera_count','selected_refinement_view_count']}
    if appearance:summary['appearance_edit_count']=appearance['accepted_face_count']
    (output/'result.json').write_text(canonical_json(summary),encoding='utf-8');(output/'artifact_manifest.json').write_text(canonical_json({'files':inventory(output)}),encoding='utf-8')
    print(canonical_json(summary));return summary

def main(argv=None):
    p=argparse.ArgumentParser(prog='zip2cad',description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('complete',help='Video + ZipMap predictions + CAD -> corrected SKP');c.add_argument('--config',required=True,type=Path);c.add_argument('--output',required=True,type=Path);c.add_argument('--sdk',type=Path)
    w=sub.add_parser('write-skp',help='Standalone SKP writeback of an approved plan');w.add_argument('--plan',required=True,type=Path);w.add_argument('--output',required=True,type=Path);w.add_argument('--sdk',type=Path)
    d=sub.add_parser('doctor');d.add_argument('--sdk',type=Path)
    demo=sub.add_parser('demo',help='Synthetic wall and pitched-roof SDK example; no neural inference');demo.add_argument('--output',required=True,type=Path);demo.add_argument('--sdk',type=Path)
    r=sub.add_parser('reconstruct',help='Extract a new video and run the official ZipMap checkpoint');r.add_argument('--video',required=True,type=Path);r.add_argument('--cad',required=True,type=Path);r.add_argument('--output',required=True,type=Path)
    r.add_argument('--zipmap-repo',required=True,type=Path);r.add_argument('--checkpoint',required=True,type=Path);r.add_argument('--inference-python',type=Path);r.add_argument('--cameras',type=Path);r.add_argument('--scope',nargs='*',type=int);r.add_argument('--sample-fps',type=float,default=1.);r.add_argument('--block-size',type=int,default=24);r.add_argument('--overlap',type=int,default=8)
    s=sub.add_parser('complete-cloud',help='Add finite arbitrary-direction planes from an attributed CAD-gauge cloud');s.add_argument('--cad',required=True,type=Path);s.add_argument('--cloud',required=True,type=Path);s.add_argument('--output',required=True,type=Path)
    s.add_argument('--train',nargs='+',required=True,type=int);s.add_argument('--heldout',nargs='+',required=True,type=int);s.add_argument('--scope',nargs='*',type=int);s.add_argument('--sdk',type=Path);s.add_argument('--transform',type=Path);s.add_argument('--thickness',type=float,default=.2)
    v=sub.add_parser('preview',help='Render before/after geometry without SketchUp UI');v.add_argument('--receipt',required=True,type=Path);v.add_argument('--output',required=True,type=Path);v.add_argument('--scope',nargs='*',type=int);v.add_argument('--sdk',type=Path)
    structures=sub.add_parser('complete-structures',help='Extend a cached run with observed finite small roof structures');structures.add_argument('--config',required=True,type=Path);structures.add_argument('--output',required=True,type=Path);structures.add_argument('--sdk',type=Path)
    a=p.parse_args(argv)
    if a.command=='complete':
        output_existed=a.output.exists()
        try:complete(a.config,a.output,a.sdk)
        except (ValueError,SDKError) as error:
            failure={'status':'completion_failed','error':str(error),'original_source_modified':False}
            if not output_existed and a.output.is_dir():(a.output/'failure.json').write_text(canonical_json(failure),encoding='utf-8')
            print(canonical_json(failure),file=sys.stderr);return 2
    elif a.command=='write-skp':write_native(a.plan,a.output,a.sdk)
    elif a.command=='complete-structures':
        from .structure_completion import complete as complete_structures
        complete_structures(a.config,a.output,a.sdk)
    elif a.command=='reconstruct':
        from .video_reconstruction import reconstruct
        print(reconstruct(a.video,a.cad,a.output,a.zipmap_repo,a.checkpoint,a.inference_python,a.cameras,a.scope,a.sample_fps,a.block_size,a.overlap))
    elif a.command=='complete-cloud':
        from .cloud_cli import complete_cloud
        complete_cloud(a.cad,a.cloud,a.output,a.train,a.heldout,a.scope,a.sdk,a.transform,a.thickness)
    elif a.command=='preview':
        from .model_preview import preview_models
        receipt=read(a.receipt);print(preview_models(receipt['source_model'],receipt['updated_model'],a.scope or [],receipt['generated_assembly_pid'],a.output,a.sdk))
    elif a.command=='demo':
        from .headless_demo import demo
        demo(a.output,a.sdk)
    else:
        with SketchUpSDK(a.sdk) as sdk:print(canonical_json({'status':'SDK_loaded','library':str(sdk.library_path),'UI_required':False,'Ruby_required':False}))
    return 0
if __name__=='__main__':raise SystemExit(main())
