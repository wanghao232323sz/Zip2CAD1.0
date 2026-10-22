"""Reusable configured ZipMap -> refined point cloud -> plain editable solids."""
import argparse,json,sys
from pathlib import Path
ROOT=Path.cwd()
import numpy as np,cv2,trimesh
from zip2cad.pointcloud_completion import CompletionPolicy,robust_similarity,refine_depth,extract_planar_components,trusted_perimeter_pixel_masks
from zip2cad.pointcloud_boundary import refine_boundaries
from zip2cad.roof_completion_geometry import clip_convex_xy
from zip2cad.pointcloud_completion import polygon_prism,infer_superseded_roof_proxies,convex_footprint_loft,trusted_exterior_candidate_overlap
from scipy.spatial import ConvexHull
from zip2cad.similarity import transform_points
from zip2cad.skp_exchange import sha256_file,canonical_json
from zip2cad.stage_ledger import inventory
from zip2cad.completion_decisions import suppress_duplicate_components

def load_target_mesh(path,prefix):
    with np.load(path) as z:
        owners=np.array([str(x).startswith(prefix) for x in z['entity_ids']]);use=owners[z['face_owner_indices']]
        mesh=trimesh.Trimesh(z['vertices_world_m'],z['faces'][use],process=False)
        opacity=z['face_material_opacity'][z['face_owner_indices'][use]] if 'face_material_opacity' in z.files else None
    valid=mesh.nondegenerate_faces();mesh.update_faces(valid)
    return mesh,opacity[valid] if opacity is not None else None

def run(config_path,output):
    config_path=Path(config_path).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh output required')
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if config.get('reconstruction_backend','ZipMap')!='ZipMap':raise ValueError('This maintained release accepts ZipMap observations only')
    output.mkdir(parents=True)
    def file(key):return (ROOT/config[key]).resolve()
    if not config['user_accepts_existing_exterior_for_metric_calibration']:raise ValueError('Metric exterior premise required')
    camera=json.loads(file('camera_calibration_report').read_text())[config['camera_calibration_key']]
    K=np.array(camera['intrinsics']);R=np.array(camera['rotations']);C=np.array(camera['centers'])
    with np.load(file('prediction')) as z:
        ids=config['prediction_frame_indices'];raw=z['world_points'][ids];confidence=z['world_points_conf'][ids];images=z['images'][ids].transpose(0,2,3,1)
    refs=config['frame_refs'];training=config['training_views'];heldout=config['heldout_views'];h,w=images.shape[1:3]
    K=np.diag([w/config['camera_native_image_size'][0],h/config['camera_native_image_size'][1],1])@K
    mesh,opacity=load_target_mesh(file('CAD'),config['CAD_scope_prefix'])
    polygon=np.asarray(config['CAD_target_polygon_xy'],np.float32);base=config['CAD_attachment_z_m']
    protected_masks,protected_triangles=trusted_perimeter_pixel_masks(mesh,polygon,base,K,R,C,(h,w),face_opacity=opacity if config.get('protect_existing_opaque_roof_geometry') else None)
    np.savez_compressed(output/'trusted_exterior_pixel_masks.npz',masks=protected_masks,frame_refs=np.array(refs))
    xx,yy=np.meshgrid(np.arange(2,w,4),np.arange(2,h,4));xy=np.c_[xx.ravel(),yy.ravel()];sources=[];targets=[];anchorrefs=[]
    for i,ref in enumerate(refs):
        rays=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(K[i]).T@R[i];rays/=np.linalg.norm(rays,axis=1)[:,None]
        hit,ray,tri=mesh.ray.intersects_location(np.repeat(C[i,None],len(xy),axis=0),rays,multiple_hits=False)
        if not len(hit):
            sources.append(np.empty((0,3)));targets.append(np.empty((0,3)));anchorrefs.append(np.empty(0,dtype=int));continue
        use=(hit[:,2]<base-.9)&(hit[:,2]>base-13)&(abs(mesh.face_normals[tri,2])<.25)
        pixels=xy[ray[use]];target=hit[use];co=confidence[i,pixels[:,1],pixels[:,0]]
        if not len(co):
            sources.append(np.empty((0,3)));targets.append(np.empty((0,3)));anchorrefs.append(np.empty(0,dtype=int));continue
        good=co>=np.percentile(co,40);pixels=pixels[good];target=target[good]
        sources.append(raw[i,pixels[:,1],pixels[:,0]]);targets.append(target);anchorrefs.append(np.full(len(target),ref))
    ar=np.concatenate(anchorrefs);source=np.concatenate(sources);target=np.concatenate(targets)
    registration_views=config.get('registration_training_views',training)
    anchor_counts={str(ref):int(np.sum(ar==ref)) for ref in refs}
    if np.sum(np.isin(ar,registration_views))<30 or sum(anchor_counts[str(ref)]>=12 for ref in registration_views)<2:
        (output/'registration_failure.json').write_text(canonical_json({'status':'calibration_required','stable_shell_anchor_counts':anchor_counts,
            'reason':'At least two training views with 12 stable CAD facade rays and 30 total correspondences are required; automatic pose may be ambiguous.'}),encoding='utf-8')
        raise ValueError('Camera initialization has insufficient stable facade overlap; supply calibrated camera inputs (see registration_failure.json)')
    registration=robust_similarity(source,target,np.isin(ar,registration_views));registration['training_frame_refs']=registration_views
    initial=transform_points(raw,registration['scale'],np.asarray(registration['rotation']),np.asarray(registration['translation']))
    metric_anchors=transform_points(source,registration['scale'],np.asarray(registration['rotation']),np.asarray(registration['translation']))
    np.savez_compressed(output/'stable_shell_anchors.npz',ZipMap_points=source,CAD_points=target,frame_refs=ar)
    yy,xx=np.mgrid[:h,:w];pixels=np.stack([xx,yy,np.ones_like(xx)],axis=-1);cloud=[];cloudrefs=[];pixelids=[];scores=[];initialrows=[];summary=[]
    for i,ref in enumerate(refs):
        # Target scope is CAD-space, with no manually drawn image contours.
        rays=pixels@np.linalg.inv(K[i]).T@R[i];d=(base-C[i,2])/rays[...,2];onbase=C[i]+rays*d[...,None]
        scope=np.array([cv2.pointPolygonTest(polygon,tuple(map(float,p)),False)>=0 for p in onbase[...,:2].reshape(-1,2)]).reshape(h,w)
        if scope.any():scope&=(d>0)&(confidence[i]>=np.percentile(confidence[i][scope],30))
        scope&=~protected_masks[i]
        prior=initial[i].copy();depth_bias=0.
        if config.get('per_frame_stable_shell_depth_bias',False):
            mask=ar==ref
            if mask.sum()>=16:
                a=((metric_anchors[mask]-C[i])@R[i].T)[:,2];b=((target[mask]-C[i])@R[i].T)[:,2]
                depth_bias=float(np.median(a-b));z=((prior-C[i])@R[i].T)[...,2]-depth_bias;prior=C[i]+rays*z[...,None]
        prior[~scope]=np.nan
        reference_indices=[refs.index(r) for r in training]
        # Heldout source images generate independent validation observations;
        # they are never passed to plane fitting or boundary selection.
        if ref in heldout:reference_indices=[i,refs.index(training[0]),refs.index(training[-1])]
        if config.get('maximum_depth_reference_views'):
            eligible=[j for j in reference_indices if j!=i]
            useful=[j for j in eligible if np.linalg.norm(C[j]-C[i])>=.5]
            if len(useful)>=2:eligible=useful
            eligible.sort(key=lambda j:np.linalg.norm(C[j]-C[i])+10*np.linalg.norm(R[j]-R[i]))
            reference_indices=[i]+eligible[:max(2,config['maximum_depth_reference_views']-1)]
        result=refine_depth(images,prior,K,R,C,i,reference_indices)
        points=result['points'];valid=result['valid']&scope&(points[...,2]>=base-.3)&(points[...,2]<base+6)
        inside=np.array([cv2.pointPolygonTest(polygon,tuple(map(float,p)),False)>=0 for p in points[...,:2].reshape(-1,2)]).reshape(h,w);valid&=inside
        cloud.append(points[valid]);cloudrefs.append(np.full(valid.sum(),ref));pixelids.append(np.flatnonzero(valid));scores.append(result['zncc'][valid]);initialrows.append(initial[i][valid])
        summary.append({'frame_ref':ref,'split':'training' if ref in training else 'heldout_observation_only','retained_points':int(valid.sum()),
            'stable_shell_depth_bias_m':depth_bias,'depth_reference_frame_refs':[refs[j] for j in reference_indices],
            'protected_existing_exterior_pixel_count':int(protected_masks[i].sum()),'protected_existing_coping_triangle_count':protected_triangles,
            'median_depth_correction_m':float(np.median(abs(result['refined_depth_CAD_m'][valid]-result['initial_depth_CAD_m'][valid]))) if valid.any() else None})
        debug=np.zeros((h,w,3),np.uint8);color=cv2.applyColorMap(np.clip((points[...,2]-base)/3*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO);debug[valid]=color[valid];cv2.imwrite(str(output/f'depth_{ref}.png'),debug)
        print('refined',summary[-1],flush=True)
    points=np.concatenate(cloud);frames=np.concatenate(cloudrefs)
    np.savez_compressed(output/'refined_ZipMap_cloud.npz',points_CAD_m=points,original_ZipMap_points_CAD_m=np.concatenate(initialrows),frame_refs=frames,source_pixel_index=np.concatenate(pixelids),ZNCC=np.concatenate(scores))
    return complete_cloud(config_path,output,config,mesh,points,frames,registration,summary)

def complete_cloud(config_path,output,config,mesh,points,frames,registration,summary,parent=None):
    def file(key):return (ROOT/config[key]).resolve()
    training=config['training_views'];heldout=config['heldout_views'];base=config['CAD_attachment_z_m']
    policy=CompletionPolicy(**config['policy']);result=extract_planar_components(points,frames,training,heldout,base,policy)
    _,opacity=load_target_mesh(file('CAD'),config['CAD_scope_prefix'])
    coping=(abs(mesh.face_normals[:,2])>.9)&(mesh.triangles_center[:,2]>=base+.25)&(mesh.triangles_center[:,2]<=base+1.2)
    if opacity is not None:coping&=opacity>=.95
    trusted_triangles=mesh.triangles[coping] if config.get('protect_existing_opaque_roof_geometry') else None
    camera=json.loads(file('camera_calibration_report').read_text())[config['camera_calibration_key']]
    with np.load(file('prediction')) as z:images=z['images'][config['prediction_frame_indices']].transpose(0,2,3,1)
    with np.load(output/config.get('attributed_cloud_file_name','refined_ZipMap_cloud.npz')) as z:pixel_indices=z['source_pixel_index']
    protected=None
    if (output/'trusted_exterior_pixel_masks.npz').is_file():
        with np.load(output/'trusted_exterior_pixel_masks.npz') as z:protected=z['masks']
    cloud_colours=np.empty((len(points),3),np.float32)
    for i,ref in enumerate(config['frame_refs']):
        selected=frames==ref;cloud_colours[selected]=images[i].reshape(-1,3)[pixel_indices[selected]]
    K=np.array(camera['intrinsics']);R=np.array(camera['rotations']);C=np.array(camera['centers']);h,w=images.shape[1:3]
    K=np.diag([w/config['camera_native_image_size'][0],h/config['camera_native_image_size'][1],1])@K
    refined=[]
    for item in result['components']:
        boundary_training=training
        if config.get('maximum_boundary_training_views'):
            source=config['frame_refs'].index(training[0]);others=sorted(training[1:],key=lambda ref:np.linalg.norm(C[config['frame_refs'].index(ref)]-C[source])+10*np.linalg.norm(R[config['frame_refs'].index(ref)]-R[source]))
            boundary_training=[training[0]]+others[:config['maximum_boundary_training_views']-1]
        refined.extend(refine_boundaries(item,points,frames,images,K,R,C,config['frame_refs'],boundary_training,heldout,config['CAD_target_polygon_xy'],base,policy,cloud_colours=cloud_colours,protected_pixel_masks=protected))
    result['components']=refined[:policy.maximum_components]
    for i,item in enumerate(result['components']):
        item['geometry_source']=config.get('geometry_source','refined_ZipMap_pointcloud')
        item['component_id']=f'cloud_patch_{i+1:02d}'
        poly=clip_convex_xy(np.asarray(item['boundary_xy']),np.asarray(config['CAD_target_polygon_xy']))
        origin=np.array(item['plane_origin']);normal=np.array(item['plane_normal']);height=origin[2]-(poly-origin[:2])@normal[:2]/normal[2]
        item['boundary_xy']=poly.tolist();item['mesh']=polygon_prism(np.c_[poly,height],np.c_[poly,np.full(len(poly),base)])
    parts=[];rows=[]
    for item in result['components']:
        geometry=item.pop('mesh');name=item['component_id'];np.savez_compressed(output/f'{name}.npz',vertices_world_CAD_m=geometry['vertices'],triangles=geometry['triangles'])
        # CAD difference uses source mesh at these same XY locations, excludes
        # new geometry and heldout points from determining the shape.
        origin=np.array(item['plane_origin']);normal=np.array(item['plane_normal']);trainpoints=points[np.isin(frames,training)]
        poly=np.array(item['boundary_xy'],np.float32);inside=np.array([cv2.pointPolygonTest(poly,tuple(map(float,p)),False)>=0 for p in trainpoints[:,:2]])
        p=trainpoints[inside&(abs((trainpoints-origin)@normal)<policy.plane_tolerance_m)]
        p=p[np.linspace(0,len(p)-1,min(len(p),500),dtype=int)]
        _,distance,_=trimesh.proximity.closest_point(mesh,p)
        item['median_distance_to_existing_CAD_m']=float(np.median(distance));item['fraction_farther_than_change_threshold']=float(np.mean(distance>policy.minimum_change_m))
        item['CAD_missing_or_different_supported']=bool(np.mean(distance>policy.minimum_change_m)>=.5)
        guard=trusted_exterior_candidate_overlap(item['boundary_xy'],config['CAD_target_polygon_xy'],registration['training_median_m'],policy.voxel_m,trusted_triangles)
        item['trusted_exterior_guard']=guard
        item['native_edit_approved']=item['accepted'] and item['CAD_missing_or_different_supported'] and not guard['trusted_existing_exterior_conflict'];rows.append(item)
        if config.get('boundary_support_area_ratio_range'):
            ratio=float(cv2.contourArea(poly)/max(item['occupied_area_m2'],1e-9));lower,upper=config['boundary_support_area_ratio_range'];item['boundary_to_training_seed_area_ratio']=ratio
            if not lower<=ratio<=upper:
                item['native_edit_approved']=False;item['writeback_rejection_reason']='finite_boundary_area_inconsistent_with_training_cloud_support'
        if not config.get('write_partial_cloud_patches',True) and not item.get('boundary_source'):
            item['native_edit_approved']=False;item['writeback_rejection_reason']='partial_cloud_patch_without_supported_finite_video_boundary'
        if item['native_edit_approved']:
            parts.append({'component_id':name,'vertices_world_CAD_m':geometry['vertices'].tolist(),'triangles':geometry['triangles'].tolist(),'top_triangle_mask':geometry['top_triangle_mask'].tolist(),
                'top_color':config['appearance']['top_color'],'side_color':config['appearance']['side_color'],'closed_solid_verified':True,'volume_CAD_units_cubed':geometry['volume_CAD_units_cubed']})
    parts,suppressed=suppress_duplicate_components(rows,parts,policy.plane_tolerance_m)
    proof={'schema':'zip2cad_pointcloud_completion_report_v1','status':'pointcloud_geometry_candidates_completed','metric_registration':registration,'refinement_rows':summary,'components':rows,'rejected_components':result['rejected'],'duplicate_suppression_rows':suppressed,
        'native_component_count':len(parts),'new_triangle_count':sum(len(c['triangles']) for c in parts),'raw_ZipMap_prediction_sha256':sha256_file(file('prediction')),
        'source_CAD_sha256':sha256_file(file('CAD')),'config_sha256':sha256_file(config_path),'module_sha256':sha256_file(Path(__file__).with_name('pointcloud_completion.py')),
        'boundary_module_sha256':sha256_file(Path(__file__).with_name('pointcloud_boundary.py')),
        'pointcloud_parent':parent,'camera_calibration_report_sha256':sha256_file(file('camera_calibration_report')),'producer_sha256':sha256_file(Path(__file__)),'photographic_texture_count':0,'hand_drawn_image_ROI_count':0,'lower_closure_is_CAD_attachment_prior':True,'whole_building_universal_success_claimed':False,'independent_survey_accuracy_claimed':False,
        'limitations':config['limitations']}
    approved=[item for item in rows if item['native_edit_approved']]
    with np.load(file('CAD')) as z:
        replacements=infer_superseded_roof_proxies(z['vertices_world_m'],z['faces'],z['face_owner_indices'],z['entity_ids'],config['CAD_scope_prefix'],approved,base,policy)
        for replacement in replacements:
            name=replacement['replacement_component_id'];row=next(item for item in rows if item['component_id']==name)
            if row.get('boundary_primitive')!='cloud_plane_video_boundary_round_prism':continue
            prefix=config['CAD_scope_prefix'].split(':')[:2];prefix=':'.join(prefix)+':'+'.'.join(map(str,replacement['source_container_pid_path']))+'.'
            selected=np.array([str(n).startswith(prefix) for n in z['entity_ids']]);old=z['vertices_world_m'][z['faces'][selected[z['face_owner_indices']]]].reshape(-1,3)
            bottom=np.unique(np.round(old[abs(old[:,2]-base)<1e-5,:2],10),axis=0)
            if len(bottom)<3:continue
            bottom=bottom[ConvexHull(bottom).vertices]
            geometry=convex_footprint_loft(row['boundary_xy'],bottom,row['plane_origin'],row['plane_normal'],base)
            part=next(p for p in parts if p['component_id']==name)
            part.update(vertices_world_CAD_m=geometry['vertices'].tolist(),triangles=geometry['triangles'].tolist(),top_triangle_mask=geometry['top_triangle_mask'].tolist(),volume_CAD_units_cubed=geometry['volume_CAD_units_cubed'])
            np.savez_compressed(output/f'{name}.npz',vertices_world_CAD_m=geometry['vertices'],triangles=geometry['triangles'])
            row['CAD_attachment_outline_xy']=bottom.tolist();row['CAD_attachment_closure_is_prior']=True;row['exact_old_opening_attachment_vertex_count']=len(bottom)
            replacement['exact_old_opening_loft_generated']=True
    proof['new_triangle_count']=sum(len(c['triangles']) for c in parts)
    proof['automatic_reversible_proxy_replacements']=replacements
    proof['approval_scope']='engineering_research_preview_under_existing_CAD_scale_and_camera_calibration'
    proof['full_library_roof_completed']=False
    proof['whole_pipeline_independent_camera_holdout_claimed']=False
    proof['reconstruction_backend']=config.get('reconstruction_backend','ZipMap')
    if proof['reconstruction_backend']!='ZipMap':
        proof['source_rendered_views_sha256']=proof.pop('raw_ZipMap_prediction_sha256')
        proof['trained_reconstruction_independent_holdout_claimed']=False
    (output/'pointcloud_completion_report.json').write_text(canonical_json(proof),encoding='utf-8')
    native=output.with_name(output.name+'_native');model=file('source_model')
    plan={'schema':'zip2cad_pointcloud_native_completion_plan_v1','source_model':str(model),'source_model_sha256':sha256_file(model),
        'output_model':str(native/'pointcloud_completed_editable.skp'),'native_report':str(native/'native_pointcloud_completion_report.json'),
        'view_scope_pid_path':config['view_scope_pid_path'],'hide_existing_pid_paths':config['hide_existing_pid_paths']+[r['source_container_pid_path'] for r in replacements],'components':parts,
        'evidence_report_sha256':sha256_file(output/'pointcloud_completion_report.json'),'textures_allowed':False,'original_geometry_protected':True}
    (output/'native_pointcloud_plan.json').write_text(canonical_json(plan),encoding='utf-8')
    (output/'artifact_manifest.json').write_text(canonical_json({'files':inventory(output)}),encoding='utf-8')
    print(canonical_json({'components':len(parts),'triangles':proof['new_triangle_count'],'candidates':rows,'rejected':len(result['rejected'])}));return proof

def run_cached(config_path,output,parent):
    import shutil
    config_path=Path(config_path).resolve();output=Path(output).resolve();parent=Path(parent).resolve()
    if output.exists():raise ValueError('Fresh output required')
    config=json.loads(config_path.read_text(encoding="utf-8"));report=json.loads((parent/'pointcloud_completion_report.json').read_text())
    if report['config_sha256']!=sha256_file(config_path):raise ValueError('Cached point cloud config differs')
    if report['raw_ZipMap_prediction_sha256']!=sha256_file(ROOT/config['prediction']):raise ValueError('ZipMap prediction changed')
    inventory_rows=json.loads((parent/'artifact_manifest.json').read_text())['files']
    for name in ['refined_ZipMap_cloud.npz','pointcloud_completion_report.json','stable_shell_anchors.npz']:
        row=next(r for r in inventory_rows if r['path']==name)
        if sha256_file(parent/name)!=row['sha256']:raise ValueError('Cached cloud provenance changed')
    output.mkdir(parents=True)
    for name in ['refined_ZipMap_cloud.npz','stable_shell_anchors.npz']:shutil.copy2(parent/name,output/name)
    if (parent/'trusted_exterior_pixel_masks.npz').is_file():
        name='trusted_exterior_pixel_masks.npz';row=next(r for r in inventory_rows if r['path']==name)
        if sha256_file(parent/name)!=row['sha256']:raise ValueError('Trusted exterior masks changed')
        shutil.copy2(parent/name,output/name)
    with np.load(output/'refined_ZipMap_cloud.npz') as z:points=z['points_CAD_m'];refs=z['frame_refs']
    with np.load(ROOT/config['CAD']) as z:
        owner=np.array([str(x).startswith(config['CAD_scope_prefix']) for x in z['entity_ids']]);mesh=trimesh.Trimesh(z['vertices_world_m'],z['faces'][owner[z['face_owner_indices']]],process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    return complete_cloud(config_path,output,config,mesh,points,refs,report['metric_registration'],report['refinement_rows'],
        {'path':str(parent),'artifact_manifest_sha256':sha256_file(parent/'artifact_manifest.json')})

