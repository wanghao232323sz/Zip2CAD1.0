"""Validated standalone SKP writeback through the official SketchUp C API."""
import ctypes as ct,json
from pathlib import Path
import numpy as np,trimesh
from .sketchup_sdk import SketchUpSDK,Ref,SDKError
from .skp_exchange import sha256_file,canonical_json
from .stage_ledger import inventory

def mesh_from_exchange(data):
    return trimesh.Trimesh(data['vertices_world_m'],data['faces'],process=True)

def verify_mesh(actual,item):
    planned=trimesh.Trimesh(np.array(item['vertices_world_CAD_m']),np.array(item['triangles']),process=True)
    if not actual.is_watertight or not actual.is_winding_consistent or actual.volume<=0:raise SDKError('SDK roundtrip is not a closed oriented solid')
    # The C API may merge coplanar triangles. Compare surfaces, not arbitrary
    # tessellation centre vertices or face count.
    q1=np.vstack([planned.vertices,planned.triangles_center]);q2=np.vstack([actual.vertices,actual.triangles_center])
    d1=trimesh.proximity.closest_point(actual,q1)[1];d2=trimesh.proximity.closest_point(planned,q2)[1]
    error=float(max(d1.max(),d2.max()))
    if error>1e-6 or abs(actual.volume-planned.volume)>max(1e-5,planned.volume*1e-7):raise SDKError('SDK roundtrip geometry differs from plan')
    return {'component_id':item['component_id'],'closed_solid_verified':True,'winding_consistent':True,'volume_m3':float(actual.volume),
        'maximum_bidirectional_surface_error_m':error,'triangulated_face_count':len(actual.faces)}

def write_native(plan_path,output,sdk_path=None):
    plan_path=Path(plan_path).resolve();output=Path(output).resolve();plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('schema') not in ['zip2cad_pointcloud_native_completion_plan_v1','zip2cad_native_completion_plan_v2']:raise ValueError('Unsupported native plan')
    source=Path(plan['source_model']);source=source if source.is_absolute() else plan_path.parent/source
    source=source.resolve()
    if output.exists() or output==source or source.is_relative_to(output):raise ValueError('A fresh output folder separate from source is required')
    if sha256_file(source)!=plan['source_model_sha256']:raise ValueError('Source SKP changed')
    if (not plan['components'] and not plan.get('face_material_updates')) or plan.get('textures_allowed')!=False:raise ValueError('Geometry or explicit appearance edits required; textures are disabled')
    for row in plan.get('face_material_updates',[]):
        color=row.get('color',[])
        if len(color)!=3 or any(not isinstance(v,int) or isinstance(v,bool) or not 0<=v<=255 for v in color):raise ValueError('Appearance color must be three integer RGB values')
    if plan.get('appearance_report_sha256'):
        appearance=plan_path.parent/'roof_appearance_report.json'
        if sha256_file(appearance)!=plan['appearance_report_sha256']:raise ValueError('Appearance evidence changed')
        if json.loads(appearance.read_text(encoding='utf-8'))['updates']!=plan.get('face_material_updates',[]):raise ValueError('Appearance plan differs from supported edits')
    if plan.get('evidence_report_sha256'):
        evidence=plan_path.parent/'pointcloud_completion_report.json'
        if sha256_file(evidence)!=plan['evidence_report_sha256']:raise ValueError('Completion evidence changed')
        report=json.loads(evidence.read_text(encoding='utf-8'))
        for item in plan['components']:
            row=next((r for r in report['components'] if r['component_id']==item['component_id']),None)
            if not row or not row['native_edit_approved']:raise ValueError('Native component lacks approval')
    for item in plan['components']:
        p=np.array(item['vertices_world_CAD_m'],float);f=np.array(item['triangles'],int)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=len(p):raise ValueError('Invalid mesh plan')
        verify_mesh(trimesh.Trimesh(p,f,process=True),item)
    output.mkdir(parents=True);target=output/'corrected.skp';receipt={};model=None
    with SketchUpSDK(sdk_path) as sdk:
        try:
            model=sdk.load(source);finish_updates=plan.get('face_material_updates',[]);sdk.debug_paths={'.'.join(map(str,r['source_face_pid_path'])) for r in finish_updates}
            before=sdk.snapshot(model);original_appearance=sdk.debug_records.copy();materials=sdk.materials(model);print('source protection snapshot completed',flush=True)
            overrides={};hide=plan.get('hide_existing_pid_paths',[])
            for path in hide:
                found=[r for r in sdk.walk(model,path) if r[2]==path]
                if len(found)!=1 or found[0][0] not in ('Group','ComponentInstance'):raise SDKError('Old proxy path is not a unique source container')
                kind,obj,_,_,_=found[0];overrides['.'.join(map(str,path))]=sdk.drawing_record(kind,obj)['hidden'];sdk.hidden(kind,obj,True)
            finishes={};finish_rows=[]
            for update in finish_updates:
                path=update['source_face_pid_path'];key='.'.join(map(str,path))
                found=[r for r in sdk.walk(model,path) if r[2]==path]
                if len(found)!=1 or found[0][0]!='Face' or key not in original_appearance:raise SDKError('Appearance edit is not an existing source face')
                signature=tuple(update['color'])
                if signature not in finishes:
                    from .sketchup_sdk import Color
                    material=sdk.ref('SUMaterialCreate');sdk.call('SUMaterialSetName',material,('Zip2CAD Roof Finish '+str(len(finishes)+1)).encode())
                    color=Color(*signature,255);sdk.call('SUMaterialSetColor',material,ct.byref(color));sdk.call('SUModelAddMaterials',model,1,(Ref*1)(material));finishes[signature]=material
                face=found[0][1];sdk.call('SUFaceSetFrontMaterial',face,finishes[signature]);sdk.call('SUFaceSetBackMaterial',face,finishes[signature]);finish_rows.append({'source_face_pid_path':path,'color':list(signature),'geometry_changed':False})
            root=sdk.ref('SUModelGetEntities',model);assembly=sdk.group(root,'Zip2CAD Completion');assembly_pid=sdk.pid('Group',assembly);inside=sdk.children('Group',assembly)
            groups=[]
            for item in plan['components']:
                group=sdk.add_solid(model,inside,item);groups.append((item,sdk.pid('Group',group)))
            sdk.mesh_cache.clear()
            if sdk.snapshot(model,exclude=[assembly_pid],hidden_overrides=overrides,appearance_overrides=original_appearance)!=before:raise SDKError('Protected original source hierarchy or geometry changed')
            current={r['pid']:r for r in sdk.materials(model)}
            if any(current.get(r['pid'])!=r for r in materials):raise SDKError('Protected source materials changed')
            sdk.completed_view(model,plan.get('view_scope_pid_path'))
            sdk.call('SUModelSaveToFile',model,str(target).encode('utf-8'));sdk.release(model);model=None
            model=sdk.load(target)
            if sdk.snapshot(model,exclude=[assembly_pid],hidden_overrides=overrides,appearance_overrides=original_appearance)!=before:raise SDKError('Original source changed after save and reload')
            current={r['pid']:r for r in sdk.materials(model)}
            if any(current.get(r['pid'])!=r for r in materials):raise SDKError('Original materials changed after reload')
            rows=[]
            for update in finish_updates:
                path=update['source_face_pid_path'];face=next(r[1] for r in sdk.walk(model,path) if r[2]==path)
                material=sdk.drawing_record('Face',face)['material'];actual=current[material]
                if actual['rgba'][:3]!=update['color'] or actual['opacity']<.999 or actual['texture_sha256'] is not None:raise SDKError('Saved roof appearance differs from approved neutral finish')
            for item,pid in groups:
                exchange=sdk.geometry(model,scope=[assembly_pid,pid]);mesh=mesh_from_exchange(exchange);row=verify_mesh(mesh,item)
                group=next(r[1] for r in sdk.walk(model,[assembly_pid,pid]) if r[2]==[assembly_pid,pid]);definition=sdk.ref('SUGroupGetDefinition',group);manifold=ct.c_bool();sdk.call('SUComponentDefinitionIsManifold',definition,ct.byref(manifold))
                if not manifold.value:raise SDKError('Saved native group is not manifold')
                row.update(group_pid=pid,native_manifold=True,native_face_count=sdk.number('SUEntitiesGetNumFaces',sdk.children('Group',group)));rows.append(row)
                # A real edit probe uses the reopened SDK model, then applies
                # its inverse and verifies the geometry again. The frozen SKP
                # is not saved a second time after this in-memory probe.
                entities=sdk.children('Group',group);face=sdk.list(entities,'Faces')[0];p0=sdk.face_mesh(face)[0].mean(0)
                from .sketchup_sdk import Transform
                delta=np.eye(4);delta[0,3]=.001/.0254;transform=Transform((ct.c_double*16)(*delta.flatten(order='F')))
                sdk.call('SUEntitiesTransform',entities,1,(Ref*1)(sdk.entity('Face',face)),ct.byref(transform));sdk.mesh_cache.clear();p1=sdk.face_mesh(face)[0].mean(0)
                if np.linalg.norm(p1-p0)<.0005:raise SDKError('Native face editing probe did not change geometry')
                delta[0,3]*=-1;transform=Transform((ct.c_double*16)(*delta.flatten(order='F')));sdk.call('SUEntitiesTransform',entities,1,(Ref*1)(sdk.entity('Face',face)),ct.byref(transform));sdk.mesh_cache.clear()
                verify_mesh(mesh_from_exchange(sdk.geometry(model,scope=[assembly_pid,pid])),item);row['native_face_transform_and_inverse_verified']=True
                np.savez_compressed(output/(item['component_id']+'_reexport.npz'),**{k:v for k,v in exchange.items() if k!='rows'})
            if sha256_file(source)!=plan['source_model_sha256']:raise SDKError('Original file was overwritten')
            receipt={'schema':'zip2cad_sdk_native_receipt_v1','status':'standalone_SDK_saved_reloaded_verified','source_model':str(source),
                'source_model_sha256':sha256_file(source),'updated_model':str(target),'updated_model_sha256':sha256_file(target),
                'plan_sha256':sha256_file(plan_path),'producer_sha256':sha256_file(Path(__file__)),'sdk_binding_sha256':sha256_file(Path(__file__).with_name('sketchup_sdk.py')),
                'sdk_library':str(sdk.library_path),'sdk_library_sha256':sha256_file(sdk.library_path),'original_snapshot':before,
                'original_geometry_hierarchy_materials_preserved':True,'source_file_unchanged':True,'save_reload_verified':True,
                'generated_assembly_pid':assembly_pid,'component_rows':rows,'new_photographic_texture_count':0,
                'appearance_update_rows':finish_rows,'appearance_edit_count':len(finish_rows),'appearance_changes_are_explicitly_whitelisted':True,
                'saved_appearance_colors_and_opacity_verified':True,'completed_preview_scene_created':True,
                'native_face_editing_probe_verified':bool(rows) and all(r['native_face_transform_and_inverse_verified'] for r in rows),
                'native_UI_used':False,'Ruby_used':False,'source_proxy_hides_are_reversible':hide,'approval_scope':'engineering_preview',
                'whole_building_survey_certified':False}
            (output/'sdk_native_receipt.json').write_text(canonical_json(receipt),encoding='utf-8')
            (output/'artifact_manifest.json').write_text(canonical_json({'files':inventory(output)}),encoding='utf-8')
            print(canonical_json({'status':receipt['status'],'components':len(rows),'output':str(target)}),flush=True);return receipt
        except Exception as e:
            (output/'failure.json').write_text(canonical_json({'status':'failed','error':repr(e)}),encoding='utf-8');raise
        finally:
            if model is not None:sdk.release(model)
