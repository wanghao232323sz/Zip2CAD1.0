"""Auditable 14-stage video-to-editable-SKP workflow; no library-specific paths."""
import re,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import trimesh
from .release_io import read,save,sha,resolve,config,similarity

STAGES=[
 'validate_inputs','extract_video_frames','reconstruct_ZipMap_blocks','align_and_fuse_blocks',
 'extract_original_CAD','register_observations_to_CAD','prepare_observed_roof_context',
 'build_observed_model_contract','extract_frozen_features','predict_learned_geometry',
 'regularize_and_export_editable_SKP','evaluate_reference_if_supplied',
 'write_CAD_change_proposal','verify_and_package_run']


def joint_frame(outline,base):
    from shapely import MultiPoint
    outline=np.asarray(outline,float);q=np.array(MultiPoint(outline).minimum_rotated_rectangle.exterior.coords)[:4]
    edges=np.roll(q,-1,axis=0)-q;axes=edges/np.linalg.norm(edges,axis=1,keepdims=True);u=axes[np.argmax(abs(axes[:,0]))]
    if u[0]<0:u=-u
    basis=np.c_[u,[-u[1],u[0]]];uv=outline@basis;low=uv.min(0);span=uv.max(0)-low
    return dict(origin_xy=(low@basis.T).tolist(),basis_xy=basis.tolist(),span_m=span.tolist(),
                base_z_m=float(base),outline_normalized=((uv-low)/span).tolist(),reference_roof_used=False)


def run(config_file,output,resume=False,to_stage=14):
    cfg,root=config(config_file);out=resolve(root,output);video=resolve(root,cfg['video']);cad=resolve(root,cfg['source_cad'])
    sdk=resolve(root,cfg['sdk']) if cfg.get('sdk') else None
    if out in [root,video.parent,cad.parent] or video.is_relative_to(out) or cad.is_relative_to(out):raise ValueError('Output must be separate from source assets')
    if out.exists() and not resume:raise FileExistsError('Use a fresh output folder or --resume')
    out.mkdir(parents=True,exist_ok=True);ledger=out/'workflow.json';config_hash=sha(config_file)
    state=read(ledger) if ledger.exists() else dict(schema='zip2cad_workflow_run_v1',config_sha256=config_hash,stages=[],status='running',independent_validation=False)
    if state['config_sha256']!=config_hash:raise ValueError('Resume config hash differs')
    completed={row['step']:row for row in state['stages'] if row['status']=='complete'}
    if resume and 1 in completed:
        bound=read(out/'01_inputs.json')
        if sha(video)!=bound['video_sha256'] or sha(cad)!=bound['CAD_sha256']:raise ValueError('Resume source video or CAD changed')
    def stage(number,name,fn):
        if number in completed:
            for path,digest in completed[number].get('output_hashes',{}).items():
                if sha(out/path)!=digest:raise ValueError('Resume output changed: '+path)
            return
        print(f'Step {number:02d}/14 {name}',flush=True);started=time.perf_counter()
        try:details=fn() or {}
        except Exception as error:
            state.update(status='failed',failed_step=number,error=str(error));save(ledger,state);raise
        state['stages']=[row for row in state['stages'] if row['step']!=number]
        state['stages'].append(dict(step=number,name=name,status='complete',elapsed_s=time.perf_counter()-started,
             details=details,output_hashes={p.relative_to(out).as_posix():sha(p) for p in out.glob(f'{number:02d}_*') if p.is_file()}))
        state.update(progress_percent=round(number/14*100,2),status='running');save(ledger,state)
    def step1():
        if not video.is_file() or not cad.is_file():raise FileNotFoundError('Survey video and original SKP are required')
        binding=dict(video_sha256=sha(video),CAD_sha256=sha(cad),config_sha256=config_hash)
        rec=cfg.get('reconstruction',{})
        if rec.get('mode')=='cached' and rec.get('source_video_sha256')!=binding['video_sha256']:raise ValueError('Cached blocks must bind to the exact source video SHA256')
        save(out/'01_inputs.json',binding);return binding
    def step2():
        from .video_preprocess import preprocess_video,VideoPreprocessConfig
        value=preprocess_video(video,out/'frames',VideoPreprocessConfig(**cfg.get('video_preprocessing',{})))
        save(out/'02_video.json',dict(summary=value['summary'],source_video=value['source_video']));return value['summary']
    def step3():
        rec=cfg.get('reconstruction',{});blocks=[]
        if rec.get('mode')=='cached':
            for item in rec['blocks']:
                path=resolve(root,item['path'])
                if item.get('sha256') and sha(path)!=item['sha256']:raise ValueError('Cached ZipMap block hash differs')
                with np.load(path) as z:
                    if len(item['frame_refs'])!=len(z['world_points']):raise ValueError('Physical frame/prediction length mismatch')
                blocks.append(dict(item,path=str(path),sha256=sha(path)))
        else:
            repo=resolve(root,rec['repository']);weights=resolve(root,rec['checkpoint']);python=rec.get('python',sys.executable)
            for block in read(out/'frames/zipmap_blocks.json')['blocks']:
                target=out/'ZipMap'/block['block_id']
                subprocess.run([str(python),str(Path(__file__).with_name('zipmap_inference.py')),'--blocks-json',str(out/'frames/zipmap_blocks.json'),
                     '--block-id',block['block_id'],'--zipmap-repo',str(repo),'--checkpoint',str(weights),'--output-dir',str(target),'--skip-glb'],check=True)
                path=target/'predictions.npz';refs=[int(re.search(r'frame_(\d+)',n).group(1))-1 for n in block['frame_files']]
                blocks.append(dict(path=str(path),frame_refs=refs,sha256=sha(path)))
        save(out/'03_blocks.json',dict(blocks=blocks,mode=rec.get('mode','official'),source_video_sha256=sha(video)));return dict(block_count=len(blocks),cached=rec.get('mode')=='cached')
    def step4():
        from .full_video_alignment import robust_similarity,compose
        blocks=read(out/'03_blocks.json')['blocks'];parent=(1.,np.eye(3),np.zeros(3));previous=None;frames={};rows=[]
        percentile=float(cfg.get('fusion',{}).get('confidence_percentile',60));stride=int(cfg.get('fusion',{}).get('pixel_stride',4))
        if not 0<=percentile<100 or stride<1:raise ValueError('Fusion confidence/stride invalid')
        for block in blocks:
            if sha(block['path'])!=block['sha256']:raise ValueError('ZipMap predictions changed after stage3')
            with np.load(block['path']) as z:
                points=z['world_points'];confidence=z['world_points_conf'];images=z['images']
            refs=block['frame_refs']
            if 'to_common' in block:t=block['to_common'];parent=(t['scale'],np.asarray(t['rotation']),np.asarray(t['translation']))
            elif previous:
                old,oldc,oldrefs=previous;shared=sorted(set(refs)&set(oldrefs));a=[];b=[]
                if not shared:raise ValueError('Adjacent blocks need shared physical frames')
                for ref in shared:
                    i=oldrefs.index(ref);j=refs.index(ref);conf=np.minimum(oldc[i],confidence[j]);yy,xx=np.mgrid[3:conf.shape[0]-3:8,3:conf.shape[1]-3:8];keep=conf[yy,xx]>=np.percentile(conf,60)
                    a.append(points[j,yy[keep],xx[keep]]);b.append(old[i,yy[keep],xx[keep]])
                parent=compose(parent,robust_similarity(np.concatenate(a),np.concatenate(b)))
            transform=dict(scale=float(parent[0]),rotation=parent[1].tolist(),translation=parent[2].tolist())
            rows.append(dict(path=block['path'],to_common=transform))
            for i,ref in enumerate(refs):
                score=float(np.nanmedian(confidence[i]))
                if ref in frames and frames[ref][0]>=score:continue
                h,w=confidence[i].shape;yy,xx=np.mgrid[:h:stride,:w:stride];keep=np.isfinite(points[i,yy,xx]).all(-1)&(confidence[i,yy,xx]>=np.nanpercentile(confidence[i],percentile))
                rgb=images[i].transpose(1,2,0) if images[i].shape[0]==3 else images[i]
                frames[ref]=(score,similarity(points[i,yy[keep],xx[keep]],transform),rgb[yy[keep],xx[keep]])
            previous=(points,confidence,refs)
        point=np.concatenate([frames[k][1] for k in sorted(frames)]);rgb=np.concatenate([frames[k][2] for k in sorted(frames)]);refs=np.concatenate([np.full(len(frames[k][1]),k,np.int32) for k in sorted(frames)])
        if rgb.max()>1:rgb=rgb/255
        np.savez_compressed(out/'04_fused_arbitrary.npz',points=point,rgb=rgb.astype(np.float32),frame_refs=refs)
        save(out/'04_fusion.json',dict(transforms=rows,physical_views=len(frames),points=len(point),scale_is_arbitrary=True,overlapping_frames_counted_once=True))
        return dict(points=len(point),views=len(frames))
    def step5():
        from .headless_cli import export_cad
        context=export_cad(cad,out/'05_original_CAD.npz',cfg.get('scope_pid_path',[]),'building',sdk)
        context.update(cfg.get('CAD_context_override',{}));save(out/'05_CAD_context.json',context);return context
    def step6():
        with np.load(out/'04_fused_arbitrary.npz') as z:p=z['points'];rgb=z['rgb'];refs=z['frame_refs']
        registration=cfg.get('registration',{});mode=registration.get('mode','manual_similarity')
        if mode=='manual_similarity':transform=registration['transform']
        elif mode=='anchors':
            from .similarity import estimate_similarity
            anchors=read(resolve(root,registration['anchors']));s,r,t=estimate_similarity(np.asarray(anchors['observed_points']),np.asarray(anchors['CAD_points']))
            transform=dict(scale=float(s),rotation=r.tolist(),translation=t.tolist())
        elif mode=='experimental_auto':
            from .auto_registration import initialize_similarity
            with np.load(out/'05_original_CAD.npz') as z:
                vertices=z['vertices_world_m'];faces=z['faces']
                scope=cfg.get('scope_pid_path',[])
                if scope:
                    prefix='skp:building:'+'.'.join(map(str,scope))+'.'
                    selected=np.array([str(v).startswith(prefix) for v in z['entity_ids']])
                    faces=faces[selected[z['face_owner_indices']]]
            if not len(faces):raise ValueError('Target CAD scope empty for automatic alignment')
            mesh=trimesh.Trimesh(vertices,faces,process=False);target=mesh.sample(min(9000,max(100,len(mesh.vertices))))
            transform=initialize_similarity(p,target)
        else:raise ValueError('Specify anchors, manual_similarity, or experimental_auto registration')
        p=similarity(p,transform)
        np.savez_compressed(out/'06_observed_CAD.npz',points_CAD_m=p,rgb=rgb,frame_refs=refs)
        save(out/'06_registration.json',dict(transform=transform,mode=mode,metric_survey_certified=False,coordinate_system='CAD_metres'))
        return dict(mode=mode,points=len(p),registration_needs_visual_review=True)
    def step7():
        from shapely import Polygon,contains_xy
        context=read(out/'05_CAD_context.json');poly=Polygon(context['CAD_target_polygon_xy']).buffer(float(cfg.get('roof_context',{}).get('xy_margin_m',1.)))
        with np.load(out/'06_observed_CAD.npz') as z:p=z['points_CAD_m'];rgb=z['rgb'];refs=z['frame_refs']
        base=context['CAD_attachment_z_m'];keep=contains_xy(poly,p[:,0],p[:,1])&(p[:,2]>=base-3)&(p[:,2]<=base+8)
        ids=np.flatnonzero(keep);voxel=float(cfg.get('roof_context',{}).get('voxel_m',.15))
        if len(ids)<128:raise ValueError('Fewer than128 roof observations: review registration, CAD scope and crop')
        _,take=np.unique(np.floor(p[ids]/voxel).astype(np.int64),axis=0,return_index=True);ids=np.sort(ids[take])
        np.savez_compressed(out/'07_roof_observations.npz',points_CAD_m=p[ids],rgb=rgb[ids],frame_refs=refs[ids],source_point_index=ids)
        trimesh.points.PointCloud(p[ids],colors=np.clip(rgb[ids]*255,0,255).astype(np.uint8)).export(out/'07_roof_observed.ply')
        return dict(roof_points=len(ids),voxel_m=voxel)
    def step8():
        context=read(out/'05_CAD_context.json');frame=joint_frame(context['CAD_target_polygon_xy'],context['CAD_attachment_z_m'])
        save(out/'08_model_frame.json',frame);save(out/'08_input_contract.json',dict(source='ZipMap',cloud_sha256=sha(out/'07_roof_observations.npz'),reference_loaded=False,frame=frame))
        return dict(contains_reference_targets=False)
    def step9():
        neural=cfg.get('neural',{})
        if not neural.get('enabled',False):save(out/'09_neural_status.json',dict(enabled=False,method='geometric baseline'));return dict(skipped=True,reason='explicit geometric baseline mode')
        from .release_semantic import sonata_features,semantics
        with np.load(out/'07_roof_observations.npz') as z:points=z['points_CAD_m'];rgb=z['rgb']
        features=sonata_features(points,rgb,resolve(root,neural['sonata_repository']),resolve(root,neural['sonata_weights']),out/'09_frozen_features.npz')
        probability=semantics(features,points,resolve(root,neural['semantic_checkpoint']))
        np.savez_compressed(out/'09_semantics.npz',probability=probability)
        return dict(encoder_frozen=True,points=len(points),training_executed=False)
    def step10():
        neural=cfg.get('neural',{})
        if not neural.get('enabled',False):save(out/'10_prediction.json',dict(components=[],neural_used=False));return dict(neural_used=False)
        from .release_neural import build_joint_inputs,joint_inference
        with np.load(out/'09_semantics.npz') as z:probability=z['probability']
        inputs=build_joint_inputs(out/'07_roof_observations.npz',probability,read(out/'08_model_frame.json'),out/'neural_inputs')
        result=joint_inference(inputs,resolve(root,neural['geometry_checkpoint']));save(out/'10_prediction.json',result)
        return dict(components=len(result['components']),inference_reference_reads=False)
    def step11():
        from .native_skp import write_native
        prediction=read(out/'10_prediction.json')
        if prediction.get('smoke_only'):raise ValueError('Smoke-only training checkpoint cannot be published as a CAD result')
        if prediction['components']:
            plan=dict(schema='zip2cad_native_completion_plan_v2',source_model=str(cad),source_model_sha256=sha(cad),components=prediction['components'],
                textures_allowed=False,hide_existing_pid_paths=[],view_scope_pid_path=cfg.get('scope_pid_path',[]),approval_scope='research_review',production_edit_accepted=False)
            save(out/'11_native_plan.json',plan);receipt=write_native(out/'11_native_plan.json',out/'native',sdk);result=dict(output_model=receipt['updated_model'],native_receipt=receipt)
        else:
            from .cloud_cli import complete_cloud
            with np.load(out/'07_roof_observations.npz') as z:views=np.unique(z['frame_refs'])
            if len(views)<4:raise ValueError('At least4 physical views for geometric train/development filtering required')
            held=views[::5].tolist();train=[int(v) for v in views if v not in held]
            result=complete_cloud(cad,out/'07_roof_observations.npz',out/'native_baseline',train,held,cfg.get('scope_pid_path',[]),sdk)
        shutil.copyfile(result['output_model'],out/'corrected_RESEARCH.skp')
        save(out/'11_export.json',dict(result,final_model='corrected_RESEARCH.skp',source_CAD_unchanged=sha(cad)==read(out/'01_inputs.json')['CAD_sha256']))
        return dict(SKP='corrected_RESEARCH.skp',formal_production_approval=False)
    def step12():
        if not cfg.get('reference_cad'):
            save(out/'12_evaluation.json',dict(status='not_evaluated',reason='No evaluation reference supplied; no accuracy claimed'));return dict(reference_evaluation=False)
        from .sketchup_sdk import SketchUpSDK
        from .delivery_metrics import compare_surfaces
        from shapely import Polygon,contains_xy
        ctx=read(out/'05_CAD_context.json');poly=Polygon(ctx['CAD_target_polygon_xy']).buffer(.5);base=ctx['CAD_attachment_z_m'];meshes=[]
        with SketchUpSDK(sdk) as api:
            for path in [out/'corrected_RESEARCH.skp',resolve(root,cfg['reference_cad'])]:
                model=api.load(path)
                try:d=api.geometry(model,visible_only=True)
                finally:api.release(model)
                tri=d['vertices_world_m'][d['faces']];c=tri.mean(1);mask=contains_xy(poly,c[:,0],c[:,1])&(tri[:,:,2].min(1)>=base-.05)&(tri[:,:,2].max(1)<=base+8)&(c[:,2]>=base+.35)
                mesh=trimesh.Trimesh(d['vertices_world_m'],d['faces'][mask],process=False);mesh.update_faces(mesh.area_faces>1e-10);meshes.append(mesh)
        if min(len(m.faces) for m in meshes)==0:raise ValueError('Reference evaluation ROI empty; inspect scope/registration')
        score,_,_=compare_surfaces(*meshes,count=int(cfg.get('evaluation_samples',20000)))
        save(out/'12_evaluation.json',dict(score,independent_validation=False,reference_used_for_evaluation_only_this_run=True));return dict(reference_evaluation=True)
    def step13():
        p=read(out/'10_prediction.json');save(out/'13_change_proposal.json',dict(source_CAD_sha256=sha(cad),candidate_sha256=sha(out/'corrected_RESEARCH.skp'),
             source_CAD_overwritten=False,actions=[dict(component_id=c['component_id'],action='add_research_candidate') for c in p['components']],
             neural_geometry_used=bool(p['components']),review_required=True))
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Zip2CAD run</title><h1>ZipMap反向修正CAD图纸</h1><p>Research candidate; source CAD preserved.</p><p><a href="corrected_RESEARCH.skp">Editable SKP</a> · <a href="12_evaluation.json">Evaluation</a> · <a href="13_change_proposal.json">Change proposal</a> · <a href="workflow.json">14-step ledger</a></p>',encoding='utf-8')
        return dict(change_proposal_written=True)
    def step14():
        original=read(out/'01_inputs.json')
        if sha(video)!=original['video_sha256'] or sha(cad)!=original['CAD_sha256']:raise ValueError('Original input changed during run')
        save(out/'14_verification.json',dict(status='passed',original_video_unchanged=True,original_CAD_unchanged=True,
             final_SKP_sha256=sha(out/'corrected_RESEARCH.skp'),full_building_accuracy_certified=False))
        return dict(source_preservation=True)
    functions=[step1,step2,step3,step4,step5,step6,step7,step8,step9,step10,step11,step12,step13,step14]
    for number,(name,fn) in enumerate(zip(STAGES,functions),1):
        if number>to_stage:break
        stage(number,name,fn)
    state.update(status='complete' if to_stage==14 else 'partial',progress_percent=round(len(state['stages'])/14*100,2));save(ledger,state)
    return out
