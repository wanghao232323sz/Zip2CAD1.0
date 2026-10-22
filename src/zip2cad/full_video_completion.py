"""Scan all cached video blocks, extend camera coverage and prepare one gauge.

Each prediction block declares a scalar similarity into the common ZipMap
frame. Camera anchors and video frame manifest are explicit portable inputs.
The module never imports a case's roof dimensions or image annotations.
"""
from pathlib import Path
import json
import cv2,numpy as np
from .video_camera_extension import background_landmarks,estimate_camera,roof_mask
from .skp_exchange import canonical_json,sha256_file

def prepare_views(config,base,video,seed_cameras,polygon,support,output):
    def path(value):
        p=Path(value);return (p if p.is_absolute() else base/p).resolve()
    blocks=config['prediction_blocks'];manifest_path=path(config['frame_manifest']);manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    frame_rows={int(r['accepted_index'])-1:r for r in manifest['frames'] if r.get('status')=='accepted'}
    sources={};images={};block_rows=[];bindings=[];capture=cv2.VideoCapture(str(video))
    try:
        for block in blocks:
            file=path(block['path']);refs=block['frame_refs'];transform=block.get('to_common',{'scale':1.,'rotation':np.eye(3).tolist(),'translation':[0,0,0]})
            r=np.asarray(transform['rotation']);s=float(transform['scale']);t=np.asarray(transform['translation'])
            if s<=0 or r.shape!=(3,3) or not np.allclose(r@r.T,np.eye(3),atol=1e-4) or np.linalg.det(r)<.999 or t.shape!=(3,):raise ValueError('Block needs a positive scalar rigid similarity')
            with np.load(file) as z:
                rgb=z['images'].transpose(0,2,3,1);conf=z['world_points_conf']
                if len(rgb)!=len(refs):raise ValueError('Block frame mapping mismatch')
                for i,ref in enumerate(refs):
                    score=float(np.median(conf[i]));previous=sources.get(ref)
                    if previous is None or score>previous['confidence_score']:
                        images[ref]=rgb[i].copy();sources[ref]={'path':str(file),'index':i,'confidence_score':score,'to_common':transform}
                # Bind a real original-video frame to every prediction block.
                ref=refs[len(refs)//2]
                if ref not in frame_rows:raise ValueError('Prediction frame not present in video manifest')
                timestamp=float(frame_rows[ref]['timestamp_s']);capture.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000);ok,frame=capture.read()
                if not ok:raise ValueError('Cannot decode bound source-video frame')
                im=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),rgb[0].shape[1::-1]).astype(float)/255
                difference=float(np.mean(abs(im-rgb[len(refs)//2])))
                if difference>.04:raise ValueError('Cached RGB does not match original video: '+str(ref)+' MAE '+str(difference))
                bindings.append({'frame_ref':ref,'timestamp_s':timestamp,'RGB_mean_absolute_error':difference})
            block_rows.append({'path':str(file),'sha256':sha256_file(file),'frame_count':len(refs),'to_common':transform})
            print('video block scanned',len(block_rows),'/',len(blocks),'unique frames',len(images),flush=True)
    finally:capture.release()
    missing=set(seed_cameras['frame_refs'])-set(images)
    if missing:raise ValueError('Camera anchors have no cached source images: '+str(missing))
    shape=images[seed_cameras['frame_refs'][0]].shape[:2];h,w=shape;native=seed_cameras['native_image_size'];scale=np.diag([w/native[0],h/native[1],1])
    cameras={ref:(scale@np.asarray(k),np.asarray(r),np.asarray(c)) for ref,k,r,c in zip(seed_cameras['frame_refs'],seed_cameras['intrinsics'],seed_cameras['rotations'],seed_cameras['centers'])}
    landmarks=background_landmarks(images,cameras,np.asarray(polygon),support,seed_cameras['training_views']);np.savez_compressed(output/'background_landmarks.npz',points_CAD_m=landmarks['points'],descriptors=landmarks['descriptors'])
    print('background landmarks',len(landmarks['points']),flush=True)
    intrinsic=np.median([v[0] for v in cameras.values()],axis=0);rows=[]
    for ref in sorted(images):
        if images[ref].shape[:2]!=shape:raise ValueError('All selected prediction images need the same dimensions')
        if ref in cameras:
            k,r,c=cameras[ref];area=int((roof_mask(shape,np.asarray(polygon),support,k,r,c,0)>0).sum());row={'status':'input_camera_anchor','frame_ref':ref,'projected_roof_pixels':area,'camera_source':'provided_calibration','p90_reprojection_px':None}
        else:
            camera,row=estimate_camera(images[ref],intrinsic,landmarks,np.asarray(polygon),support,maximum_p90_px=config.get('camera_maximum_p90_px',3.))
            row['frame_ref']=ref
            if camera is not None:cameras[ref]=camera
        rows.append(row)
        if (ref+1)%32==0:print('camera frames examined',ref+1,'accepted',len(cameras),flush=True)
    maximum=config.get('maximum_refine_views',16);selected=list(seed_cameras['frame_refs'])
    candidates=[r for r in rows if r['status']=='roof_camera_accepted' and r['frame_ref'] not in selected]
    # Source visibility and camera diversity select training views; known
    # heldout anchors remain frozen and do not rank the extra training cameras.
    while candidates and len(selected)<maximum:
        def rank(row):
            ref=row['frame_ref'];distance=min(np.linalg.norm(cameras[ref][2]-cameras[q][2]) for q in selected)
            return min(distance,20.)*np.sqrt(row['projected_roof_pixels'])/max(1.,row['p90_reprojection_px'])
        chosen=max(candidates,key=rank);candidates.remove(chosen);selected.append(chosen['frame_ref'])
    known_training=list(seed_cameras['training_views']);known_heldout=list(seed_cameras['heldout_views']);new=sorted(set(selected)-set(seed_cameras['frame_refs']));training=known_training+new;heldout=known_heldout
    # Put a calibrated training source first for the finite boundary extractor.
    selected=[known_training[0]]+[r for r in selected if r!=known_training[0]]
    point_rows=[];confidence=[];rgb_rows=[]
    for ref in selected:
        info=sources[ref];tr=info['to_common']
        with np.load(info['path']) as z:
            p=z['world_points'][info['index']];point_rows.append(tr['scale']*p@np.asarray(tr['rotation']).T+np.asarray(tr['translation']));confidence.append(z['world_points_conf'][info['index']]);rgb_rows.append(images[ref].transpose(2,0,1))
    prediction=output/'selected_predictions.npz';np.savez_compressed(prediction,world_points=np.asarray(point_rows,np.float32),world_points_conf=np.asarray(confidence,np.float32),images=np.asarray(rgb_rows,np.float32))
    camera_out={'frame_refs':selected,'training_views':training,'heldout_views':heldout,'registration_training_views':known_training,
        'intrinsics':[(np.linalg.inv(scale)@cameras[r][0]).tolist() for r in selected],'rotations':[cameras[r][1].tolist() for r in selected],
        'centers':[cameras[r][2].tolist() for r in selected],'native_image_size':native}
    (output/'extended_cameras.json').write_text(canonical_json(camera_out),encoding='utf-8')
    report={'schema':'zip2cad_full_video_camera_coverage_v1','unique_video_frames_examined':len(images),'prediction_block_count':len(blocks),
        'accepted_background_camera_count':sum(r['status']=='roof_camera_accepted' for r in rows),'input_anchor_camera_count':len(seed_cameras['frame_refs']),
        'selected_refinement_frame_refs':selected,'selected_refinement_view_count':len(selected),'frame_rows':rows,'blocks':block_rows,
        'source_video_RGB_bindings':bindings,'source_frame_manifest_sha256':sha256_file(manifest_path),'landmark_count':len(landmarks['points']),
        'landmark_pair_rows':landmarks['pair_rows'],'roof_geometry_used_for_camera_extension':False,
        'all_camera_poses_independently_survey_certified':False,'camera_anchors_are_calibration_inputs':True}
    (output/'video_coverage_report.json').write_text(canonical_json(report),encoding='utf-8')
    print('selected refinement views',selected,flush=True);return prediction,camera_out,report
