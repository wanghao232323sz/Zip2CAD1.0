"""Video-supported neutral roof finishes, without photographs or new dimensions.

Only existing upper horizontal CAD faces are considered. Colour and opacity
updates are an explicit edit class, separately reported from geometry additions.
"""
import numpy as np,cv2

def sample(image,points,K,R,C):
    p=(points-C)@R.T@K.T;uv=p[:,:2]/np.maximum(p[:,2:],1e-8);h,w=image.shape[:2]
    visible=(p[:,2]>0)&(uv[:,0]>1)&(uv[:,0]<w-2)&(uv[:,1]>1)&(uv[:,1]<h-2)
    rgb=cv2.remap(image.astype(np.float32),uv[:,0,None].astype(np.float32),uv[:,1,None].astype(np.float32),cv2.INTER_LINEAR).reshape(-1,3)
    return rgb,visible

def propose_roof_finishes(cad,images,K,R,C,frame_refs,training,heldout,scope_prefix,base,outline,hidden_paths,top_color=(160,160,156),side_color=(231,231,226)):
    vertices=cad['vertices_world_m'];faces=cad['faces'];owners=cad['face_owner_indices'];ids=cad['entity_ids'];opacity=cad.get('face_material_opacity',np.ones(len(ids)))
    triangles=vertices[faces];cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);area=np.linalg.norm(cross,axis=1)/2
    normal=cross/np.maximum(2*area[:,None],1e-12);centres=triangles.mean(1);polygon=np.asarray(outline,np.float32);updates=[];rows=[]
    for owner in np.unique(owners):
        name=str(ids[owner])
        if not name.startswith(scope_prefix):continue
        path=list(map(int,name.split(':')[-1].split('.')))
        if any(path[:len(p)]==p for p in hidden_paths):continue
        sel=(owners==owner)&(abs(normal[:,2])>.98)&(area>.02)&(centres[:,2]>=base-.06)&(centres[:,2]<=base+1.2)
        if not sel.any():continue
        a=float(area[sel].sum());height=float(np.average(centres[sel,2],weights=area[sel]));surface='roof_finish' if abs(height-base)<.08 else 'upper_roof_coping'
        if surface=='roof_finish' and a<3:continue
        if surface=='upper_roof_coping' and a<.2:continue
        t=triangles[sel];order=np.argsort(-area[sel])[:12];t=t[order]
        samples=np.concatenate([t.mean(1),.6*t[:,0]+.2*t[:,1]+.2*t[:,2],.2*t[:,0]+.6*t[:,1]+.2*t[:,2],.2*t[:,0]+.2*t[:,1]+.6*t[:,2]])
        inside=np.array([cv2.pointPolygonTest(polygon,tuple(map(float,p)),False)>=0 for p in samples[:,:2]])
        samples=samples[inside]
        if len(samples)<4:continue
        observations=[]
        for i,ref in enumerate(frame_refs):
            rgb,visible=sample(images[i],samples,K[i],R[i],C[i])
            if visible.sum()<max(3,len(samples)*.5):continue
            observations.append({'frame_ref':int(ref),'median_RGB':np.median(rgb[visible],axis=0).tolist(),'visible_samples':int(visible.sum())})
        train=[r for r in observations if r['frame_ref'] in training];valid=[r for r in observations if r['frame_ref'] in heldout]
        if len(train)<2 or not valid:continue
        colour=np.median([r['median_RGB'] for r in train],axis=0);scatter=np.linalg.norm(np.array([r['median_RGB'] for r in train])-colour,axis=1)
        matched=[r for r in train if np.linalg.norm(np.array(r['median_RGB'])-colour)<.18];tested=[r for r in valid if np.linalg.norm(np.array(r['median_RGB'])-colour)<.18]
        # Roof surfaces with unresolved view-dependent appearance are retained.
        eligible=len(matched)>=2 and bool(tested) and float(np.median(scatter))<.12
        if surface=='upper_roof_coping':eligible&=float(np.mean(colour))>.42
        if surface=='roof_finish':eligible&=float(np.mean(colour))<.65
        row={'source_face_pid_path':path,'source_face_id':name,'surface_type':surface,'area_m2':a,'source_effective_opacity':float(opacity[owner]),
            'observed_training_median_RGB':colour.tolist(),'training_rows':train,'heldout_rows':valid,'accepted':bool(eligible),
            'geometry_changed':False,'appearance_is_schematic_neutral_finish':True,'physical_opacity_is_engineering_roof_prior':True}
        rows.append(row)
        if eligible:updates.append({'source_face_pid_path':path,'color':list(top_color if surface=='roof_finish' else side_color),
            'component_id':'roof_finish' if surface=='roof_finish' else 'roof_coping_finish','opacity':1.,'evidence':'calibrated_training_and_heldout_video_RGB',
            'material_policy':'neutral_opaque_roof_finish','original_geometry_preserved':True})
    # SDK faces belong to definitions. Editing a shared leaf also changes its
    # other instances; authorize it only when every occurrence was supported
    # with the same finish. Unobserved/background instances remain untouched.
    occurrences={}
    for name in ids:
        leaf=str(name).split(':')[-1].split('.')[-1];occurrences.setdefault(leaf,set()).add(str(name).split(':')[-1])
    proposed={}
    for update in updates:
        leaf=str(update['source_face_pid_path'][-1]);proposed.setdefault(leaf,[]).append(update)
    safe=[]
    for leaf,items in proposed.items():
        paths={'.'.join(map(str,i['source_face_pid_path'])) for i in items};colours={tuple(i['color']) for i in items}
        if paths==occurrences[leaf] and len(colours)==1:safe.extend(items)
        else:
            for row in rows:
                if str(row['source_face_pid_path'][-1])==leaf:row['accepted']=False;row['rejection_reason']='shared_source_definition_has_unapproved_or_conflicting_instances'
    updates=safe
    return {'updates':updates,'candidate_rows':rows,'accepted_face_count':len(updates),'new_photographic_texture_count':0,
        'geometry_additions_count':0,'full_physical_material_identification_certified':False}
