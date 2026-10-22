"""Experimental training-only PCA/trimmed Sim(3) CAD initialization.

This selects an engineering pose hypothesis. Symmetric or background-dominated
observations can remain ambiguous; a calibrated camera input overrides this
initializer when available. It is not an identity or survey certificate.
"""
from itertools import permutations,product
import numpy as np
from scipy.spatial import cKDTree
from .similarity import estimate_similarity,transform_points

def initialize_similarity(source,target,iterations=18,trim=.65):
    source,target=np.asarray(source,float),np.asarray(target,float)
    if source.ndim!=2 or source.shape[1]!=3 or target.ndim!=2 or target.shape[1]!=3 or min(len(source),len(target))<100:raise ValueError('Dense source and target geometry required')
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Finite registration geometry required')
    source=source[np.linspace(0,len(source)-1,min(2400,len(source)),dtype=int)];target=target[np.linspace(0,len(target)-1,min(9000,len(target)),dtype=int)]
    sc,tc=np.median(source,axis=0),np.median(target,axis=0);_,_,sv=np.linalg.svd(source-sc,full_matrices=False);_,_,tv=np.linalg.svd(target-tc,full_matrices=False)
    extent_s=np.percentile(np.linalg.norm(source-sc,axis=1),75);extent_t=np.percentile(np.linalg.norm(target-tc,axis=1),75);initial=extent_t/extent_s;tree=cKDTree(target);rows=[]
    for perm in permutations(range(3)):
        for signs in product([-1,1],repeat=3):
            q=np.eye(3)[list(perm)]*np.array(signs)[:,None];rotation=tv.T@q@sv
            if np.linalg.det(rotation)<0:continue
            for factor in [.7,1.,1.4]:
                scale=initial*factor;translation=tc-scale*rotation@sc
                for _ in range(iterations):
                    transformed=transform_points(source,scale,rotation,translation);distance,nearest=tree.query(transformed)
                    keep=np.argsort(distance)[:max(50,int(len(source)*trim))]
                    s,r,t=estimate_similarity(source[keep],target[nearest[keep]])
                    if not initial*.1<s<initial*10:break
                    scale,rotation,translation=s,r,t
                distance=tree.query(transform_points(source,scale,rotation,translation))[0];score=float(np.mean(np.sort(distance)[:int(len(source)*trim)]))
                rows.append({'scale':float(scale),'rotation':rotation.tolist(),'translation':translation.tolist(),'training_trimmed_mean_distance_m':score})
    rows.sort(key=lambda r:r['training_trimmed_mean_distance_m']);best=rows[0]
    best.update(status='experimental_CAD_conditioned_alignment',candidate_count=len(rows),alternative_score=rows[1]['training_trimmed_mean_distance_m'],identity_certified=False,
        alternative_hypotheses=rows[1:8])
    return best

def initialize_cameras(prediction,indices,frame_refs,mesh,native_image_size):
    images=prediction['images'][indices];world=prediction['world_points'][indices];confidence=prediction['world_points_conf'][indices]
    extrinsic=prediction['extrinsic'][indices];intrinsic=prediction['intrinsic'][indices];sample=[];h,w=images.shape[-2:]
    for i in range(len(indices)):
        if i%5==0:continue # freeze validation frames before pose fitting
        r=extrinsic[i,:3,:3];c=-np.linalg.solve(r,extrinsic[i,:3,3]);distance=np.linalg.norm(world[i]-c,axis=2)
        yy,xx=np.mgrid[:h,:w];roi=(xx>w*.12)&(xx<w*.88)&(yy>h*.12)&(yy<h*.9)&(confidence[i]>=np.percentile(confidence[i],65))
        if roi.sum()<100:continue
        roi&=distance<=np.percentile(distance[roi],55);p=world[i][roi];sample.append(p[np.linspace(0,len(p)-1,min(1600,len(p)),dtype=int)])
    if len(sample)<3:raise ValueError('At least three usable ZipMap images for camera initialization')
    rng=np.random.default_rng(77);tri=rng.choice(len(mesh.faces),9000,p=mesh.area_faces/mesh.area_faces.sum());bary=rng.random((9000,2));bary[bary.sum(1)>1]=1-bary[bary.sum(1)>1]
    target=mesh.triangles[tri,0]+bary[:,0,None]*(mesh.triangles[tri,1]-mesh.triangles[tri,0])+bary[:,1,None]*(mesh.triangles[tri,2]-mesh.triangles[tri,0])
    registration=initialize_similarity(np.concatenate(sample),target)
    registration['initialization_training_frame_refs']=[ref for i,ref in enumerate(frame_refs) if i%5!=0]
    registration['heldout_geometry_used_for_initialization']=False
    centres=np.array([-np.linalg.solve(e[:3,:3],e[:3,3]) for e in extrinsic])
    candidates=[{k:v for k,v in registration.items() if k in ['scale','rotation','translation','training_trimmed_mean_distance_m']}]+registration['alternative_hypotheses']
    # Roof-mode drone initialization uses a declared CAD height/above-roof
    # prior to reject upside-down PCA hypotheses; it is not hidden evidence.
    usable=[q for q in candidates if np.mean((q['scale']*centres@np.array(q['rotation']).T+q['translation'])[:,2]>mesh.bounds[1,2])>=.5]
    if usable:
        chosen=min(usable,key=lambda q:q['training_trimmed_mean_distance_m']);registration.update(chosen)
    s=registration['scale'];r=np.array(registration['rotation']);t=np.array(registration['translation']);K=[];R=[];C=[]
    scale=np.diag([native_image_size[0]/w,native_image_size[1]/h,1])
    for e,k in zip(extrinsic,intrinsic):
        center=-np.linalg.solve(e[:3,:3],e[:3,3]);C.append((s*r@center+t).tolist());R.append((e[:3,:3]@r.T).tolist());K.append((scale@k).tolist())
    training=[ref for i,ref in enumerate(frame_refs) if i%5!=0];heldout=[ref for i,ref in enumerate(frame_refs) if i%5==0]
    return {'frame_refs':frame_refs,'training_views':training,'heldout_views':heldout,'intrinsics':K,'rotations':R,'centers':C,
        'native_image_size':native_image_size,'initialization_report':registration,'calibration_source':'automatic_ZipMap_CAD_pose_hypothesis'}
