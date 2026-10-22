"""Multi-scale finite structural patches with training-only shape estimation.

This module accepts attributed points in any metric CAD frame. It does not
contain building names, coordinates, dimensions, image masks or painted assets.
Unobserved thickness is explicit; observed opposing surfaces can replace it.
"""
from dataclasses import dataclass,asdict
import cv2,numpy as np,trimesh
from scipy import ndimage
from scipy.spatial import cKDTree,ConvexHull
from .pointcloud_completion import polygon_prism

@dataclass(frozen=True)
class StructurePolicy:
    voxel_m:float=.12
    plane_tolerance_m:float=.10
    minimum_area_m2:float=.30
    minimum_change_m:float=.18
    minimum_training_views:int=2
    minimum_validation_points:int=12
    maximum_components:int=48
    thickness_prior_m:float=.12
    normal_radius_m:float=.55
    minimum_occupancy:float=.40
    normal_angle_degrees:float=25.
    minimum_rectangularity:float=.75

def _basis(n):
    seed=np.eye(3)[np.argmin(abs(n))];u=np.cross(n,seed);u/=np.linalg.norm(u);return np.c_[u,np.cross(n,u)]

def extract_structures(points,frame_refs,cad_mesh,training,validation,policy=StructurePolicy(),protected_parts=(),normal_hints=None):
    p=np.asarray(points,float);refs=np.asarray(frame_refs,int);training=set(map(int,training));validation=set(map(int,validation))
    if p.ndim!=2 or p.shape[1]!=3 or refs.shape!=(len(p),) or not np.isfinite(p).all():raise ValueError('Finite attributed metric points required')
    if not training or not validation or training&validation:raise ValueError('Disjoint training and validation views required')
    if min(policy.voxel_m,policy.plane_tolerance_m,policy.minimum_area_m2,policy.thickness_prior_m)<=0:raise ValueError('Positive structure policy required')
    ids=np.flatnonzero(np.isin(refs,list(training)));cells=np.floor(p[ids]/policy.voxel_m).astype(np.int64)
    _,keep=np.unique(cells,axis=0,return_index=True);ids=ids[keep];v=p[ids]
    if len(v)<24:return {'components':[],'candidates':[],'policy':asdict(policy)}
    tree=cKDTree(v);distance,near=tree.query(v,k=min(18,len(v)),workers=-1);local=v[near];center=local.mean(1);delta=local-center[:,None]
    values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',delta,delta));normals=vectors[:,:,0]
    good=(distance[:,-1]<=policy.normal_radius_m)&(values[:,0]/np.maximum(values[:,1],1e-12)<.24)
    if normal_hints is not None:
        hint=np.asarray(normal_hints,float)
        if hint.shape!=p.shape or not np.isfinite(hint).all() or np.any(np.linalg.norm(hint,axis=1)<1e-8):raise ValueError('Finite nonzero source-bound normal hypotheses required')
        normals=hint[ids]/np.linalg.norm(hint[ids],axis=1)[:,None]
        good=distance[:,-1]<=policy.normal_radius_m
    remaining=np.flatnonzero(good);initial_normal_count=len(remaining);rng=np.random.default_rng(31);patches=[];rows=[];cos=np.cos(np.deg2rad(policy.normal_angle_degrees))
    for iteration in range(policy.maximum_components*3):
        if len(remaining)<24:break
        # Seed normals prevent horizontal roof observations from consuming
        # vertical strips. Every proposal and boundary uses training only.
        seeds=rng.choice(remaining,min(160,len(remaining)),False);best=np.array([],int)
        for seed in seeds:
            n=normals[seed];select=remaining[(abs((v[remaining]-v[seed])@n)<=policy.plane_tolerance_m)&(abs(normals[remaining]@n)>=cos)]
            if len(select)>len(best):best=select
        if len(best)<24:break
        remaining=remaining[~np.isin(remaining,best)];origin=v[best].mean(0);_,_,vh=np.linalg.svd(v[best]-origin,full_matrices=False);n=vh[-1]
        if n[np.argmax(abs(n))]<0:n=-n
        basis=_basis(n);xy=(v[best]-origin)@basis;cells=np.floor(xy/policy.voxel_m).astype(int);low=cells.min(0);size=cells.max(0)-low+3
        if np.prod(size)>2000000:continue
        grid=np.zeros(tuple(size),bool);grid[tuple((cells-low+1).T)]=True
        labels,count=ndimage.label(ndimage.binary_dilation(grid),np.ones((3,3)))
        for label in range(1,count+1):
            take=best[labels[tuple((cells-low+1).T)]==label]
            if len(take)<20:continue
            origin2=v[take].mean(0);q=(v[take]-origin2)@basis
            try:hull=q[ConvexHull(q).vertices]
            except Exception:continue
            area=float(cv2.contourArea(hull.astype(np.float32)));occupied=len(np.unique(np.floor(q/policy.voxel_m),axis=0))*policy.voxel_m**2
            if area<policy.minimum_area_m2 or occupied/max(area,1e-9)<policy.minimum_occupancy:continue
            boundary=cv2.approxPolyDP(hull.astype(np.float32).reshape(-1,1,2),policy.voxel_m*.7,True).reshape(-1,2).astype(float)
            if len(boundary)<3:continue
            # Snap only genuinely rectangular patches, never invent a large
            # rectangle from a sparse L-shaped set of observations.
            rect=cv2.boxPoints(cv2.minAreaRect(q.astype(np.float32)));rectarea=float(cv2.contourArea(rect))
            rectangularity=area/max(rectarea,1e-9)
            if rectangularity>policy.minimum_rectangularity:boundary=rect.astype(float)
            area=float(cv2.contourArea(boundary.astype(np.float32)));allxy=(p-origin2)@basis
            # Local finite support rather than all points on an infinite plane.
            near2=cKDTree(q);d,_=near2.query(allxy,workers=-1);on=(d<=policy.voxel_m*2)&(abs((p-origin2)@n)<=policy.plane_tolerance_m*1.5)
            train_counts={int(ref):int(np.sum(on&(refs==ref))) for ref in sorted(training)}
            validation_counts={int(ref):int(np.sum(on&(refs==ref))) for ref in sorted(validation)}
            sample=v[take][np.linspace(0,len(take)-1,min(250,len(take)),dtype=int)]
            cad_dist=trimesh.proximity.closest_point(cad_mesh,sample)[1]
            novel=float(np.mean(cad_dist>policy.minimum_change_m));duplicate=False
            for part in protected_parts:
                m=trimesh.Trimesh(part['vertices_world_CAD_m'],part['triangles'],process=False)
                if np.all(sample.max(0)>=m.bounds[0]-.2) and np.all(sample.min(0)<=m.bounds[1]+.2):
                    if np.mean(trimesh.proximity.closest_point(m,sample)[1]<policy.plane_tolerance_m*2)>.65:duplicate=True;break
            accepted=sum(c>=12 for c in train_counts.values())>=policy.minimum_training_views and max(validation_counts.values(),default=0)>=policy.minimum_validation_points and novel>=.65 and not duplicate
            ext=np.ptp(q,axis=0);kind='vertical_patch' if abs(n[2])<.35 else ('horizontal_patch' if abs(n[2])>.85 else 'inclined_patch')
            if max(ext)/max(min(ext),1e-9)>=4:kind='linear_'+kind
            row={'component_id':f'structure_{len(rows)+1:03d}','surface_type':kind,'plane_origin':origin2.tolist(),'plane_normal':n.tolist(),'boundary_local':boundary.tolist(),'basis':basis.tolist(),'area_m2':area,'training_occupied_area_m2':occupied,'training_counts':train_counts,'validation_counts':validation_counts,'novel_fraction':novel,'duplicate_existing_addition':duplicate,'native_edit_approved':bool(accepted),'thickness_prior_m':policy.thickness_prior_m,'thickness_measured':False,'boundary_is_observed_patch_not_full_semantic_component':True,'rectangularity':rectangularity,'rectangular_constraint_applied':rectangularity>policy.minimum_rectangularity,'training_sample_indices':ids[take].tolist()}
            rows.append(row)
            if accepted:patches.append(row)
            if len(patches)>=policy.maximum_components:break
        if len(patches)>=policy.maximum_components:break
    parts=[]
    for row in patches:
        n=np.asarray(row['plane_normal']);origin=np.asarray(row['plane_origin']);basis=np.asarray(row['basis']);boundary=np.asarray(row['boundary_local']);thickness=policy.thickness_prior_m;direction=-1
        # Pair opposing observed patches. Parameters are recovered from
        # training planes; validation points cannot set the thickness.
        for other in rows:
            if other is row or not other['native_edit_approved']:continue
            no=np.asarray(other['plane_normal']);offset=float((np.asarray(other['plane_origin'])-origin)@n)
            if abs(n@no)>.985 and policy.thickness_prior_m<=abs(offset)<=1.:
                center=(np.asarray(other['plane_origin'])-origin)@basis
                if cv2.pointPolygonTest(boundary.astype(np.float32),tuple(map(float,center)),False)>=0 and other['area_m2']>=row['area_m2']*.5:
                    thickness=abs(offset);direction=1 if offset>0 else -1;row['thickness_measured']=True;row['paired_surface_id']=other['component_id'];break
        top=np.c_[boundary,np.zeros(len(boundary))];bottom=top+[0,0,direction*thickness]
        try:mesh=polygon_prism(top,bottom)
        except ValueError:row['native_edit_approved']=False;row['rejection_reason']='invalid_finite_mesh';continue
        transform=np.c_[basis,n];vertices=mesh['vertices']@transform.T+origin
        # Duplicate pairing solids are removed by surface distance rather
        # than naming or case-specific component IDs.
        candidate=trimesh.Trimesh(vertices,mesh['triangles'],process=True)
        duplicate=False
        for part in parts:
            prev=trimesh.Trimesh(part['vertices_world_CAD_m'],part['triangles'],process=False)
            if np.all(candidate.bounds[1]>=prev.bounds[0]) and np.all(candidate.bounds[0]<=prev.bounds[1]):
                if np.mean(trimesh.proximity.closest_point(prev,candidate.vertices)[1]<.04)>.85:duplicate=True;break
        if duplicate:row['native_edit_approved']=False;row['rejection_reason']='paired_solid_duplicate';continue
        row['solid_thickness_m']=thickness
        parts.append({'component_id':row['component_id'],'vertices_world_CAD_m':vertices.tolist(),'triangles':mesh['triangles'].tolist(),'top_triangle_mask':mesh['top_triangle_mask'].tolist(),'top_color':[175,175,170],'side_color':[230,230,224],'volume_CAD_units_cubed':mesh['volume_CAD_units_cubed'],'closed_solid_verified':True})
    # Record geometric adjacency for review without inventing a semantic
    # label or boolean-unioning uncertain intersections.
    adjacency=[]
    for i,a in enumerate(parts):
        pa=np.asarray(a['vertices_world_CAD_m'])
        for b in parts[i+1:]:
            pb=np.asarray(b['vertices_world_CAD_m']);gap=float(cKDTree(pa).query(pb)[0].min())
            if gap<=policy.voxel_m*2:adjacency.append({'a':a['component_id'],'b':b['component_id'],'vertex_gap_m':gap,'semantic_connection_verified':False})
    return {'schema':'zip2cad_finite_structures_v1','components':parts,'candidates':rows,'adjacency_candidates':adjacency,'policy':asdict(policy),'shape_estimation_uses_validation':False,'full_semantic_component_inventory_verified':False,'training_voxel_samples':len(v),'planar_normal_samples':initial_normal_count,'unassigned_planar_samples':len(remaining)}
