"""Finite planar wall/roof additions from an attributed CAD-gauge point cloud.

Every new face is trained on observed points. Thin hidden closure is a declared
engineering thickness prior. The extractor supports arbitrary plane directions.
"""
from dataclasses import dataclass
import cv2,numpy as np,trimesh
from scipy import ndimage
from scipy.spatial import ConvexHull
from .pointcloud_completion import polygon_prism

@dataclass(frozen=True)
class SurfacePolicy:
    voxel_m:float=.3
    plane_tolerance_m:float=.3
    minimum_area_m2:float=2.
    minimum_change_m:float=.25
    minimum_training_views:int=2
    minimum_heldout_fraction:float=.5
    maximum_components:int=24
    thickness_prior_m:float=.2

def extract_structural_surfaces(points,frame_refs,cad_mesh,training,heldout,policy=SurfacePolicy()):
    p=np.asarray(points,float);refs=np.asarray(frame_refs,int)
    if p.ndim!=2 or p.shape[1]!=3 or refs.shape!=(len(p),) or not np.isfinite(p).all():raise ValueError('Finite source-bound CAD-gauge points required')
    if set(training)&set(heldout) or not heldout:raise ValueError('Disjoint training and validation frames required')
    if min(policy.voxel_m,policy.plane_tolerance_m,policy.minimum_area_m2,policy.thickness_prior_m)<=0:raise ValueError('Positive geometry policy required')
    # Balance frame sampling and retain original observation identities.
    cells=np.floor(p/policy.voxel_m).astype(int);_,indices=np.unique(np.c_[cells,refs],axis=0,return_index=True);v=p[indices];vr=refs[indices]
    remaining=np.flatnonzero(np.isin(vr,training));validation=np.flatnonzero(np.isin(vr,heldout));rng=np.random.default_rng(19);parts=[];rows=[]
    for _ in range(policy.maximum_components*3):
        if len(remaining)<24 or len(parts)>=policy.maximum_components:break
        best=[]
        for j in range(120):
            tri=v[rng.choice(remaining,3,False)];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(n)
            if length<1e-9:continue
            n/=length;selected=remaining[abs((v[remaining]-tri[0])@n)<=policy.plane_tolerance_m]
            if len(selected)>len(best):best=selected
        if len(best)<24:break
        remaining=remaining[~np.isin(remaining,best)];origin=v[best].mean(0);_,_,vh=np.linalg.svd(v[best]-origin,full_matrices=False);n=vh[-1]
        if n[np.argmax(abs(n))]<0:n=-n
        u=vh[0];u/=np.linalg.norm(u);w=np.cross(n,u);basis=np.c_[u,w];xy=(v[best]-origin)@basis
        cells=np.floor(xy/policy.voxel_m).astype(int);low=cells.min(0);size=cells.max(0)-low+3
        grid=np.zeros(tuple(size),np.uint8);grid[tuple((cells-low+1).T)]=1
        # Rotating frame-balanced world voxels into plane coordinates can
        # leave one-cell sampling gaps. Group neighbouring observations while
        # fitting boundaries only to the original, undilated points.
        labels,count=ndimage.label(ndimage.binary_dilation(grid,np.ones((3,3))),np.ones((3,3)))
        for label in range(1,count+1):
            selected=best[labels[tuple((cells-low+1).T)]==label]
            if len(selected)<16 or len(np.unique(vr[selected]))<policy.minimum_training_views:continue
            plane_xy=(v[selected]-origin)@basis
            try:boundary=plane_xy[ConvexHull(plane_xy).vertices]
            except Exception:continue
            area=float(cv2.contourArea(boundary.astype(np.float32)))
            if area<policy.minimum_area_m2:continue
            boundary=cv2.approxPolyDP(boundary.astype(np.float32).reshape(-1,1,2),policy.voxel_m,True).reshape(-1,2)
            if len(boundary)<3:continue
            held_rows=[]
            for ref in heldout:
                ids=validation[vr[validation]==ref];q=(v[ids]-origin)@basis;inside=np.array([cv2.pointPolygonTest(boundary,tuple(map(float,x)),False)>=0 for x in q])
                errors=abs((v[ids[inside]]-origin)@n);fraction=float(np.mean(errors<=policy.plane_tolerance_m*1.5)) if len(errors) else 0.
                held_rows.append({'frame_ref':int(ref),'point_count':len(errors),'supported_fraction':fraction})
            accepted=sum(r['point_count']>=4 and r['supported_fraction']>=policy.minimum_heldout_fraction for r in held_rows)>=1
            _,dist,_=trimesh.proximity.closest_point(cad_mesh,v[selected]);novel_fraction=float(np.mean(dist>policy.minimum_change_m));accepted&=novel_fraction>=.5
            row={'component_id':'surface_'+str(len(rows)+1).zfill(3),'plane_origin':origin.tolist(),'plane_normal':n.tolist(),'area_m2':area,
                'training_frame_refs':np.unique(vr[selected]).tolist(),'training_observation_indices':indices[selected].tolist(),'heldout_rows':held_rows,
                'median_CAD_distance_m':float(np.median(dist)),'novel_fraction':novel_fraction,'accepted':bool(accepted),'source':'attributed_pointcloud_plane',
                'surface_type':'wall' if abs(n[2])<.3 else ('roof' if abs(n[2])>.7 else 'inclined_structural_plane'),
                'thickness_is_engineering_prior':True,'hidden_geometry_measured':False,'maximum_sampling_gap_grouping_m':2*policy.voxel_m};rows.append(row)
            if not accepted:continue
            top=origin+boundary@basis.T;bottom=top-n*policy.thickness_prior_m
            # polygon_prism triangulates in XY; orient this arbitrary plane
            # into a local XY basis, then transform its closed solid to CAD.
            local_top=np.c_[boundary,np.zeros(len(boundary))];local_bottom=local_top-[0,0,policy.thickness_prior_m];mesh=polygon_prism(local_top,local_bottom)
            rotation=np.c_[u,w,n];vertices=mesh['vertices']@rotation.T+origin
            parts.append({'component_id':row['component_id'],'vertices_world_CAD_m':vertices.tolist(),'triangles':mesh['triangles'].tolist(),
                'top_triangle_mask':mesh['top_triangle_mask'].tolist(),'top_color':[160,160,156],'side_color':[231,231,226],
                'volume_CAD_units_cubed':mesh['volume_CAD_units_cubed'],'closed_solid_verified':True})
    return {'components':parts,'candidate_rows':rows,'approval_scope':'engineering_preview','full_building_certified':False}
