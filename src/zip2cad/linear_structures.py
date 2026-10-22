"""Recover finite beam/post candidates along trusted architectural axes.

Observed cross-section bounds are partial: their unseen completion is an
engineering box assumption, not a survey measurement.
"""
import numpy as np,trimesh
from .roof_structures import StructurePolicy

def extract_linear_structures(points,refs,cad,training,validation,polygon,protected_parts=(),minimum_length_m=2.5,radius_m=.25,maximum_parts=24):
    p=np.asarray(points,float);refs=np.asarray(refs,int);poly=np.asarray(polygon,float)
    if set(training)&set(validation) or not validation:raise ValueError('Disjoint beam training and validation views required')
    train=np.isin(refs,training);v=p[train];vr=refs[train]
    if len(v)<50:return {'components':[],'candidates':[]}
    edges=np.roll(poly,-1,axis=0)-poly;edge=edges[np.argmax(np.linalg.norm(edges,axis=1))];a=np.r_[edge/np.linalg.norm(edge),0.];b=np.cross([0,0,1],a)
    axes=[a,b,np.array([0.,0.,1.])];parts=[];rows=[]
    for axis_index,d in enumerate(axes):
        u=axes[1-axis_index] if axis_index<2 else a;w=np.cross(d,u);basis=np.c_[d,u,w];local=v@basis
        cross=np.floor(local[:,1:]/radius_m).astype(int);cells,counts=np.unique(cross,axis=0,return_counts=True);order=np.argsort(counts)[::-1]
        used=np.zeros(len(v),bool)
        for cell in cells[order[:250]]:
            centre=(cell+.5)*radius_m;select=(np.linalg.norm(local[:,1:]-centre,axis=1)<=radius_m*1.25)&~used
            selected=np.flatnonzero(select)
            if len(selected)<50:continue
            along=local[selected,0];bins=np.unique(np.floor(along/.2).astype(int));gaps=np.flatnonzero(np.diff(bins)>.6/.2)+1
            for segment in np.split(bins,gaps):
                low,high=segment[0]*.2,(segment[-1]+1)*.2
                if high-low<minimum_length_m or len(segment)*.2/(high-low)<.55:continue
                ids=selected[(along>=low)&(along<=high)]
                if len(ids)<50 or len(np.unique(vr[ids]))<2:continue
                q=local[ids];bounds=np.percentile(q,[5,95],axis=0);extent=bounds[1]-bounds[0]
                if extent[0]<minimum_length_m or min(extent[1:])<.04:continue
                # Thin observed side faces cannot determine a full section;
                # a minimum section is an explicit engineering prior.
                extent[1:]=np.maximum(extent[1:],.08);centre_local=bounds.mean(0);centre_world=centre_local@basis.T
                allq=p@basis;inside=np.all(abs(allq-centre_local)<=extent/2+.12,axis=1)
                validation_counts={int(r):int(np.sum(inside&(refs==r))) for r in validation}
                if max(validation_counts.values(),default=0)<12:continue
                dist=trimesh.proximity.closest_point(cad,v[ids][::max(1,len(ids)//150)])[1]
                if np.mean(dist>.18)<.65:continue
                box=trimesh.creation.box(extent);box.vertices=box.vertices@basis.T+centre_world
                duplicate=False
                for item in list(protected_parts)+parts:
                    m=trimesh.Trimesh(item['vertices_world_CAD_m'],item['triangles'],process=False)
                    if np.all(box.bounds[1]>=m.bounds[0]) and np.all(box.bounds[0]<=m.bounds[1]):
                        # Both an already represented surface and an already
                        # represented finite volume suppress the proposal.
                        distance=trimesh.proximity.closest_point(m,box.vertices)[1]
                        if np.mean(distance<.2)>.65:duplicate=True;break
                if duplicate:continue
                name=f'linear_{len(parts)+1:03d}';used[ids]=True
                rows.append({'component_id':name,'surface_type':'post_candidate' if axis_index==2 else 'beam_candidate','native_edit_approved':True,'extent_m':extent.tolist(),'centre_world_m':centre_world.tolist(),'axis':d.tolist(),'training_frame_refs':np.unique(vr[ids]).tolist(),'validation_counts':validation_counts,'along_axis_occupied_fraction':len(segment)*.2/(high-low),'section_is_partial_observation_box_prior':True,'semantic_component_identity_verified':False})
                parts.append({'component_id':name,'vertices_world_CAD_m':box.vertices.tolist(),'triangles':box.faces.tolist(),'top_triangle_mask':(abs(box.face_normals[:,2])>.8).tolist(),'top_color':[175,175,170],'side_color':[230,230,224],'volume_CAD_units_cubed':float(box.volume),'closed_solid_verified':True})
                if len(parts)>=maximum_parts:return {'components':parts,'candidates':rows}
    return {'components':parts,'candidates':rows}
