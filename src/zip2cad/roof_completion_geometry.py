"""Image-plane footprint extraction and closed editable roof component meshes."""
import cv2
import numpy as np
import trimesh
from zip2cad.roof_rim_geometry import circle_basis


def planar_surface_mask(images,intrinsics,rotations,centers,normal,origin,offset,roi,seed,training_views,*,window=9,minimum_zncc=.55,appearance_seed_polygon=None):
    n=np.asarray(normal,float);n/=np.linalg.norm(n);u,v=circle_basis(n);p=np.asarray(origin)+offset*n
    k,r,c=(np.asarray(x,float) for x in (intrinsics,rotations,centers));basis=np.c_[u,v]
    maps=[kk@rr@np.c_[basis,p-cc] for kk,rr,cc in zip(k,r,c)]
    source=cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY).astype(np.float32);height,width=source.shape
    def mean(x):return cv2.boxFilter(x,cv2.CV_32F,(window,window),normalize=True)
    m0=mean(source);var0=np.maximum(mean(source*source)-m0*m0,0);correlations=[];valid=[]
    for i in range(1,len(images)):
        h=maps[i]@np.linalg.inv(maps[0]);im=cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY).astype(np.float32)
        warped=cv2.warpPerspective(im,h,(width,height),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        visible=cv2.warpPerspective(np.ones(im.shape,np.float32),h,(width,height),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>.99
        mj=mean(warped);varj=np.maximum(mean(warped*warped)-mj*mj,0)
        corr=(mean(source*warped)-m0*mj)/np.maximum(np.sqrt(var0*varj),1e-6)
        ok=visible&(var0>=9)&(varj>=9)
        correlations.append(np.clip(corr,-1,1));valid.append(ok)
    scope=np.zeros(source.shape,np.uint8);cv2.fillPoly(scope,[np.asarray(roi,np.int32)],1)
    votes=sum((correlations[i-1]>=minimum_zncc)&valid[i-1] for i in training_views)
    appearance=np.ones(source.shape,bool);color_cut=None
    if appearance_seed_polygon is not None:
        color_scope=np.zeros(source.shape,np.uint8);cv2.fillPoly(color_scope,[np.asarray(appearance_seed_polygon,np.int32)],1)
        rb=images[0][:,:,2].astype(float)-images[0][:,:,0]
        samples=rb[(color_scope>0)&(source<110)]
        if len(samples)<100:raise ValueError('Insufficient RGB appearance seed samples')
        color_cut=float(np.percentile(samples,25));appearance=rb>=color_cut
    raw=(scope>0)&(source<140)&(votes>=2)&appearance
    mask=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    label=int(labels[int(seed[1]),int(seed[0])])
    if label==0:
        candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=100]
        if not candidates:raise ValueError('No photometrically supported roof component')
        label=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    selected=(labels==label).astype(np.uint8);contours,_=cv2.findContours(selected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea);hull=cv2.approxPolyDP(cv2.convexHull(contour),2.,True).reshape(-1,2)
    held=[i for i in range(1,len(images)) if i not in training_views]
    held_rows=[{'view_index':i,'fraction_with_ZNCC_at_least_threshold':float(np.mean((correlations[i-1]>=minimum_zncc)[selected>0])),
        'valid_fraction':float(np.mean(valid[i-1][selected>0]))} for i in held]
    return {'mask':selected,'hull_xy':hull,'source_pixel_count':int(selected.sum()),'heldout_rows':held_rows,
        'heldout_pixels_used_to_select_mask':False,'minimum_zncc':minimum_zncc,'window':window,
        'source_only_R_minus_B_lower_quartile':color_cut}


def clip_convex_xy(subject,clip_polygon):
    subject=np.asarray(subject,float);clip=np.asarray(clip_polygon,float)
    sign=np.sign(np.sum(clip[:,0]*np.roll(clip[:,1],-1)-clip[:,1]*np.roll(clip[:,0],-1)))
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    result=subject.copy()
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        output=[]
        if len(result)<3:raise ValueError('Roof footprint is outside the trusted exterior')
        previous=result[-1];dp=sign*cross(b-a,previous-a)
        for current in result:
            dc=sign*cross(b-a,current-a)
            if (dc>=0)!=(dp>=0):output.append(previous+(current-previous)*dp/(dp-dc))
            if dc>=0:output.append(current)
            previous,dp=current,dc
        result=np.array(output)
    return result


def closed_prism(top,bottom):
    top,bottom=np.asarray(top,float),np.asarray(bottom,float);n=len(top)
    if n<3 or bottom.shape!=top.shape:raise ValueError('Matching polygon rings required')
    vertices=np.vstack([top,bottom,top.mean(0),bottom.mean(0)]);faces=[];top_flags=[]
    for i in range(n):
        j=(i+1)%n
        faces.extend([[2*n,i,j],[2*n+1,n+j,n+i],[i,n+i,n+j],[i,n+j,j]])
        top_flags.extend([True,False,False,False])
    return checked_mesh(vertices,faces,top_flags)


def checked_mesh(vertices,faces,top_flags):
    mesh=trimesh.Trimesh(vertices=np.asarray(vertices,float),faces=np.asarray(faces,int),process=False);mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0 or np.any(mesh.area_faces<=1e-10):
        raise ValueError('Roof component must be closed, oriented and nondegenerate')
    return {'vertices':mesh.vertices.copy(),'triangles':mesh.faces.copy(),'top_triangle_mask':np.asarray(top_flags,bool),
        'watertight':True,'volume_CAD_units_cubed':float(mesh.volume),'surface_area_CAD_units_squared':float(mesh.area)}


def roof_floor_with_cap(outer,base_center,base_u,base_v,radius,top_center,top_u,top_v,*,thickness=.8,samples=96):
    """Shared base ring, exact convex outer outline and closed component solids."""
    outer=np.asarray(outer,float);bc=np.asarray(base_center,float);bu=np.asarray(base_u,float);bv=np.asarray(base_v,float)
    if outer.shape!=(4,3) or thickness<=0 or radius<=0:raise ValueError('Convex outer roof and positive dimensions required')
    uv=np.c_[bu[:2],bv[:2]];corner_coordinates=(outer[:,:2]-bc[:2])@np.linalg.inv(uv).T
    # Corner rays can coincide with regular samples up to floating rounding.
    # Quantize phases before deduplication to prevent zero-area native faces.
    phase=np.unique(np.round(np.r_[np.arange(samples)*2*np.pi/samples,np.arctan2(corner_coordinates[:,1],corner_coordinates[:,0])%(2*np.pi)],12))
    phase.sort();inner=bc+radius*(np.cos(phase)[:,None]*bu+np.sin(phase)[:,None]*bv);outside=[]
    for point in inner:
        direction=point[:2]-bc[:2];hits=[]
        for a,b in zip(outer,np.roll(outer,-1,axis=0)):
            matrix=np.c_[direction,-(b-a)[:2]]
            if abs(np.linalg.det(matrix))<1e-12:continue
            t,along=np.linalg.solve(matrix,(a-bc)[:2])
            if t>1 and -1e-8<=along<=1+1e-8:hits.append((t,a+np.clip(along,0,1)*(b-a)))
        if not hits:raise ValueError('Circle must lie strictly inside trusted outer roof')
        outside.append(min(hits,key=lambda pair:pair[0])[1])
    outside=np.array(outside);n=len(inner);top=np.vstack([outside,inner]);bottom=top-[0,0,thickness];vertices=np.vstack([top,bottom]);faces=[];top_flags=[]
    for i in range(n):
        j=(i+1)%n
        quads=[[i,j,n+j,n+i],[2*n+i,3*n+i,3*n+j,2*n+j],
            [i,2*n+i,2*n+j,j],[n+i,n+j,3*n+j,3*n+i]]
        for index,quad in enumerate(quads):
            faces.extend([[quad[0],quad[1],quad[2]],[quad[0],quad[2],quad[3]]]);top_flags.extend([index==0,index==0])
    floor=checked_mesh(vertices,faces,top_flags)
    tc=np.asarray(top_center);tu=np.asarray(top_u);tv=np.asarray(top_v);top_ring=tc+radius*(np.cos(phase)[:,None]*tu+np.sin(phase)[:,None]*tv)
    cap=closed_prism(top_ring,inner)
    return {'floor':floor,'cap':cap,'shared_ring':inner,'phase':phase}
