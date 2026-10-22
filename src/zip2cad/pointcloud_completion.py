"""Geometry-only completion from calibrated ZipMap point maps.

No building identifiers, image ROI, texture, or architectural dimensions are
embedded here. Camera/metric calibration and target CAD scope are inputs.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.spatial import ConvexHull,cKDTree
from scipy import ndimage
from .similarity import estimate_similarity,transform_points
from .roof_completion_geometry import closed_prism,checked_mesh

def polygon_prism(top,bottom):
    """Ear-clipped simple polygon; convex fans would fill an L-shaped void."""
    top,bottom=np.asarray(top,float),np.asarray(bottom,float);xy=top[:,:2];n=len(xy)
    area=np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
    if area<0:top=top[::-1];bottom=bottom[::-1];xy=xy[::-1]
    remaining=list(range(n));triangles=[]
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    while len(remaining)>3:
        found=False
        for j in range(len(remaining)):
            a,b,c=remaining[j-1],remaining[j],remaining[(j+1)%len(remaining)]
            if cross(xy[b]-xy[a],xy[c]-xy[b])<=1e-10:continue
            other=[i for i in remaining if i not in (a,b,c)]
            inside=[all(cross(xy[y]-xy[x],xy[i]-xy[x])>=-1e-10 for x,y in [(a,b),(b,c),(c,a)]) for i in other]
            if any(inside):continue
            triangles.append([a,b,c]);remaining.pop(j);found=True;break
        if not found:raise ValueError('Self-intersecting or degenerate polygon')
    triangles.append(remaining);faces=triangles+[[c+n,b+n,a+n] for a,b,c in triangles];flags=[True]*len(triangles)+[False]*len(triangles)
    for i in range(n):
        j=(i+1)%n;faces.extend([[i,i+n,j+n],[i,j+n,j]]);flags.extend([False,False])
    return checked_mesh(np.vstack([top,bottom]),faces,flags)

def infer_superseded_roof_proxies(vertices,faces,owners,entity_ids,scope_prefix,components,base_z,policy):
    """Infer reversible old roof containers from area overlap and elevation.

    Persistent paths are extracted from CAD inputs, never stored in code. Tall
    walls and the supporting slab cannot meet the vertical bounds predicate.
    """
    groups={}
    for owner in np.unique(owners):
        name=str(entity_ids[owner])
        if not name.startswith(scope_prefix):continue
        path=tuple(int(x) for x in name.split(':')[-1].split('.')[:-1])
        groups.setdefault(path,[]).append(int(owner))
    rows=[]
    for path,ids in groups.items():
        triangles=vertices[faces[np.isin(owners,ids)]]
        if len(triangles)<8:continue
        low=triangles.min(axis=(0,1));high=triangles.max(axis=(0,1))
        if low[2]<base_z-.05 or high[2]-low[2]<.2:continue
        cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);area=np.linalg.norm(cross,axis=1)/2
        normals=cross/np.maximum(area[:,None]*2,1e-12);centers=triangles.mean(1)
        top=(abs(normals[:,2])>.9)&(centers[:,2]>high[2]-.05)&(area>1e-10)
        if top.sum()<3:continue
        for item in components:
            poly=np.array(item['boundary_xy'],np.float32);origin=np.array(item['plane_origin']);normal=np.array(item['plane_normal'])
            new_z=origin[2]-(centers[top,:2]-origin[:2])@normal[:2]/normal[2]
            inside=np.array([cv2.pointPolygonTest(poly,tuple(map(float,p)),False)>=0 for p in centers[top,:2]])
            coverage=float(np.sum(area[top][inside])/np.sum(area[top]));new_area=cv2.contourArea(poly)
            if not inside.any() or np.median(new_z[inside])-high[2]<policy.plane_tolerance_m:continue
            # A smaller replacement may cover less than 80% of the old proxy.
            # Verify the smaller footprint instead of forbidding a real shrink.
            old_outline=ConvexHull(triangles[top,:,:2].reshape(-1,2)).vertices
            old_points=triangles[top,:,:2].reshape(-1,2);old_outline=old_points[old_outline].astype(np.float32)
            if abs(cv2.contourArea(cv2.convexHull(poly))-new_area)>new_area*.1:continue
            overlap,_=cv2.intersectConvexConvex(old_outline,poly)
            smaller_coverage=float(overlap/max(min(cv2.contourArea(old_outline),new_area),1e-9))
            if smaller_coverage<.8 or np.sum(area[top])>new_area*1.4 or new_area>np.sum(area[top])*1.4:continue
            rows.append({'source_container_pid_path':list(path),'replacement_component_id':item['component_id'],
                'old_top_area_overlap_fraction':coverage,'smaller_footprint_overlap_fraction':smaller_coverage,'old_container_bounds_CAD_m':[low.tolist(),high.tolist()],
                'action':'hide_original_proxy_reversibly_in_generated_view','original_geometry_deleted':False,
                'physical_identity_survey_certified':False});break
    return rows

def convex_footprint_loft(top_xy,bottom_xy,plane_origin,plane_normal,base_z):
    """Keep observed top, attach to the exact existing convex CAD opening.

    Both convex outlines contribute corner rays. The shared phase mesh avoids
    the crescent gap caused by moving a lid onto a different existing opening.
    CAD lower faces are an attachment prior, not video measurements.
    """
    top_xy,bottom_xy=map(lambda x:np.asarray(x,float),(top_xy,bottom_xy))
    tc,bc=top_xy.mean(0),bottom_xy.mean(0)
    bottom_phases=np.arctan2(*(bottom_xy-bc)[:,::-1].T)%(2*np.pi)
    top_phases=np.arctan2(*(top_xy-tc)[:,::-1].T)%(2*np.pi)
    # Keep exact CAD corner rays. A nearby fitted top ray is the same corner,
    # not a sub-nanometre edge SketchUp could create as an independent face.
    phases=list(bottom_phases)
    for phase in top_phases:
        difference=abs((np.asarray(phases)-phase+np.pi)%(2*np.pi)-np.pi)
        if difference.min()>1e-7:phases.append(float(phase))
    phases=np.sort(phases)
    def intersect(poly,center,phase):
        direction=np.array([np.cos(phase),np.sin(phase)]);hits=[]
        for a,b in zip(poly,np.roll(poly,-1,axis=0)):
            matrix=np.c_[direction,-(b-a)]
            if abs(np.linalg.det(matrix))<1e-10:continue
            distance,along=np.linalg.solve(matrix,a-center)
            if distance>0 and -1e-8<=along<=1+1e-8:hits.append((distance,center+distance*direction))
        if not hits:raise ValueError('Convex attachment outline required')
        return min(hits,key=lambda pair:pair[0])[1]
    top=np.array([intersect(top_xy,tc,p) for p in phases]);bottom=np.array([intersect(bottom_xy,bc,p) for p in phases])
    origin=np.asarray(plane_origin);normal=np.asarray(plane_normal);z=origin[2]-(top-origin[:2])@normal[:2]/normal[2]
    return closed_prism(np.c_[top,z],np.c_[bottom,np.full(len(bottom),base_z)])

@dataclass(frozen=True)
class CompletionPolicy:
    voxel_m:float=.5
    minimum_area_m2:float=15.
    plane_tolerance_m:float=.30
    maximum_components:int=6
    minimum_training_views:int=2
    minimum_heldout_fraction:float=.7
    minimum_change_m:float=.5
    maximum_boundary_vertices:int=12
    curved_boundary_vertices:int=48
    maximum_roof_slope_degrees:float=20.

    def __post_init__(self):
        if not all(np.isfinite(x) and x>0 for x in (self.voxel_m,self.minimum_area_m2,self.plane_tolerance_m,self.minimum_change_m)):
            raise ValueError('Positive finite metric thresholds required')
        if not 0<self.minimum_heldout_fraction<=1 or self.minimum_training_views<2 or self.maximum_components<1 or self.maximum_boundary_vertices<3 or not 8<=self.curved_boundary_vertices<=256:
            raise ValueError('Valid support and complexity limits required')
        if not 0<self.maximum_roof_slope_degrees<85:raise ValueError('Roof slope limit must be between 0 and 85 degrees')

def robust_similarity(source,target,training_mask,trim=.7):
    """Only paired stable CAD surfaces set one scalar metric registration."""
    source,target=np.asarray(source,float),np.asarray(target,float);training=np.asarray(training_mask,bool)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or training.shape!=(len(source),) or training.sum()<16:
        raise ValueError('At least 16 paired finite training anchors are required')
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Nonfinite anchors')
    keep=training.copy()
    for _ in range(8):
        scale,rotation,translation=estimate_similarity(source[keep],target[keep])
        errors=np.linalg.norm(transform_points(source,scale,rotation,translation)-target,axis=1)
        keep=training&(errors<=np.quantile(errors[training],trim))
    return {'scale':scale,'rotation':rotation.tolist(),'translation':translation.tolist(),
        'training_median_m':float(np.median(errors[training])),
        'heldout_median_m':float(np.median(errors[~training])) if (~training).any() else None,
        'heldout_used_for_fit':False}

def refine_depth(images,initial_points,intrinsics,rotations,centers,source_index,training_views,
                 *,depth_radius_m=12.,depth_step_m=.2,minimum_zncc=.7,window=7,minimum_texture_std=2/255):
    """Bounded plane sweep initialized by real ZipMap points, training only.

    Uses supplied cameras in the CAD gauge; it does not adjust cameras or use
    heldout frames to select depths. Returns pixel-source mapping and ambiguity.
    """
    images=np.asarray(images,float);K,R,C=map(np.asarray,(intrinsics,rotations,centers));source=source_index
    if source not in training_views or len(training_views)<3:raise ValueError('Source and two other training views required')
    if depth_radius_m<=0 or depth_step_m<=0:raise ValueError('Positive depth search policy required')
    gray=np.array([cv2.cvtColor(im.astype(np.float32),cv2.COLOR_RGB2GRAY) for im in images])
    h,w=gray[0].shape;yy,xx=np.mgrid[:h,:w];pixels=np.stack([xx,yy,np.ones_like(xx)],axis=-1)
    rays=pixels@np.linalg.inv(K[source]).T@R[source]
    initial_z=((initial_points-C[source])@R[source].T)[...,2]
    def avg(a):return cv2.boxFilter(a.astype(np.float32),-1,(window,window),normalize=True)
    src=gray[source];m=avg(src);variance=avg(src*src)-m*m
    best=np.full((h,w),-2.,np.float32);best_depth=np.zeros((h,w),np.float32)
    # Keep score curve for a separated second peak; adjacent samples of one
    # smooth peak do not represent an alternative correspondence.
    depth_low=max(1.,float(np.percentile(initial_z[np.isfinite(initial_z)],1))-depth_radius_m)
    depth_high=float(np.percentile(initial_z[np.isfinite(initial_z)],99))+depth_radius_m
    levels=np.arange(depth_low,depth_high+depth_step_m/2,depth_step_m)
    scores=[]
    for depth in levels:
        points=C[source]+rays*depth;rows=[]
        for view in training_views:
            if view==source:continue
            projection=(points-C[view])@R[view].T@K[view].T;xy=projection[...,:2]/np.maximum(projection[...,2:],1e-6)
            target=cv2.remap(gray[view],xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            tm=avg(target);tv=avg(target*target)-tm*tm
            corr=(avg(src*target)-m*tm)/np.sqrt(np.maximum(variance*tv,1e-8))
            valid=(xy[...,0]>window)&(xy[...,0]<w-window)&(xy[...,1]>window)&(xy[...,1]<h-window)&(projection[...,2]>0)
            rows.append(np.where(valid,np.clip(corr,-1,1),-1))
        score=np.minimum.reduce(rows).astype(np.float32)
        score[abs(initial_z-depth)>depth_radius_m]=-2
        score[~np.isfinite(initial_z)]=-2
        better=score>best;best[better]=score[better];best_depth[better]=depth;scores.append(score)
    alternative=np.full((h,w),-2.,np.float32)
    for depth,score in zip(levels,scores):alternative=np.maximum(alternative,np.where(abs(best_depth-depth)>=1.,score,-2))
    valid=(best>=minimum_zncc)&(variance>minimum_texture_std**2)&((best-alternative)>=.025)&(best_depth>0)
    border=np.zeros((h,w),bool);border[window:-window,window:-window]=True;valid&=border
    refined=C[source]+rays*best_depth[...,None]
    return {'points':refined,'valid':valid,'zncc':best,'alternative_margin':best-alternative,
        'initial_depth_CAD_m':initial_z,'refined_depth_CAD_m':best_depth,
        'heldout_used_for_depth_selection':False,'depth_radius_m':depth_radius_m,'depth_step_m':depth_step_m}

def trusted_perimeter_pixel_masks(mesh,target_polygon,base_z,K,R,C,image_shape,band_m=1.,face_opacity=None):
    """Project existing near-horizontal exterior coping, without source IDs.

    The user accepts the existing exterior. Already present roof-edge surfaces
    are therefore protected input geometry, not missing interior roof parts.
    """
    polygon=np.asarray(target_polygon,np.float32);centers=mesh.triangles_center
    distance=np.array([cv2.pointPolygonTest(polygon,tuple(map(float,p)),True) for p in centers[:,:2]])
    use=(abs(mesh.face_normals[:,2])>.9)&(centers[:,2]>=base_z+.25)&(centers[:,2]<=base_z+1.2)&(abs(distance)<=band_m)
    if face_opacity is not None:
        # Existing opaque coping is protected even in a concave/curved CAD
        # envelope. Transparent simplified roof proxies remain replaceable.
        use=(abs(mesh.face_normals[:,2])>.9)&(centers[:,2]>=base_z+.25)&(centers[:,2]<=base_z+1.2)&(np.asarray(face_opacity)>=.95)
    triangles=mesh.triangles[use];h,w=image_shape;masks=[]
    for k,r,c in zip(K,R,C):
        p=(triangles-c)@r.T@k.T;good=(p[...,2]>0).all(axis=1);uv=p[good,:,:2]/p[good,:,2:]
        mask=np.zeros((h,w),np.uint8)
        for tri in uv:
            if np.isfinite(tri).all():cv2.fillConvexPoly(mask,np.rint(tri).astype(np.int32),1)
        masks.append(cv2.dilate(mask,np.ones((3,3),np.uint8)).astype(bool))
    return np.asarray(masks),int(use.sum())

def trusted_exterior_candidate_overlap(boundary,target_polygon,training_registration_error,voxel_m,trusted_roof_triangles=None):
    """Abstain on exterior-band candidates under the trusted-outline premise.

    Training registration uncertainty expands the protected CAD perimeter. A
    heldout result does not tune this guard; interior roof changes remain free.
    """
    boundary=np.asarray(boundary,np.float32);target=np.asarray(target_polygon,np.float32)
    low=boundary.min(0);high=boundary.max(0);step=float(voxel_m)
    yy,xx=np.mgrid[low[1]+step/2:high[1]:step,low[0]+step/2:high[0]:step]
    samples=np.c_[xx.ravel(),yy.ravel()]
    samples=samples[[cv2.pointPolygonTest(boundary,tuple(map(float,p)),False)>=0 for p in samples]]
    band=max(2*step,3*float(training_registration_error))
    distances=np.array([cv2.pointPolygonTest(target,tuple(map(float,p)),True) for p in samples])
    if trusted_roof_triangles is not None and len(trusted_roof_triangles):
        anchor=[]
        for tri in np.asarray(trusted_roof_triangles):
            for a,b in zip(tri,np.roll(tri,-1,axis=0)):
                count=max(2,int(np.ceil(np.linalg.norm(b[:2]-a[:2])/step))+1);anchor.extend(np.linspace(a[:2],b[:2],count))
        distances=np.minimum(distances,cKDTree(np.asarray(anchor)).query(samples)[0])
    fraction=float(np.mean(distances<=band)) if len(samples) else 1.
    return {'protected_band_CAD_m':band,'training_registration_error_CAD_m':float(training_registration_error),
        'sample_count':len(samples),'candidate_area_fraction_in_protected_band':fraction,
        'trusted_existing_exterior_conflict':fraction>=.5,'heldout_used_to_choose_band':False}

def voxel_consensus(points,refs,training_views,voxel_m):
    """Per-frame medians prevent a dense view from dominating geometry."""
    points=np.asarray(points,float);refs=np.asarray(refs,int)
    cell=np.floor(points[:,:2]/voxel_m).astype(int);keys=np.c_[cell,refs]
    unique,inv=np.unique(keys,axis=0,return_inverse=True);med=[]
    for i in range(len(unique)):med.append(np.median(points[inv==i],axis=0))
    return np.asarray(med),unique[:,2],unique[:,:2]

def extract_planar_components(points,refs,training_views,heldout_views,base_z,policy=CompletionPolicy()):
    """Train-only connected plane patches -> simple closed prisms.

    Lower faces are explicitly an attachment to the known CAD support plane,
    never an observed hidden surface. No unsupported points fill missing data.
    """
    p=np.asarray(points,float);refs=np.asarray(refs,int)
    if p.ndim!=2 or p.shape[1]!=3 or refs.shape!=(len(p),) or not np.isfinite(p).all():raise ValueError('Finite attributed points required')
    if set(training_views)&set(heldout_views):raise ValueError('Physical frames must be split')
    if not training_views or not heldout_views:raise ValueError('Training and validation frames required')
    v,vr,cells=voxel_consensus(p,refs,training_views,policy.voxel_m)
    train=np.isin(vr,training_views);remaining=np.flatnonzero(train&(v[:,2]>=base_z+policy.minimum_change_m));held=np.flatnonzero(np.isin(vr,heldout_views))
    rng=np.random.default_rng(81);components=[];rejected=[]
    for attempt in range(policy.maximum_components*3):
        if len(remaining)<40 or len(components)>=policy.maximum_components:break
        best=None
        for _ in range(240):
            tri=v[rng.choice(remaining,3,replace=False)];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(normal)
            if length<1e-8:continue
            normal/=length
            if abs(normal[2])<np.cos(np.deg2rad(policy.maximum_roof_slope_degrees)):continue
            ids=remaining[abs((v[remaining]-tri[0])@normal)<=policy.plane_tolerance_m]
            occupancy=len(np.unique(cells[ids],axis=0))
            if best is None or occupancy>best[0]:best=(occupancy,ids)
        if best is None:break
        ids=best[1];remaining=remaining[~np.isin(remaining,ids)]
        origin=v[ids].mean(0);_,_,vh=np.linalg.svd(v[ids]-origin,full_matrices=False);normal=vh[-1];normal*=np.sign(normal[2])
        xy_cells=cells[ids];low=xy_cells.min(0);shape=xy_cells.max(0)-low+1
        grid=np.zeros(tuple(shape+2),np.uint8);grid[tuple((xy_cells-low+1).T)]=1
        labels,count=ndimage.label(grid,np.ones((3,3)))
        for label in range(1,count+1):
            local=ids[labels[tuple((xy_cells-low+1).T)]==label]
            occupied=len(np.unique(cells[local],axis=0));area=occupied*policy.voxel_m**2
            if area<policy.minimum_area_m2 or len(set(vr[local]))<policy.minimum_training_views:continue
            xy=v[local,:2];hull=xy[ConvexHull(xy).vertices]
            # Fit each disconnected patch separately, even when two buildings
            # share a plane. Keep concave boundaries instead of a convex bridge.
            localorigin=v[local].mean(0);_,_,localvh=np.linalg.svd(v[local]-localorigin,full_matrices=False)
            localnormal=localvh[-1];localnormal*=np.sign(localnormal[2])
            component=(labels==label).astype(np.uint8)
            component=cv2.morphologyEx(component,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour=max(contours,key=cv2.contourArea).reshape(-1,2)
            # OpenCV rows/columns correspond to CAD y/x after this transpose.
            boundary=(contour[:,::-1]+low-1+.5)*policy.voxel_m
            convex_area=float(cv2.contourArea(hull.astype(np.float32)))
            if convex_area>0 and area/convex_area>=.55:boundary=hull
            original_boundary_area=float(cv2.contourArea(boundary.astype(np.float32)))
            poly=None
            # Complexity is selected using training support only. Preserve area
            # while removing sub-metre noise; never increase the vertex budget.
            for epsilon in np.linspace(policy.voxel_m,policy.voxel_m*4,7):
                candidate=cv2.approxPolyDP(boundary.astype(np.float32).reshape(-1,1,2),float(epsilon),True).reshape(-1,2).astype(float)
                candidate_area=float(cv2.contourArea(candidate.astype(np.float32)))
                if len(candidate)<=policy.maximum_boundary_vertices and len(candidate)>=3 and abs(candidate_area-original_boundary_area)<=max(1.,original_boundary_area*.2):
                    poly=candidate;break
            if poly is None:poly=boundary
            if len(poly)>policy.maximum_boundary_vertices:
                rejected.append({'reason':'complex_boundary','occupied_area_m2':area});continue
            if len(poly)<3:continue
            # Do not bridge disconnected patches or fill a large unseen hole.
            hull_area=float(cv2.contourArea(poly.astype(np.float32)))
            if hull_area<=0 or area/hull_area<.55:
                rejected.append({'reason':'sparse_or_concave_support','occupied_area_m2':area});continue
            heights=localorigin[2]-(poly-localorigin[:2])@localnormal[:2]/localnormal[2]
            if heights.min()<=base_z+.15:continue
            heldrows=[]
            for ref in heldout_views:
                hid=held[vr[held]==ref];near=np.array([cv2.pointPolygonTest(poly.astype(np.float32),tuple(map(float,x)),False)>=0 for x in v[hid,:2]])
                hid=hid[near];errors=abs((v[hid]-localorigin)@localnormal)
                fraction=float(np.mean(errors<=policy.plane_tolerance_m*1.5)) if len(hid) else 0.
                heldrows.append({'frame_ref':int(ref),'point_count':len(hid),'supported_fraction':fraction,'p90_plane_error_m':float(np.percentile(errors,90)) if len(hid) else None})
            accepted=bool(heldrows and all(row['point_count']>=8 and row['supported_fraction']>=policy.minimum_heldout_fraction for row in heldrows))
            item={'training_views':sorted(set(vr[local].tolist())),'training_voxel_point_count':len(local),'occupied_area_m2':area,
                'boundary_xy':poly.tolist(),'plane_origin':localorigin.tolist(),'plane_normal':localnormal.tolist(),'heldout_rows':heldrows,'accepted':accepted,
                'hidden_bottom_source':'known_CAD_support_plane','geometry_source':'refined_ZipMap_pointcloud','heldout_used_for_geometry':False}
            if accepted:
                top=np.c_[poly,heights];bottom=np.c_[poly,np.full(len(poly),base_z)]
                try:mesh=polygon_prism(top,bottom)
                except ValueError as error:
                    rejected.append({'reason':'invalid_finite_component_mesh','occupied_area_m2':area,'boundary_xy':poly.tolist(),'error':str(error)});continue
                item.update(component_id=f'cloud_patch_{len(components)+1:02d}',mesh=mesh);components.append(item)
            else:rejected.append(item)
    return {'components':components,'rejected':rejected,'voxel_points':v,'voxel_refs':vr}
