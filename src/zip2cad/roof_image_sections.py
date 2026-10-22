"""Image-supported section hypotheses anchored to predicted top edges.

No target geometry is accepted by this API. Rectangular sections and inward
extrusion are explicit architectural priors, not outputs of a trained model.
"""
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
import trimesh

LONG_EDGES = [(0, 1), (3, 2), (4, 5), (7, 6)]


def beam_vertices(a, b, inward, width, depth):
    a, b, inward = np.asarray(a, float), np.asarray(b, float), np.asarray(inward, float)
    if not np.isfinite(np.r_[a,b,inward,width,depth]).all() or width <= 0 or depth <= 0:
        raise ValueError('Finite geometry and positive section required')
    if np.linalg.norm(b[:2]-a[:2]) < .75 or np.linalg.norm(inward[:2]) < .99:
        raise ValueError('Degenerate beam axis or inward direction')
    shift = width * inward
    top = np.array([a,b,b+shift,a+shift])
    return np.vstack([top,top-[0,0,depth]])


def beam_mesh(vertices):
    # Convex hull retains the predicted endpoint heights (a sloping top edge).
    return trimesh.convex.convex_hull(np.asarray(vertices, float))


def project(points, camera):
    local = (np.asarray(points)-camera['C']) @ camera['R'].T
    h = local @ camera['K'].T
    if (local[:,2] <= 0).any(): raise ValueError('Geometry behind camera')
    return h[:,:2]/h[:,2:]


def direction_distance(image, a, b, exclude_fixed_top_px=0.):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),.6)
    edge=cv2.Canny(gray,30,90)>0
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0); gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    tangent=np.asarray(b)-a; tangent=tangent/np.linalg.norm(tangent)
    # Edge tangent must be parallel to the predicted beam in this view.
    magnitude=np.sqrt(gx*gx+gy*gy)+1e-6
    compatible=np.abs((gx*tangent[0]+gy*tangent[1])/magnitude)<np.sin(np.deg2rad(20))
    selected=edge&compatible
    if exclude_fixed_top_px>0:
        yy,xx=np.indices(gray.shape)
        delta_x,delta_y=xx-a[0],yy-a[1]
        perpendicular=np.abs(delta_x*tangent[1]-delta_y*tangent[0])
        along=delta_x*tangent[0]+delta_y*tangent[1]
        selected &= ~((perpendicular<exclude_fixed_top_px)&(along>-10)&(along<np.linalg.norm(np.asarray(b)-a)+10))
    return cv2.distanceTransform((~selected).astype(np.uint8),cv2.DIST_L2,5)


def silhouette_lines(vertices, camera):
    uv=project(vertices,camera)
    hull=cv2.convexHull(uv.astype(np.float32),returnPoints=False).ravel().tolist()
    boundary={frozenset((hull[i],hull[(i+1)%len(hull)])) for i in range(len(hull))}
    # Exclude the fixed top edge: it cannot constrain a section's width/depth.
    return np.array([[uv[i],uv[j]] for i,j in LONG_EDGES[1:] if frozenset((i,j)) in boundary]).reshape(-1,2,2)


def contour_score(vertices, camera, distance, shift=None, gray=None):
    lines=silhouette_lines(vertices,camera)
    if len(lines)==0:return 20.
    t=np.linspace(.15,.85,32)
    samples=(lines[:,0,None,:]*(1-t)[None,:,None]+lines[:,1,None,:]*t[None,:,None]).reshape(-1,2)
    if shift is not None:samples+=shift
    values=map_coordinates(distance,[samples[:,1],samples[:,0]],order=1,mode='constant',cval=20.)
    cost=float(np.mean(np.minimum(values,12.)))
    if gray is not None:
        # White concrete hypothesis: visible silhouette should bound a brighter
        # inner beam face. The cue is deliberately explicit and not a label.
        centre=project(vertices,camera).mean(axis=0)
        if shift is not None:centre+=shift
        normal=centre-samples;normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-6)
        inside=samples+normal*2.;outside=samples-normal*2.
        inner=map_coordinates(gray,[inside[:,1],inside[:,0]],order=1,mode='nearest')
        outer=map_coordinates(gray,[outside[:,1],outside[:,0]],order=1,mode='nearest')
        cost+=float(np.mean(np.clip((outer-inner+12)/32,0,2)))*2.
    return cost


def fit_section(a,b,inward,views,fit_refs,check_refs,widths=None,depths=None,return_surface=False):
    widths=np.round(np.arange(.08,.801,.04),3) if widths is None else np.asarray(widths)
    depths=np.round(np.arange(.08,1.001,.04),3) if depths is None else np.asarray(depths)
    candidates=[]
    for width in widths:
        for depth in depths:
            v=beam_vertices(a,b,inward,width,depth)
            score=np.mean([contour_score(v,views[r]['camera'],views[r]['distance'],gray=views[r].get('gray')) for r in fit_refs])
            candidates.append((float(score),float(width),float(depth)))
    candidates.sort()
    score,width,depth=candidates[0]
    # This is an objective plateau, not a statistical confidence interval.
    near=[c for c in candidates if c[0]<=score+.5]
    v=beam_vertices(a,b,inward,width,depth)
    baseline=beam_vertices(a,b,inward,.32,.32)
    checks={}
    for ref in check_refs:
        view=views[ref];cost=contour_score(v,view['camera'],view['distance'],gray=view.get('gray'))
        controls=[contour_score(v,view['camera'],view['distance'],np.array(shift),gray=view.get('gray')) for shift in [(8,0),(-8,0),(0,8),(0,-8)]]
        checks[str(ref)]=dict(candidate_px=cost,generic_032m_px=contour_score(baseline,view['camera'],view['distance'],gray=view.get('gray')),
                             shifted_control_median_px=float(np.median(controls)),shift_margin_px=float(np.median(controls)-cost))
    range_w=[min(c[1] for c in near),max(c[1] for c in near)]
    range_d=[min(c[2] for c in near),max(c[2] for c in near)]
    boundary=width in (float(widths.min()),float(widths.max())) or depth in (float(depths.min()),float(depths.max()))
    distinct=sum(c['shift_margin_px']>.5 for c in checks.values())
    stable=range_w[1]-range_w[0]<=.24 and range_d[1]-range_d[0]<=.24
    report=dict(width_m=width,depth_m=depth,fit_oriented_contour_cost_px=score,
                objective_plateau_width_m=range_w,objective_plateau_depth_m=range_d,
                parameter_bound_hit=bool(boundary),check_views=checks,
                distinctive_check_views=distinct,dimension_identified=bool(stable and not boundary and distinct>=2),
                rectangle_and_inward_extrusion_are_priors=True,section_is_neural_prediction=False,
                reference_parameters_used=False)
    if return_surface:report['search_surface']=candidates
    return report,v


def shared_depth_hypothesis(surface_a,surface_b):
    """A shared beam depth within one frame, never across left and right frames."""
    depths=sorted(set(c[2] for c in surface_a)&set(c[2] for c in surface_b))
    if not depths:raise ValueError('No common section search depths')
    rows=[]
    for depth in depths:
        a=min((c for c in surface_a if c[2]==depth))
        b=min((c for c in surface_b if c[2]==depth))
        rows.append((.5*(a[0]+b[0]),depth,a[1],b[1]))
    best=min(rows);near=[r for r in rows if r[0]<=best[0]+.5]
    return dict(combined_objective=best[0],depth_m=best[1],width_a_m=best[2],width_b_m=best[3],
                objective_plateau_depth_m=[min(r[1] for r in near),max(r[1] for r in near)],
                shared_stock_depth_is_architectural_prior=True)


def outer_face_depth(a,b,inward,views,fit_refs,check_refs):
    """Identify a white beam's outer-face lower edge only in visible views.

    Camera position determines face visibility. This avoids forcing hidden
    faces to match roof textures. Costs remain heuristic image evidence.
    """
    a,b,inward=np.array(a),np.array(b),np.array(inward)
    eligible=[r for r in fit_refs if (views[r]['camera']['C']-(a+b)/2)@inward<-.5]
    if len(eligible)<2:raise ValueError('Too few outer-face-visible images')
    def cost(depth,ref):
        view=views[ref];camera=view['camera'];gray=view['gray']
        t=np.linspace(.2,.8,48)[:,None]
        top=a*(1-t)+b*t;bottom=top-[0,0,depth]
        uv=project(bottom,camera);topuv=project(top,camera)
        normal=topuv-uv;normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-6)
        inner=uv+normal*2.;outer=uv-normal*2.
        gi=map_coordinates(gray,[inner[:,1],inner[:,0]],order=1,mode='nearest')
        go=map_coordinates(gray,[outer[:,1],outer[:,0]],order=1,mode='nearest')
        distance=map_coordinates(view['distance'],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=15.)
        value=.15*np.minimum(distance,12.)-np.clip(gi-go,-80,80)/40+np.maximum(110-gi,0)/80
        return float(np.mean(value))
    rows=[]
    for depth in np.round(np.arange(.12,1.001,.02),3):
        values=sorted(cost(depth,r) for r in eligible)
        objective=float(np.mean(values[1:-1] if len(values)>=5 else values))
        rows.append((objective,float(depth)))
    objective,depth=min(rows);near=[d for c,d in rows if c<=objective+.15]
    return dict(depth_m=depth,fit_visible_views=eligible,objective=objective,
                heuristic_objective_plateau_depth_m=[min(near),max(near)],
                search_curve=rows,check_visible_views={str(r):cost(depth,r) for r in check_refs if (views[r]['camera']['C']-(a+b)/2)@inward<-.5},
                cue='visible outer face; bright inside versus dark outside below; camera-facing test'),depth


def connected_frame(meshes):
    # Boolean union removes overlapping internal faces; no gap-filling bridge.
    value=trimesh.boolean.union(meshes,engine='manifold')
    if value is None or not value.is_watertight or not value.is_winding_consistent or value.volume<=0:
        raise ValueError('Invalid section union')
    if len(value.split(only_watertight=False)) != 1:
        raise ValueError('Section hypotheses do not meet; refuse invented bridge')
    return value
