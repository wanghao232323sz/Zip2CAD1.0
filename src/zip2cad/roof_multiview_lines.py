"""Triangulate parallel section edges without assuming a white top face.

Each image line defines a plane through its camera. Its intersection with the
beam's transverse plane gives a constraint on horizontal offset and height.
"""
import cv2,numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment
from scipy.optimize import differential_evolution
from .roof_image_sections import project


def extract_parallel_lines(image,a,b,camera,corridor_px=32.):
    local=(np.asarray([a,b])-camera['C'])@camera['R'].T
    if np.any(local[:,2]<=.1):return []
    uv=project([a,b],camera);start,end=uv;axis=end-start;length=np.linalg.norm(axis)
    midpoint=uv.mean(axis=0)
    if length<18 or np.any(midpoint<[-corridor_px,-corridor_px]) or np.any(midpoint>[image.shape[1]+corridor_px,image.shape[0]+corridor_px]):return []
    unit=axis/length
    detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines=detector.detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))[0]
    rows=[]
    for raw in [] if lines is None else lines.reshape(-1,4):
        p,q=raw[:2],raw[2:];v=q-p;size=np.linalg.norm(v)
        if size<max(9.,length*.16):continue
        if abs(v@unit)/size<np.cos(np.deg2rad(10)):continue
        t=np.sort(np.array([(p-start)@unit,(q-start)@unit]))
        overlap=max(0.,min(length*.9,t[1])-max(length*.1,t[0]))
        if overlap<length*.16:continue
        normal=np.array([-v[1],v[0]])/size;d=-normal@p
        distance=abs(normal@((start+end)/2)+d)
        if distance>corridor_px:continue
        rows.append(dict(endpoints=raw.tolist(),equation=[*normal.tolist(),float(d)],coverage=float(min(overlap/(length*.8),1.))))
    return rows


def section_queries(a,b,side,h,z):
    return (np.array(a)+np.array(b))/2+np.asarray(h)[...,None]*side+np.asarray(z)[...,None]*[0,0,1]


def score_queries(points,observations):
    perview={}
    for ref,view in observations.items():
        lines=view['lines']
        if not lines:continue
        uv=project(points,view['camera']);eq=np.array([r['equation'] for r in lines]);weights=np.array([r['coverage'] for r in lines])
        distance=np.abs(np.c_[uv,np.ones(len(uv))]@eq.T)
        values=np.exp(-.5*(distance/1.5)**2)*np.minimum(weights/.5,1.)
        selected=np.argmax(values,axis=1)
        perview[ref]=dict(score=values[np.arange(len(points)),selected],distance=distance[np.arange(len(points)),selected],line_index=selected)
    return perview


def triangulate_section_edges(a,b,side,observations,fit_refs,check_refs,h_bounds=(-.8,1.5),z_bounds=(-1.4,.5),cell=.025):
    side=np.array(side,float);h=np.arange(h_bounds[0],h_bounds[1]+cell/2,cell);z=np.arange(z_bounds[0],z_bounds[1]+cell/2,cell)
    hh,zz=np.meshgrid(h,z);points=section_queries(a,b,side,hh.ravel(),zz.ravel())
    scores=score_queries(points,{r:observations[r] for r in fit_refs})
    if len(scores)<3:raise ValueError('At least three views with line evidence required')
    # Equal temporal-group weight, no held-out-view selection.
    groups=[[r for r in scores if r<290],[r for r in scores if r>=290]];groups=[g for g in groups if g]
    value=np.mean([np.mean([scores[r]['score'] for r in g],axis=0) for g in groups],axis=0).reshape(hh.shape)
    peak=(value>=maximum_filter(value,size=7))&(value>.25)
    cells=np.argwhere(peak);cells=sorted(cells,key=lambda ij:value[tuple(ij)],reverse=True)[:50]
    candidates=[]
    for iz,ih in cells:
        x=np.clip(np.array([h[ih],z[iz]]),[h_bounds[0]+1e-9,z_bounds[0]+1e-9],[h_bounds[1]-1e-9,z_bounds[1]-1e-9])
        for _ in range(3):
            q=section_queries(a,b,side,[x[0]],[x[1]])
            assignment=score_queries(q,{r:observations[r] for r in fit_refs})
            chosen=[(r,v['line_index'][0]) for r,v in assignment.items() if v['distance'][0]<3.]
            if len(chosen)<3:break
            def residual(state):
                point=section_queries(a,b,side,[state[0]],[state[1]])
                return [float(np.r_[project(point,observations[r]['camera'])[0],1.]@np.array(observations[r]['lines'][int(i)]['equation'])) for r,i in chosen]
            fitted=least_squares(residual,x,bounds=([h_bounds[0],z_bounds[0]],[h_bounds[1],z_bounds[1]]),loss='soft_l1',f_scale=1.5)
            x=fitted.x
        if any(np.linalg.norm(x-np.array([v['horizontal_offset_m'],v['height_offset_m']]))<.10 for v in candidates):continue
        q=section_queries(a,b,side,[x[0]],[x[1]]);evidence=score_queries(q,observations)
        fitgood=[r for r in fit_refs if r in evidence and evidence[r]['distance'][0]<2.5]
        checkgood=[r for r in check_refs if r in evidence and evidence[r]['distance'][0]<2.5]
        candidates.append(dict(horizontal_offset_m=float(x[0]),height_offset_m=float(x[1]),grid_score=float(value[iz,ih]),fit_support_refs=fitgood,check_support_refs=checkgood,
            median_fit_line_distance_px=float(np.median([evidence[r]['distance'][0] for r in fitgood])) if fitgood else None,
            per_view={str(r):dict(distance_px=float(v['distance'][0]),detected_line_index=int(v['line_index'][0])) for r,v in evidence.items()},
            at_search_bound=bool(min(x[0]-h_bounds[0],h_bounds[1]-x[0],x[1]-z_bounds[0],z_bounds[1]-x[1])<.04)))
    candidates.sort(key=lambda v:(len(v['fit_support_refs']),v['grid_score']),reverse=True)
    return candidates,dict(horizontal_axis_m=h,height_axis_m=z,fit_score=value)


def rectangle_hypotheses(candidates,face_top,face_depth):
    """Rank three-edge rectangles; missing fourth edge stays explicitly unseen."""
    rectangles=[]
    for front in candidates:
        h,z=front['horizontal_offset_m'],front['height_offset_m']
        # Mask faces were sampled on the uncorrected seed plane. They supply a
        # coarse height range, not an exact plane offset or a measured top.
        if abs(h)>.8 or abs(z-face_top)>.5 or len(front['fit_support_refs'])<4:continue
        for lower in candidates:
            depth=z-lower['height_offset_m']
            if abs(lower['horizontal_offset_m']-h)>.15 or not .2<depth<1.3 or abs(depth-face_depth)>.3:continue
            if len(lower['fit_support_refs'])<4:continue
            for inner in candidates:
                width=inner['horizontal_offset_m']-h
                if not .12<width<1.3 or abs(inner['height_offset_m']-z)>.15:continue
                if len(inner['fit_support_refs'])<4:continue
                # No check-view observations participate in candidate ranking.
                fit=sum(v['grid_score'] for v in [front,lower,inner])
                fit-=abs(z-face_top)+abs(depth-face_depth)+2*abs(inner['height_offset_m']-z)+2*abs(lower['horizontal_offset_m']-h)
                rectangles.append(dict(width_m=float(width),depth_m=float(depth),front_plane_offset_m=h,top_z_offset_m=z,fit_objective=float(fit),
                    top_front=front,top_inner=inner,bottom_front=lower,fourth_edge_observed=False,
                    check_support_min=min(len(v['check_support_refs']) for v in [front,lower,inner]),
                    rectangular_section_is_prior=True,reference_geometry_used=False))
    rectangles.sort(key=lambda v:v['fit_objective'],reverse=True)
    return rectangles


def distinct_rectangle_support(rectangle,a,b,side,observations,fit_refs,check_refs):
    """One detected image line cannot count as two observed section edges."""
    edges=[rectangle[k] for k in ['top_front','top_inner','bottom_front']]
    points=section_queries(a,b,np.array(side),[r['horizontal_offset_m'] for r in edges],[r['height_offset_m'] for r in edges])
    details={}
    for ref,view in observations.items():
        if len(view['lines'])<3:continue
        uv=project(points,view['camera']);eq=np.array([v['equation'] for v in view['lines']])
        distance=np.abs(np.c_[uv,np.ones(3)]@eq.T)
        ri,ci=linear_sum_assignment(distance)
        separation=float(min(np.linalg.norm(uv[i]-uv[j]) for i in range(3) for j in range(i)))
        distances=distance[ri,ci]
        details[str(ref)]=dict(line_indices=ci.tolist(),distances_px=distances.tolist(),minimum_edge_separation_px=separation,
            three_distinct_resolved_edges=bool(np.max(distances)<2.5 and separation>=1.5))
    goodfit=[r for r in fit_refs if details.get(str(r),{}).get('three_distinct_resolved_edges')]
    goodcheck=[r for r in check_refs if details.get(str(r),{}).get('three_distinct_resolved_edges')]
    return dict(three_distinct_fit_refs=goodfit,three_distinct_check_refs=goodcheck,per_view=details,
                candidate_width_has_distinct_edge_support=bool(len(goodfit)>=2 and len(goodcheck)>=1))


def refine_beam_axis(a,b,observations,fit_refs,max_angle_degrees=6.):
    """Infer a shared direction from line vanishing constraints, keeping span."""
    a,b=np.array(a),np.array(b);centre=(a+b)/2;span=np.linalg.norm(b-a);d0=(b-a)/span
    horizontal=np.cross([0,0,1],d0);horizontal/=np.linalg.norm(horizontal)
    vertical=np.cross(d0,horizontal)
    def direction(state):
        d=d0+state[0]*horizontal+state[1]*vertical
        return d/np.linalg.norm(d)
    def perview(d,refs):
        result={}
        for ref in refs:
            v=observations[ref]
            if not v['lines']:continue
            uv=project([centre-d*span/2,centre+d*span/2],v['camera']);tangent=uv[1]-uv[0];tangent/=np.linalg.norm(tangent)
            residual=np.array([abs(np.array(row['equation'][:2])@tangent) for row in v['lines']])
            order=np.argsort(residual)[:min(2,len(residual))]
            result[ref]=float(np.mean(np.degrees(np.arcsin(np.clip(residual[order],0,1)))))
        return result
    def objective(state):
        values=perview(direction(state),fit_refs)
        groups=[[v for r,v in values.items() if r<290],[v for r,v in values.items() if r>=290]]
        scores=[]
        for group in groups:
            if group:
                group=sorted(group);scores.append(np.mean(group[:max(1,int(np.ceil(len(group)*.8)))]))
        return float(np.mean(scores))+.01*float(np.linalg.norm(state)/.05)
    limit=np.tan(np.deg2rad(max_angle_degrees))
    fit=differential_evolution(objective,[(-limit,limit)]*2,seed=1020,maxiter=35,popsize=8,tol=.001,polish=True)
    d=direction(fit.x);before=perview(d0,fit_refs);after=perview(d,fit_refs)
    accepted=objective(fit.x)<objective([0.,0.])-.05
    if not accepted:d=d0
    return centre-d*span/2,centre+d*span/2,dict(accepted=accepted,rotation_degrees=float(np.degrees(np.arccos(np.clip(d@d0,-1,1)))),
        fit_objective_before=objective([0.,0.]),fit_objective_after=objective(fit.x),fit_angular_residual_before_deg={str(k):v for k,v in before.items()},fit_angular_residual_after_deg={str(k):v for k,v in after.items()},
        neural_span_preserved=True,midpoint_preserved=True,reference_used=False)


def fit_joint_rectangle(rectangle,a,b,side,observations,fit_refs,check_refs):
    """Fit an exact rectangle with injective, fixed image-line correspondences.

    Leave-one-view results measure sensitivity conditional on these matches;
    they are not independent validation or a confidence interval.
    """
    initial=np.array([rectangle['front_plane_offset_m'],rectangle['top_z_offset_m'],rectangle['width_m'],rectangle['depth_m']])
    def points(x):return section_queries(a,b,np.array(side),[x[0],x[0]+x[2],x[0]],[x[1],x[1],x[1]-x[3]])
    assignments={}
    for ref,v in observations.items():
        if not v['lines']:continue
        uv=project(points(initial),v['camera']);eq=np.array([r['equation'] for r in v['lines']]);dist=np.abs(np.c_[uv,np.ones(3)]@eq.T)
        rr,cc=linear_sum_assignment(dist)
        assignments[ref]=[(int(i),int(j)) for i,j in zip(rr,cc) if dist[i,j]<3.]
    def residual(x,refs):
        result=[]
        for ref in refs:
            if ref not in assignments:continue
            view=observations[ref];uv=project(points(x),view['camera'])
            for i,j in assignments[ref]:result.append(float(np.r_[uv[i],1.]@view['lines'][j]['equation']))
        return np.array(result)
    def fit(refs):
        if len(residual(initial,refs))<6:raise ValueError('Too few distinct edge constraints')
        result=least_squares(lambda x:residual(x,refs),initial,bounds=([-1.,-.7,.1,.1],[1.,.7,1.4,1.4]),loss='soft_l1',f_scale=1.)
        return result.x
    x=fit(fit_refs);jackknife=[]
    for ref in fit_refs:
        try:estimate=fit([r for r in fit_refs if r!=ref])
        except ValueError:continue
        jackknife.append(dict(omitted_view=ref,front_plane_offset_m=float(estimate[0]),top_z_offset_m=float(estimate[1]),width_m=float(estimate[2]),depth_m=float(estimate[3])))
    report=dict(front_plane_offset_m=float(x[0]),top_z_offset_m=float(x[1]),width_m=float(x[2]),depth_m=float(x[3]),
        median_fit_matched_line_distance_px=float(np.median(abs(residual(x,fit_refs)))),
        median_check_matched_line_distance_px=float(np.median(abs(residual(x,check_refs)))) if len(residual(x,check_refs)) else None,
        leave_one_view_sensitivity=jackknife,assignments={str(r):v for r,v in assignments.items()},
        sensitivity_is_conditional_on_fixed_matches=True,independent_validation=False)
    if jackknife:
        report['conditional_width_range_m']=[min(v['width_m'] for v in jackknife),max(v['width_m'] for v in jackknife)]
        report['conditional_depth_range_m']=[min(v['depth_m'] for v in jackknife),max(v['depth_m'] for v in jackknife)]
    return report
