"""Continuous editable route envelopes from image cues and ZipMap source depth.

This is a hybrid candidate, not an end-to-end neural prediction or survey truth.
"""
import numpy as np
from shapely.geometry import LineString
from .neural_components import polygon_solid


def project(points,K,R,C):
    local=(np.asarray(points)-C)@R.T;hom=local@K.T
    return hom[:,:2]/local[:,2:],local[:,2]


def route_solid(nodes_xy,top_z,width,height):
    nodes=np.asarray(nodes_xy,float)
    if nodes.shape!=(3,2) or not np.isfinite(nodes).all():raise ValueError('Three finite route nodes required')
    if min(width,height)<=0:raise ValueError('Positive width and envelope height required')
    if np.any(np.linalg.norm(np.diff(nodes,axis=0),axis=1)<width):raise ValueError('Route segments too short for the width')
    footprint=LineString(nodes).buffer(width/2,cap_style=2,join_style=2,mitre_limit=3)
    if not footprint.is_valid or footprint.geom_type!='Polygon' or len(footprint.interiors):raise ValueError('Invalid route footprint')
    part=polygon_solid(footprint,[0,0,float(top_z)],np.array([[1,0],[0,1],[0,0]]),[0,0,1],float(height))
    return part,footprint


def fit_route(image_nodes,width_pixels,K,R,C,source_points,top_initial,fit_views=None):
    from scipy.optimize import least_squares
    image_nodes=np.asarray(image_nodes,float);source_points=np.asarray(source_points,float)
    if fit_views is None:fit_views=np.arange(len(K))
    fit_views=np.asarray(fit_views,int)
    if len(fit_views)<2:raise ValueError('At least two fitting views required')
    # Ray/plane initialization is only a starting estimate; no CAD vertices copied.
    i=int(fit_views[0]);ray=np.linalg.solve(K[i],np.c_[image_nodes[i],np.ones(3)].T).T
    ray=np.linalg.solve(R[i],ray.T).T;depth=(top_initial-C[i,2])/ray[:,2]
    nodes=(C[i]+depth[:,None]*ray)[:,:2]
    x0=np.r_[nodes.ravel(),top_initial,.5]
    centre=nodes.mean(0);lower=np.r_[np.tile(centre-15,3),top_initial-2,.15];upper=np.r_[np.tile(centre+15,3),top_initial+2,1.2]
    def unpack(x):return x[:6].reshape(3,2),x[6],x[7]
    def objective(x):
        xy,z,width=unpack(x);p=np.c_[xy,np.full(3,z)];res=[]
        delta=xy[2]-xy[1];side=np.array([-delta[1],delta[0]])/max(np.linalg.norm(delta),1e-8)
        cross=np.c_[((xy[1]+xy[2])/2)[None]+np.array([-.5,.5])[:,None]*width*side,np.full(2,z)]
        for j in fit_views:
            uv,d=project(p,K[j],R[j],C[j]);res.extend(((uv-image_nodes[j])/3.).ravel())
            cut,_=project(cross,K[j],R[j],C[j]);res.append((np.linalg.norm(cut[1]-cut[0])-width_pixels[j])/3.)
            res.extend(np.maximum(.1-d,0)*100)
        # All source points are image-context evidence, not exact labelled edge points.
        # Robust scalar plane-height prior avoids forcing their XY to a CAD template.
        if len(source_points):res.append((z-np.median(source_points[:,2]))/.3)
        else:res.append((z-top_initial)/.5)
        a=xy[1]-xy[0];b=xy[2]-xy[1]
        # Soft approximate right-angle architectural prior, not a learned fact.
        res.append((a@b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8)/.3)
        res.extend(np.maximum(width-np.linalg.norm(np.diff(xy,axis=0),axis=1),0)*10)
        return np.asarray(res)
    fit=least_squares(objective,x0,bounds=(lower,upper),loss='soft_l1',f_scale=2,max_nfev=500)
    xy,z,width=unpack(fit.x)
    return xy,z,width,dict(optimizer_success=bool(fit.success),nfev=int(fit.nfev),cost=float(fit.cost),
        fitting_view_indices=fit_views.tolist(),top_height_source_context_prior=True,
        right_angle_is_soft_engineering_prior=True,closed_envelope_height_not_estimated_from_images=True,
        reached_bound=bool(np.any(np.minimum(fit.x-lower,upper-fit.x)<1e-5)))
