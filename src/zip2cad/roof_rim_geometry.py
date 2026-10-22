"""Finite RGB rim extraction and multi-view circle geometry, without CAD labels."""
import cv2
import numpy as np
from scipy.optimize import least_squares


def ellipse_distance(points, ellipse):
    center, axes, angle=ellipse
    theta=np.deg2rad(angle)
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    local=(np.asarray(points,float)-center)@rotation
    axes=np.asarray(axes,float)/2
    value=np.sum((local/axes)**2,axis=1)-1
    gradient=2*local/axes**2
    return value/np.maximum(np.linalg.norm(gradient,axis=1),1e-12)


def extract_rim(image, center, semi_axes, angle_degrees, *, annulus_width=12, appearance_mask=False):
    if (image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or
        not np.isfinite([*center,*semi_axes,angle_degrees,annulus_width]).all() or
        min(semi_axes)<=0 or annulus_width<=0):
        raise ValueError('finite RGB ellipse scope required')
    edge=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),40,100)
    if appearance_mask:
        # Separate the dark top disk from its bright lower fascia before fitting
        # a conic. The coarse ellipse supplies foreground/background samples,
        # never an exact measured edge or a camera-conditioned selection.
        cx,cy=center;radius=max(semi_axes)*1.5
        x0,y0=max(0,int(cx-radius)),max(0,int(cy-radius))
        x1,y1=min(image.shape[1],int(cx+radius)+1),min(image.shape[0],int(cy+radius)+1)
        crop=image[y0:y1,x0:x1];yy,xx=np.indices(crop.shape[:2]);theta=np.deg2rad(angle_degrees)
        lx=(xx+x0-cx)*np.cos(theta)+(yy+y0-cy)*np.sin(theta)
        ly=-(xx+x0-cx)*np.sin(theta)+(yy+y0-cy)*np.cos(theta)
        rho=np.sqrt((lx/semi_axes[0])**2+(ly/semi_axes[1])**2)
        mask=np.full(crop.shape[:2],cv2.GC_PR_BGD,np.uint8)
        mask[rho<1.05]=cv2.GC_PR_FGD;mask[rho<.65]=cv2.GC_FGD;mask[rho>1.35]=cv2.GC_BGD
        cv2.setRNGSeed(812)
        cv2.grabCut(crop,mask,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
        foreground=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)*255
        contours,_=cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not contours:return {'supported':False,'reason':'no_disk_appearance_component'}
        contour=max(contours,key=cv2.contourArea)[:,0].astype(float)+[x0,y0]
        candidates=contour
    else:
        candidates=None
    ys,xs=np.nonzero(edge);points=np.c_[xs,ys].astype(float)
    scope=(tuple(center),tuple(2*np.asarray(semi_axes)),float(angle_degrees))
    if candidates is None:candidates=points[abs(ellipse_distance(points,scope))<=annulus_width]
    if len(candidates)<40:return {'supported':False,'reason':'insufficient_rim_edges'}
    rng=np.random.default_rng(812)
    best=None
    # RANSAC uses only the frozen RGB annulus. CAD and camera reprojections
    # cannot choose an ellipse or relabel the front lower fascia edge.
    for _ in range(1800):
        subset=candidates[rng.choice(len(candidates),5,replace=False)]
        try:ellipse=cv2.fitEllipse(subset.astype(np.float32))
        except cv2.error:continue
        if (np.linalg.norm(np.asarray(ellipse[0])-center)>annulus_width*2 or
            min(ellipse[1])<min(semi_axes)*1.4 or max(ellipse[1])>max(semi_axes)*2.4):continue
        inliers=abs(ellipse_distance(candidates,ellipse))<=1.5
        score=int(inliers.sum())
        if best is None or score>best[0]:best=(score,ellipse,inliers)
    if best is None:return {'supported':False,'reason':'no_finite_ellipse'}
    ellipse=best[1]
    for _ in range(3):
        inliers=abs(ellipse_distance(candidates,ellipse))<=1.5
        ellipse=cv2.fitEllipse(candidates[inliers].astype(np.float32))
    if appearance_mask:
        # Mask quantization and shadow can truncate an arc. Recover native
        # image edges in a narrow band around the appearance-selected top rim.
        # The lower fascia is outside this band in the frozen image scopes.
        candidates=points[abs(ellipse_distance(points,ellipse))<=4.]
        for _ in range(3):
            inliers=abs(ellipse_distance(candidates,ellipse))<=2.
            if inliers.sum()<40:break
            ellipse=cv2.fitEllipse(candidates[inliers].astype(np.float32))
    inliers=abs(ellipse_distance(candidates,ellipse))<=1.5
    # One median point per small angular sector balances textured arcs.
    xy=candidates[inliers];a=np.deg2rad(ellipse[2]);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    local=(xy-ellipse[0])@rot/(np.asarray(ellipse[1])/2)
    phases=np.arctan2(local[:,1],local[:,0]);bins=np.floor((phases+np.pi)/(2*np.pi)*72).astype(int)%72
    balanced=np.asarray([np.median(xy[bins==b],axis=0) for b in sorted(set(bins))])
    return {'supported':len(balanced)>=54,'ellipse':ellipse,'contour_xy':balanced.tolist(),
            'occupied_angular_bins':len(balanced),'total_angular_bins':72,'edge_inlier_count':len(xy),
            'image_fit_p90_pixels':float(np.percentile(abs(ellipse_distance(xy,ellipse)),90))}


def circle_basis(normal):
    normal=np.asarray(normal,float);normal=normal/np.linalg.norm(normal)
    axis=np.eye(3)[np.argmin(abs(normal))]
    u=np.cross(normal,axis);u/=np.linalg.norm(u)
    return u,np.cross(normal,u)


def projected_circle_conic(center,normal,radius,k,r,c):
    if (not np.isfinite(np.r_[center,normal,radius]).all() or radius<=0 or np.linalg.norm(normal)<1e-10):
        raise ValueError('positive finite circle and nonzero normal required')
    u,v=circle_basis(normal)
    h=np.asarray(k)@np.asarray(r)@np.c_[u*radius,v*radius,np.asarray(center)-c]
    inv=np.linalg.inv(h)
    conic=inv.T@np.diag([1,1,-1])@inv
    return conic/np.linalg.norm(conic)


def conic_distance(xy,conic):
    q=np.c_[xy,np.ones(len(xy))];cq=q@conic
    return np.sum(q*cq,axis=1)/np.maximum(2*np.linalg.norm(cq[:,:2],axis=1),1e-12)


def fit_multiview_circle(contours,intrinsics,rotations,centers,training_indices,initial_center,initial_radius):
    if (len(contours)!=len(intrinsics) or len(training_indices)<3 or
        not np.isfinite(np.r_[initial_center,initial_radius]).all() or initial_radius<=0):
        raise ValueError('at least three training views and finite initial circle required')
    def unpack(x):return x[:3],np.r_[x[3:5],1.]/np.linalg.norm(np.r_[x[3:5],1.]),float(np.exp(x[5]))
    def errors(x,i):
        p,n,rad=unpack(x)
        return conic_distance(np.asarray(contours[i]),projected_circle_conic(p,n,rad,intrinsics[i],rotations[i],centers[i]))
    def residual(x):return np.concatenate([errors(x,i) for i in training_indices])
    trials=[]
    for dz in (-2.,0.,2.):
        start=np.r_[np.asarray(initial_center)+[0,0,dz],0.,0.,np.log(initial_radius)]
        fit=least_squares(residual,start,loss='soft_l1',f_scale=1.5,max_nfev=500)
        trials.append(fit)
    fit=min(trials,key=lambda f:np.mean(abs(residual(f.x))))
    p,n,rad=unpack(fit.x)
    rows=[{'view_index':i,'split':'training' if i in training_indices else 'heldout',
           'p90_contour_pixels':float(np.percentile(abs(errors(fit.x,i)),90)),
           'median_contour_pixels':float(np.median(abs(errors(fit.x,i))))} for i in range(len(contours))]
    depth=[float((np.asarray(r)@(p-c))[2]) for r,c in zip(rotations,centers)]
    return {'center_conditioned_units':p.tolist(),'normal_in_RGB_gauge':n.tolist(),
            'radius_conditioned_units':rad,'optimizer_success':bool(fit.success),'optimizer_evaluations':fit.nfev,
            'jacobian_rank':int(np.linalg.matrix_rank(fit.jac)),'view_rows':rows,
            'circle_geometry_supported':bool(fit.success and min(depth)>rad and all(x['p90_contour_pixels']<=5.5 for x in rows)),
            'training_multistart_centers':[unpack(f.x)[0].tolist() for f in trials],
            'multistart_solution_rows':[{'center':unpack(f.x)[0].tolist(),'normal':unpack(f.x)[1].tolist(),
                'radius':unpack(f.x)[2],'training_mean_absolute_pixels':float(np.mean(abs(residual(f.x))))} for f in trials],
            'yaw_observable_from_circle':False,'metric_scale_certified':False,'can_writeback':False}
