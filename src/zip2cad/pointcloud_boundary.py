"""Training-video boundary refinement seeded by attributed 3D cloud patches."""
import numpy as np,cv2
from scipy.optimize import least_squares
from .pointcloud_completion import polygon_prism

def fit_round_boundary(points,area,seed=81):
    """Robust circle fit tolerates attached paths; rejects elongated patches."""
    points=np.asarray(points,float);rng=np.random.default_rng(seed);best=None
    for _ in range(320):
        a,b,c=points[rng.choice(len(points),3,replace=False)];matrix=2*np.array([b-a,c-a])
        if abs(np.linalg.det(matrix))<1e-8:continue
        center=np.linalg.solve(matrix,np.array([b@b-a@a,c@c-a@a]));radius=np.linalg.norm(a-center)
        if radius<=0 or abs(np.pi*radius**2-area)>area*.25:continue
        errors=abs(np.linalg.norm(points-center,axis=1)-radius);inliers=errors<=max(.3,radius*.08)
        rank=(int(inliers.sum()),-float(np.median(errors)))
        if best is None or rank>best[0]:best=(rank,center,radius,inliers)
    if best is None or np.mean(best[3])<.65:return None
    _,center,radius,inliers=best
    fit=least_squares(lambda x:np.linalg.norm(points[inliers]-x[:2],axis=1)-x[2],[*center,radius],loss='soft_l1',f_scale=.2)
    center,radius=fit.x[:2],float(fit.x[2]);error=abs(np.linalg.norm(points-center,axis=1)-radius)
    support=float(np.mean(error<=max(.3,radius*.08)))
    if support<.65 or abs(np.pi*radius**2-area)>area*.25:return None
    return {'center':center,'radius':radius,'boundary_supported_fraction':support}

def detect_round_primitive(warped,source,training_indices,seeds,mask,low,step,minimum_area):
    """Classical image edges constrain a cloud-supported circular plane."""
    gray=np.clip(cv2.cvtColor(warped[source],cv2.COLOR_RGB2GRAY)*255,0,255).astype(np.uint8)
    circles=cv2.HoughCircles(cv2.GaussianBlur(gray,(5,5),1),cv2.HOUGH_GRADIENT,1.,20,
        param1=80,param2=32,minRadius=max(8,int(np.sqrt(minimum_area/np.pi)/step)),maxRadius=int(min(gray.shape)/3))
    if circles is None:return None
    phase=np.arange(180)*2*np.pi/180;best=None
    for cx,cy,radius in circles[0]:
        center=np.array([cx,cy],dtype=float)*step+low;r=float(radius)*step
        seed_fraction=float(np.mean(np.linalg.norm(seeds-center,axis=1)<=r))
        if seed_fraction<.8:continue
        yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]];disc=(xx-cx)**2+(yy-cy)**2<=radius**2
        if np.mean(mask[disc])<.6:continue
        rows=[]
        for i in training_indices:
            g=np.clip(cv2.cvtColor(warped[i],cv2.COLOR_RGB2GRAY)*255,0,255).astype(np.uint8)
            edges=cv2.Canny(g,40,80);near=cv2.dilate(edges,np.ones((5,5),np.uint8))
            coords=np.rint(np.c_[cx+radius*np.cos(phase),cy+radius*np.sin(phase)]).astype(int)
            valid=(coords[:,0]>=0)&(coords[:,0]<g.shape[1])&(coords[:,1]>=0)&(coords[:,1]<g.shape[0])
            fraction=float(np.mean(near[coords[valid,1],coords[valid,0]]>0)) if valid.all() else 0.
            rows.append(fraction)
        if min(rows)<.6:continue
        rank=(min(rows),float(np.mean(mask[disc])))
        if best is None or rank>best[0]:best=(rank,{'center':center,'radius':r,'boundary_supported_fraction':min(rows),'training_edge_fractions':rows,'method':'cloud_seeded_training_Hough_circle'})
    return best[1] if best else None

def detect_ellipse_primitive(warped,source,training_indices,seeds,mask,low,step,minimum_area):
    """Finite edge contours permit an ellipse without assuming a round roof."""
    gray=np.clip(cv2.cvtColor(warped[source],cv2.COLOR_RGB2GRAY)*255,0,255).astype(np.uint8)
    edges=cv2.Canny(gray,40,80);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    maskcontours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    edge_contour_count=len(contours);contours=list(contours)+list(maskcontours);best=None
    phase=np.arange(180)*2*np.pi/180
    for contour_index,contour in enumerate(contours):
        if len(contour)<20 or cv2.contourArea(contour)*step**2<minimum_area:continue
        center,axes,angle=cv2.fitEllipse(contour);axes=np.asarray(axes)/2
        if axes.min()<=0 or axes.max()/axes.min()>1.5:continue
        if abs(np.pi*np.prod(axes)*step**2-mask.sum()*step**2)>mask.sum()*step**2*.3:continue
        q=np.array([[np.cos(np.deg2rad(angle)),-np.sin(np.deg2rad(angle))],[np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))]])
        normalized=((contour.reshape(-1,2)-center)@q)/axes;error=abs(np.linalg.norm(normalized,axis=1)-1)
        if np.percentile(error,75)>.15:continue
        seedcoords=(((seeds-low)/step-center)@q)/axes
        if np.mean(np.linalg.norm(seedcoords,axis=1)<=1)<.8:continue
        yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]];coords=(np.stack([xx,yy],axis=-1)-center)@q/axes;disc=np.linalg.norm(coords,axis=2)<=1
        if np.mean(mask[disc])<.6:continue
        ring=np.array(center)+(np.c_[np.cos(phase),np.sin(phase)]*axes)@q.T;pixel=np.rint(ring).astype(int)
        if (pixel<0).any() or (pixel[:,0]>=gray.shape[1]).any() or (pixel[:,1]>=gray.shape[0]).any():continue
        rows=[]
        for i in training_indices:
            g=np.clip(cv2.cvtColor(warped[i],cv2.COLOR_RGB2GRAY)*255,0,255).astype(np.uint8);near=cv2.dilate(cv2.Canny(g,40,80),np.ones((5,5),np.uint8))
            rows.append(float(np.mean(near[pixel[:,1],pixel[:,0]]>0)))
        # A visible mask can also supply a schematic ellipse. Keep the failed
        # edge verification explicitly recorded; this is shape regularization,
        # not a certificate of the complete physical perimeter.
        edge_verified=min(rows)>=.6
        if not edge_verified and contour_index<edge_contour_count:continue
        rank=(min(rows),-float(np.percentile(error,75)))
        if best is None or rank>best[0]:best=(rank,{'center':np.array(center)*step+low,'axes':axes*step,'rotation':q,'training_edge_fractions':rows,'training_edge_verified':edge_verified,
            'parametric_shape_regularization':not edge_verified})
    return best[1] if best else None

def detect_cloud_seeded_colour_circle(warped,training_indices,seeds,seed_colors,mask,low,step):
    """Multi-view finite colour transition; attached paths do not set radius."""
    if len(seeds)<30:return None
    center=np.median(seeds,axis=0);initial=np.sqrt(mask.sum()*step**2/np.pi)
    if initial<1:return None
    phase=np.arange(90)*2*np.pi/90;unit=np.c_[np.cos(phase),np.sin(phase)];palette=np.median(seed_colors,axis=0)
    best=None
    for iteration in range(2):
        if iteration==0:
            centre_step=max(.4,initial*.04);radius_step=max(.2,initial*.025)
            dx=np.arange(-initial*.35,initial*.35+centre_step/2,centre_step);radii=np.arange(initial*.55,initial*1.15,radius_step)
        else:
            center=best[1][:2];dx=np.arange(-.4,.41,.1);radii=np.arange(best[1][2]-.4,best[1][2]+.41,.1)
        xx,yy,rr=np.meshgrid(dx,dx,radii,indexing='ij');candidates=np.c_[center[0]+xx.ravel(),center[1]+yy.ravel(),rr.ravel()]
        score=np.ones(len(candidates));fractions=[]
        for i in training_indices:
            rows=[]
            for delta in [-.35,.35]:
                coords=(candidates[:,None,:2]+(candidates[:,None,2:]+delta)*unit-low)/step
                sampled=cv2.remap(warped[i].astype(np.float32),coords[...,0].astype(np.float32),coords[...,1].astype(np.float32),cv2.INTER_LINEAR)
                rows.append(np.linalg.norm(sampled-palette,axis=2))
            transition=rows[1]-rows[0];fraction=np.mean(transition>.04,axis=1);fractions.append(fraction)
            score=np.minimum(score,np.mean(np.clip(transition,-.1,.2),axis=1))
        for j in np.argsort(-score)[:40]:
            c=candidates[j];seed_fraction=np.mean(np.linalg.norm(seeds-c[:2],axis=1)<=c[2])
            if seed_fraction<.8 or min(f[j] for f in fractions)<.5:continue
            yy,xx=np.mgrid[:mask.shape[0],:mask.shape[1]];dist=np.linalg.norm((np.stack([xx,yy],axis=-1)*step+low)-c[:2],axis=2)
            disc=dist<=c[2]
            if not disc.any() or np.mean(mask[disc])<.5:continue
            rank=float(score[j])
            if best is None or rank>best[0]:best=(rank,c.copy(),[float(f[j]) for f in fractions])
            break
        if best is None:return None
    return {'center':best[1][:2],'radius':float(best[1][2]),'boundary_supported_fraction':min(best[2]),'training_transition_fractions':best[2],
        'method':'cloud_seeded_multiview_colour_transition_circle','geometry_boundary_is_approximation':True}

def refine_boundaries(item,cloud_points,cloud_refs,images,K,R,C,refs,training,heldout,target_polygon,base_z,policy,cloud_colours=None,protected_pixel_masks=None):
    source=refs.index(training[0]);origin=np.array(item['plane_origin']);normal=np.array(item['plane_normal']);poly=np.array(item['boundary_xy'],np.float32)
    candidates=cloud_points[(cloud_refs==training[0])&(abs((cloud_points-origin)@normal)<policy.plane_tolerance_m)]
    candidates=candidates[np.array([cv2.pointPolygonTest(poly,tuple(map(float,p)),False)>=0 for p in candidates[:,:2]])]
    if len(candidates)<30:return [item]
    step=.15;target_polygon=np.asarray(target_polygon,np.float32);low=target_polygon.min(0);high=target_polygon.max(0)
    yy,xx=np.mgrid[:int(np.ceil((high[1]-low[1])/step))+1,:int(np.ceil((high[0]-low[0])/step))+1]
    xy=np.stack([xx,yy],axis=-1)*step+low;zz=origin[2]-(xy-origin[:2])@normal[:2]/normal[2];points=np.dstack([xy,zz])
    shape=xx.shape;warped=[];visible=[];protected=[]
    for view in range(len(refs)):
        p=(points-C[view])@R[view].T@K[view].T;uv=p[...,:2]/np.maximum(p[...,2:],1e-6)
        warped.append(cv2.remap(images[view].astype(np.float32),uv[...,0].astype(np.float32),uv[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
        visible.append((uv[...,0]>4)&(uv[...,0]<images.shape[2]-4)&(uv[...,1]>4)&(uv[...,1]<images.shape[1]-4)&(p[...,2]>0))
        protected.append(cv2.remap(np.asarray(protected_pixel_masks[view],np.float32),uv[...,0].astype(np.float32),uv[...,1].astype(np.float32),cv2.INTER_NEAREST)>0
            if protected_pixel_masks is not None else np.zeros(shape,bool))
    src=warped[source];gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    def avg(x):return cv2.boxFilter(x.astype(np.float32),-1,(7,7))
    m=avg(gray);variance=avg(gray*gray)-m*m;votes=np.zeros(shape,int)
    for ref in training[1:]:
        i=refs.index(ref);g=cv2.cvtColor(warped[i],cv2.COLOR_RGB2GRAY);gm=avg(g);gv=avg(g*g)-gm*gm
        ncc=(avg(gray*g)-m*gm)/np.sqrt(np.maximum(variance*gv,1e-8));diff=np.mean(abs(src-warped[i]),axis=2)
        votes+=visible[i]&~protected[i]&((ncc>.65)|(diff<.025))
    scope=np.array([cv2.pointPolygonTest(target_polygon,tuple(map(float,p)),False)>=0 for p in xy.reshape(-1,2)]).reshape(shape)
    ij=np.rint((candidates[:,:2]-low)/step).astype(int);colors=src[ij[:,1],ij[:,0]]
    # Distinct materials on one plane become independent connected components.
    # The split is derived from point-cloud samples, with no building colour.
    groups=np.zeros(len(colors),int)
    if item['occupied_area_m2']>40 and np.std(colors.mean(1))>.07:
        cv2.setRNGSeed(81);_,labels,centres=cv2.kmeans(colors.astype(np.float32),2,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.001),4,cv2.KMEANS_PP_CENTERS);groups=labels.ravel()
    palettes=[np.median(colors[groups==g],axis=0) for g in sorted(set(groups.tolist()))]
    ownership=np.argmin(np.stack([np.mean((src-palette)**2,axis=2) for palette in palettes]),axis=0)
    results=[];diagnostics=[]
    for group in sorted(set(groups.tolist())):
        select=groups==group
        if select.sum()<20:continue
        sample=colors[select];median=np.median(sample,axis=0);mad=np.maximum(np.median(abs(sample-median),axis=0)*1.4826,.025)
        difference=np.sqrt(np.mean(((src-median)/mad)**2,axis=2))
        # A reflective or occluded view need not veto a boundary supported by
        # the source plus another training view. The 3D patch and fixed heldout
        # support gates remain unchanged.
        allowed=(difference<=3.)&(votes>=policy.minimum_training_views-1)&visible[source]&~protected[source]&scope&(ownership==group)
        # Pixels at a textured edge can have unreliable correlation. Small
        # closing repairs gaps, but each component still needs real cloud seeds.
        allowed=cv2.morphologyEx(allowed.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        # The schematic model omits narrow service paths. Break those colour
        # connections before tracing a main roof footprint.
        width=max(3,int(round(policy.voxel_m*2/step))|1)
        allowed=cv2.morphologyEx(allowed,cv2.MORPH_OPEN,np.ones((width,width),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(allowed,8)
        seedlabels=labels[ij[select,1],ij[select,0]];seedlabels=seedlabels[seedlabels>0]
        if not len(seedlabels):continue
        ids,votes_count=np.unique(seedlabels,return_counts=True);label=int(ids[np.argmax(votes_count)]);mask=(labels==label).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);contour=max(contours,key=cv2.contourArea)
        contour_world=contour.reshape(-1,2)*step+low;area=float(cv2.contourArea(contour_world.astype(np.float32)))
        diagnostics.append({'colour_group':group,'cloud_seed_count':int(select.sum()),'segmented_area_m2':area,'source_colour_median':median.tolist()})
        if area<policy.minimum_area_m2:continue
        perimeter=cv2.arcLength(contour.astype(np.float32),True);circularity=4*np.pi*cv2.contourArea(contour)/max(perimeter**2,1)
        primitive='simplified_polygon';boundary=None
        center=contour_world.mean(0);radius=np.median(np.linalg.norm(contour_world-center,axis=1))
        fit=least_squares(lambda x:np.linalg.norm(contour_world-x[:2],axis=1)-x[2],[*center,radius],loss='soft_l1',f_scale=step*2)
        center,radius=fit.x[:2],float(fit.x[2]);error=abs(np.linalg.norm(contour_world-center,axis=1)-radius)
        diagnostics[-1]['round_fit']={'center':center.tolist(),'radius_m':radius,'p75_relative_error':float(np.percentile(error,75)/max(radius,1e-8)),
            'area_relative_error':float(abs(np.pi*radius**2-area)/area)}
        # Round shapes are chosen by geometric residual and footprint area,
        # independently of roof colour, building type or heldout outcomes.
        round_model=fit_round_boundary(contour_world,area)
        image_round=detect_round_primitive(warped,source,[refs.index(r) for r in training],candidates[select,:2],mask,low,step,policy.minimum_area_m2)
        if image_round is not None:round_model=image_round
        colour_round=detect_cloud_seeded_colour_circle(warped,[refs.index(r) for r in training],candidates[select,:2],colors[select],mask,low,step)
        if colour_round is not None:round_model=colour_round
        ellipse=detect_ellipse_primitive(warped,source,[refs.index(r) for r in training],candidates[select,:2],mask,low,step,policy.minimum_area_m2)
        if round_model is not None:
            center,radius=round_model['center'],round_model['radius']
            phase=np.arange(policy.curved_boundary_vertices)*2*np.pi/policy.curved_boundary_vertices
            boundary=center+radius*np.c_[np.cos(phase),np.sin(phase)];primitive='cloud_plane_video_boundary_round_prism'
            diagnostics[-1]['robust_round_fit']={'center':center.tolist(),'radius_m':radius,'boundary_supported_fraction':round_model['boundary_supported_fraction'],
                'method':round_model.get('method','robust_contour_circle'),'training_transition_fractions':round_model.get('training_transition_fractions')}
        elif ellipse is not None:
            phase=np.arange(policy.curved_boundary_vertices)*2*np.pi/policy.curved_boundary_vertices
            boundary=ellipse['center']+(np.c_[np.cos(phase),np.sin(phase)]*ellipse['axes'])@ellipse['rotation'].T
            primitive='cloud_plane_video_boundary_ellipse_prism'
            diagnostics[-1]['ellipse_fit']={'center':ellipse['center'].tolist(),'axes_m':ellipse['axes'].tolist(),'training_edge_fractions':ellipse['training_edge_fractions'],
                'complete_perimeter_edge_verified':ellipse['training_edge_verified'],'parametric_shape_regularization':ellipse['parametric_shape_regularization']}
        if boundary is None:
            for eps in np.linspace(step*2,policy.voxel_m*4,12):
                b=cv2.approxPolyDP(contour_world.astype(np.float32).reshape(-1,1,2),float(eps),True).reshape(-1,2).astype(float)
                if 3<=len(b)<=policy.maximum_boundary_vertices and abs(cv2.contourArea(b.astype(np.float32))-area)<area*.2:
                    boundary=b;break
        if boundary is None:continue
        # Record actual original heldout cloud observations in the refined area.
        heldrows=[]
        for ref in heldout:
            refmask=cloud_refs==ref;hp=cloud_points[refmask];inside=np.array([cv2.pointPolygonTest(boundary.astype(np.float32),tuple(map(float,p)),False)>=0 for p in hp[:,:2]]);hp=hp[inside]
            all_errors=abs((hp-origin)@normal);unfiltered_count=len(hp)
            # A lid and its bright vertical skirt have different surface
            # membership. Use the frozen training colour model, never a heldout
            # plane residual, to classify validation observations.
            if cloud_colours is not None:
                hc=np.asarray(cloud_colours)[refmask][inside]
                member=np.sqrt(np.mean(((hc-median)/mad)**2,axis=1))<=3.
                hp=hp[member]
            errors=abs((hp-origin)@normal);fraction=float(np.mean(errors<=policy.plane_tolerance_m*1.5)) if len(hp) else 0.
            heldrows.append({'frame_ref':ref,'point_count':len(hp),'supported_fraction':fraction,'p90_plane_error_m':float(np.percentile(errors,90)) if len(hp) else None,
                'unfiltered_inside_boundary_point_count':unfiltered_count,'unfiltered_supported_fraction':float(np.mean(all_errors<=policy.plane_tolerance_m*1.5)) if unfiltered_count else 0.,
                'surface_membership_source':'frozen_training_RGB_colour_model' if cloud_colours is not None else 'all_inside_boundary_points',
                'heldout_plane_residual_used_to_select_members':False})
        accepted=all(row['point_count']>=8 and row['supported_fraction']>=policy.minimum_heldout_fraction for row in heldrows)
        diagnostics[-1]['heldout_rows']=heldrows;diagnostics[-1]['accepted']=accepted
        if not accepted:continue
        z=origin[2]-(boundary-origin[:2])@normal[:2]/normal[2]
        if z.min()<=base_z+.15:continue
        new=dict(item);new.update(boundary_xy=boundary.tolist(),video_segmented_area_m2=area,
            simplified_boundary_area_m2=float(cv2.contourArea(boundary.astype(np.float32))),heldout_rows=heldrows,
            boundary_source='training_video_colour_and_plane_warp_seeded_by_ZipMap_cloud',boundary_primitive=primitive,
            cloud_seed_point_count=int(select.sum()),manual_image_ROI_used=False,heldout_used_for_boundary=False,
            boundary_refinement_diagnostics=diagnostics,
            boundary_is_approximation=True,complete_physical_boundary_certified=False,
            mesh=polygon_prism(np.c_[boundary,z],np.c_[boundary,np.full(len(boundary),base_z)]))
        results.append(new)
    if not results:item['boundary_refinement_diagnostics']=diagnostics
    return results or [item]
