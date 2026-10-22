"""Extend calibrated CAD-gauge cameras using video background landmarks.

Landmarks are triangulated from training images outside the projected target
roof. New camera poses come from SIFT/PnP, not new roof geometry or CAD fitting.
Rejected frames and correspondence residuals are preserved for inspection.
"""
import cv2,numpy as np

def project(points,K,R,C):
    p=(np.asarray(points)-C)@R.T@K.T;return p[:,:2]/p[:,2:],p[:,2]

def roof_mask(shape,polygon,base,K,R,C,margin=10):
    h,w=shape;xyz=np.c_[polygon,np.full(len(polygon),base)];uv,z=project(xyz,K,R,C)
    if not (z>0).all() or not np.isfinite(uv).all() or abs(uv).max()>1e6:return np.zeros((h,w),np.uint8)
    mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(uv).astype(np.int32)],255)
    if margin:mask=cv2.dilate(mask,np.ones((margin*2+1,margin*2+1),np.uint8))
    return mask

def background_landmarks(images,cameras,polygon,base,training,minimum_angle=.25):
    sift=cv2.SIFT_create(nfeatures=1800);features={};rows=[];descriptor=[];points=[];used=set();matcher=cv2.BFMatcher()
    for ref in training:
        K,R,C=cameras[ref];im=images[ref];mask=255-roof_mask(im.shape[:2],polygon,base,K,R,C)
        k,d=sift.detectAndCompute(cv2.cvtColor(np.clip(im*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY),mask);features[ref]=(k,d)
    pairs=[(a,b) for i,a in enumerate(training) for b in training[i+1:]]
    pairs.sort(key=lambda q:-np.linalg.norm(cameras[q[0]][2]-cameras[q[1]][2]))
    for a,b in pairs:
        ka,da=features[a];kb,db=features[b]
        if da is None or db is None or len(db)<2:continue
        matches=[m for m,n in matcher.knnMatch(da,db,k=2) if m.distance<.75*n.distance and (a,m.queryIdx) not in used and (b,m.trainIdx) not in used]
        if len(matches)<8:continue
        xa=np.array([ka[m.queryIdx].pt for m in matches]);xb=np.array([kb[m.trainIdx].pt for m in matches]);K,R,C=cameras[a];k,r,c=cameras[b]
        pa=K@np.c_[R,-R@C];pb=k@np.c_[r,-r@c];hom=cv2.triangulatePoints(pa,pb,xa.T,xb.T);world=(hom[:3]/hom[3]).T
        ua,za=project(world,K,R,C);ub,zb=project(world,k,r,c);ray1=world-C;ray2=world-c
        angle=np.degrees(np.arccos(np.clip(np.sum(ray1*ray2,axis=1)/(np.linalg.norm(ray1,axis=1)*np.linalg.norm(ray2,axis=1)),-1,1)))
        good=(za>1)&(zb>1)&(np.linalg.norm(ua-xa,axis=1)<2)&(np.linalg.norm(ub-xb,axis=1)<2)&(angle>=minimum_angle)&np.isfinite(world).all(1)
        for j,m in enumerate(matches):
            if not good[j]:continue
            points.append(world[j]);descriptor.append(da[m.queryIdx]);used.update([(a,m.queryIdx),(b,m.trainIdx)])
        rows.append({'training_pair':[int(a),int(b)],'matched_count':len(matches),'accepted_landmarks':int(good.sum())})
    if len(points)<20:raise ValueError('Not enough background landmarks to extend video camera poses')
    return {'points':np.asarray(points),'descriptors':np.asarray(descriptor,np.float32),'pair_rows':rows,'roof_geometry_used_for_landmarks':False}

def estimate_camera(image,K,landmarks,polygon,base,minimum_inliers=12,maximum_p90_px=3.):
    gray=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);k,d=cv2.SIFT_create(nfeatures=1800).detectAndCompute(gray,None)
    if d is None:return None,{'status':'no_image_features'}
    matches=[m for m,n in cv2.BFMatcher().knnMatch(d,landmarks['descriptors'],k=2) if m.distance<.72*n.distance]
    # Every physical landmark may appear once in the PnP fit.
    chosen={}
    for m in matches:
        if m.trainIdx not in chosen or m.distance<chosen[m.trainIdx].distance:chosen[m.trainIdx]=m
    matches=list(chosen.values());row={'feature_matches':len(matches),'camera_source':'background_SIFT_triangulation_PnP'}
    if len(matches)<minimum_inliers:return None,{**row,'status':'insufficient_unique_correspondences'}
    world=np.array([landmarks['points'][m.trainIdx] for m in matches]);pixel=np.array([k[m.queryIdx].pt for m in matches]);cv2.setRNGSeed(82)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(world,pixel,K,None,iterationsCount=800,reprojectionError=maximum_p90_px,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<minimum_inliers:return None,{**row,'status':'PnP_consensus_failed'}
    selected=inliers.ravel();rvec,tvec=cv2.solvePnPRefineLM(world[selected],pixel[selected],K,None,rvec,tvec);R=cv2.Rodrigues(rvec)[0];C=-R.T@tvec.ravel()
    uv,z=project(world[selected],K,R,C);error=np.linalg.norm(uv-pixel[selected],axis=1);p90=float(np.percentile(error,90));row.update(inlier_count=len(selected),inlier_fraction=len(selected)/len(world),p90_reprojection_px=p90,center_CAD_m=C.tolist())
    if p90>maximum_p90_px or np.mean(z>0)<.95 or len(selected)/len(world)<.25:return None,{**row,'status':'camera_quality_failed'}
    if C[2]<=base+.8:return None,{**row,'status':'camera_below_roof_support'}
    mask=roof_mask(image.shape[:2],polygon,base,K,R,C,margin=0);area=int((mask>0).sum());row['projected_roof_pixels']=area
    if area<max(500,image.shape[0]*image.shape[1]*.015):return None,{**row,'status':'insufficient_roof_visibility'}
    return (K,R,C),{**row,'status':'roof_camera_accepted'}
