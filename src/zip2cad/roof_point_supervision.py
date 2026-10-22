"""Conservative CAD proximity supervision; UNKNOWN is never background."""
import numpy as np
from scipy.spatial import cKDTree
from zip2cad.context_surface_support import complete_radius_surface_support


def correspondence(points, triangles, classes, limit=.20, margin=.05):
    classes=np.asarray(classes)
    if limit<=0 or margin<=0 or len(triangles)!=len(classes):
        raise ValueError('Positive thresholds and triangle classes required')
    distances=np.full((len(points),9),np.inf)
    for k in range(9):
        if np.any(classes==k):
            distances[:,k]=complete_radius_surface_support(points,triangles[classes==k],radius=limit+margin)[0]
    order=np.argsort(distances,axis=1,kind='stable')
    best=distances[np.arange(len(points)),order[:,0]]
    second=distances[np.arange(len(points)),order[:,1]]
    gap=np.zeros(len(points));finite=np.isfinite(best)
    gap[finite]=second[finite]-best[finite]
    valid=(best<=limit)&(gap>=margin)
    labels=np.full(len(points),-1,np.int16);labels[valid]=order[valid,0]
    return labels,best.astype(np.float32),gap.astype(np.float32)


def estimate_normals(points, radius=.8, neighbors=24):
    """PCA normals from observed points only; deterministic sign, not camera facing."""
    points=np.asarray(points,float);tree=cKDTree(points)
    normals=np.zeros_like(points,dtype=np.float32);valid=np.zeros(len(points),bool)
    curvature=np.ones(len(points),np.float32)
    for start in range(0,len(points),2048):
        q=points[start:start+2048];d,idx=tree.query(q,k=min(neighbors,len(points)),workers=4)
        if idx.ndim!=2 or idx.shape[1]<8:continue
        within=d<=radius;count=within.sum(1);xyz=points[idx]
        mean=(xyz*within[...,None]).sum(1)/np.maximum(count,1)[:,None]
        delta=(xyz-mean[:,None])*within[...,None]
        cov=np.einsum('nki,nkj->nij',delta,delta)/np.maximum(count,1)[:,None,None]
        eig,vec=np.linalg.eigh(cov);eig=np.maximum(eig,0)
        c=eig[:,0]/np.maximum(eig.sum(1),1e-12)
        good=(count>=8)&(eig[:,1]/np.maximum(eig[:,2],1e-12)>=.08)&(c<=.08)
        n=vec[:,:,0];axis=np.argmax(abs(n),axis=1)
        n*=np.where(n[np.arange(len(n)),axis]<0,-1,1)[:,None]
        n[~good]=0;normals[start:start+len(q)]=n
        valid[start:start+len(q)]=good;curvature[start:start+len(q)]=c
    return normals,valid,curvature


def spatial_chunks(points, maximum=4096):
    """Disjoint median splits, every point once; no label-dependent selection."""
    if maximum<128 or len(points)<128:raise ValueError('At least 128 points required')
    pending=[np.arange(len(points))];result=[]
    while pending:
        ids=pending.pop()
        if len(ids)<=maximum:result.append(ids);continue
        axis=np.argmax(np.ptp(points[ids],axis=0))
        ids=ids[np.argsort(points[ids,axis],kind='stable')];mid=len(ids)//2
        pending.extend([ids[mid:],ids[:mid]])
    return result
