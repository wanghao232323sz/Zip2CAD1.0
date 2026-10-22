"""Exact finite candidate-triangle proximity: support upper bounds, never absence."""
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def candidate_surface_distance(points,triangles,candidates=32,batch=512):
    p,t=np.asarray(points,float),np.asarray(triangles,float)
    if p.ndim!=2 or p.shape[1]!=3 or t.ndim!=3 or t.shape[1:]!=(3,3) or not len(t) or not np.isfinite(p).all() or not np.isfinite(t).all() or candidates<1 or batch<1:raise ValueError('finite points, triangles and positive query budget required')
    k=min(candidates,len(t));tree=cKDTree(t.mean(1));dist=np.full(len(p),np.inf);ids=np.full(len(p),-1,int)
    for start in range(0,len(p),batch):
        q=p[start:start+batch];_,near=tree.query(q,k=k);near=np.asarray(near).reshape(len(q),k)
        cp=trimesh.triangles.closest_point(t[near].reshape(-1,3,3),np.repeat(q,k,axis=0)).reshape(len(q),k,3)
        d=np.linalg.norm(cp-q[:,None],axis=2);best=np.argmin(d,axis=1)
        dist[start:start+len(q)]=d[np.arange(len(q)),best];ids[start:start+len(q)]=near[np.arange(len(q)),best]
    return dist,ids


def complete_radius_surface_support(points,triangles,radius=1.25,batch=128):
    """All triangles whose AABBs touch the radius box, then exact finite distance.

    Complete for deciding distance < radius, NOT a global nearest-distance query.
    Large triangles with distant centroids cannot be omitted by this search.
    """
    p,t=np.asarray(points,float),np.asarray(triangles,float)
    if p.ndim!=2 or p.shape[1]!=3 or t.ndim!=3 or t.shape[1:]!=(3,3) or not len(t) or not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isfinite(radius) or radius<=0 or batch<1 or int(batch)!=batch:raise ValueError('finite geometry and positive radius/batch required')
    tree=trimesh.util.bounds_tree(np.stack((t.min(1),t.max(1)),axis=1));distance=np.full(len(p),np.inf);ids=np.full(len(p),-1,int)
    for start in range(0,len(p),batch):
        q=p[start:start+batch];lists=[sorted(tree.intersection(np.r_[v-radius,v+radius])) for v in q]
        counts=np.array([len(v) for v in lists]);nonempty=np.flatnonzero(counts)
        if not len(nonempty):continue
        flat=np.concatenate([np.asarray(lists[i],int) for i in nonempty]);repeat=np.repeat(nonempty,counts[nonempty])
        closest=trimesh.triangles.closest_point(t[flat],q[repeat]);d=np.linalg.norm(closest-q[repeat],axis=1)
        offset=0
        for i in nonempty:
            count=counts[i];best=offset+int(np.argmin(d[offset:offset+count]));distance[start+i]=d[best];ids[start+i]=flat[best];offset+=count
    return distance,ids
