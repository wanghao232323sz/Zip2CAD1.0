"""Partial front-frame geometry from frozen photo masks on a seed plane."""
import numpy as np
from scipy.ndimage import label


def intervals(mask):
    regions,n=label(np.asarray(mask,bool))
    return [np.flatnonzero(regions==i) for i in range(1,n+1)]


def summarize_front_surface(probability,along_m,z_offset_m,min_support=.55):
    """Summarize a consensus raster without copying a target frame template.

    probability is (vertical, along-beam). z offsets increase upwards.
    Only stable spans and posts connected to the beam are retained.
    """
    p=np.asarray(probability,float);x=np.asarray(along_m,float);z=np.asarray(z_offset_m,float)
    if p.shape!=(len(z),len(x)) or len(x)<10 or len(z)<10 or not np.isfinite(p).all():raise ValueError('Invalid front-surface raster')
    if not np.all(np.diff(x)>0) or not np.all(np.diff(z)>0):raise ValueError('Ascending surface axes required')
    centre=(x>=x.min()+.2*np.ptp(x))&(x<=x.min()+.8*np.ptp(x))
    coverage=np.mean(p[:,centre]>=min_support,axis=1)
    bands=[ix for ix in intervals(coverage>=.65) if z[ix[-1]]>=-1.3 and z[ix[0]]<=.4]
    if not bands:raise ValueError('No cross-view supported horizontal beam')
    band=max(bands,key=lambda ix:len(ix))
    bottom,top=float(z[band[0]]),float(z[band[-1]])
    if not .12<=top-bottom<=1.6:raise ValueError('Beam face thickness unidentifiable')
    beam_rows=(z>=bottom)&(z<=top)
    column_coverage=np.mean(p[beam_rows]>=min_support,axis=0)
    spans=intervals(column_coverage>=.6)
    span=max(spans,key=len) if spans else np.arange(len(x))
    # A post must extend at least 0.35m below the beam across several rows.
    low=(z<bottom-.2)&(z>=bottom-2.8)
    count=np.sum(p[low]>=min_support,axis=0)
    dz=float(np.median(np.diff(z)));dx=float(np.median(np.diff(x)))
    posts=[]
    # A broad end panel can join a narrow deep post in the 0.35m tier. Probe
    # deeper tiers instead of rejecting that entire connected component.
    for minimum_extension in [.35,.6,.9,1.2,1.5]:
        for ix in intervals(count*dz>=minimum_extension):
            width=float(x[ix[-1]]-x[ix[0]]+dx)
            if not .1<=width<=1.4:continue
            left,right=float(x[ix[0]]-dx/2),float(x[ix[-1]]+dx/2)
            if any(min(right,v['end_along_m'])>max(left,v['start_along_m']) for v in posts):continue
            support=np.mean(p[:,ix]>=min_support,axis=1)
            vertical=[v for v in intervals(support>=.6) if z[v[-1]]>=bottom-.12]
            if not vertical:continue
            v=min(vertical,key=lambda ii:z[ii[0]])
            lower=float(z[v[0]])
            if bottom-lower<.35:continue
            posts.append(dict(start_along_m=left,end_along_m=right,width_along_m=width,
                              lower_z_offset_m=lower,upper_z_offset_m=top,continues_below_beam_m=bottom-lower,
                              discovery_depth_tier_m=minimum_extension))
    return dict(top_z_offset_m=top,bottom_z_offset_m=bottom,beam_depth_m=top-bottom,
                beam_start_along_m=float(x[span[0]]),beam_end_along_m=float(x[span[-1]]),
                posts=posts,horizontal_coverage=coverage.tolist(),mask_support_threshold=min_support,
                estimated_face_only=True,transverse_thickness_identified=False)
