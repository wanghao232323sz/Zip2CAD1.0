"""Scientific model preview rendered from SDK-extracted triangles, without UI."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .sketchup_sdk import SketchUpSDK

def render_mesh(data,output,title,size=(1600,1100),basis=None,bounds=None,camera=None):
    p=data['vertices_world_m'];f=data['faces'];owner=data['face_owner_indices']
    if basis is None:
        view=np.array([-.95,-1.,1.15]);view/=np.linalg.norm(view);right=np.cross(view,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,view);basis=np.c_[right,up,view]
    w,h=size
    if camera is None:
        q=p@basis;xy=q[:,:2]
        if bounds is None:bounds=np.array([xy.min(0),xy.max(0)])
        scale=min((w-100)/np.ptp(bounds[:,0]),(h-150)/np.ptp(bounds[:,1]));centre=bounds.mean(0)
        uv=(xy-centre)*scale+[w/2,h/2+25];uv[:,1]=h-uv[:,1]
    else:
        cam=(p-np.asarray(camera['center']))@np.asarray(camera['rotation']).T
        if np.any(cam[:,2]<=0):raise ValueError('Preview geometry must be in front of the matched camera')
        projection=cam@np.asarray(camera['intrinsics']).T;uv=projection[:,:2]/projection[:,2,None]
        # Screen-space interpolation of reciprocal depth is perspective-correct.
        q=np.c_[cam[:,:2],1/cam[:,2]]
    triangles=p[f];normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);length=np.linalg.norm(normal,axis=1);normal/=np.maximum(length[:,None],1e-12)
    light=np.array([-.3,-.4,1.]);light/=np.linalg.norm(light);shade=.7+.3*abs(normal@light)
    rgb=np.array(data['face_material_rgb'],float)[owner];alpha=np.array(data['face_material_opacity'])[owner,None]
    rgb=np.clip(rgb*shade[:,None],0,255)
    pixels=np.full((h,w,3),[245,248,250],dtype=float);depth=np.full((h,w),-np.inf)
    # Per-pixel depth avoids ordering artefacts where large roof faces cross
    # small raised solids. Transparency is composited over opaque CAD pixels.
    for transparent in [False,True]:
        glass_depth=np.full((h,w),-np.inf);opaque_pixels=pixels.copy()
        for index in np.argsort(q[f,2].mean(1)):
            opacity=float(alpha[index,0])
            if length[index]<1e-12 or (opacity<.999)!=transparent:continue
            tri=uv[f[index]];lo=np.maximum(np.floor(tri.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(tri.max(0)).astype(int),[w-1,h-1])
            if np.any(hi<lo):continue
            yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];x=xx+.5;y=yy+.5
            a,b,c=tri;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den)<1e-10:continue
            l1=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
            l2=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den;l3=1-l1-l2
            z=l1*q[f[index,0],2]+l2*q[f[index,1],2]+l3*q[f[index,2],2]
            ys=slice(lo[1],hi[1]+1);xs=slice(lo[0],hi[0]+1);local=depth[ys,xs]
            keep=(l1>=-1e-8)&(l2>=-1e-8)&(l3>=-1e-8)&(z>=local-1e-7)
            if transparent:
                gd=glass_depth[ys,xs];keep&=z>gd;gd[keep]=z[keep]
                pixels[ys,xs][keep]=rgb[index]*opacity+opaque_pixels[ys,xs][keep]*(1-opacity)
            else:
                local[keep]=z[keep];pixels[ys,xs][keep]=rgb[index]
    image=Image.fromarray(np.clip(pixels,0,255).astype(np.uint8));draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,w,55),fill=(255,255,255));draw.text((25,20),title,fill=(30,40,50));image.save(output)
    return basis,bounds

def preview_models(source,corrected,scope,assembly_pid,output,sdk_path=None,after_title='Corrected CAD - SDK geometry preview'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with SketchUpSDK(sdk_path) as sdk:
        model=sdk.load(source)
        try:before=sdk.geometry(model,scope=scope,include_appearance=True,visible_only=True)
        finally:sdk.release(model)
        model=sdk.load(corrected)
        try:
            original=sdk.geometry(model,scope=scope,include_appearance=True,visible_only=True);addition=sdk.geometry(model,scope=[assembly_pid],include_appearance=True,visible_only=True)
        finally:sdk.release(model)
    after={k:v.copy() for k,v in original.items() if k!='rows'}
    if len(addition['faces']):
        offset=len(after['vertices_world_m']);face_offset=len(after['entity_ids'])
        after['vertices_world_m']=np.vstack([after['vertices_world_m'],addition['vertices_world_m']]);after['faces']=np.vstack([after['faces'],addition['faces']+offset])
        after['face_owner_indices']=np.r_[after['face_owner_indices'],addition['face_owner_indices']+face_offset]
        for k in ['entity_ids','face_material_opacity','face_material_rgb']:after[k]=np.concatenate([after[k],addition[k]])
    basis,bounds=render_mesh(before,output/'before.png','Original CAD - SDK geometry preview')
    render_mesh(after,output/'after.png',after_title,basis=basis,bounds=bounds)
    left=Image.open(output/'before.png');right=Image.open(output/'after.png');combined=Image.new('RGB',(left.width+right.width,left.height));combined.paste(left,(0,0));combined.paste(right,(left.width,0));combined.save(output/'comparison.png')
    return {'renderer':'deterministic_SDK_triangle_preview','photographic_textures_rendered':False,'native_UI_used':False}
