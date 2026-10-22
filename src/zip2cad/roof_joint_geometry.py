"""Neural structured roof parameters. No target/ref imports or file reads.

Scoped topology: one rectangular room and two three-node main ducts. Building
axes come from unchanged CAD exterior only; dimensions and positions are learned.
"""
import numpy as np
import torch
from torch import nn

PARAMETER_COUNT=25
# [room cx,cy,length,depth,bottom,height,angle; duct nodes(2,3,2);
#  duct widths(2), heights(2), bottoms(2)] in normalized building coordinates.


class JointGeometryNet(nn.Module):
    def __init__(self,structured=True):
        super().__init__();self.structured=structured
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,stride=2,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())
        self.encoder=nn.Sequential(block(15,24),block(24,48),block(48,64),nn.AdaptiveAvgPool2d((4,4)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*16,192),nn.SiLU(),nn.Dropout(.15),nn.Linear(192,PARAMETER_COUNT))

    def forward(self,x):
        raw=self.head(self.encoder(x));sig=torch.sigmoid(raw)
        room=torch.cat([sig[:,:2],.03+.90*sig[:,2:4],.4*torch.tanh(raw[:,4:5]),
            .01+.7*sig[:,5:6],torch.tanh(raw[:,6:7])*np.pi/4],1)
        nodes=sig[:,7:19]
        # Shared outputs tie cross-sections, not the two routes or their lengths.
        if self.structured:
            dims=torch.stack([raw[:,19:21].mean(1),raw[:,21:23].mean(1),raw[:,23:25].mean(1)],1)
            widths=(.003+.12*torch.sigmoid(dims[:,0:1])).expand(-1,2)
            heights=(.01+.3*torch.sigmoid(dims[:,1:2])).expand(-1,2)
            bottoms=(.4*torch.tanh(dims[:,2:3])).expand(-1,2)
        else:
            widths=.003+.12*sig[:,19:21];heights=.01+.3*sig[:,21:23];bottoms=.4*torch.tanh(raw[:,23:25])
        return torch.cat([room,nodes,widths,heights,bottoms],1)


def geometry_loss(pred,target,span,height_scale=6.,structured=True):
    """Metric parameter loss + differentiable route/order/axis consistency."""
    scale=pred.new_tensor([span[0],span[1],span[0],span[1],height_scale,height_scale,2.,
        *([span[0],span[1]]*6),span[0],span[0],height_scale,height_scale,height_scale,height_scale])
    err=(pred-target)*scale
    supervised=nn.functional.smooth_l1_loss(err,torch.zeros_like(err),beta=.25)
    nodes=pred[:,7:19].reshape(-1,2,3,2)*pred.new_tensor(span)
    legs=nodes[:,:,1:]-nodes[:,:,:-1];length=legs.norm(dim=-1)
    short=torch.relu(.8-length).square().mean()
    # Slots are along CAD u: near and far, not a hardcoded world position.
    order=torch.relu((nodes[:,0,:,0].mean(1)-nodes[:,1,:,0].mean(1))+.3).square().mean()
    axis=torch.sin(pred[:,6]).square().mean()
    total=supervised+.05*short+.005*order+(.2*axis if structured else 0.)
    return total,dict(parameter_Huber=supervised,short_segment_penalty=short,slot_order_penalty=order,axis_penalty=axis)


def metric_parameters(vector,frame):
    p=np.asarray(vector,float);span=np.array(frame['span_m']);basis=np.asarray(frame['basis_xy']);origin=np.asarray(frame['origin_xy'])
    if p.shape!=(25,) or not np.isfinite(p).all():raise ValueError('Finite predicted parameter vector required')
    xy=lambda q:origin+(np.asarray(q)*span)@basis.T
    angle=float(p[6]);u=basis@np.array([np.cos(angle),np.sin(angle)]);v=basis@np.array([-np.sin(angle),np.cos(angle)])
    return dict(room=dict(centre_xy=xy(p[:2]).tolist(),length_m=float(p[2]*span[0]),depth_m=float(p[3]*span[1]),
        bottom_z_m=float(frame['base_z_m']+p[4]*6),height_m=float(p[5]*6),axis_u=u.tolist(),axis_v=v.tolist()),
        ducts=[dict(nodes_xy=xy(p[7:19].reshape(2,3,2)[i]).tolist(),width_m=float(p[19+i]*span[0]),
            height_m=float(p[21+i]*6),bottom_z_m=float(frame['base_z_m']+p[23+i]*6)) for i in range(2)])


def grid_inputs(points,rgb,refs,probabilities,frame,size=96):
    """Only observed ZipMap geometry/color and frozen target-free semantics."""
    from .roof_grid import observed_grid
    from shapely.geometry import Polygon
    from shapely import contains_xy
    span=np.asarray(frame['span_m']);uv=(np.asarray(points)[:,:2]-frame['origin_xy'])@np.asarray(frame['basis_xy'])
    # observed_grid has square cells: scale UV into normalized raster coordinates.
    q=np.c_[uv/span*size,np.asarray(points)[:,2]]
    features,_,_=observed_grid(q,rgb,refs,[0,0],1,(size,size),frame['base_z_m'])
    ij=np.floor(q[:,:2]).astype(int);valid=((ij>=0)&(ij<size)).all(1);flat=ij[valid,1]*size+ij[valid,0]
    count=np.bincount(flat,minlength=size*size);sem=[]
    for k in [4,5]:
        sem.append((np.bincount(flat,weights=probabilities[valid,k],minlength=size*size)/np.maximum(count,1)).reshape(size,size))
    y,x=np.mgrid[:size,:size];coords=np.stack([(x+.5)/size,(y+.5)/size])
    polygon=Polygon(frame['outline_normalized']);mask=contains_xy(polygon,coords[0],coords[1]).astype(float)
    return np.concatenate([features,np.array(sem),coords,mask[None]]).astype(np.float32)


def augment_grid(x,target,rng):
    """Rigid grid translation of observation and labels together; no ref fit."""
    n=x.shape[-1];dx,dy=rng.integers(-6,7,2);result=x.copy()
    for channel in list(range(12))+[14]:
        result[channel]=np.roll(x[channel],(dy,dx),(0,1))
        if dx>0:result[channel,:,:dx]=0
        elif dx<0:result[channel,:,dx:]=0
        if dy>0:result[channel,:dy,:]=0
        elif dy<0:result[channel,dy:,:]=0
    y=target.copy();shift=np.array([dx/n,dy/n]);y[:2]+=shift;y[7:19]=(y[7:19].reshape(6,2)+shift).ravel()
    keep=rng.random((n,n))>.12;result[:12]*=keep[None]
    # Drop rectangular view holes without changing true object positions.
    if rng.random()<.5:
        xx,yy=rng.integers(0,n-12,2);result[:12,yy:yy+12,xx:xx+12]=0
    result[:4]+=rng.normal(0,.006,result[:4].shape).astype(np.float32)*(result[9:10]>0)
    return result,y


def procedural_sample(rng,span,size=96):
    """Vary geometry, not a resampling of the standard roof. Synthetic only."""
    y=np.zeros(25,np.float32);y[:2]=[rng.uniform(.4,.6),rng.uniform(.70,.83)]
    y[2:4]=[rng.uniform(.35,.70),rng.uniform(.07,.17)];y[4]=rng.uniform(-.03,.08);y[5]=rng.uniform(.2,.5);y[6]=rng.uniform(-.07,.07)
    starts=np.array([[rng.uniform(.13,.25),rng.uniform(.73,.89)],[rng.uniform(.75,.87),rng.uniform(.73,.89)]])
    elbows=starts+[0,-rng.uniform(.12,.23)];ends=np.array([[rng.uniform(.34,.44),rng.uniform(.43,.58)],[rng.uniform(.56,.66),rng.uniform(.43,.58)]])
    nodes=np.stack([starts,elbows,ends],axis=1);y[7:19]=nodes.ravel()
    y[19:21]=rng.uniform(.012,.045);y[21:23]=rng.uniform(.025,.13);y[23:25]=rng.uniform(-.015,.07)
    yy,xx=np.mgrid[:size,:size];coords=np.stack([(xx+.5)/size,(yy+.5)/size]);xy=np.moveaxis(coords,0,-1)
    d=xy-y[:2];a=y[6];lx=d[...,0]*np.cos(a)+d[...,1]*np.sin(a);ly=-d[...,0]*np.sin(a)+d[...,1]*np.cos(a)
    room=(abs(lx)<y[2]/2)&(abs(ly)<y[3]/2)
    duct=np.zeros((size,size),bool)
    for route in nodes:
        for start,end in zip(route[:-1],route[1:]):
            q=(xy-start)*span;dv=(end-start)*span;t=np.clip((q*dv).sum(-1)/np.dot(dv,dv),0,1)
            distance=np.linalg.norm(q-t[...,None]*dv,axis=-1);duct|=distance<y[19]*span[0]/2
    high=rng.normal(0,.014,(size,size)).astype(np.float32);high[room]=y[4]+y[5];high[duct]=y[23]+y[21]
    # Nuisance round roof, scattered devices and bounded low-frequency depth error.
    circle=((xy[...,0]-.5)/rng.uniform(.16,.21))**2+((xy[...,1]-.46)/rng.uniform(.13,.18))**2<1
    high[circle]=rng.uniform(.08,.16)
    for _ in range(10):
        i,j=rng.integers(0,size-3,2);high[i:i+3,j:j+3]=rng.uniform(.03,.28)
    high+=rng.normal(0,.008,high.shape).astype(np.float32)
    keep=rng.random(high.shape)>rng.uniform(.1,.4)
    x=np.zeros((15,size,size),np.float32);x[0]=high-rng.uniform(.01,.05);x[1:3]=high;x[3]=rng.uniform(.005,.03)
    x[4:6]=rng.uniform(.1,.8,(2,size,size));x[6:9]=rng.uniform(.25,.7,(3,size,size));x[9]=1
    x[10]=np.clip(duct*.8+rng.normal(0,.12,duct.shape),0,1);x[11]=np.clip(room*.8+rng.normal(0,.12,room.shape),0,1)
    x[:12]*=keep[None];x[12:14]=coords;x[14]=1
    return augment_grid(x,y,rng)
