"""Small observation-supervised PointNet keypoint head for a research prototype."""
import torch
from torch import nn


class RoofKeypointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU())
        self.attention=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,6))
        self.offset=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,18))

    def forward(self,x):
        local=self.point(x);global_feature=local.max(1).values
        attention=self.attention(torch.cat([local,global_feature[:,None].expand(-1,x.shape[1],-1)],-1)).transpose(1,2).softmax(-1)
        corners=attention@x[:,:,:3]
        return corners+.25*torch.tanh(self.offset(global_feature).reshape(-1,6,3))


def frame_regularity(points):
    """Generic horizontal frame relationships; no stored reference dimensions."""
    a,b,c,d,e,f=points.unbind(1)
    def unit(v):return v[:,:2]/v[:,:2].norm(dim=1,keepdim=True).clamp_min(1e-5)
    left,right,long_left,long_right=map(unit,[b-a,c-d,e-a,f-d])
    return ((left*long_left).sum(1).square()+(right*long_right).sum(1).square()+.5*(1-(left*right).sum(1).square())+.5*(1-(long_left*long_right).sum(1).square())).mean()


def metric_keypoint_loss(prediction,truth,weights,scale_m):
    """Metre-scale corner and observed edge supervision prevents collapsed spans.

    Generic angle regularity remains a small auxiliary term. Targets enter this
    training loss only and are never stored as coordinates in the checkpoint.
    """
    predicted_m=prediction*scale_m;truth_m=truth*scale_m
    corner=(torch.nn.functional.smooth_l1_loss(predicted_m,truth_m,beta=.25,reduction='none').mean(-1)*weights).mean()
    starts=[0,0,2,3];ends=[1,4,3,5]
    pv=predicted_m[:,ends]-predicted_m[:,starts];tv=truth_m[:,ends]-truth_m[:,starts]
    vector=torch.nn.functional.smooth_l1_loss(pv,tv,beta=.25)
    span=torch.nn.functional.smooth_l1_loss(pv.norm(dim=-1),tv.norm(dim=-1),beta=.25)
    angle=frame_regularity(prediction)
    loss=corner+.5*vector+.25*span+.002*angle
    return loss,dict(corner_m=corner,edge_vector_m=vector,edge_length_m=span,angle=angle)
