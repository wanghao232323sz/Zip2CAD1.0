"""Independent learned room and duct branches. No reference I/O or fixed geometry."""
import hashlib
import torch
from torch import nn
from .roof_joint_geometry import JointGeometryNet


def state_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class BranchSeparatedGeometryNet(nn.Module):
    def __init__(self,freeze_room=True):
        super().__init__();self.freeze_room=bool(freeze_room);self.structured=True
        self.room_model=JointGeometryNet(True);self.duct_model=JointGeometryNet(True)
        if self.freeze_room:self.room_model.requires_grad_(False)
        self.train(True)

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_room:self.room_model.eval()
        return self

    def forward(self,x):
        if self.freeze_room:
            with torch.no_grad():room=self.room_model(x)[:,:7]
        else:room=self.room_model(x)[:,:7]
        return torch.cat([room,self.duct_model(x)[:,7:]],dim=1)
