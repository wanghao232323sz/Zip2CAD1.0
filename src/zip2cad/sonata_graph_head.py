"""Frozen-Sonata local graph head with SEMANTIC boundary supervision.

Edges distinguish semantic classes, not physical instances or CAD connectivity.
Unknown neighbors contribute observed features, never invented target labels.
"""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from .sonata_roof_head import RoofHead

def observed_graph(points,neighbors=8,radius=.60):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite Nx3 points required')
    if len(points)<2 or neighbors<1 or radius<=0:raise ValueError('Positive graph settings and two points required')
    d,idx=cKDTree(points).query(points,k=min(neighbors+1,len(points)),workers=4)
    output=np.full((len(points),neighbors),-1,np.int32);relative=np.zeros((len(points),neighbors,3),np.float32)
    for j in range(neighbors):
        if j+1>=idx.shape[1]:break
        valid=(d[:,j+1]<=radius)&(idx[:,j+1]!=np.arange(len(points)))
        output[valid,j]=idx[valid,j+1]
        relative[valid,j]=(points[output[valid,j]]-points[valid])/radius
    return output,relative

def supervised_edge_targets(center_ids,neighbors,truth,train):
    centers=np.asarray(center_ids);index=neighbors[centers];safe=np.maximum(index,0)
    valid=(index>=0)&train[centers,None]&train[safe]&(truth[centers,None]>=0)&(truth[safe]>=0)
    same=truth[centers,None]==truth[safe]
    return same.astype(np.float32),valid

class GraphRoofHead(nn.Module):
    def __init__(self):
        super().__init__();self.base=RoofHead()
        self.edge=nn.Sequential(nn.Linear(132,32),nn.GELU(),nn.Linear(32,1))
        self.residual=nn.Linear(128,9)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
    def project(self,x):return self.base.net[:3](x)
    def from_embeddings(self,center,neighbor,relative,valid):
        diff=neighbor-center[:,None];distance=torch.linalg.vector_norm(relative,dim=-1,keepdim=True)
        edge=self.edge(torch.cat([diff.abs(),relative,distance],dim=-1)).squeeze(-1)
        weights=torch.sigmoid(edge)*valid.to(edge.dtype)
        # Residual neighbor differences; empty neighborhoods reduce to base head.
        context=(diff*weights[...,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1.)
        return self.base.net[3](center)+self.residual(context),edge
    def forward(self,center,neighbor,relative,valid):
        return self.from_embeddings(self.project(center),self.project(neighbor),relative,valid)

def balanced_edge_loss(logits,same,valid):
    selected=torch.nn.functional.binary_cross_entropy_with_logits(logits,same,reduction='none')
    terms=[]
    for mask in [valid&(same>.5),valid&(same<.5)]:
        if mask.any():terms.append(selected[mask].mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0

def facade_exclusion_loss(logits,classes=(4,6,7)):
    mass=logits.softmax(-1)[:,list(classes)].sum(-1)
    return -torch.log1p(-mass.clamp(max=1-1e-6)).mean()
