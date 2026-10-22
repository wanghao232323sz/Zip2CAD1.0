"""Small semantic head on cached frozen Sonata features, no CAD generator."""
import numpy as np
import torch
from torch import nn
from pathlib import Path
from zip2cad.zipmap_learning import read_json,sha256_file


class RoofHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1088,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,9))
    def forward(self,x):return self.net(x)


def load_features(directory):
    directory=Path(directory);manifest=read_json(directory/'feature_manifest.json')
    if manifest['status']!='complete_frozen_features_not_semantics':raise ValueError('Completed frozen features required')
    features=np.empty((manifest['point_count'],1088),np.float16);seen=np.zeros(len(features),bool)
    for row in manifest['chunks']:
        path=directory/row['file']
        if sha256_file(path)!=row['sha256']:raise ValueError('Feature shard changed')
        with np.load(path,allow_pickle=False) as z:
            ids=z['dataset_point_index'];value=z['features']
            if len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(features)) or seen[ids].any():raise ValueError('Invalid/duplicated feature indices')
            if value.shape!=(len(ids),1088) or not np.isfinite(value).all():raise ValueError('Invalid features')
            features[ids]=value;seen[ids]=True
    if not seen.all():raise ValueError('Missing roof feature points')
    return features,manifest


def reference_metrics(predicted,truth):
    valid=truth>=0;cm=np.bincount((truth[valid]*9+predicted[valid]).astype(int),minlength=81).reshape(9,9)
    rows=[]
    for k in range(9):
        tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp)
        rows.append(dict(class_id=k,support=int(cm[k].sum()),precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None,IoU=tp/(tp+fp+fn) if tp+fp+fn else None))
    return dict(scope='training_reference_only_not_independent_validation',confusion_matrix=cm.tolist(),classes=rows,
        mean_IoU=float(np.mean([v['IoU'] for v in rows if v['IoU'] is not None])),
        known_accuracy=float(np.mean(predicted[valid]==truth[valid])),unknown_points_ignored=int((~valid).sum()))
