"""Small public IO contract. Config paths are relative to project_root."""
import hashlib,json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def resolve(root,value):
    p=Path(value).expanduser();return (p if p.is_absolute() else Path(root)/p).resolve()
def config(path):
    path=Path(path).resolve();value=read(path)
    if value.get('schema')!='zip2cad_pipeline_v1':raise ValueError('Expected zip2cad_pipeline_v1')
    root=resolve(path.parent,value.get('project_root','..'))
    return value,root
def similarity(points,transform):
    s=float(transform['scale']);r=np.asarray(transform['rotation'],float);t=np.asarray(transform['translation'],float)
    if s<=0 or r.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(np.r_[s,r.ravel(),t]).all():raise ValueError('Invalid Sim(3)')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or np.linalg.det(r)<.99:raise ValueError('Proper rotation required')
    return s*np.asarray(points)@r.T+t
