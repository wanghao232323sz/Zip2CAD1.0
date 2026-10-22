"""Optional official frozen Sonata + portable semantic heads for fresh observations."""
import sys,os
from pathlib import Path
import numpy as np
from .release_io import save,sha


def sonata_features(points,rgb,source,weights,output,max_chunk=4096):
    runtime=Path(output).resolve().parent/'backbone_runtime_cache'
    for key,name in [('TEMP','tmp'),('TMP','tmp'),('CUDA_CACHE_PATH','cuda'),('TORCH_HOME','torch'),('HF_HOME','huggingface')]:
        folder=runtime/name;folder.mkdir(parents=True,exist_ok=True);os.environ[key]=str(folder)
    import torch
    from .roof_point_supervision import estimate_normals,spatial_chunks
    if not torch.cuda.is_available():raise RuntimeError('Official Sonata/spconv feature extraction requires the configured CUDA environment')
    if not (Path(source)/'sonata').is_dir():raise ValueError('Official Sonata source directory required')
    sys.path.insert(0,str(Path(source).resolve()))
    try:
        import pccm.constants,pccm.builder.inliner
        pccm.constants.PCCM_INLINE_LIBRARY_PATH=runtime/'pccm_inline';pccm.builder.inliner.PCCM_INLINE_LIBRARY_PATH=runtime/'pccm_inline'
    except ImportError:pass
    try:import torch_scatter
    except ImportError:
        from .sonata_native_segments import install_frozen_forward_adapter
        install_frozen_forward_adapter()
    import sonata
    normals,_,_=estimate_normals(points)
    local=(points-np.median(points,axis=0)).astype(np.float32)
    model=sonata.load(str(weights),custom_config=dict(enable_flash=False,enc_patch_size=[128]*5)).cuda().eval();model.requires_grad_(False)
    chunks=spatial_chunks(points,maximum=max_chunk)
    result=np.empty((len(points),1088),np.float16)
    for j,ids in enumerate(chunks):
        ids=np.asarray(ids);torch.manual_seed(17+j)
        point=sonata.transform.default()(dict(coord=local[ids].copy(),color=rgb[ids].copy(),normal=normals[ids].copy()))
        point={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in point.items()}
        with torch.inference_mode():
            encoded=model(point)
            for _ in range(2):
                parent=encoded.pop('pooling_parent');inverse=encoded.pop('pooling_inverse')
                parent.feat=torch.cat([parent.feat,encoded.feat[inverse]],dim=-1);encoded=parent
            while 'pooling_parent' in encoded:
                parent=encoded.pop('pooling_parent');inverse=encoded.pop('pooling_inverse');parent.feat=encoded.feat[inverse];encoded=parent
            features=encoded.feat[encoded.inverse].cpu().numpy().astype(np.float16)
        if features.shape!=(len(ids),1088) or not np.isfinite(features).all():raise ValueError('Sonata restored feature shape invalid')
        result[ids]=features
        print(f'Sonata {j+1}/{len(chunks)}',flush=True)
    np.savez_compressed(output,features=result)
    return result


def semantics(features,points,checkpoint):
    import torch
    from .sonata_roof_head import RoofHead
    from .sonata_graph_head import GraphRoofHead,observed_graph
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    graph=state.get('condition','control_mlp')!='control_mlp' and 'condition' in state
    model=(GraphRoofHead() if graph else RoofHead()).eval();model.load_state_dict(state['model']);torch.set_num_threads(4)
    mean=state['feature_mean'].cpu().numpy();std=state['feature_std'].cpu().numpy()
    x=(np.asarray(features,dtype=np.float32)-mean)/std
    if not np.isfinite(x).all() or x.shape!=(len(points),1088):raise ValueError('Invalid frozen feature matrix')
    output=np.empty((len(points),9),np.float32)
    with torch.inference_mode():
        if not graph:
            for start in range(0,len(x),4096):output[start:start+4096]=model(torch.from_numpy(x[start:start+4096])).softmax(-1).numpy()
        else:
            neighbors,relative=observed_graph(points);embedding=torch.cat([model.project(torch.from_numpy(x[i:i+4096])) for i in range(0,len(x),4096)])
            for start in range(0,len(x),2048):
                ni=neighbors[start:start+2048];center=embedding[start:start+len(ni)]
                logits,_=model.from_embeddings(center,embedding[np.maximum(ni,0)],torch.from_numpy(relative[start:start+len(ni)]),torch.from_numpy(ni>=0))
                output[start:start+len(ni)]=logits.softmax(-1).numpy()
    return output


def train_semantic(features_file,labels_file,output,steps=1500,seed=17,points_file=None,graph=False):
    import torch
    from .sonata_roof_head import RoofHead
    from .sonata_graph_head import GraphRoofHead,observed_graph,balanced_edge_loss,facade_exclusion_loss
    if steps<1:raise ValueError('Positive training steps required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with np.load(features_file) as z:x=z['features'].astype(np.float32)
    with np.load(labels_file) as z:
        labels=z['semantic'].astype(np.int64);mask=z['train_mask'].astype(bool) if 'train_mask' in z else labels>=0
        weak=np.flatnonzero(z['facade_weak_mask']) if 'facade_weak_mask' in z else np.array([],int)
    mask&=labels>=0;ids=np.flatnonzero(mask)
    if not len(ids) or x.shape!=(len(labels),1088):raise ValueError('Known 0..8 training labels and Nx1088 features required')
    if (labels[ids]>8).any():raise ValueError('Semantic class outside 0..8')
    mean=x[ids].mean(0);std=np.maximum(x[ids].std(0),1e-4);x=(x-mean)/std
    if graph:
        if not points_file:raise ValueError('--points is required for graph-head training')
        with np.load(points_file) as z:
            points=z['points_CAD_m'] if 'points_CAD_m' in z else z['coord']+z['CAD_translation_m']
        if len(points)!=len(labels):raise ValueError('Graph points/features/labels must have identical row ordering')
        neighbors,relative=observed_graph(points)
    torch.manual_seed(seed);torch.set_num_threads(4);model=GraphRoofHead() if graph else RoofHead();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(seed)
    for step in range(steps):
        selected=rng.choice(ids,256,replace=True)
        if graph:
            ni=neighbors[selected];safe=np.maximum(ni,0)
            logits,edge=model(torch.from_numpy(x[selected]),torch.from_numpy(x[safe]),torch.from_numpy(relative[selected]),torch.from_numpy(ni>=0))
            valid=(ni>=0)&mask[safe];same=labels[selected,None]==labels[safe]
            loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(labels[selected]))+.2*balanced_edge_loss(edge,torch.from_numpy(same.astype(np.float32)),torch.from_numpy(valid))
            if len(weak):
                w=rng.choice(weak,128,replace=True);wn=neighbors[w]
                wl,_=model(torch.from_numpy(x[w]),torch.from_numpy(x[np.maximum(wn,0)]),torch.from_numpy(relative[w]),torch.from_numpy(wn>=0))
                loss=loss+.03*facade_exclusion_loss(wl)
        else:loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(x[selected])),torch.from_numpy(labels[selected]))
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%100==0 or step+1==steps:print(f'Semantic {step+1}/{steps}: {loss.item():.6f}',flush=True)
    torch.save(dict(model=model.state_dict(),feature_mean=torch.from_numpy(mean),feature_std=torch.from_numpy(std),condition='graph_boundary' if graph else 'control_mlp',step=steps),output/'checkpoint.pt')
    save(output/'training_summary.json',dict(steps=steps,known_training_points=len(ids),features_sha256=sha(features_file),labels_sha256=sha(labels_file),encoder_frozen=True,independent_validation=False))

