"""Portable trained heads and input contracts. References are training-only."""
from pathlib import Path
import numpy as np
from .release_io import read,save,sha


def corner_inference(inputs,checkpoint,output,contract=None):
    import torch
    from .research_roof_keypoints import RoofKeypointHead
    torch.set_num_threads(4)
    with np.load(inputs) as z:
        points=z['points_world_m'] if 'points_world_m' in z else z['points_CAD_m']
        rgb=z['rgb'];refs=z['source_refs'] if 'source_refs' in z else z['frame_refs']
    frame=read(contract)['frame'] if contract else dict(centre_m=np.median(points,axis=0).tolist(),scale_m=float(np.max(np.ptp(points,axis=0))))
    centre=np.asarray(frame['centre_m']);scale=float(frame['scale_m'])
    if scale<=0:raise ValueError('Nondegenerate normalization frame required')
    model=RoofKeypointHead().eval();state=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
    rng=np.random.default_rng(111);ids=rng.choice(len(points),4096,replace=len(points)<4096)
    x=np.c_[(points[ids]-centre)/scale,rgb[ids]].astype(np.float32)
    with torch.inference_mode():pred=model(torch.from_numpy(x)[None])[0].numpy()*scale+centre
    result=dict(landmark_ids=list('ABCDEF'),predictions_world_m=pred.tolist(),frame=frame,
                checkpoint_sha256=sha(checkpoint),observations_sha256=sha(inputs),reference_read=False,
                physical_sections_predicted=False,scope='single-building six-role local geometry head')
    save(output,result);return result


def joint_inference(inputs,checkpoint):
    import torch
    from .roof_branch_geometry import BranchSeparatedGeometryNet,state_digest
    from .roof_joint_geometry import JointGeometryNet,metric_parameters
    from .neural_components import polygon_solid
    from .continuous_route_candidate import route_solid
    from shapely.geometry import Polygon
    folder=Path(inputs);contract=read(folder/'input_contract.json')
    entry=next(x for x in contract['files'] if x['file']=='all.npz')
    if sha(folder/'all.npz')!=entry['sha256'] or contract['reference_geometry_in_inputs']:raise ValueError('Observed input contract invalid')
    with np.load(folder/'all.npz') as z:x=z['features']
    if x.shape!=(15,contract['grid_size'],contract['grid_size']) or not np.isfinite(x).all():raise ValueError('15-channel observed grid required')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['schema']=='zip2cad_branch_geometry_checkpoint_v1':
        model=BranchSeparatedGeometryNet(state['freeze_room'])
    elif state['schema']=='zip2cad_joint_geometry_checkpoint_v1':model=JointGeometryNet(state['structured'])
    else:raise ValueError('Unsupported geometry checkpoint')
    model.load_state_dict(state['model']);model.eval();torch.set_num_threads(4)
    if state.get('freeze_room') and state_digest(model.room_model)!=state['initial_room_state_digest']:raise ValueError('Room protection digest mismatch')
    with torch.inference_mode():vector=model(torch.from_numpy(x[None]))[0].numpy()
    params=metric_parameters(vector,contract['frame']);room=params['room'];u=np.array(room['axis_u']);v=np.array(room['axis_v']);centre=np.array(room['centre_xy'])
    xy=centre+(np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*[room['length_m']/2,room['depth_m']/2])@np.c_[u,v].T
    part=polygon_solid(Polygon(xy),[0,0,room['bottom_z_m']+room['height_m']],np.array([[1,0],[0,1],[0,0]]),[0,0,1],room['height_m'])
    part.update(component_id='NEURAL_PRIMARY_ROOM',top_color=[96,101,101],side_color=[231,231,224]);parts=[part]
    for i,duct in enumerate(params['ducts']):
        part,_=route_solid(np.array(duct['nodes_xy']),duct['bottom_z_m']+duct['height_m'],duct['width_m'],duct['height_m'])
        part.update(component_id=f'NEURAL_MAIN_DUCT_{i}',top_color=[155,160,163],side_color=[128,133,136]);parts.append(part)
    return dict(components=parts,physical_parameters=params,normalized_parameters=vector.tolist(),
                checkpoint_sha256=sha(checkpoint),input_sha256=entry['sha256'],reference_loaded_at_inference=False,
                smoke_only=state.get('smoke_only',False),topology_scope='one room and two three-node ducts')


def build_joint_inputs(cloud,probability,frame,output):
    from .roof_joint_geometry import grid_inputs
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with np.load(cloud) as z:points=z['points_CAD_m'];rgb=z['rgb'];refs=z['frame_refs']
    if np.asarray(probability).shape!=(len(points),9):raise ValueError('Nine-class probabilities for every input point required')
    features=grid_inputs(points,rgb,refs,probability,frame)
    np.savez_compressed(output/'all.npz',features=features)
    save(output/'input_contract.json',dict(schema='zip2cad_joint_geometry_inputs_v1',frame=frame,grid_size=96,channels=15,
         reference_geometry_in_inputs=False,files=[dict(file='all.npz',sha256=sha(output/'all.npz'))],source='ZipMap'))
    return output


def train_corners(inputs,targets,output,steps=1500,batch=4,seed=20,contract=None,resume=None):
    import torch
    from .research_roof_keypoints import RoofKeypointHead,metric_keypoint_loss
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if steps<1 or batch<1:raise ValueError('Positive steps/batch required')
    with np.load(inputs) as z:points=z['points_world_m'] if 'points_world_m' in z else z['points_CAD_m'];rgb=z['rgb']
    with np.load(targets) as z:truth=z['target_world_m'];weights=z['weights'] if 'weights' in z else np.ones(6)
    if truth.shape!=(6,3):raise ValueError('ABCDEF targets must be 6x3')
    frame=read(contract)['frame'] if contract else dict(centre_m=np.median(points,axis=0).tolist(),scale_m=float(np.max(np.ptp(points,axis=0))))
    scale=frame['scale_m'];centre=np.array(frame['centre_m']);points=(points-centre)/scale;truth=(truth-centre)/scale
    torch.manual_seed(seed);torch.set_num_threads(4);rng=np.random.default_rng(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=RoofKeypointHead().to(device);previous=0
    if resume:
        state=torch.load(resume,map_location=device,weights_only=True);model.load_state_dict(state['model']);previous=state['steps']
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    target=torch.tensor(truth,dtype=torch.float32,device=device);weight=torch.tensor(weights,dtype=torch.float32,device=device)
    losses=[]
    for step in range(steps):
        chosen=rng.choice(len(points),(batch,2048),replace=True);x=np.concatenate([points[chosen],rgb[chosen]],axis=-1).astype(np.float32)
        x[:,:,:3]+=rng.normal(0,.002,x[:,:,:3].shape)
        pred=model(torch.from_numpy(x).to(device));loss,_=metric_keypoint_loss(pred,target[None].expand(batch,-1,-1),weight,scale)
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();losses.append(float(loss.item()))
        if (step+1)%100==0 or step+1==steps:print(f'Corners {step+1}/{steps}: loss={loss.item():.6f}',flush=True)
    torch.save(dict(model=model.cpu().state_dict(),steps=previous+steps,architecture='PointNet_attention_six_corners_v1',smoke_only=steps<100),output/'checkpoint.pt')
    save(output/'input_contract.json',dict(frame=frame,input_sha256=sha(inputs)))
    save(output/'training_summary.json',dict(steps=steps,total_steps=previous+steps,parameters=sum(p.numel() for p in model.parameters()),
         device=device,losses=losses,targets_sha256=sha(targets),independent_validation=False,scope='single-building adaptation; not Sonata backbone training'))
    return output


def train_joint(inputs,targets,output,steps=3000,seed=20,resume=None,freeze_room=False):
    import torch
    from .roof_joint_geometry import JointGeometryNet,geometry_loss,procedural_sample,augment_grid
    from .roof_branch_geometry import BranchSeparatedGeometryNet,state_digest
    if steps<1 or (freeze_room and not resume):raise ValueError('Positive steps; protected room training requires --resume')
    inputs=Path(inputs);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    contract=read(inputs/'input_contract.json')
    with np.load(inputs/'all.npz') as z:x=z['features'].astype(np.float32)
    with np.load(targets) as z:target=z['parameters'].astype(np.float32)
    if x.shape!=(15,96,96) or target.shape!=(25,):raise ValueError('Expected 15x96x96 input and 25 normalized geometry targets')
    torch.manual_seed(seed);torch.set_num_threads(4);rng=np.random.default_rng(seed);model=BranchSeparatedGeometryNet(True) if freeze_room else JointGeometryNet(True)
    previous=0;state={}
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True);previous=int(state['step'])
        if freeze_room and state['schema']=='zip2cad_joint_geometry_checkpoint_v1':
            model.room_model.load_state_dict(state['model']);model.duct_model.load_state_dict(state['model'])
        else:model.load_state_dict(state['model'])
    room_digest=state_digest(model.room_model) if freeze_room else None
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.001);span=contract['frame']['span_m']
    for step in range(steps):
        batch_x=[];batch_y=[]
        for _ in range(4):
            a,b=augment_grid(x,target,rng);batch_x.append(a);batch_y.append(b)
        prediction=model(torch.from_numpy(np.stack(batch_x)));loss,_=geometry_loss(prediction,torch.from_numpy(np.stack(batch_y)),span)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if (step+1)%100==0 or step+1==steps:print(f'Joint {step+1}/{steps}: {loss.item():.6f}',flush=True)
    if freeze_room:assert state_digest(model.room_model)==room_digest
    checkpoint=dict(schema='zip2cad_branch_geometry_checkpoint_v1' if freeze_room else 'zip2cad_joint_geometry_checkpoint_v1',model=model.state_dict(),structured=True,smoke_only=steps<100,step=previous+steps)
    if freeze_room:checkpoint.update(freeze_room=True,initial_room_state_digest=room_digest,room_effective_training_step=state.get('room_effective_training_step',previous),duct_effective_training_step=state.get('duct_effective_training_step',previous)+steps)
    torch.save(checkpoint,output/'checkpoint.pt')
    save(output/'training_summary.json',dict(steps=steps,input_sha256=sha(inputs/'all.npz'),target_sha256=sha(targets),independent_validation=False,topology='one room and two main ducts'))
