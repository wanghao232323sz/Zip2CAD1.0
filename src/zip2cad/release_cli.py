"""Public commands for the complete research workflow and local library1.0 case."""
import argparse,importlib.util,os,shutil,sys
from pathlib import Path
from .release_io import read,save,sha


def library_replay(case,output,sdk=None):
    import numpy as np
    from .release_neural import joint_inference,corner_inference
    from .native_skp import write_native
    case=Path(case).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Fresh output required')
    out.mkdir(parents=True)
    release=case/'release';rep=release/'reproducibility'
    learned=joint_inference(case/'joint_inputs',rep/'room_duct_composed_weights.pt');save(out/'joint_prediction.json',learned)
    corner_inference(rep/'corner_inputs.npz',rep/'observed_keypoint_v2.pt',out/'corner_prediction.json',rep/'corner_input_contract.json')
    expected=read(release/'evidence/trained_room_duct_prediction.json')['physical_parameters']
    def numbers(value):
        if isinstance(value,dict):return [n for k in sorted(value) for n in numbers(value[k])]
        if isinstance(value,list):return [n for v in value for n in numbers(v)]
        if isinstance(value,(int,float)):return [value]
        return []
    error=float(np.max(np.abs(np.asarray(numbers(expected))-np.asarray(numbers(learned['physical_parameters'])))))
    if error>1e-4:raise ValueError('Learned branch replay differs from frozen case')
    plan=read(release/'evidence/current_geometry_plan.json');source=release/'models/M1_learned_room_duct.skp'
    if sha(source)!=plan['source_model_sha256']:raise ValueError('Library protected base hash differs')
    plan['source_model']=str(source);save(out/'current_geometry_plan.json',plan)
    receipt=write_native(out/'current_geometry_plan.json',out/'native',sdk)
    shutil.copyfile(receipt['updated_model'],out/'library1.0_RESEARCH.skp')
    save(out/'replay_report.json',dict(status='passed',mode='cached_case_geometry_replay_with_fresh_learned_head_inference',
        output_skp='library1.0_RESEARCH.skp',joint_parameter_max_difference=error,reference_SKP_read=False,
        original_survey_video_reconstructed_in_this_replay=False,latest_photo_constrained_sections_replayed_from_saved_algorithm_plan=True,
        production_accuracy_certified=False))
    return out


def main(argv=None):
    p=argparse.ArgumentParser(prog='zip2cad',description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('run');c.add_argument('--config',type=Path,required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--resume',action='store_true');c.add_argument('--to-step',type=int,choices=range(1,15),default=14)
    sub.add_parser('steps')
    d=sub.add_parser('doctor');d.add_argument('--sdk',type=Path)
    demo=sub.add_parser('demo');demo.add_argument('--output',type=Path,required=True);demo.add_argument('--sdk',type=Path)
    case=sub.add_parser('library-replay');case.add_argument('--case',type=Path,default=Path('artifacts/library1.0'));case.add_argument('--output',type=Path,required=True);case.add_argument('--sdk',type=Path)
    corner=sub.add_parser('infer-corners')
    for key in ['inputs','checkpoint','output']:corner.add_argument('--'+key,type=Path,required=True)
    corner.add_argument('--contract',type=Path)
    joint=sub.add_parser('infer-joint')
    for key in ['inputs','checkpoint','output']:joint.add_argument('--'+key,type=Path,required=True)
    for name in ['train-corners','train-joint','train-semantic']:
        train=sub.add_parser(name);train.add_argument('--inputs',type=Path,required=True);train.add_argument('--targets',type=Path,required=True);train.add_argument('--output',type=Path,required=True);train.add_argument('--steps',type=int,default=1500)
        if name=='train-corners':train.add_argument('--contract',type=Path);train.add_argument('--resume',type=Path)
        if name=='train-joint':train.add_argument('--resume',type=Path);train.add_argument('--freeze-room',action='store_true')
        if name=='train-semantic':train.add_argument('--points',type=Path);train.add_argument('--graph',action='store_true')
    a=p.parse_args(argv)
    if a.command=='steps':
        from .release_pipeline import STAGES
        for i,name in enumerate(STAGES,1):print(f'Step{i:02d}: {name}')
    elif a.command=='run':
        from .release_pipeline import run
        print(run(a.config,a.output,a.resume,a.to_step))
    elif a.command=='doctor':
        import json
        checks={name:importlib.util.find_spec(name) is not None for name in ['numpy','scipy','cv2','trimesh','shapely','rtree','torch']}
        try:
            from .sketchup_sdk import discover_sdk
            checks['SketchUp_SDK']=str(discover_sdk(a.sdk))
        except Exception as error:checks['SketchUp_SDK']=str(error)
        print(json.dumps(dict(python=sys.version,dependencies=checks),indent=2))
    elif a.command=='demo':
        from .headless_demo import demo
        demo(a.output,a.sdk)
    elif a.command=='library-replay':print(library_replay(a.case,a.output,a.sdk))
    elif a.command=='infer-corners':
        from .release_neural import corner_inference
        corner_inference(a.inputs,a.checkpoint,a.output,a.contract)
    elif a.command=='infer-joint':
        from .release_neural import joint_inference
        save(a.output,joint_inference(a.inputs,a.checkpoint))
    elif a.command=='train-corners':
        from .release_neural import train_corners
        train_corners(a.inputs,a.targets,a.output,a.steps,contract=a.contract,resume=a.resume)
    elif a.command=='train-joint':
        from .release_neural import train_joint
        train_joint(a.inputs,a.targets,a.output,a.steps,resume=a.resume,freeze_room=a.freeze_room)
    elif a.command=='train-semantic':
        from .release_semantic import train_semantic
        train_semantic(a.inputs,a.targets,a.output,a.steps,points_file=a.points,graph=a.graph)
    return 0


if __name__=='__main__':raise SystemExit(main())
