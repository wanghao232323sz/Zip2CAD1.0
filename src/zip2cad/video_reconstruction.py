"""Fresh video -> official ZipMap blocks -> portable completion project."""
import json,subprocess,re,sys
from pathlib import Path
import numpy as np
from .video_preprocess import preprocess_video,VideoPreprocessConfig
from .full_video_alignment import robust_similarity,compose
from .skp_exchange import canonical_json,sha256_file
from .stage_ledger import inventory

def reconstruct(video,cad,output,repo,checkpoint,inference_python=None,cameras=None,scope=None,sample_fps=1.,block_size=24,overlap=8):
    video,cad,output,repo,checkpoint=map(lambda p:Path(p).resolve(),(video,cad,output,repo,checkpoint))
    if output.exists():raise ValueError('Fresh reconstruction output required')
    if not video.is_file() or not cad.is_file() or not checkpoint.is_file() or not (repo/'zipmap').is_dir():raise ValueError('Video, CAD, official ZipMap repository and checkpoint are required')
    output.mkdir(parents=True);extraction=output/'video';manifest=preprocess_video(video,extraction,VideoPreprocessConfig(sample_fps=sample_fps,block_size=block_size,block_overlap=overlap))
    blocks_file=extraction/'zipmap_blocks.json';blocks=json.loads(blocks_file.read_text(encoding='utf-8'))['blocks'];records=[];transforms={}
    parent=(1.,np.eye(3),np.zeros(3));previous=None
    for block in blocks:
        folder=output/block['block_id'];command=[str(inference_python or sys.executable),str(Path(__file__).with_name('zipmap_inference.py')),
            '--blocks-json',str(blocks_file),'--block-id',block['block_id'],'--zipmap-repo',str(repo),'--checkpoint',str(checkpoint),'--output-dir',str(folder),'--skip-glb']
        subprocess.run(command,check=True)
        prediction=folder/'predictions.npz';refs=[int(re.search(r'frame_(\d+)',name).group(1))-1 for name in block['frame_files']]
        if previous is not None:
            old,old_refs=previous;shared=sorted(set(refs)&set(old_refs));a=[];b=[]
            if not shared:raise ValueError('Sequential reconstruction blocks have no shared physical frames')
            with np.load(old) as x,np.load(prediction) as y:
                for ref in shared:
                    i=old_refs.index(ref);j=refs.index(ref);confidence=np.minimum(x['world_points_conf'][i],y['world_points_conf'][j]);h,w=confidence.shape
                    yy,xx=np.mgrid[3:h-3:8,3:w-3:8];keep=confidence[yy,xx]>=np.percentile(confidence,60)
                    a.append(y['world_points'][j,yy[keep],xx[keep]]);b.append(x['world_points'][i,yy[keep],xx[keep]])
            child=robust_similarity(np.concatenate(a),np.concatenate(b));parent=compose(parent,child)
        transform={'scale':float(parent[0]),'rotation':parent[1].tolist(),'translation':parent[2].tolist()};transforms[block['block_id']]=transform
        records.append({'path':str(prediction),'frame_refs':refs,'to_common':transform});previous=(prediction,refs)
    project={'schema':'zip2cad_headless_project_v1','project_id':'building','video':str(video),'source_model':str(cad),'prediction_blocks':records,
        'frame_manifest':str(extraction/'frame_manifest.json'),'maximum_refine_views':16,'scope_pid_path':scope or [],'profile':'engineering',
        'trusted_existing_exterior':True,'roof_finish_correction':True}
    if cameras:project['camera_inputs']=str(Path(cameras).resolve())
    (output/'completion_project.json').write_text(canonical_json(project),encoding='utf-8');(output/'block_transforms.json').write_text(canonical_json(transforms),encoding='utf-8')
    (output/'reconstruction_provenance.json').write_text(canonical_json({'source_video_sha256':sha256_file(video),'source_CAD_sha256':sha256_file(cad),
        'checkpoint_sha256':sha256_file(checkpoint),'block_count':len(records),'accepted_frame_count':manifest['summary']['accepted_count'],'camera_initialization':'provided' if cameras else 'experimental_automatic',
        'producer_sha256':sha256_file(Path(__file__)),'ZipMap_repository_is_external':True}),encoding='utf-8')
    (output/'artifact_manifest.json').write_text(canonical_json({'files':inventory(output)}),encoding='utf-8');return output/'completion_project.json'
