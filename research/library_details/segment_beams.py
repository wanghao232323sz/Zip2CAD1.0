"""Frozen MobileSAM photo masks prompted by predicted beam hypotheses.

Masks are automatically generated evidence, never human-reviewed labels.
The reference CAD roof and all landmark supervision are excluded.
"""
import sys,json,hashlib,argparse
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from zip2cad.roof_image_sections import project,beam_vertices


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refs',nargs='+',type=int,default=[264,270,276,308,315,320]);parser.add_argument('--prompt-mode',choices=['basic','depth_sweep'],default='basic');parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    base=args.case.resolve();settings=read(base/'inputs.json');manifest=read(ROOT/'models/manifest.json')['mobilesam']
    receipt=dict(source=str(ROOT/'third_party/MobileSAM'),weights=str(ROOT/'models/mobilesam/mobile_sam.pt'),commit=manifest['source_commit'],weights_sha256=manifest['weight_sha256'])
    if sha(Path(receipt['weights']))!=receipt['weights_sha256']:raise ValueError('MobileSAM weight hash differs')
    sys.path[:0]=[receipt['source'],str(ROOT/'third_party/mobilesam_deps')]
    from mobile_sam import sam_model_registry,SamPredictor
    torch.set_num_threads(4);model=sam_model_registry['vit_t'](checkpoint=receipt['weights']).eval().cuda()
    predictor=SamPredictor(model)
    pp=base/settings['sections'];prior=read(pp)
    cp=base/settings['cameras']
    fp=base/settings['frames'];frames={v['ref']:str((base/v['image_path']).resolve()) for v in read(fp)['source_frames']}
    with np.load(cp) as z:cameras={r:{k:z[k][list(z['camera_refs']).index(r)] for k in ['R','C','K']} for r in args.refs}
    reads=[pp,cp,fp,Path(receipt['weights'])]+[Path(frames[r]) for r in args.refs];hashes={str(p):sha(p) for p in reads}
    out=args.output.resolve();out.mkdir();records=[]
    font=ImageFont.load_default()
    for ref in args.refs:
        image=cv2.cvtColor(cv2.imread(frames[ref]),cv2.COLOR_BGR2RGB)
        for name,part in prior['sections'].items():
            a,b=np.array(part['predicted_top_edge']);side=np.array(part['inward_direction'])
            # The prompt is a generic projected section envelope; dimensions
            # are not supplied by reference CAD or manually clicked targets.
            vertices=beam_vertices(a,b,side,max(.5,part['width_m']),max(.6,part['depth_m']))
            uv=project(vertices,cameras[ref]);box=np.r_[uv.min(0)-6,uv.max(0)+6]
            box=np.clip(box,[0,0,0,0],[image.shape[1]-1,image.shape[0]-1,image.shape[1]-1,image.shape[0]-1])
            crop=np.r_[np.maximum(box[:2]-35,0),np.minimum(box[2:]+35,[image.shape[1],image.shape[0]])].astype(int)
            x,y,x2,y2=crop;patch=image[y:y2,x:x2];scale=min(3.,1024/max(patch.shape[:2]))
            patch=cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            # Use the visible outer face when facing it, otherwise a point on
            # the top face; sample along the middle of the beam, away from joins.
            t=np.array([.3,.5,.7])[:,None];seed=a*(1-t)+b*t
            if (cameras[ref]['C']-(a+b)/2)@side<0:seed-=np.array([0,0,min(.18,part['depth_m']*.4)])
            else:seed+=side*min(.15,part['width_m']*.3)
            prompts=(project(seed,cameras[ref])-[x,y])*scale
            localbox=(box-np.r_[crop[:2],crop[:2]])*scale
            with torch.inference_mode():
                predictor.set_image(patch)
                if args.prompt_mode=='basic':
                    masks,scores,_=predictor.predict(point_coords=prompts,point_labels=np.ones(3,dtype=int),box=localbox,multimask_output=True)
                    allprompts=[prompts]*len(masks)
                else:
                    allmasks=[];allscores=[];allprompts=[]
                    for depth in [.1,.3,.5,.7,1.]:
                        seed=a*(1-t)+b*t-[0,0,depth]
                        ps=(project(seed,cameras[ref])-[x,y])*scale
                        ms,ss,_=predictor.predict(point_coords=ps,point_labels=np.ones(3,dtype=int),box=localbox,multimask_output=True)
                        allmasks.extend(ms);allscores.extend(ss);allprompts.extend([ps]*len(ms))
                    masks=np.array(allmasks);scores=np.array(allscores)
            # Retain all hypotheses. Selection uses only the model mask score;
            # it is not a physical-instance confidence or geometric accuracy.
            selection=scores.copy()
            cue=[]
            for mask in masks:
                colors=patch[mask].astype(float)
                white=float(np.mean((colors.mean(1)>150)&(colors.max(1)-colors.min(1)<65))) if len(colors) else 0.
                fraction=float(mask.mean());cue.append(white)
                if args.prompt_mode=='depth_sweep':selection[len(cue)-1]+=.6*white-2*max(0.,fraction-.5)
            selected=int(np.argmax(selection));mask=masks[selected];prompts=allprompts[selected]
            fullpatch=np.zeros((y2-y,x2-x),np.uint8)
            fullpatch[:]=cv2.resize(mask.astype(np.uint8),((x2-x),(y2-y)),interpolation=cv2.INTER_NEAREST)
            np.savez_compressed(out/f'{name}_{ref}_mask.npz',masks=masks,crop=crop,scale=scale,scores=scores,selection_objective=selection,white_face_fractions=cue,selected=selected,mask_native_crop=fullpatch,prompts_full_image=prompts/scale+[x,y],box_full_image=box)
            vis=patch.copy();vis[mask]=(.45*vis[mask]+.55*np.array([20,220,240])).astype(np.uint8)
            canvas=Image.fromarray(vis);d=ImageDraw.Draw(canvas)
            for p in prompts:d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill='yellow')
            canvas.save(out/f'{name}_{ref}_overlay.png')
            records.append(dict(segment=name,ref=ref,mask_file=str(out/f'{name}_{ref}_mask.npz'),mask_score=float(scores[selected]),mask_area_native_pixels=int(fullpatch.sum()),white_face_fraction=cue[selected],crop=crop.tolist(),human_verified=False,selected_by='frozen model score plus explicit white-face prior' if args.prompt_mode=='depth_sweep' else 'frozen model mask score only'))
            print(name,ref,'mask_score',float(scores[selected]),'native_area',int(fullpatch.sum()),flush=True)
    assert all(sha(p)==v for p,v in hashes.items())
    report=dict(model='MobileSAM vit_t',prompt_mode=args.prompt_mode,commit=receipt['commit'],parameters=sum(p.numel() for p in model.parameters()),weights_sha256=receipt['weights_sha256'],neural_training_steps=0,records=records,read_hashes=hashes,reference_geometry_read=False,supervision_read=False,automatic_masks_not_training_labels=True,prompt_depends_on_previous_neural_hypothesis=True)
    save(out/'mask_report.json',report)
    page='<meta charset="utf-8"><title>MobileSAM梁面候选</title><style>body{font:16px Microsoft YaHei;max-width:1100px;margin:30px auto}img{max-width:100%}</style><h1>MobileSAM梁面候选</h1><p>固定官方预训练模型，仅做照片推理。黄色点来自上一轮神经几何候选投影，不是人工真值。蓝色为自动候选掩膜；模型分数不等于真实构件准确率。所有候选保留在NPZ，不能直接转为训练标签。</p>'
    page+=''.join(f'<h2>{r["segment"]} 视图{r["ref"]}</h2><img src="{r["segment"]}_{r["ref"]}_overlay.png">' for r in records)
    (out/'index.html').write_text(page,encoding='utf-8');print('REPORT',out,flush=True)


if __name__=='__main__':main()
