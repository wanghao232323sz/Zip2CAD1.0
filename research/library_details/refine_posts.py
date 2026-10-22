"""Re-prompt vertical posts beyond the old beam-sized segmentation box.

Only extends existing, image-derived partial posts. No CAD roof is loaded.
"""
import sys,json,hashlib,argparse
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from scipy.ndimage import map_coordinates,label
from zip2cad.roof_image_sections import project,beam_vertices,beam_mesh,connected_frame
from zip2cad.native_skp import write_native


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    base=args.case.resolve();settings=read(base/'inputs.json');parent=(base/settings['front_frames']).parent;pp=base/settings['front_frames'];prior=read(pp)
    modelinfo=dict(source=str(ROOT/'third_party/MobileSAM'),weights=str(ROOT/'models/mobilesam/mobile_sam.pt'));sys.path[:0]=[modelinfo['source'],str(ROOT/'third_party/mobilesam_deps')]
    from mobile_sam import sam_model_registry,SamPredictor
    torch.set_num_threads(4);predictor=SamPredictor(sam_model_registry['vit_t'](checkpoint=modelinfo['weights']).eval().cuda())
    cp=base/settings['cameras'];fp=base/settings['frames']
    original=read(base/settings['sections'])['sections'];frames={r['ref']:str((base/r['image_path']).resolve()) for r in read(fp)['source_frames']}
    fit=settings['fit_refs'];check=settings['check_refs'];refs=fit+check
    with np.load(cp) as z:cameras={r:{k:z[k][list(z['camera_refs']).index(r)] for k in ['R','C','K']} for r in refs}
    images={r:cv2.cvtColor(cv2.imread(frames[r]),cv2.COLOR_BGR2RGB) for r in refs}
    out=args.output.resolve();out.mkdir();results={};parts=[];newmeshes={}
    hashes={str(p):sha(p) for p in [pp,cp,fp,Path(modelinfo['weights'])]+[Path(frames[r]) for r in refs]}
    for name,face in prior['front_frames'].items():
        a,b=np.array(original[name]['predicted_top_edge']);side=np.array(original[name]['inward_direction']);width=face['transverse_width_hypothesis_m']
        if face.get('line_section'):
            section=face['line_section'];a+=side*section['front_plane_offset_m']+[0,0,section['top_z_offset_m']];b+=side*section['front_plane_offset_m']+[0,0,section['top_z_offset_m']]
        axis=b-a;length=np.linalg.norm(axis[:2]);top=face['top_z_offset_m'];bottom=face['bottom_z_offset_m']
        s0=face['beam_start_along_m'] if name=='CD' else 0.;s1=face['beam_end_along_m'] if name=='CD' else length
        beam=beam_mesh(beam_vertices(a+axis*s0/length+[0,0,top],a+axis*s1/length+[0,0,top],side,width,top-bottom));solids=[beam]
        result=[]
        for idx,post in enumerate(face['posts']):
            middle=(post['start_along_m']+post['end_along_m'])/2;axiswidth=post['end_along_m']-post['start_along_m']
            pa=a+axis*post['start_along_m']/length+[0,0,top];pb=a+axis*post['end_along_m']/length+[0,0,top]
            upper=np.array([pa,pb,pb+side*width,pa+side*width]);lower=upper-[0,0,3.2]
            along=np.linspace(post['start_along_m'],post['end_along_m'],20);zoffset=np.arange(-3.4,top+.01,.02)
            points=a+(along/length)[None,:,None]*axis+zoffset[:,None,None]*[0,0,1]
            viewgrids={};records=[]
            for ref in refs:
                im=images[ref];uv=project(np.vstack([upper,lower]),cameras[ref]);box=np.r_[uv.min(0)-4,uv.max(0)+4]
                box=np.clip(box,[0,0,0,0],[im.shape[1]-1,im.shape[0]-1,im.shape[1]-1,im.shape[0]-1])
                crop=np.r_[np.maximum(box[:2]-30,0),np.minimum(box[2:]+30,[im.shape[1],im.shape[0]])].astype(int)
                x,y,x2,y2=crop;patch=im[y:y2,x:x2];scale=min(4.,1024/max(patch.shape[:2]));patch=cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                # Positive points remain in the previously observed partial post.
                seed=a+axis*middle/length+[0,0,(bottom+post['lower_z_offset_m'])/2]
                xy=(project([seed],cameras[ref])-[x,y])*scale
                localbox=(box-np.r_[crop[:2],crop[:2]])*scale
                with torch.inference_mode():
                    predictor.set_image(patch);masks,scores,_=predictor.predict(point_coords=xy,point_labels=np.ones(1,dtype=int),box=localbox,multimask_output=True)
                choices=[]
                for mask,score in zip(masks,scores):
                    area=float(mask.mean());choices.append(float(score)-max(0.,area-.5))
                selected=int(np.argmax(choices));mask=masks[selected]
                native=cv2.resize(mask.astype(np.uint8),(x2-x,y2-y),interpolation=cv2.INTER_NEAREST).astype(float)
                q=project(points.reshape(-1,3),cameras[ref]).reshape(points.shape[:-1]+(2,))-crop[:2]
                sampled=map_coordinates(native,[q[:,:,1],q[:,:,0]],order=1,mode='constant',cval=0.)
                viewgrids[ref]=sampled
                vis=patch.copy();vis[mask]=(.45*vis[mask]+.55*np.array([25,220,240])).astype(np.uint8)
                Image.fromarray(vis).save(out/f'{name}_{idx}_{ref}_mask.png')
                np.savez_compressed(out/f'{name}_{idx}_{ref}_mask.npz',mask_native_crop=native,crop=crop,scores=scores,selected=selected)
                records.append(dict(ref=ref,model_score=float(scores[selected]),native_area=int(native.sum())))
            consensus=np.mean([viewgrids[r] for r in fit],axis=0)
            supported=np.mean(consensus>=.5,axis=1)>=.6
            labs,n=label(supported);runs=[np.flatnonzero(labs==i) for i in range(1,n+1)]
            # Extension must overlap the old visible stub. Do not bridge gaps.
            runs=[ii for ii in runs if zoffset[ii[-1]]>=post['lower_z_offset_m']+.1 and zoffset[ii[0]]<=bottom-.2]
            estimated=min(float(zoffset[ii[0]]) for ii in runs) if runs else post['lower_z_offset_m']
            newbottom=min(estimated,post['lower_z_offset_m'])
            # Detect cropped/truncated lower limits rather than call them bases.
            report=dict(**post,refined_lower_z_offset_m=newbottom,extension_m=post['lower_z_offset_m']-newbottom,
                reaches_search_bound=bool(newbottom<=zoffset[0]+.04),post_base_measured=False,
                per_view_new_lower_support={str(r):float(np.mean(viewgrids[r][abs(zoffset-newbottom)<.06])) for r in refs},records=records)
            bottom_vertices=upper.copy();bottom_vertices[:,2]-=top-newbottom
            solids.append(beam_mesh(np.vstack([upper,bottom_vertices])));result.append(report)
            np.savez_compressed(out/f'{name}_{idx}_consensus.npz',probability=consensus,along_m=along,z_offset_m=zoffset)
            print(name,'old bottom',post['lower_z_offset_m'],'new bottom',newbottom,'extension',report['extension_m'],flush=True)
        mesh=connected_frame(solids);newmeshes[name]=mesh
        parts.append(dict(component_id='RESEARCH_POST_MASK_FRAME_'+name,vertices_world_CAD_m=mesh.vertices.tolist(),triangles=mesh.faces.tolist(),top_triangle_mask=(mesh.face_normals[:,2]>.5).tolist(),top_color=[240,183,76],side_color=[255,218,160],closed_solid_verified=True,volume_CAD_units_cubed=float(mesh.volume)))
        results[name]=result
    # Preserve the two separate uncertain side strips from the input plan.
    oldplan=read(base/settings['front_native_plan']);parts.extend([p for p in oldplan['components'] if p['component_id'].startswith('PREVIOUS_UNCERTAIN_SIDE_')])
    for ref in refs:
        im=Image.fromarray(images[ref]);draw=ImageDraw.Draw(im);alluv=[]
        for mesh in newmeshes.values():
            uv=project(mesh.vertices,cameras[ref]);alluv.extend(uv.tolist())
            for i,j in mesh.edges_unique:draw.line([tuple(uv[i]),tuple(uv[j])],fill='#ffa520',width=2)
        lo=np.min(alluv,axis=0)-[60,60];hi=np.max(alluv,axis=0)+[60,60]
        im.crop((max(0,int(lo[0])),max(0,int(lo[1])),min(im.width,int(hi[0])),min(im.height,int(hi[1])))).save(out/f'overlay_{ref}.png')
    source=(base/settings['source_model']).resolve();source_hash=sha(source);plan=dict(oldplan,source_model=str(source),source_model_sha256=source_hash,components=parts,approval_scope='academic_post_mask_extension_only')
    save(out/'native_plan.json',plan);receipt=write_native(out/'native_plan.json',out/'hypothesis_SKP')
    assert sha(source)==source_hash and all(sha(p)==v for p,v in hashes.items())
    save(out/'post_report.json',dict(posts=results,parent_frame=str(parent),read_hashes=hashes,reference_geometry_read=False,manual_supervision_read=False,model_training_steps=0,native_receipt=receipt,post_bases_not_certified=True,default_candidate_changed=False))
    print('REPORT',out,flush=True)


if __name__=='__main__':main()
