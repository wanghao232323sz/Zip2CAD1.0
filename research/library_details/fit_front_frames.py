"""Fit front-beam faces/posts from MobileSAM consensus without CAD targets."""
import sys,json,hashlib,argparse
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,cv2
from scipy.ndimage import map_coordinates
from PIL import Image,ImageDraw,ImageFont
from zip2cad.roof_image_sections import project,beam_vertices,beam_mesh,connected_frame
from zip2cad.roof_mask_surface import summarize_front_surface
from zip2cad.native_skp import write_native
from zip2cad.roof_multiview_lines import rectangle_hypotheses,distinct_rectangle_support


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--line-sections',action='store_true');parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    base=args.case.resolve();settings=read(base/'inputs.json')
    masks=base/settings['beam_masks'];mr=read(masks/'mask_report.json')
    pp=base/settings['sections'];prior=read(pp)
    cp=base/settings['cameras']
    fp=base/settings['frames'];frames={v['ref']:str((base/v['image_path']).resolve()) for v in read(fp)['source_frames']}
    fit=settings['fit_refs'];check=settings['check_refs'];refs=fit+check
    with np.load(cp) as z:cameras={r:{k:z[k][list(z['camera_refs']).index(r)] for k in ['R','C','K']} for r in refs}
    out=args.output.resolve();out.mkdir()
    rows={};parts=[];render_meshes={};hashes={str(p):sha(p) for p in [pp,cp,fp,masks/'mask_report.json']}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    section_changes={}
    if args.line_sections:
        linespath=base/settings['lines'];lr=read(linespath);hashes[str(linespath)]=sha(linespath)
        current_faces=read(base/settings['seed_faces'])['front_frames']
        name='CD';previous=prior['sections'][name];a,b=np.array(previous['predicted_top_edge']);side=np.array(previous['inward_direction'])
        obs={int(r):dict(camera=cameras[int(r)],lines=v) for r,v in lr['segments'][name]['observations'].items()}
        rectangles=rectangle_hypotheses(lr['segments'][name]['edge_candidates'],current_faces[name]['top_z_offset_m'],current_faces[name]['beam_depth_m'])
        for rect in rectangles:rect['distinct_edge_support']=distinct_rectangle_support(rect,a,b,side,obs,lr['fit_refs'],lr['check_refs'])
        fitted=[v for v in rectangles if len(v['distinct_edge_support']['three_distinct_fit_refs'])>=2]
        if not fitted:raise ValueError('No fit-only three-distinct-line section hypothesis')
        selected=fitted[0]
        if not selected['distinct_edge_support']['three_distinct_check_refs']:raise ValueError('Selected section fails extra-view distinct-line check')
        previous['predicted_top_edge']=(np.array([a,b])+side*selected['front_plane_offset_m']+[0,0,selected['top_z_offset_m']]).tolist()
        previous['width_m']=selected['width_m'];previous['depth_m']=selected['depth_m'];section_changes[name]=selected
    for name in ['AB','CD']:
        previous=prior['sections'][name];a,b=np.array(previous['predicted_top_edge']);axis=b-a;length=float(np.linalg.norm(axis[:2]));side=np.array(previous['inward_direction'])
        buffer=.6 if args.line_sections and name=='CD' else 0.
        along=np.arange(-buffer,length+buffer+.001,.04);zoffset=np.arange(-3.,.501,.02)
        points=a[None,None,:]+(along/length)[None,:,None]*axis+zoffset[:,None,None]*[0,0,1]
        projections={};eligible=[];fitgrids=[]
        for ref in refs:
            record=next(r for r in mr['records'] if r['segment']==name and r['ref']==ref)
            if record['mask_area_native_pixels']<50 or record['white_face_fraction']<.35 or record['mask_score']<.8:continue
            if (cameras[ref]['C']-(a+b)/2)@side>=-.5:continue
            path=(base/record['mask_file']).resolve();hashes[str(path)]=sha(path)
            with np.load(path) as zz:mask=zz['mask_native_crop'].astype(float);crop=zz['crop']
            uv=project(points.reshape(-1,3),cameras[ref]).reshape(points.shape[:-1]+(2,))-crop[:2]
            sample=map_coordinates(mask,[uv[:,:,1],uv[:,:,0]],order=1,mode='constant',cval=0.)
            projections[ref]=sample
            if ref in fit:fitgrids.append(sample);eligible.append(ref)
        if len(eligible)<3:raise ValueError('Insufficient eligible mask views')
        consensus=np.mean(fitgrids,axis=0)
        np.savez_compressed(out/f'{name}_surface.npz',probability=consensus,along_m=along,z_offset_m=zoffset,fit_refs=eligible,per_view_probability=np.array(fitgrids))
        result=summarize_front_surface(consensus,along,zoffset,min_support=.5)
        groups={}
        for group,rr in [('earlier',[r for r in eligible if r<290]),('later',[r for r in eligible if r>=290])]:
            if len(rr)>=2:
                try:groups[group]=summarize_front_surface(np.mean([projections[r] for r in rr],axis=0),along,zoffset,min_support=.5)
                except ValueError as exc:groups[group]=dict(failed=str(exc))
        result.update(eligible_fit_refs=eligible,temporal_groups=groups,previous_depth_m=previous['depth_m'],transverse_width_hypothesis_m=previous['width_m'],line_section=section_changes.get(name))
        # Keep hypotheses at the original seed's transverse plane. No image
        # fit here is allowed to claim that transverse width was recovered.
        top=result['top_z_offset_m'];bottom=result['bottom_z_offset_m']
        start=result['beam_start_along_m'] if buffer else 0.;end=result['beam_end_along_m'] if buffer else length
        beam_a=a+axis*(start/length)+[0,0,top];beam_b=a+axis*(end/length)+[0,0,top]
        beam=beam_mesh(beam_vertices(beam_a,beam_b,side,previous['width_m'],top-bottom));meshes=[beam]
        for post in result['posts']:
            lo,hi=post['start_along_m'],post['end_along_m']
            pa=a+axis*(lo/length)+[0,0,top];pb=a+axis*(hi/length)+[0,0,top]
            # A short box along the beam uses the same local planar basis.
            shift=side*previous['width_m'];vertices=np.array([pa,pb,pb+shift,pa+shift])
            lower=vertices.copy();lower[:,2]-=top-post['lower_z_offset_m']
            meshes.append(beam_mesh(np.vstack([vertices,lower])))
        combined=connected_frame(meshes)
        render_meshes[name]=combined
        parts.append(dict(component_id='RESEARCH_MASK_FRONT_FRAME_'+name,vertices_world_CAD_m=combined.vertices.tolist(),triangles=combined.faces.tolist(),top_triangle_mask=(combined.face_normals[:,2]>.5).tolist(),top_color=[240,183,76],side_color=[255,218,160],closed_solid_verified=True,volume_CAD_units_cubed=float(combined.volume)))
        result['check_views']={}
        beamregion=(along[None,:]>.2*length)&(along[None,:]<.8*length)&(zoffset[:,None]>=bottom)&(zoffset[:,None]<=top)
        for ref in check:
            if ref in projections:result['check_views'][str(ref)]=dict(predicted_beam_face_mask_support=float(np.mean(projections[ref][beamregion])))
        rows[name]=result
        # Raster evidence itself, not a rendering of reference geometry.
        rgb=np.stack([255*(1-consensus),255*np.ones_like(consensus),255*(1-consensus)],axis=-1).astype(np.uint8)[::-1]
        image=Image.fromarray(rgb).resize((900,480));d=ImageDraw.Draw(image)
        def y(v):return (zoffset[-1]-v)/np.ptp(zoffset)*479
        for zz in [top,bottom]:d.line((0,y(zz),899,y(zz)),fill='red',width=2)
        for post in result['posts']:
            d.rectangle((post['start_along_m']/length*899,y(top),post['end_along_m']/length*899,y(post['lower_z_offset_m'])),outline='blue',width=2)
        image.save(out/f'{name}_surface.png')
        print(name,json.dumps({k:v for k,v in result.items() if k not in ['horizontal_coverage','temporal_groups']}),flush=True)
    # Retain the two uncertain side strips as separate hypotheses, not as newly
    # detected walls. Distinguish them in names and colors in the editable file.
    for name in ['AE','DF']:
        p=prior['sections'][name];a,b=np.array(p['predicted_top_edge']);mesh=beam_mesh(beam_vertices(a,b,p['inward_direction'],p['width_m'],p['depth_m']))
        parts.append(dict(component_id='PREVIOUS_UNCERTAIN_SIDE_'+name,vertices_world_CAD_m=mesh.vertices.tolist(),triangles=mesh.faces.tolist(),top_triangle_mask=(mesh.face_normals[:,2]>.5).tolist(),top_color=[180,180,180],side_color=[180,180,180],closed_solid_verified=True,volume_CAD_units_cubed=float(mesh.volume)))
    for ref in refs:
        image=Image.open(frames[ref]).convert('RGB');d=ImageDraw.Draw(image);alluv=[]
        for name,mesh in render_meshes.items():
            uv=project(mesh.vertices,cameras[ref]);alluv.extend(uv.tolist())
            for i,j in mesh.edges_unique:d.line([tuple(uv[i]),tuple(uv[j])],fill='#ffa520',width=2)
        lo=np.min(alluv,axis=0)-[60,65];hi=np.max(alluv,axis=0)+[60,60]
        image.crop((max(0,int(lo[0])),max(0,int(lo[1])),min(image.width,int(hi[0])),min(image.height,int(hi[1])))).save(out/f'overlay_{ref}.png')
    source=(base/settings['source_model']).resolve();source_hash=sha(source)
    plan=dict(schema='zip2cad_native_completion_plan_v2',source_model=str(source),source_model_sha256=source_hash,components=parts,textures_allowed=False,approval_scope='academic_mask_consensus_front_frames',production_edit_accepted=False,reference_target_parameters_used=False)
    save(out/'native_plan.json',plan);receipt=write_native(out/'native_plan.json',out/'hypothesis_SKP')
    assert sha(source)==source_hash and all(sha(p)==v for p,v in hashes.items())
    report=dict(front_frames=rows,model_mask_directory=str(masks),reference_geometry_read=False,manual_supervision_read=False,read_hashes=hashes,transverse_width_not_identified=True,image_constrained_section_count=len(section_changes),complete_physical_sections_certified=0,transverse_sections_are_research_hypotheses=True,line_section_changes=section_changes,side_strips_are_previous_uncertain_hypotheses=True,independent_validation=False,default_candidate_changed=False,native_receipt=receipt)
    save(out/'front_frame_report.json',report)
    page='<meta charset="utf-8"><title>前横梁与立柱掩膜共识</title><style>body{font:16px Microsoft YaHei;max-width:1100px;margin:35px auto;line-height:1.8}img{max-width:100%}</style><h1>ZipMap反向修正CAD图纸 前横梁与立柱掩膜共识</h1><p>MobileSAM冻结推理掩膜映射到神经预测前横梁所在竖直平面，六视图开发共识提取梁面与下方连通立柱。横向厚度并未恢复，仍保留上一轮不确定值；不能将面内重建当完整真实截面。灰色侧条仍为上一轮假设，不是新识别侧墙。</p>'
    for name,result in rows.items():page+=f'<h2>{name}</h2><p>面内梁高{result["beam_depth_m"]:.2f}m，连通立柱候选{len(result["posts"])}个。与相机和平面假设有关，非测量真值。</p><img src="{name}_surface.png">'
    page+=''.join(f'<h2>照片{ref}</h2><img src="overlay_{ref}.png">' for ref in refs)
    page+='<p><a href="front_frame_report.json">完整参数、分组诊断与输入保护记录</a></p>'
    (out/'index.html').write_text(page,encoding='utf-8');print('REPORT',out,flush=True)


if __name__=='__main__':main()
