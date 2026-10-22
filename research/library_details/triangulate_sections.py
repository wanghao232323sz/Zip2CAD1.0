"""Cross-view line triangulation diagnostic; no reference roof geometry."""
import sys,json,hashlib,argparse
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from zip2cad.roof_multiview_lines import extract_parallel_lines,triangulate_section_edges,rectangle_hypotheses,distinct_rectangle_support,refine_beam_axis


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all-views',action='store_true');parser.add_argument('--refine-axis',action='store_true');parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    base=args.case.resolve();settings=read(base/'inputs.json');pp=base/settings['sections'];prior=read(pp)
    maskpath=base/settings['seed_faces'];faces=read(maskpath)['front_frames']
    cp=base/settings['cameras'];fp=base/settings['frames']
    frames={r['ref']:str((base/r['image_path']).resolve()) for r in read(fp)['source_frames']};fit=settings['fit_refs'];check=settings['check_refs']
    with np.load(cp) as z:
        if args.all_views:fit=[int(r) for r in z['camera_refs'] if r not in check]
        refs=fit+check;cameras={r:{k:z[k][list(z['camera_refs']).index(r)] for k in ['R','C','K']} for r in refs}
    images={r:cv2.imread(frames[r]) for r in refs}
    out=args.output.resolve();out.mkdir();report={}
    hashes={str(p):sha(p) for p in [pp,maskpath,cp,fp]+[Path(frames[r]) for r in refs]}
    for name in ['AB','CD','AE','DF']:
        p=prior['sections'][name];a,b=np.array(p['predicted_top_edge']);side=np.array(p['inward_direction'])
        observations={r:dict(camera=cameras[r],lines=extract_parallel_lines(images[r],a,b,cameras[r])) for r in refs}
        axis_report=None
        if args.refine_axis:
            a,b,axis_report=refine_beam_axis(a,b,observations,fit)
            newside=np.r_[-(b-a)[1],(b-a)[0],0.];newside/=np.linalg.norm(newside)
            if newside@side<0:newside=-newside
            side=newside
        candidates,grid=triangulate_section_edges(a,b,side,observations,fit,check)
        face=faces.get(name,dict(top_z_offset_m=0.,beam_depth_m=p['depth_m']))
        rectangles=rectangle_hypotheses(candidates,face['top_z_offset_m'],face['beam_depth_m'])
        for rectangle in rectangles:rectangle['distinct_edge_support']=distinct_rectangle_support(rectangle,a,b,side,observations,fit,check)
        report[name]=dict(line_counts={str(r):len(v['lines']) for r,v in observations.items()},edge_candidates=candidates,rectangles=rectangles[:15],observations={str(r):v['lines'] for r,v in observations.items()},axis_refinement=axis_report,seed_endpoints=[a.tolist(),b.tolist()],side_direction=side.tolist())
        np.savez_compressed(out/f'{name}_grid.npz',**grid)
        print(name,'lines',report[name]['line_counts'],'edge peaks',len(candidates),'rectangles',len(rectangles),flush=True)
        for r in rectangles[:3]:print('rectangle',{k:v for k,v in r.items() if k not in ['top_front','top_inner','bottom_front','distinct_edge_support']},'distinctfit',r['distinct_edge_support']['three_distinct_fit_refs'],'distinctcheck',r['distinct_edge_support']['three_distinct_check_refs'],flush=True)
    save(out/'line_report.json',dict(segments=report,read_hashes=hashes,fit_refs=fit,check_refs=check,reference_geometry_read=False,manual_supervision_read=False,independent_validation=False))
    for ref in [270,315]:
        im=Image.fromarray(cv2.cvtColor(images[ref],cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
        for name,color in [('AB','orange'),('CD','cyan')]:
            for row in report[name]['observations'][str(ref)]:d.line([tuple(row['endpoints'][:2]),tuple(row['endpoints'][2:])],fill=color,width=2)
        im.crop((700,220,1480,530) if ref==270 else (190,130,860,530)).save(out/f'lines_{ref}.png')
    assert all(sha(p)==v for p,v in hashes.items());print('REPORT',out,flush=True)


if __name__=='__main__':main()
