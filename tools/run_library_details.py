"""Re-run the published library beam/line/post algorithms on explicit case inputs.

Dataset-specific seeds/cameras are declared in inputs.json; no fixed experiment
directory and no reference roof geometry are read. Output is always fresh.
"""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zip2cad.release_io import read,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reuse-beam-masks',action='store_true');args=parser.parse_args()
    case=args.case.resolve();out=args.output.resolve()
    if out.exists() or case.is_relative_to(out) or out.is_relative_to(case):raise ValueError('Fresh output separate from case inputs required')
    settings=read(case/'inputs.json');out.mkdir(parents=True);context=out/'context';context.mkdir()
    original_hash=sha(case/'inputs.json')
    for key in ['sections','seed_faces','cameras','frames','lines','front_frames','front_native_plan','source_model','beam_masks']:settings[key]=str((case/settings[key]).resolve())
    frames=read(settings['frames'])
    for row in frames['source_frames']:row['image_path']=str((case/row['image_path']).resolve())
    save(context/'frames.json',frames);settings['frames']=str(context/'frames.json')
    if args.reuse_beam_masks:
        masks=context/'cached_beam_masks';masks.mkdir();report=read(Path(settings['beam_masks'])/'mask_report.json')
        for row in report['records']:row['mask_file']=str((case/row['mask_file']).resolve())
        save(masks/'mask_report.json',report);settings['beam_masks']=str(masks)
    def flush():save(context/'inputs.json',settings)
    def run(script,output,extra=()):
        flush();subprocess.run([sys.executable,str(ROOT/'research/library_details'/script),'--case',str(context),'--output',str(output),*extra],cwd=ROOT,check=True)
    if not args.reuse_beam_masks:
        masks=out/'01_beam_masks';run('segment_beams.py',masks,['--prompt-mode','depth_sweep','--refs',*[str(i) for i in settings['fit_refs']+settings['check_refs']]])
        settings['beam_masks']=str(masks)
    initial=out/'02_initial_faces';run('fit_front_frames.py',initial);settings['seed_faces']=str(initial/'front_frame_report.json')
    lines=out/'03_lines';run('triangulate_sections.py',lines);settings['lines']=str(lines/'line_report.json')
    fronts=out/'04_line_constrained_frames';run('fit_front_frames.py',fronts,['--line-sections'])
    settings['front_frames']=str(fronts/'front_frame_report.json');settings['front_native_plan']=str(fronts/'native_plan.json')
    posts=out/'05_posts';run('refine_posts.py',posts)
    assert sha(case/'inputs.json')==original_hash
    save(out/'DETAIL_RESEARCH_NOTICE.json',dict(status='completed',reference_roof_used=False,long_neural_training=False,
         case_specific_camera_and_seed_contract=True,beam_masks_reused=args.reuse_beam_masks,
         final_SKP=str(posts/'hypothesis_SKP/corrected.skp'),full_physical_sections_verified=False))
    print(out)


if __name__=='__main__':main()
