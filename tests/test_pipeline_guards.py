import json,tempfile,unittest
from pathlib import Path
from zip2cad.release_pipeline import run


class PipelineGuards(unittest.TestCase):
    def make(self,path,cached=False):
        path=Path(path);(path/'video.avi').write_bytes(b'dummy input for preflight only');(path/'original.skp').write_bytes(b'original immutable CAD')
        cfg=dict(schema='zip2cad_pipeline_v1',project_root='.',video='video.avi',source_cad='original.skp',neural=dict(enabled=False))
        if cached:cfg['reconstruction']=dict(mode='cached',source_video_sha256='incorrect',blocks=[])
        file=path/'project.json';file.write_text(json.dumps(cfg),encoding='utf8');return file

    def temp(self):
        base=Path(__file__).resolve().parents[1]/'outputs/unit_test_temp';base.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=base)

    def test_resume_rejects_changed_original_input(self):
        with self.temp() as folder:
            path=Path(folder);cfg=self.make(path);run(cfg,'run',to_stage=1)
            (path/'video.avi').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source video or CAD changed'):run(cfg,'run',resume=True,to_stage=1)

    def test_resume_rejects_config_change(self):
        with self.temp() as folder:
            path=Path(folder);cfg=self.make(path);run(cfg,'run',to_stage=1)
            data=json.loads(cfg.read_text());data['neural']['enabled']=True;cfg.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'config hash differs'):run(cfg,'run',resume=True,to_stage=1)

    def test_cached_predictions_require_video_binding(self):
        with self.temp() as folder:
            path=Path(folder);cfg=self.make(path,cached=True)
            with self.assertRaisesRegex(ValueError,'exact source video'):run(cfg,'run',to_stage=1)

    def test_output_cannot_replace_input_directory(self):
        with self.temp() as folder:
            path=Path(folder);cfg=self.make(path)
            with self.assertRaises(ValueError):run(cfg,path,to_stage=1)
            self.assertEqual((path/'original.skp').read_bytes(),b'original immutable CAD')


if __name__=='__main__':unittest.main()
