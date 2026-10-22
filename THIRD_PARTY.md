# Third-party components

| Component | Official source | Distribution policy |
|---|---|---|
| ZipMap | https://github.com/Haian-Jin/ZipMap | Pinned source/weights downloaded separately; retain upstream Apache-2.0 notice |
| Sonata | https://github.com/facebookresearch/sonata | Frozen official backbone; pinned source/weights downloaded separately; upstream Apache-2.0 applies |
| MobileSAM | https://github.com/ChaoningZhang/MobileSAM | Optional image research dependency; source/weights separate; retain upstream notices |
| SketchUp C SDK | https://extensions.sketchup.com/developers/sketchup_c_api/sketchup/ | User-supplied proprietary runtime; no binary redistribution |

Exact revisions and weight hashes are in `models/manifest.json`. Python dependencies retain their own licenses. This repository's MIT license covers the project's own code, not a relicensing of these external packages or user survey/CAD data.

## Citation

Please cite ZipMap and Sonata according to their official repositories when using their reconstructions/features. This project adds CAD registration, evidence handling, local task heads, geometric decoding, protected native SKP output and research evaluation; it does not claim authorship of those backbones.

## Project task weights

The three original task heads in `models/pretrained/` are released under MIT (see their model card and LICENSE). This grant does not cover upstream backbone weights or campus CAD and survey data.
