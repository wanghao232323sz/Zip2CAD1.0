"""Run the official ZipMap model on one block from zipmap_blocks.json.

This adapter deliberately lives in the project rather than modifying the
vendored ZipMap repository. It records the exact input selection and model
revision beside every prediction so that later CAD alignment is reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


MODEL_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": {
        "ttt_mode": True,
        "params": {
            "bias": True,
            "head_dim": 1024,
            "inter_multi": 2,
            "base_lr": 0.01,
            "muon_update_steps": 5,
            "use_gate_fn": True,
        },
    },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-json", type=Path, required=True)
    parser.add_argument("--block-id", default="block_001")
    parser.add_argument("--zipmap-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Evenly sample at most this many frames; useful for a VRAM smoke test.",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=50.0,
        help="Discard this percentage of low-confidence points in the GLB.",
    )
    parser.add_argument("--skip-glb", action="store_true")
    return parser.parse_args()


def choose_frames(frame_files: list[str], max_frames: int | None) -> list[str]:
    if max_frames is None or max_frames >= len(frame_files):
        return frame_files
    if max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
    return [frame_files[index] for index in indices]


def main() -> int:
    args = parse_args()
    # ZipMap decorates its TTT kernels with torch.compile. Triton is not
    # available on native Windows, so use the supported eager fallback.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    for required_path in (args.blocks_json, args.zipmap_repo, args.checkpoint):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    with args.blocks_json.open("r", encoding="utf-8") as handle:
        block_manifest = json.load(handle)
    blocks = {block["block_id"]: block for block in block_manifest["blocks"]}
    if args.block_id not in blocks:
        raise KeyError(f"Unknown block {args.block_id!r}; available: {sorted(blocks)}")

    frame_dir = Path(block_manifest["frame_directory"])
    selected_names = choose_frames(blocks[args.block_id]["frame_files"], args.max_frames)
    image_paths = [frame_dir / name for name in selected_names]
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing input images: {missing_images[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": "1.0",
        "block_id": args.block_id,
        "source_blocks_json": str(args.blocks_json.resolve()),
        "source_frame_count": blocks[args.block_id]["frame_count"],
        "selected_frame_count": len(image_paths),
        "selection_method": "evenly_spaced" if args.max_frames else "all",
        "frame_files": selected_names,
        "zipmap_repo": str(args.zipmap_repo.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_config": MODEL_CONFIG,
    }
    with (args.output_dir / "input_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)

    sys.path.insert(0, str(args.zipmap_repo))
    import torch
    from zipmap.models.ZipMap import ZipMap
    from zipmap.utils.geometry import (
        closed_form_inverse_se3,
        homogenize_points,
        unproject_depth_map_to_point_map,
    )
    from zipmap.utils.load_fn import load_and_preprocess_images
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")

    started = time.perf_counter()
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = ZipMap(**MODEL_CONFIG)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model architecture mismatch: missing={missing}, unexpected={unexpected}")
    if missing:
        print(f"Missing state keys: {missing}")
    if unexpected:
        print(f"Unexpected state keys: {unexpected}")
    del checkpoint, state_dict
    model.eval().to(device)

    images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)
    print(f"Input tensor: {tuple(images.shape)}")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    arrays: dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            arrays[key] = value.detach().cpu().float().numpy().squeeze(0)
    arrays["images"] = images.detach().cpu().float().numpy()

    frame_count = arrays["extrinsic"].shape[0]
    extrinsics_h = np.concatenate(
        [
            arrays["extrinsic"],
            np.zeros((frame_count, 1, 4), dtype=arrays["extrinsic"].dtype),
        ],
        axis=-2,
    )
    extrinsics_h[:, -1, -1] = 1.0
    first_cam_inv = closed_form_inverse_se3(extrinsics_h[0:1])[0]
    arrays["extrinsic"] = (extrinsics_h @ first_cam_inv)[:, :3, :]

    arrays["world_points_from_depth"] = unproject_depth_map_to_point_map(
        arrays["depth"], arrays["extrinsic"], arrays["intrinsic"]
    )
    if "local_points" in arrays:
        cam_to_world = closed_form_inverse_se3(arrays["extrinsic"])
        arrays["world_points"] = np.einsum(
            "sij,shwj->shwi",
            cam_to_world,
            homogenize_points(arrays["local_points"]),
        )[..., :3]
        arrays["world_points_conf"] = arrays.get(
            "local_points_conf", arrays["depth_conf"]
        )

    prediction_path = args.output_dir / "predictions.npz"
    np.savez_compressed(prediction_path, **arrays)

    if not args.skip_glb:
        from visual_util import predictions_to_glb
        scene = predictions_to_glb(
            arrays,
            conf_thres=args.confidence_percentile,
            filter_by_frames="all",
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=str(args.output_dir),
            prediction_mode="Predicted Pointmap",
        )
        scene.export(args.output_dir / "reconstruction.glb")

    elapsed = time.perf_counter() - started
    run_summary = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
        "prediction_keys": sorted(arrays),
        "predictions_file": str(prediction_path.resolve()),
        "glb_file": (
            str((args.output_dir / "reconstruction.glb").resolve())
            if not args.skip_glb
            else None
        ),
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
