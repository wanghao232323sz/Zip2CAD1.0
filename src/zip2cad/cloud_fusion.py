"""Provenance-preserving arbitrary-scale preview, explicitly not validated CAD."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .similarity import transform_points
from .skp_exchange import ExchangeError, canonical_json, read_json


def _preview(points, colours, path):
    image = Image.new("RGB", (1500, 540), "white")
    draw = ImageDraw.Draw(image)
    views = []
    for panel, (a, b, label) in enumerate([(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
        projected = points[:, [a, b]]
        low, high = np.percentile(projected, [0.5, 99.5], axis=0)
        span = max(float(np.max(high - low)), 1e-12)
        centre = (high + low) / 2
        pixel = np.rint((projected - centre) / span * 440 + 250).astype(int)
        visible = (pixel >= 20).all(axis=1) & (pixel < 480).all(axis=1)
        indices = np.flatnonzero(visible)
        # Deterministic sampled preview; no clipping changes the saved cloud.
        for i in indices:
            image.putpixel((panel * 500 + int(pixel[i, 0]), 510 - int(pixel[i, 1])), tuple(int(c) for c in colours[i]))
        draw.text((panel * 500 + 25, 20), f"{label} / arbitrary units / NOT metric CAD", fill="black")
        views.append({"view": label, "display_quantiles": [0.5, 99.5], "displayed_fraction": float(np.mean(visible))})
    draw.text((25, 522), "UNVALIDATED overlap-fusion preview; no semantic masking, loop closure or writeback approval", fill="black")
    image.save(path)
    return views


def fuse_preview(blocks_json: Path, block_outputs: dict[str, Path], alignment: Path, output: Path, points_per_block=20000, confidence_percentile=60.):
    if not 16 <= points_per_block <= 200000 or not np.isfinite(confidence_percentile) or not 0 <= confidence_percentile < 100:
        raise ExchangeError("FUSION_CONFIG", "Invalid preview sampling settings")
    blocks = read_json(blocks_json)["blocks"]
    transform_payload = read_json(alignment / "block_transforms.json")
    if transform_payload["coordinate_system"] != "zipmap_global_arbitrary_scale":
        raise ExchangeError("FUSION_COORDINATES", "Preview requires arbitrary-scale block transforms")
    ids = [b["block_id"] for b in blocks]
    transforms = transform_payload["transforms"]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(transforms) or set(ids) != set(block_outputs):
        raise ExchangeError("FUSION_BLOCKS", "Exactly one transform and prediction per block is required")
    point_rows, colours_rows, block_rows, frame_rows, pixel_rows, provenance = [], [], [], [], [], []
    for index, block in enumerate(blocks):
        block_id = block["block_id"]
        folder = block_outputs[block_id]
        names = read_json(folder / "input_selection.json")["frame_files"]
        if names != block["frame_files"]:
            raise ExchangeError("FUSION_SELECTION", "Full manifest source frames are required")
        transform = transforms[block_id]
        scale, rotation, translation = transform["scale"], np.asarray(transform["rotation"]), np.asarray(transform["translation"])
        if not np.isfinite(scale) or scale <= 0 or rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
            raise ExchangeError("FUSION_TRANSFORM", "Invalid proper similarity transform")
        with np.load(folder / "predictions.npz", allow_pickle=False) as data:
            points = data["world_points"]
            confidence = data["world_points_conf"]
            images = data["images"]
        if points.ndim != 4 or points.shape[0] != len(names) or points.shape[-1] != 3 or confidence.size != np.prod(points.shape[:-1]):
            raise ExchangeError("FUSION_ARRAYS", "Point dimensions differ from source frames/confidence")
        if images.shape == (len(names), 3, points.shape[1], points.shape[2]):
            images = images.transpose(0, 2, 3, 1)
        if images.shape != points.shape:
            raise ExchangeError("FUSION_ARRAYS", "Source image dimensions differ from points")
        pixels_per_frame = points.shape[1] * points.shape[2]
        flat, conf, colour = points.reshape(-1, 3), confidence.ravel(), images.reshape(-1, 3)
        valid = np.isfinite(flat).all(axis=1) & np.isfinite(conf) & (conf > 0) & np.isfinite(colour).all(axis=1)
        if valid.sum() < 16:
            raise ExchangeError("FUSION_POINTS", "Insufficient valid positive-confidence points")
        cutoff = float(np.percentile(conf[valid], confidence_percentile))
        keep = np.flatnonzero(valid & (conf >= cutoff))
        keep = keep[np.linspace(0, len(keep) - 1, min(len(keep), points_per_block), dtype=int)]
        point_rows.append(transform_points(flat[keep], scale, rotation, translation))
        colours_rows.append(np.clip(colour[keep] * 255, 0, 255).astype(np.uint8))
        block_rows.append(np.full(len(keep), index, dtype=np.int32))
        frame_rows.append((keep // pixels_per_frame).astype(np.int32))
        pixel_rows.append((keep % pixels_per_frame).astype(np.int32))
        provenance.append({"block_index": index, "block_id": block_id, "source_frame_files": names,
                           "prediction_image_shape": list(points.shape[1:3]), "confidence_cutoff": cutoff, "sample_count": len(keep)})
    points, colours = np.concatenate(point_rows), np.concatenate(colours_rows)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "candidate_cloud.npz", points=points, colours=colours,
        source_block_index=np.concatenate(block_rows), source_frame_index=np.concatenate(frame_rows), source_pixel_index=np.concatenate(pixel_rows))
    vertices = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
    vertices["red"], vertices["green"], vertices["blue"] = colours.T
    with (output / "candidate_cloud.ply").open("wb") as handle:
        handle.write(("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(points)}\n" + "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
        vertices.tofile(handle)
    views = _preview(points, colours, output / "candidate_cloud_preview.png")
    report = {"status": "unvalidated_fusion_preview", "coordinate_system": "zipmap_global_arbitrary_scale",
        "point_count": len(points), "block_count": len(blocks), "points_per_block_limit": points_per_block,
        "confidence_percentile": confidence_percentile, "provenance": provenance, "views": views,
        "source_mapping_complete": True, "can_generate_candidates": False, "can_writeback": False,
        "limitations": ["PLY is a float32 display export; authoritative source indices are in NPZ/report",
            "Confidence filtering and sampling do not establish scene semantics or building coverage",
            "No metric CAD registration, drift closure or editable SKP output"]}
    (output / "fusion_report.json").write_text(canonical_json(report), encoding="utf-8")
    return report
