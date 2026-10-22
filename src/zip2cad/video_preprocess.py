from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .json_io import write_json


@dataclass(frozen=True)
class VideoPreprocessConfig:
    sample_fps: float = 1.0
    min_sharpness: float = 700.0
    min_brightness: float = 25.0
    max_brightness: float = 235.0
    max_clipped_fraction: float = 0.55
    min_frame_difference: float = 0.025
    analysis_width: int = 320
    jpeg_quality: int = 94
    block_size: int = 120
    block_overlap: int = 20
    block_max_overflow_fraction: float = 0.10
    contact_sheet_count: int = 36


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec_fourcc: str


def probe_video(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata reported for: {path}")
    codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=frame_count / fps,
        codec_fourcc=codec,
    )


def frame_quality_metrics(
    frame_bgr: np.ndarray,
    analysis_width: int,
) -> tuple[dict[str, float], np.ndarray]:
    height, width = frame_bgr.shape[:2]
    analysis_height = max(1, int(round(height * analysis_width / width)))
    small = cv2.resize(
        frame_bgr,
        (analysis_width, analysis_height),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    dark_fraction = float(np.mean(gray <= 8))
    bright_fraction = float(np.mean(gray >= 247))
    comparison = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    metrics = {
        "sharpness": sharpness,
        "brightness": brightness,
        "dark_clipped_fraction": dark_fraction,
        "bright_clipped_fraction": bright_fraction,
    }
    return metrics, comparison


def frame_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(left.astype(np.float32) - right.astype(np.float32))
        )
        / 255.0
    )


def build_overlapping_blocks(
    frame_records: list[dict[str, object]],
    block_size: int,
    overlap: int,
    max_overflow_fraction: float = 0.10,
) -> list[dict[str, object]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if overlap < 0 or overlap >= block_size:
        raise ValueError("block_overlap must satisfy 0 <= overlap < block_size")
    if not frame_records:
        return []

    blocks: list[dict[str, object]] = []
    start = 0
    while start < len(frame_records):
        end = min(start + block_size, len(frame_records))
        maximum_overflow = max(1, int(round(block_size * max_overflow_fraction)))
        if end < len(frame_records) and len(frame_records) - end <= maximum_overflow:
            end = len(frame_records)
        selected = frame_records[start:end]
        blocks.append(
            {
                "block_id": f"block_{len(blocks) + 1:03d}",
                "start_accepted_index": int(selected[0]["accepted_index"]),
                "end_accepted_index": int(selected[-1]["accepted_index"]),
                "start_timestamp_s": float(selected[0]["timestamp_s"]),
                "end_timestamp_s": float(selected[-1]["timestamp_s"]),
                "frame_count": len(selected),
                "frame_files": [record["output_file"] for record in selected],
            }
        )
        if end == len(frame_records):
            break
        start = end - overlap
    return blocks


def _write_contact_sheet(
    frame_dir: Path,
    accepted: list[dict[str, object]],
    output_path: Path,
    requested_count: int,
) -> None:
    if not accepted:
        return
    count = min(requested_count, len(accepted))
    indices = np.linspace(0, len(accepted) - 1, count, dtype=int)
    columns = 6
    thumb_width = 320
    thumb_height = 180
    label_height = 28
    rows = int(np.ceil(count / columns))
    sheet = np.full(
        (rows * (thumb_height + label_height), columns * thumb_width, 3),
        28,
        dtype=np.uint8,
    )
    for sheet_index, accepted_index in enumerate(indices):
        record = accepted[int(accepted_index)]
        image = cv2.imread(str(frame_dir / str(record["output_file"])))
        if image is None:
            continue
        thumb = cv2.resize(
            image,
            (thumb_width, thumb_height),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(sheet_index, columns)
        y = row * (thumb_height + label_height)
        x = column * thumb_width
        sheet[y : y + thumb_height, x : x + thumb_width] = thumb
        label = (
            f"#{record['accepted_index']}  "
            f"t={float(record['timestamp_s']):.1f}s"
        )
        cv2.putText(
            sheet,
            label,
            (x + 8, y + thumb_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(
        str(output_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def preprocess_video(
    input_path: Path,
    output_dir: Path,
    config: VideoPreprocessConfig | None = None,
) -> dict[str, object]:
    config = config or VideoPreprocessConfig()
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(input_path)
    if config.sample_fps <= 0.0 or config.sample_fps > metadata.fps:
        raise ValueError(
            f"sample_fps must be in (0, source_fps={metadata.fps}]"
        )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {input_path}")

    started = perf_counter()
    candidate_index = 0
    accepted_index = 0
    source_frame_index = 0
    next_sample_frame = 0
    previous_accepted_small: np.ndarray | None = None
    all_records: list[dict[str, object]] = []
    accepted_records: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if source_frame_index < next_sample_frame:
                source_frame_index += 1
                continue

            candidate_index += 1
            timestamp_s = source_frame_index / metadata.fps
            metrics, comparison = frame_quality_metrics(
                frame,
                config.analysis_width,
            )
            reasons: list[str] = []
            if metrics["sharpness"] < config.min_sharpness:
                reasons.append("blur")
            if metrics["brightness"] < config.min_brightness:
                reasons.append("too_dark")
            elif metrics["brightness"] > config.max_brightness:
                reasons.append("too_bright")
            if (
                metrics["dark_clipped_fraction"] > config.max_clipped_fraction
                or metrics["bright_clipped_fraction"] > config.max_clipped_fraction
            ):
                reasons.append("exposure_clipping")

            difference = None
            if previous_accepted_small is not None:
                difference = frame_difference(comparison, previous_accepted_small)
                if difference < config.min_frame_difference:
                    reasons.append("near_duplicate")
            metrics["difference_from_previous_accepted"] = difference

            record: dict[str, object] = {
                "candidate_index": candidate_index,
                "source_frame_index": source_frame_index,
                "timestamp_s": timestamp_s,
                "status": "rejected" if reasons else "accepted",
                "rejection_reasons": reasons,
                "metrics": metrics,
                "output_file": None,
            }
            if reasons:
                rejection_counts.update(reasons)
            else:
                accepted_index += 1
                filename = (
                    f"frame_{accepted_index:06d}_"
                    f"t{int(round(timestamp_s * 1000)):09d}ms.jpg"
                )
                output_path = frame_dir / filename
                written = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
                )
                if not written:
                    raise IOError(f"Could not write extracted frame: {output_path}")
                record["accepted_index"] = accepted_index
                record["output_file"] = filename
                previous_accepted_small = comparison
                accepted_records.append(record)
            all_records.append(record)

            next_sample_frame = int(
                round(candidate_index * metadata.fps / config.sample_fps)
            )
            source_frame_index += 1
    finally:
        capture.release()

    contact_sheet_path = output_dir / "contact_sheet.jpg"
    _write_contact_sheet(
        frame_dir,
        accepted_records,
        contact_sheet_path,
        config.contact_sheet_count,
    )
    blocks = build_overlapping_blocks(
        accepted_records,
        config.block_size,
        config.block_overlap,
        config.block_max_overflow_fraction,
    )
    blocks_path = output_dir / "zipmap_blocks.json"
    write_json(
        blocks_path,
        {
            "schema_version": "1.0",
            "frame_directory": str(frame_dir),
            "block_size": config.block_size,
            "block_overlap": config.block_overlap,
            "block_count": len(blocks),
            "blocks": blocks,
        },
    )

    manifest = {
        "schema_version": "1.0",
        "source_video": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            **asdict(metadata),
        },
        "config": asdict(config),
        "summary": {
            "candidate_count": candidate_index,
            "accepted_count": accepted_index,
            "rejected_count": candidate_index - accepted_index,
            "acceptance_rate": (
                accepted_index / candidate_index if candidate_index else 0.0
            ),
            "rejection_counts": dict(rejection_counts),
            "processing_time_s": perf_counter() - started,
            "zipmap_block_count": len(blocks),
        },
        "artifacts": {
            "frame_directory": str(frame_dir),
            "contact_sheet": str(contact_sheet_path),
            "zipmap_blocks": str(blocks_path),
        },
        "frames": all_records,
    }
    manifest_path = output_dir / "frame_manifest.json"
    manifest["artifacts"]["manifest"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest
