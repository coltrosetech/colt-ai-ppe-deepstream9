#!/usr/bin/env python3
"""Convert ``deepstream-app`` GIE KITTI dumps to person-detection JSONL.

DeepStream writes one file per source frame, including an empty file when no
object survives post-processing.  Keeping those empty frames is essential for
ground-truth recall and frame-alignment checks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "deepsafe.person-detections/v1"
KITTI_NAME = re.compile(
    r"^(?P<app_index>\d{2})_(?P<source_id>\d{3})_(?P<frame_index>\d{6,})\.txt$"
)


def _load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not labels or any(not label for label in labels):
        raise ValueError(f"{path}: labels must be non-empty lines")
    return labels


def _parse_kitti_row(raw: str, source: Path, line_number: int) -> tuple[str, list[float]]:
    # The KITTI writer does not quote labels, while COCO contains labels such as
    # "traffic light".  DeepStream's numeric tail contains exactly 15 fields, so
    # parse from the right instead of splitting on the first whitespace.
    fields = raw.split()
    if len(fields) < 16:
        raise ValueError(f"{source}:{line_number}: expected a label and 15 KITTI values")
    label = " ".join(fields[:-15])
    if not label:
        raise ValueError(f"{source}:{line_number}: empty object label")
    try:
        values = [float(value) for value in fields[-15:]]
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: non-numeric KITTI field") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{source}:{line_number}: non-finite KITTI field")
    return label, values


def caviar_frame_indices(path: Path) -> set[int]:
    """Return the explicitly annotated frame numbers in one CAVIAR CVML XML."""

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{path}: invalid CAVIAR XML") from exc
    frames: set[int] = set()
    for frame in root.findall("frame"):
        try:
            index = int(frame.attrib["number"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path}: invalid CAVIAR frame number") from exc
        if index in frames:
            raise ValueError(f"{path}: duplicate CAVIAR frame {index}")
        frames.add(index)
    if not frames:
        raise ValueError(f"{path}: no CAVIAR frames")
    return frames


def _discover_frames(
    directory: Path, *, app_index: int, source_id: int
) -> tuple[dict[int, Path], int]:
    selected: dict[int, Path] = {}
    recognized = 0
    for path in sorted(directory.glob("*.txt")):
        match = KITTI_NAME.fullmatch(path.name)
        if not match:
            continue
        recognized += 1
        if (
            int(match["app_index"]) != app_index
            or int(match["source_id"]) != source_id
        ):
            continue
        index = int(match["frame_index"])
        if index in selected:
            raise ValueError(f"{directory}: duplicate KITTI frame {index}")
        selected[index] = path
    if not selected:
        raise ValueError(
            f"{directory}: no KITTI files for app {app_index}, source {source_id}"
        )
    return selected, recognized


def _expected_indices(frame_files: dict[int, Path], expected_frames: int | None) -> set[int]:
    if expected_frames is not None:
        if expected_frames <= 0:
            raise ValueError("expected_frames must be positive")
        expected = set(range(expected_frames))
    else:
        expected = set(range(max(frame_files) + 1))
    actual = set(frame_files)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "KITTI frame sequence mismatch: "
            f"missing={missing[:10]} extra={extra[:10]} "
            f"expected_count={len(expected)} actual_count={len(actual)}"
        )
    return expected


def convert_kitti_directory(
    directory: Path,
    output: Path,
    *,
    sequence_id: str,
    image_width: int,
    image_height: int,
    labels_path: Path,
    coordinate_width: int | None = None,
    coordinate_height: int | None = None,
    app_index: int = 0,
    source_id: int = 0,
    expected_frames: int | None = None,
    include_frames: Iterable[int] | None = None,
    fps: float | None = None,
    source_uri: str | None = None,
    model_id: str | None = None,
) -> dict[str, object]:
    """Convert a complete single-source KITTI dump and return conversion stats."""

    if not sequence_id.strip():
        raise ValueError("sequence_id must be non-empty")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    coordinate_width = coordinate_width or image_width
    coordinate_height = coordinate_height or image_height
    if coordinate_width <= 0 or coordinate_height <= 0:
        raise ValueError("KITTI coordinate dimensions must be positive")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be a positive finite number")

    labels = _load_labels(labels_path)
    class_ids = {label.casefold(): index for index, label in enumerate(labels)}
    frame_files, recognized_files = _discover_frames(
        directory, app_index=app_index, source_id=source_id
    )
    decoded_frames = _expected_indices(frame_files, expected_frames)
    exported_frames = decoded_frames if include_frames is None else set(include_frames)
    unknown_includes = exported_frames - decoded_frames
    if unknown_includes:
        raise ValueError(
            f"requested export frames absent from KITTI dump: {sorted(unknown_includes)[:10]}"
        )

    stats: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "sequence_id": sequence_id,
        "decoded_frame_files": len(decoded_frames),
        "exported_frame_records": len(exported_frames),
        "skipped_unannotated_frames": len(decoded_frames - exported_frames),
        "recognized_kitti_files_all_sources": recognized_files,
        "json_image_dimensions": [image_width, image_height],
        "kitti_coordinate_dimensions": [coordinate_width, coordinate_height],
        "person_detections": 0,
        "ignored_non_person_detections": 0,
        "clipped_person_boxes": 0,
        "dropped_degenerate_person_boxes": 0,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for frame_index in sorted(exported_frames):
                detections: list[dict[str, object]] = []
                source = frame_files[frame_index]
                for line_number, raw in enumerate(
                    source.read_text(encoding="utf-8").splitlines(), 1
                ):
                    if not raw.strip():
                        continue
                    label, values = _parse_kitti_row(raw, source, line_number)
                    if label.casefold() != "person":
                        stats["ignored_non_person_detections"] = int(
                            stats["ignored_non_person_detections"]
                        ) + 1
                        continue
                    confidence = values[-1]
                    if not 0 <= confidence <= 1:
                        raise ValueError(
                            f"{source}:{line_number}: confidence outside [0, 1]"
                        )
                    left, top, right, bottom = values[3:7]
                    clipped = (
                        max(0.0, min(float(coordinate_width), left)),
                        max(0.0, min(float(coordinate_height), top)),
                        max(0.0, min(float(coordinate_width), right)),
                        max(0.0, min(float(coordinate_height), bottom)),
                    )
                    if clipped != (left, top, right, bottom):
                        stats["clipped_person_boxes"] = int(stats["clipped_person_boxes"]) + 1
                    left, top, right, bottom = clipped
                    if right <= left or bottom <= top:
                        stats["dropped_degenerate_person_boxes"] = int(
                            stats["dropped_degenerate_person_boxes"]
                        ) + 1
                        continue
                    detections.append(
                        {
                            "class_id": class_ids.get("person", 0),
                            "class_name": "person",
                            "confidence": confidence,
                            "bbox_norm_xywh": [
                                round(left / coordinate_width, 10),
                                round(top / coordinate_height, 10),
                                round((right - left) / coordinate_width, 10),
                                round((bottom - top) / coordinate_height, 10),
                            ],
                        }
                    )
                    stats["person_detections"] = int(stats["person_detections"]) + 1

                record: dict[str, object] = {
                    "schema_version": SCHEMA_VERSION,
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "image_width": image_width,
                    "image_height": image_height,
                    "detections": detections,
                }
                if fps is not None:
                    record["timestamp_ns"] = round(frame_index * 1_000_000_000 / fps)
                if source_uri is not None:
                    record["source_uri"] = source_uri
                if model_id is not None:
                    record["model_id"] = model_id
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument(
        "--coordinate-width",
        type=int,
        help="KITTI/mux width when different from original JSON image width",
    )
    parser.add_argument(
        "--coordinate-height",
        type=int,
        help="KITTI/mux height when different from original JSON image height",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--app-index", type=int, default=0)
    parser.add_argument("--source-id", type=int, default=0)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument(
        "--caviar-ground-truth",
        type=Path,
        help="export only frame numbers explicitly present in this CAVIAR XML",
    )
    parser.add_argument("--fps", type=float)
    parser.add_argument("--source-uri")
    parser.add_argument("--model-id")
    parser.add_argument("--stats-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (args.coordinate_width is None) != (args.coordinate_height is None):
            raise ValueError("coordinate width and height must be supplied together")
        include_frames = (
            caviar_frame_indices(args.caviar_ground_truth)
            if args.caviar_ground_truth
            else None
        )
        stats = convert_kitti_directory(
            args.kitti_dir,
            args.output,
            sequence_id=args.sequence_id,
            image_width=args.image_width,
            image_height=args.image_height,
            labels_path=args.labels,
            coordinate_width=args.coordinate_width,
            coordinate_height=args.coordinate_height,
            app_index=args.app_index,
            source_id=args.source_id,
            expected_frames=args.expected_frames,
            include_frames=include_frames,
            fps=args.fps,
            source_uri=args.source_uri,
            model_id=args.model_id,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    rendered = json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
