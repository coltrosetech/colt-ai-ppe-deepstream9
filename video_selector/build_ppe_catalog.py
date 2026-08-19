#!/usr/bin/env python3
"""Build the selector PPE catalog from the reviewed candidate manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


CAMERA_LABELS = {
    "eye_level": "Göz hizası",
    "eye_level_rear": "Göz hizası · arkadan",
}

FRAMING_LABELS = {
    "wide": "geniş",
    "medium_wide": "orta-geniş",
    "medium": "orta",
    "medium_full": "orta · tam vücut",
    "full": "tam vücut",
}

SCENARIO_LABELS = {
    "dense_rebar_construction": "Donatı sahası",
    "crane_construction_crew": "Vinçli şantiye",
    "concrete_frame_construction": "Betonarme şantiye",
    "masonry_crew": "Duvar ekibi",
    "construction_management": "Şantiye yönetimi",
    "construction_plan_review": "Plan inceleme",
    "truck_crane_maintenance": "Araç bakımı",
    "rebar_work": "Donatı işi",
    "dense_construction_crew": "Kalabalık şantiye",
    "street_maintenance": "Yol bakımı",
    "rail_maintenance": "Ray bakımı",
    "brick_transport": "Tuğla taşıma",
    "busy_building_site": "Yoğun şantiye",
    "construction_site_inspection": "Saha denetimi",
    "warehouse_forklift": "Depo · forklift",
    "construction_supervision": "Şantiye gözetimi",
    "construction_inspection_pair": "Şantiye denetimi",
    "warehouse_inventory": "Depo · raf kontrolü",
    "warehouse_checklist": "Depo · kontrol",
    "rebar_installation": "Donatı montajı",
}


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    duration = float(payload["format"]["duration"])
    raw_frame_count = stream.get("nb_frames")
    frame_count = (
        int(raw_frame_count)
        if raw_frame_count not in {None, "N/A"}
        else round(duration * fps)
    )
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_seconds": duration,
        "frame_count": frame_count,
    }


def build_catalog(manifest_path: Path, raw_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != (
        "colt-ai.ppe-better-video-shortlist/v1"
    ):
        raise RuntimeError("PPE shortlist manifest schema is invalid")

    videos: list[dict[str, Any]] = []
    for asset in manifest["assets"]:
        source_id = str(asset["id"])
        video_id = source_id.removeprefix("ppe-").upper()
        source_path = raw_root / f"{source_id}.mp4"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        probe = _probe(source_path)

        declared_media = asset["media"]
        if (
            probe["width"] != int(declared_media["width"])
            or probe["height"] != int(declared_media["height"])
        ):
            raise RuntimeError(f"{source_id} media dimensions do not match manifest")

        ppe_preview = asset["ppe_preview"]
        helmet_tag = (
            "Kask ihlali adayı"
            if ppe_preview["helmet"] in {"absent", "weak_or_absent"}
            else "Kask"
        )
        hi_vis_tag = (
            "Hi-vis yok"
            if ppe_preview["high_visibility"] == "absent"
            else "Hi-vis"
        )
        camera = CAMERA_LABELS.get(
            asset["camera_angle"],
            str(asset["camera_angle"]).replace("_", " "),
        )
        framing = FRAMING_LABELS.get(
            asset["framing"],
            str(asset["framing"]).replace("_", " "),
        )

        videos.append(
            {
                "video_id": video_id,
                "title": asset["title"],
                "scene": ppe_preview["note"],
                "camera": f"{camera} · {framing}",
                "tags": [
                    SCENARIO_LABELS.get(asset["scenario"], asset["scenario"]),
                    helmet_tag,
                    hi_vis_tag,
                    f"Öncelik {asset['priority']}",
                ],
                "media_filename": f"{video_id}.mp4",
                "poster_filename": f"{video_id}.jpg",
                **probe,
                "processing_source_path": (
                    f"content/ppe-candidate-gallery-r2/raw/{source_id}.mp4"
                ),
                "ground_truth_path": None,
                "source_url": asset["page_url"],
                "license": "Pexels",
            }
        )

    return {
        "schema_version": "colt-ai.video-catalog/v1",
        "catalog_revision": "ppe-20260725-r2",
        "videos": videos,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/ppe-better-video-shortlist-20260725.json"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("content/ppe-candidate-gallery-r2/raw"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video_selector/ppe-catalog.json"),
    )
    args = parser.parse_args()

    payload = build_catalog(args.manifest, args.raw_root)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
