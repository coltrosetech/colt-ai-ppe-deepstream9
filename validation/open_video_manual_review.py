#!/usr/bin/env python3
"""Versioned source-frame observations and qualitative overlay review QA.

This module deliberately does not run inference.  It validates source-only
visual observations, prepares pending 640/960 decision rows, checks human
overlay decisions, and merges the two evidence layers without turning a
sampled qualitative review into dataset-level metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.strict_json import loads as strict_json_loads  # noqa: E402


SOURCE_SCHEMA_VERSION = "deepsafe.open-video-source-frame-review/v1"
OVERLAY_SCHEMA_VERSION = "deepsafe.open-video-overlay-decision/v1"
MERGED_SCHEMA_VERSION = "deepsafe.open-video-frame-review-merged/v1"
QA_SCHEMA_VERSION = "deepsafe.open-video-manual-review-qa/v1"

DEFAULT_SCENE_MANIFEST = REPO_ROOT / "validation/open_video_review/scenes.json"
DEFAULT_SOURCE_RECORDS = (
    REPO_ROOT / "validation/open_video_review/source-frame-reviews-v1.jsonl"
)
DEFAULT_NORMALIZATION_REPORT = REPO_ROOT / "validation/results/open-video-normalization.tsv"

SOURCE_STATUSES = {
    "ai_reviewed_needs_human_qa",
    "human_verified",
    "needs_review",
    "closed_review",
    "rejected",
}
PERSON_EXPECTED = {"yes", "no", "uncertain", "not_reviewed"}
SCALE_CLASSES = {"small", "medium", "close"}
OCCLUSION_LEVELS = {"none", "light", "moderate", "heavy", "unknown"}
SEGMENT_ROLES = {
    "person_visible",
    "likely_empty",
    "partial_body_only",
    "non_content",
    "unscored",
    "closed_review",
}
NEGATIVE_WINDOW_ROLES = {"likely_empty", "partial_body_only", "non_content"}
OVERLAY_STATUSES = {
    "pending_overlay",
    "pending_review",
    "reviewed",
    "ambiguous",
    "excluded",
    "closed_review",
}
HUMAN_REVIEWER_TYPES = {"human", "human_with_ai_assist"}
EVIDENCE_HASH_FIELDS = {
    "review_report_path": "review_report_sha256",
    "overlay_image_path": "overlay_image_sha256",
    "predictions_path": "predictions_sha256",
}


class ReviewValidationError(ValueError):
    """Raised when a record violates the manual-review contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReviewValidationError(f"{context}: expected an object")
    return value


def _require_keys(value: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ReviewValidationError(f"{context}: missing fields {', '.join(missing)}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = strict_json_loads(raw)
            except json.JSONDecodeError as exc:
                raise ReviewValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ReviewValidationError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise ReviewValidationError(f"{path}: JSONL has no records")
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_path(value: str, workspace_root: Path, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReviewValidationError(f"{context}: path must be a non-empty string")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    try:
        resolved.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise ReviewValidationError(f"{context}: path leaves workspace: {value}") from exc
    return resolved


def contact_sheet_frame_index(tile_index: int, frame_count: int, tiles: int = 12) -> int:
    """Reconstruct the CFR input frame selected at the centre of an fps-filter bin."""

    if not _is_int(tile_index) or not 0 <= tile_index < tiles:
        raise ReviewValidationError(f"tile_index must be in [0, {tiles - 1}]")
    if not _is_int(frame_count) or frame_count <= 0 or tiles <= 0:
        raise ReviewValidationError("frame_count and tiles must be positive integers")
    return math.floor((tile_index + 0.5) * frame_count / tiles)


def _load_scene_contract(
    scene_manifest: Path = DEFAULT_SCENE_MANIFEST,
    normalization_report: Path = DEFAULT_NORMALIZATION_REPORT,
) -> dict[str, dict[str, Any]]:
    payload = strict_json_loads(scene_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "deepsafe.open-video-review-scenes/v1":
        raise ReviewValidationError(f"{scene_manifest}: unsupported scene schema")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        raise ReviewValidationError(f"{scene_manifest}: scenes must be an array")

    with normalization_report.open(encoding="utf-8", newline="") as handle:
        normalizations = {row["id"]: row for row in csv.DictReader(handle, delimiter="\t")}

    result: dict[str, dict[str, Any]] = {}
    for raw in scenes:
        scene = _require_object(raw, "scene")
        scene_id = scene.get("id")
        normalization_id = scene.get("normalization_id")
        if not isinstance(scene_id, str) or normalization_id not in normalizations:
            raise ReviewValidationError(f"{scene_manifest}: invalid scene/normalization id")
        proof = normalizations[normalization_id]
        frame_count = int(proof["output_frames"])
        if int(proof["source_frames"]) != frame_count:
            raise ReviewValidationError(f"{scene_id}: normalization did not preserve frames")
        fps_fraction = proof["fps"]
        fps = float(Fraction(fps_fraction))
        segments = scene.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ReviewValidationError(f"{scene_id}: no scene segments")
        result[scene_id] = {
            "id": scene_id,
            "sensitive": scene.get("sensitive") is True,
            "normalization_id": normalization_id,
            "frame_count": frame_count,
            "fps_fraction": fps_fraction,
            "fps": fps,
            "segments": segments,
        }
    return result


def _segment_at(scene: dict[str, Any], timestamp: float) -> dict[str, Any]:
    duration = scene["frame_count"] / scene["fps"]
    for raw in scene["segments"]:
        start = float(raw["start_seconds"])
        end = duration if raw.get("end_seconds") is None else float(raw["end_seconds"])
        if start <= timestamp < end:
            return raw
    return scene["segments"][-1]


def _validate_count_range(value: object, context: str, *, nullable: bool) -> tuple[int | None, int | None]:
    obj = _require_object(value, context)
    _require_keys(obj, ("min", "max"), context)
    minimum, maximum = obj["min"], obj["max"]
    if nullable and minimum is None and maximum is None:
        return None, None
    if not _is_int(minimum) or minimum < 0:
        raise ReviewValidationError(f"{context}.min: expected a non-negative integer")
    if maximum is not None and (not _is_int(maximum) or maximum < minimum):
        raise ReviewValidationError(f"{context}.max: expected null or integer >= min")
    return minimum, maximum


def _value_in_range(value: int, bounds: tuple[int | None, int | None]) -> bool:
    minimum, maximum = bounds
    if minimum is None:
        return False
    return value >= minimum and (maximum is None or value <= maximum)


def validate_source_records(
    records: Sequence[dict[str, Any]],
    *,
    scene_manifest: Path = DEFAULT_SCENE_MANIFEST,
    normalization_report: Path = DEFAULT_NORMALIZATION_REPORT,
    workspace_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate source-only observations and return a coverage summary."""

    scenes = _load_scene_contract(scene_manifest, normalization_report)
    ids: set[str] = set()
    scene_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    negative_count = 0
    closed_count = 0

    for index, raw in enumerate(records, 1):
        context = f"source record {index}"
        record = _require_object(raw, context)
        _require_keys(
            record,
            (
                "schema_version",
                "record_id",
                "scene_id",
                "sensitive",
                "frame",
                "evidence",
                "window",
                "observation",
                "review",
                "overlay_link",
                "metric_guardrail",
            ),
            context,
        )
        if record["schema_version"] != SOURCE_SCHEMA_VERSION:
            raise ReviewValidationError(f"{context}: unsupported schema_version")
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ReviewValidationError(f"{context}: record_id must be non-empty")
        if record_id in ids:
            raise ReviewValidationError(f"{context}: duplicate record_id {record_id}")
        ids.add(record_id)

        scene_id = record["scene_id"]
        if scene_id not in scenes:
            raise ReviewValidationError(f"{context}: unknown scene_id {scene_id!r}")
        scene = scenes[scene_id]
        if record["sensitive"] is not scene["sensitive"]:
            raise ReviewValidationError(f"{context}: sensitive flag disagrees with scene policy")

        frame = _require_object(record["frame"], f"{context}.frame")
        _require_keys(
            frame,
            (
                "index",
                "timestamp_seconds",
                "fps_fraction",
                "mapping_method",
                "mapping_uncertainty_frames",
            ),
            f"{context}.frame",
        )
        frame_index = frame["index"]
        if not _is_int(frame_index) or not 0 <= frame_index < scene["frame_count"]:
            raise ReviewValidationError(f"{context}.frame.index: outside decoded video")
        if frame["fps_fraction"] != scene["fps_fraction"]:
            raise ReviewValidationError(f"{context}.frame: FPS provenance mismatch")
        timestamp = frame["timestamp_seconds"]
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            raise ReviewValidationError(f"{context}.frame.timestamp_seconds: expected number")
        expected_timestamp = frame_index / scene["fps"]
        if not math.isclose(float(timestamp), expected_timestamp, abs_tol=1e-6):
            raise ReviewValidationError(
                f"{context}.frame: timestamp {timestamp} != frame/FPS {expected_timestamp:.9f}"
            )
        if not _is_int(frame["mapping_uncertainty_frames"]) or not 0 <= frame["mapping_uncertainty_frames"] <= 1:
            raise ReviewValidationError(f"{context}.frame: mapping uncertainty must be 0 or 1")

        evidence = _require_object(record["evidence"], f"{context}.evidence")
        _require_keys(
            evidence,
            ("kind", "contact_sheet_path", "tile_index", "tile_grid", "review_visibility"),
            f"{context}.evidence",
        )
        if scene["sensitive"]:
            if evidence["kind"] != "closed_review_policy_placeholder" or any(
                evidence[key] is not None for key in ("contact_sheet_path", "tile_index", "tile_grid")
            ):
                raise ReviewValidationError(
                    f"{context}: sensitive scene may not expose contact-sheet evidence"
                )
            if evidence["review_visibility"] != "closed":
                raise ReviewValidationError(f"{context}: sensitive evidence must remain closed")
            if frame["mapping_method"] != "closed_review_policy_placeholder_v1":
                raise ReviewValidationError(f"{context}: wrong closed-review frame mapping")
        else:
            if evidence["kind"] != "existing_contact_sheet_tile":
                raise ReviewValidationError(f"{context}: expected existing contact-sheet evidence")
            if evidence["review_visibility"] != "project_internal":
                raise ReviewValidationError(f"{context}: unexpected evidence visibility")
            if evidence["tile_grid"] != [4, 3]:
                raise ReviewValidationError(f"{context}: contact sheet grid must be [4,3]")
            tile_index = evidence["tile_index"]
            expected_frame = contact_sheet_frame_index(tile_index, scene["frame_count"])
            if frame_index != expected_frame:
                raise ReviewValidationError(
                    f"{context}: frame {frame_index} does not match contact-sheet tile "
                    f"{tile_index} ({expected_frame})"
                )
            if frame["mapping_method"] != "equal_interval_contact_sheet_midpoint_v1":
                raise ReviewValidationError(f"{context}: wrong contact-sheet frame mapping")
            sheet = _workspace_path(
                evidence["contact_sheet_path"], workspace_root, f"{context}.evidence.contact_sheet_path"
            )
            if not sheet.is_file():
                raise ReviewValidationError(f"{context}: contact sheet is missing: {sheet}")

        window = _require_object(record["window"], f"{context}.window")
        _require_keys(window, ("segment_label", "segment_role", "negative_window"), f"{context}.window")
        segment = _segment_at(scene, float(timestamp))
        if window["segment_label"] != segment.get("label") or window["segment_role"] != segment.get("role"):
            raise ReviewValidationError(f"{context}: source record does not match scene segment")
        if window["segment_role"] not in SEGMENT_ROLES:
            raise ReviewValidationError(f"{context}: invalid segment role")
        expected_negative = window["segment_role"] in NEGATIVE_WINDOW_ROLES
        if window["negative_window"] is not expected_negative:
            raise ReviewValidationError(f"{context}: negative_window disagrees with segment policy")

        observation = _require_object(record["observation"], f"{context}.observation")
        _require_keys(
            observation,
            (
                "person_expected",
                "visible_person_count_range",
                "scorable_person_count_range",
                "scale_classes",
                "dominant_scale",
                "occlusion",
                "top_view",
                "high_oblique",
                "medium_close",
                "partial_body_only",
                "ambiguity",
                "notes",
            ),
            f"{context}.observation",
        )
        if observation["person_expected"] not in PERSON_EXPECTED:
            raise ReviewValidationError(f"{context}: invalid person_expected value")
        visible = _validate_count_range(
            observation["visible_person_count_range"],
            f"{context}.observation.visible_person_count_range",
            nullable=scene["sensitive"],
        )
        scorable = _validate_count_range(
            observation["scorable_person_count_range"],
            f"{context}.observation.scorable_person_count_range",
            nullable=scene["sensitive"],
        )
        if visible[0] is not None and scorable[0] is not None:
            if scorable[0] > (visible[1] if visible[1] is not None else math.inf):
                raise ReviewValidationError(f"{context}: scorable minimum exceeds visible maximum")
            if scorable[1] is not None and visible[1] is not None and scorable[1] > visible[1]:
                raise ReviewValidationError(f"{context}: scorable maximum exceeds visible maximum")
        scales = observation["scale_classes"]
        if not isinstance(scales, list) or len(scales) != len(set(scales)) or not set(scales) <= SCALE_CLASSES:
            raise ReviewValidationError(f"{context}: invalid or duplicate scale_classes")
        dominant = observation["dominant_scale"]
        if dominant is not None and dominant not in scales:
            raise ReviewValidationError(f"{context}: dominant_scale must occur in scale_classes")
        if visible == (0, 0) and (scales or dominant is not None):
            raise ReviewValidationError(f"{context}: empty frame cannot have person scales")
        if visible[0] is not None and visible[0] > 0 and not scales:
            raise ReviewValidationError(f"{context}: visible people require a qualitative scale")
        if observation["occlusion"] not in OCCLUSION_LEVELS:
            raise ReviewValidationError(f"{context}: invalid occlusion")
        for flag in ("top_view", "high_oblique", "medium_close", "partial_body_only"):
            if not isinstance(observation[flag], bool):
                raise ReviewValidationError(f"{context}: {flag} must be boolean")
        for field in ("ambiguity", "notes"):
            if not isinstance(observation[field], list) or not all(
                isinstance(item, str) and item for item in observation[field]
            ):
                raise ReviewValidationError(f"{context}: {field} must contain non-empty strings")

        review = _require_object(record["review"], f"{context}.review")
        _require_keys(
            review, ("status", "reviewer_id", "reviewer_type", "reviewed_at"), f"{context}.review"
        )
        if review["status"] not in SOURCE_STATUSES:
            raise ReviewValidationError(f"{context}: invalid source reviewer status")
        if scene["sensitive"]:
            closed_count += 1
            if review["status"] != "closed_review" or observation["person_expected"] != "not_reviewed":
                raise ReviewValidationError(f"{context}: sensitive observation must remain unreviewed/closed")
            if visible != (None, None) or scorable != (None, None):
                raise ReviewValidationError(f"{context}: closed observation cannot expose counts")
            if any(review[field] is not None for field in ("reviewer_id", "reviewer_type", "reviewed_at")):
                raise ReviewValidationError(f"{context}: closed placeholder must not claim a reviewer")
        else:
            if review["status"] == "closed_review":
                raise ReviewValidationError(f"{context}: non-sensitive scene cannot be closed_review")
            if not all(isinstance(review[field], str) and review[field] for field in ("reviewer_id", "reviewer_type", "reviewed_at")):
                raise ReviewValidationError(f"{context}: reviewed source record lacks reviewer provenance")
            if observation["person_expected"] == "yes" and (scorable[0] is None or scorable[0] < 1):
                raise ReviewValidationError(f"{context}: person_expected=yes requires scorable person(s)")

        overlay = _require_object(record["overlay_link"], f"{context}.overlay_link")
        if overlay != {"status": "awaiting_model_overlays", "profiles": [640, 960]}:
            raise ReviewValidationError(f"{context}: source-only overlay guardrail was modified")
        if not isinstance(record["metric_guardrail"], str) or "not ground truth" not in record["metric_guardrail"].lower():
            raise ReviewValidationError(f"{context}: missing not-ground-truth metric guardrail")

        scene_counts[scene_id] += 1
        status_counts[review["status"]] += 1
        negative_count += bool(window["negative_window"])

    missing_scenes = sorted(set(scenes) - set(scene_counts))
    if missing_scenes:
        raise ReviewValidationError(f"source records do not cover scenes: {', '.join(missing_scenes)}")
    return {
        "schema_version": QA_SCHEMA_VERSION,
        "kind": "source_records",
        "valid": True,
        "record_count": len(records),
        "scene_count": len(scene_counts),
        "closed_review_records": closed_count,
        "negative_window_records": negative_count,
        "status_counts": dict(sorted(status_counts.items())),
        "records_per_scene": dict(sorted(scene_counts.items())),
        "metric_guardrail": "Qualitative sampled-frame observations only; no AP/precision/recall extrapolation.",
    }


def _blank_overlay_decision(source: dict[str, Any], model_input: int) -> dict[str, Any]:
    closed = source["sensitive"]
    return {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "decision_id": f"{source['record_id']}:{model_input}",
        "source_review_id": source["record_id"],
        "scene_id": source["scene_id"],
        "frame_index": source["frame"]["index"],
        "model_input": model_input,
        "review_visibility": "closed" if closed else "project_internal",
        "overlay_evidence": {
            "review_report_path": None,
            "review_report_sha256": None,
            "overlay_image_path": None,
            "overlay_image_sha256": None,
            "predictions_path": None,
            "predictions_sha256": None,
        },
        "decision": {
            "status": "closed_review" if closed else "pending_overlay",
            "detection_count_reviewed": None,
            "visible_person_count_confirmed": None,
            "scorable_person_count_confirmed": None,
            "true_positive_count": None,
            "false_positive_count": None,
            "false_negative_count": None,
            "ignored_detection_count": None,
            "unscorable_visible_person_count": None,
            "reasons": [
                "Sensitive media remains in the closed-review workflow."
                if closed
                else "Overlay has not been generated or reviewed yet."
            ],
        },
        "review": {
            "reviewer_id": None,
            "reviewer_type": None,
            "reviewed_at": None,
        },
        "metric_guardrail": (
            "A reviewed row is a qualitative sampled-frame decision, not dense ground truth; "
            "never extrapolate precision, recall or AP from this JSONL."
        ),
    }


def make_overlay_template(
    source_records: Sequence[dict[str, Any]],
    *,
    profiles: Sequence[int] = (640, 960),
    include_sensitive: bool = False,
) -> list[dict[str, Any]]:
    if not profiles or any(profile not in (640, 960) for profile in profiles):
        raise ReviewValidationError("profiles must be a non-empty subset of 640,960")
    if len(set(profiles)) != len(profiles):
        raise ReviewValidationError("profiles must be unique")
    rows = []
    for source in source_records:
        if source["sensitive"] and not include_sensitive:
            continue
        rows.extend(_blank_overlay_decision(source, profile) for profile in profiles)
    return rows


def _validate_evidence(
    evidence: dict[str, Any],
    *,
    context: str,
    workspace_root: Path,
    required: bool,
) -> None:
    expected_keys = set(EVIDENCE_HASH_FIELDS) | set(EVIDENCE_HASH_FIELDS.values())
    if set(evidence) != expected_keys:
        raise ReviewValidationError(f"{context}: unexpected/missing overlay evidence fields")
    for path_field, hash_field in EVIDENCE_HASH_FIELDS.items():
        path_value, hash_value = evidence[path_field], evidence[hash_field]
        if path_value is None and hash_value is None:
            if required:
                raise ReviewValidationError(f"{context}: {path_field} is required")
            continue
        if path_value is None or hash_value is None:
            raise ReviewValidationError(f"{context}: path/hash must be both null or both present")
        path = _workspace_path(path_value, workspace_root, f"{context}.{path_field}")
        if not path.is_file():
            raise ReviewValidationError(f"{context}: evidence file does not exist: {path}")
        if not isinstance(hash_value, str) or len(hash_value) != 64 or _sha256(path) != hash_value.lower():
            raise ReviewValidationError(f"{context}: SHA-256 mismatch for {path_field}")


def validate_overlay_decisions(
    source_records: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    workspace_root: Path = REPO_ROOT,
    require_profiles: Sequence[int] = (),
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate overlay decisions and cross-check every source/frame/profile link."""

    sources = {record["record_id"]: record for record in source_records}
    decision_ids: set[str] = set()
    keys: set[tuple[str, int]] = set()
    status_counts: Counter[str] = Counter()
    profile_counts: dict[int, Counter[str]] = defaultdict(Counter)

    for index, raw in enumerate(decisions, 1):
        context = f"overlay decision {index}"
        decision_row = _require_object(raw, context)
        _require_keys(
            decision_row,
            (
                "schema_version",
                "decision_id",
                "source_review_id",
                "scene_id",
                "frame_index",
                "model_input",
                "review_visibility",
                "overlay_evidence",
                "decision",
                "review",
                "metric_guardrail",
            ),
            context,
        )
        if decision_row["schema_version"] != OVERLAY_SCHEMA_VERSION:
            raise ReviewValidationError(f"{context}: unsupported schema_version")
        decision_id = decision_row["decision_id"]
        if not isinstance(decision_id, str) or not decision_id or decision_id in decision_ids:
            raise ReviewValidationError(f"{context}: duplicate/invalid decision_id")
        decision_ids.add(decision_id)
        source_id = decision_row["source_review_id"]
        if source_id not in sources:
            raise ReviewValidationError(f"{context}: unknown source_review_id {source_id!r}")
        source = sources[source_id]
        profile = decision_row["model_input"]
        if profile not in (640, 960):
            raise ReviewValidationError(f"{context}: model_input must be 640 or 960")
        key = (source_id, profile)
        if key in keys:
            raise ReviewValidationError(f"{context}: duplicate source/profile decision")
        keys.add(key)
        if decision_row["scene_id"] != source["scene_id"] or decision_row["frame_index"] != source["frame"]["index"]:
            raise ReviewValidationError(f"{context}: decision is linked to the wrong scene/frame")
        expected_visibility = "closed" if source["sensitive"] else "project_internal"
        if decision_row["review_visibility"] != expected_visibility:
            raise ReviewValidationError(f"{context}: review visibility violates source policy")

        value = _require_object(decision_row["decision"], f"{context}.decision")
        count_fields = (
            "detection_count_reviewed",
            "visible_person_count_confirmed",
            "scorable_person_count_confirmed",
            "true_positive_count",
            "false_positive_count",
            "false_negative_count",
            "ignored_detection_count",
            "unscorable_visible_person_count",
        )
        _require_keys(value, ("status", *count_fields, "reasons"), f"{context}.decision")
        status = value["status"]
        if status not in OVERLAY_STATUSES:
            raise ReviewValidationError(f"{context}: invalid overlay status")
        reasons = value["reasons"]
        if not isinstance(reasons, list) or not all(isinstance(item, str) and item for item in reasons):
            raise ReviewValidationError(f"{context}: reasons must contain non-empty strings")
        review = _require_object(decision_row["review"], f"{context}.review")
        _require_keys(review, ("reviewer_id", "reviewer_type", "reviewed_at"), f"{context}.review")
        evidence = _require_object(decision_row["overlay_evidence"], f"{context}.overlay_evidence")

        if source["sensitive"] and status != "closed_review":
            raise ReviewValidationError(f"{context}: sensitive source must remain closed_review")
        if not source["sensitive"] and status == "closed_review":
            raise ReviewValidationError(f"{context}: ordinary source cannot be closed_review")

        if status in {"pending_overlay", "closed_review"}:
            if any(value[field] is not None for field in count_fields):
                raise ReviewValidationError(f"{context}: pending/closed decision cannot contain counts")
            if any(review[field] is not None for field in review):
                raise ReviewValidationError(f"{context}: pending/closed decision cannot claim review")
            _validate_evidence(evidence, context=f"{context}.overlay_evidence", workspace_root=workspace_root, required=False)
            if any(evidence[field] is not None for field in evidence):
                raise ReviewValidationError(f"{context}: pending_overlay/closed_review evidence must be null")
        elif status == "pending_review":
            if any(value[field] is not None for field in count_fields):
                raise ReviewValidationError(f"{context}: pending_review cannot contain counts")
            if any(review[field] is not None for field in review):
                raise ReviewValidationError(f"{context}: pending_review cannot claim review")
            _validate_evidence(evidence, context=f"{context}.overlay_evidence", workspace_root=workspace_root, required=True)
        elif status in {"ambiguous", "excluded"}:
            if any(value[field] is not None for field in count_fields):
                raise ReviewValidationError(f"{context}: ambiguous/excluded row cannot assert counts")
            if not reasons:
                raise ReviewValidationError(f"{context}: ambiguous/excluded row needs a reason")
            if review["reviewer_type"] not in HUMAN_REVIEWER_TYPES or not all(
                isinstance(review[field], str) and review[field]
                for field in ("reviewer_id", "reviewed_at")
            ):
                raise ReviewValidationError(f"{context}: final exception needs human review provenance")
            _validate_evidence(evidence, context=f"{context}.overlay_evidence", workspace_root=workspace_root, required=True)
        else:
            assert status == "reviewed"
            for field in count_fields:
                if not _is_int(value[field]) or value[field] < 0:
                    raise ReviewValidationError(f"{context}.decision.{field}: expected non-negative integer")
            if value["detection_count_reviewed"] != (
                value["true_positive_count"]
                + value["false_positive_count"]
                + value["ignored_detection_count"]
            ):
                raise ReviewValidationError(f"{context}: detection TP+FP+ignored reconciliation failed")
            if value["scorable_person_count_confirmed"] != (
                value["true_positive_count"] + value["false_negative_count"]
            ):
                raise ReviewValidationError(f"{context}: scorable TP+FN reconciliation failed")
            if value["visible_person_count_confirmed"] != (
                value["scorable_person_count_confirmed"]
                + value["unscorable_visible_person_count"]
            ):
                raise ReviewValidationError(f"{context}: visible=scorable+unscorable reconciliation failed")
            source_observation = source["observation"]
            visible_bounds = (
                source_observation["visible_person_count_range"]["min"],
                source_observation["visible_person_count_range"]["max"],
            )
            scorable_bounds = (
                source_observation["scorable_person_count_range"]["min"],
                source_observation["scorable_person_count_range"]["max"],
            )
            if not _value_in_range(value["visible_person_count_confirmed"], visible_bounds):
                raise ReviewValidationError(
                    f"{context}: confirmed visible count is outside source review range; revise source record"
                )
            if not _value_in_range(value["scorable_person_count_confirmed"], scorable_bounds):
                raise ReviewValidationError(
                    f"{context}: confirmed scorable count is outside source review range; revise source record"
                )
            if review["reviewer_type"] not in HUMAN_REVIEWER_TYPES or not all(
                isinstance(review[field], str) and review[field]
                for field in ("reviewer_id", "reviewed_at")
            ):
                raise ReviewValidationError(f"{context}: FP/FN decisions require a human reviewer")
            _validate_evidence(evidence, context=f"{context}.overlay_evidence", workspace_root=workspace_root, required=True)

        if not isinstance(decision_row["metric_guardrail"], str) or "not dense ground truth" not in decision_row["metric_guardrail"].lower():
            raise ReviewValidationError(f"{context}: missing sampled-review metric guardrail")
        status_counts[status] += 1
        profile_counts[profile][status] += 1

    required = tuple(require_profiles)
    if any(profile not in (640, 960) for profile in required) or len(set(required)) != len(required):
        raise ReviewValidationError("require_profiles must contain unique values from 640,960")
    eligible = [record for record in source_records if not record["sensitive"]]
    missing: list[str] = []
    incomplete: list[str] = []
    terminal = {"reviewed", "ambiguous", "excluded"}
    for source in eligible:
        for profile in required:
            key = (source["record_id"], profile)
            if key not in keys:
                missing.append(f"{source['record_id']}:{profile}")
                continue
            if require_complete:
                row = next(
                    item
                    for item in decisions
                    if item["source_review_id"] == source["record_id"] and item["model_input"] == profile
                )
                if row["decision"]["status"] not in terminal:
                    incomplete.append(f"{source['record_id']}:{profile}")
    if missing:
        raise ReviewValidationError(f"missing required profile decisions: {', '.join(missing[:12])}")
    if incomplete:
        raise ReviewValidationError(f"non-terminal required decisions: {', '.join(incomplete[:12])}")

    return {
        "schema_version": QA_SCHEMA_VERSION,
        "kind": "overlay_decisions",
        "valid": True,
        "decision_count": len(decisions),
        "eligible_source_record_count": len(eligible),
        "required_profiles": list(required),
        "require_complete": require_complete,
        "status_counts": dict(sorted(status_counts.items())),
        "per_profile": {
            str(profile): dict(sorted(counts.items()))
            for profile, counts in sorted(profile_counts.items())
        },
        "metric_guardrail": "Sampled qualitative decisions only; no dataset-level metrics.",
    }


def merge_reviews(
    source_records: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    profiles: Sequence[int] = (640, 960),
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        grouped[decision["source_review_id"]].append(decision)
    merged: list[dict[str, Any]] = []
    for source in source_records:
        overlays = sorted(grouped.get(source["record_id"], []), key=lambda row: row["model_input"])
        by_profile = {row["model_input"]: row for row in overlays}
        if source["sensitive"]:
            comparison_status = "closed_review"
        elif any(profile not in by_profile for profile in profiles):
            comparison_status = "awaiting_overlays"
        else:
            statuses = {by_profile[profile]["decision"]["status"] for profile in profiles}
            if statuses == {"reviewed"}:
                comparison_status = "ready_for_sampled_640_960_comparison"
            elif statuses <= {"reviewed", "ambiguous", "excluded"}:
                comparison_status = "reviewed_with_exceptions"
            elif statuses == {"pending_overlay"}:
                comparison_status = "awaiting_overlays"
            else:
                comparison_status = "partial_review"
        merged.append(
            {
                "schema_version": MERGED_SCHEMA_VERSION,
                "source_review": source,
                "overlay_reviews": overlays,
                "comparison_status": comparison_status,
                "metric_guardrail": (
                    "This merge preserves sampled qualitative evidence only; it is not dense "
                    "ground truth and must not produce AP, precision or recall."
                ),
            }
        )
    return merged


def _profiles(value: str) -> tuple[int, ...]:
    try:
        profiles = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("profiles must be comma-separated integers") from exc
    if not profiles or len(set(profiles)) != len(profiles) or any(item not in (640, 960) for item in profiles):
        raise argparse.ArgumentTypeError("profiles must be a unique subset of 640,960")
    return profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_source = subparsers.add_parser("validate-source", help="Validate source-frame JSONL")
    validate_source.add_argument("--source", type=Path, default=DEFAULT_SOURCE_RECORDS)

    template = subparsers.add_parser("template", help="Create pending 640/960 overlay decisions")
    template.add_argument("--source", type=Path, default=DEFAULT_SOURCE_RECORDS)
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--profiles", type=_profiles, default=(640, 960))
    template.add_argument("--include-sensitive", action="store_true")
    template.add_argument("--force", action="store_true")

    qa = subparsers.add_parser("qa-overlays", help="Cross-check overlay decisions against source rows")
    qa.add_argument("--source", type=Path, default=DEFAULT_SOURCE_RECORDS)
    qa.add_argument("--decisions", type=Path, required=True)
    qa.add_argument("--require-profiles", type=_profiles, default=())
    qa.add_argument("--require-complete", action="store_true")

    merge = subparsers.add_parser("merge", help="Merge source and overlay JSONL after QA")
    merge.add_argument("--source", type=Path, default=DEFAULT_SOURCE_RECORDS)
    merge.add_argument("--decisions", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--profiles", type=_profiles, default=(640, 960))
    merge.add_argument("--require-complete", action="store_true")
    merge.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = _read_jsonl(args.source)
        source_qa = validate_source_records(source)
        if args.command == "validate-source":
            print(json.dumps(source_qa, indent=2, ensure_ascii=False))
            return 0
        if args.command == "template":
            if args.output.exists() and not args.force:
                raise ReviewValidationError(f"refusing to overwrite {args.output}; use --force")
            decisions = make_overlay_template(
                source, profiles=args.profiles, include_sensitive=args.include_sensitive
            )
            _write_jsonl_atomic(args.output, decisions)
            print(
                json.dumps(
                    {
                        "schema_version": QA_SCHEMA_VERSION,
                        "kind": "overlay_template",
                        "output": str(args.output),
                        "decision_count": len(decisions),
                        "profiles": list(args.profiles),
                        "sensitive_included": args.include_sensitive,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        decisions = _read_jsonl(args.decisions)
        required = args.require_profiles if args.command == "qa-overlays" else args.profiles
        qa = validate_overlay_decisions(
            source,
            decisions,
            require_profiles=required,
            require_complete=args.require_complete,
        )
        if args.command == "qa-overlays":
            print(json.dumps(qa, indent=2, ensure_ascii=False))
            return 0
        if args.output.exists() and not args.force:
            raise ReviewValidationError(f"refusing to overwrite {args.output}; use --force")
        merged = merge_reviews(source, decisions, profiles=args.profiles)
        _write_jsonl_atomic(args.output, merged)
        print(
            json.dumps(
                {
                    **qa,
                    "kind": "merged_reviews",
                    "output": str(args.output),
                    "merged_record_count": len(merged),
                    "created_at": utc_now(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ReviewValidationError, json.JSONDecodeError) as exc:
        print(f"manual review validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
