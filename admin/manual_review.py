"""Fail-closed administration service for sampled open-video overlay reviews.

The service intentionally has no inference or pipeline controls.  Read-only
source and asset manifests describe what may be reviewed; only human decisions
and an append-only audit trail are written below ``DEEPSAFE_DATA``.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import re
import sqlite3
import stat
import threading
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from validation import open_video_manual_review as review_contract
from validation.strict_json import loads as strict_json_loads


ASSET_SCHEMA_VERSION = "deepsafe.open-video-manual-assets/v1"
AUDIT_SCHEMA_VERSION = "deepsafe.open-video-manual-review-audit/v1"
EXPORT_SCHEMA_VERSION = review_contract.OVERLAY_SCHEMA_VERSION
TERMINAL_STATUSES = {"reviewed", "ambiguous", "excluded"}
IMAGE_KINDS = {"source_image", "overlay_image"}
REQUIRED_EVIDENCE_KINDS = {"overlay_image", "review_report", "predictions"}
BOUND_ASSET_KINDS = REQUIRED_EVIDENCE_KINDS | {"source_image"}
ALLOWED_KINDS = IMAGE_KINDS | {"review_report", "predictions"}
ALLOWED_MEDIA_TYPES = {
    "source_image": {"image/png"},
    "overlay_image": {"image/png"},
    "review_report": {"application/json"},
    "predictions": {"application/x-ndjson"},
}
ASSET_ID_RE = re.compile(r"^a_[0-9a-f]{64}$")
BOUND_ID_RE = re.compile(r"^b_[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
DEFAULT_MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_AUDIT_EVENTS = 100_000
_DATABASE_INIT_LOCK = threading.Lock()
_INITIALIZED_DATABASES: set[str] = set()
_EXPORT_LOCK = threading.Lock()

AUDIT_EVENT_FIELDS = {
    "schema_version",
    "decision_id",
    "revision",
    "recorded_at",
    "reviewer_id",
    "reviewer_type",
    "source_record_sha256",
    "source_file_sha256",
    "asset_index_sha256",
    "evidence_asset_ids",
    "decision_row_sha256",
    "previous_event_sha256",
    "decision_row",
    "event_sha256",
}
AUDIT_TRIGGER_SQL = {
    "audit_events_no_update": (
        "create trigger audit_events_no_update before update on audit_events begin "
        "select raise(abort, 'audit events are immutable'); end"
    ),
    "audit_events_no_delete": (
        "create trigger audit_events_no_delete before delete on audit_events begin "
        "select raise(abort, 'audit events are immutable'); end"
    ),
    "decision_ledger_no_update": (
        "create trigger decision_ledger_no_update before update on decision_ledger begin "
        "select raise(abort, 'decision ledger is immutable'); end"
    ),
    "decision_ledger_no_delete": (
        "create trigger decision_ledger_no_delete before delete on decision_ledger begin "
        "select raise(abort, 'decision ledger is immutable'); end"
    ),
}


class ManualReviewError(RuntimeError):
    """A safe, externally mappable manual-review failure."""

    def __init__(self, state: str, *, status_code: int = 503):
        super().__init__(state)
        self.state = state
        self.status_code = status_code


@dataclass(frozen=True)
class ManualReviewConfig:
    source_path: Path
    asset_index_path: Path
    asset_root: Path
    data_dir: Path
    max_asset_bytes: int

    @classmethod
    def from_env(cls) -> "ManualReviewConfig":
        try:
            maximum = int(
                os.getenv(
                    "DEEPSAFE_MANUAL_REVIEW_MAX_ASSET_BYTES",
                    str(DEFAULT_MAX_ASSET_BYTES),
                )
            )
        except ValueError as exc:
            raise ManualReviewError("invalid_configuration") from exc
        if not 1 <= maximum <= 128 * 1024 * 1024:
            raise ManualReviewError("invalid_configuration")
        data_root = Path(os.getenv("DEEPSAFE_DATA", "/data"))
        return cls(
            source_path=Path(
                os.getenv(
                    "DEEPSAFE_MANUAL_REVIEW_SOURCE",
                    "/workspace/manual-review/source-frame-reviews-v1.jsonl",
                )
            ),
            asset_index_path=Path(
                os.getenv(
                    "DEEPSAFE_MANUAL_REVIEW_ASSET_INDEX",
                    "/workspace/validation-results/open-video-review/manual-assets/index.json",
                )
            ),
            asset_root=Path(
                os.getenv(
                    "DEEPSAFE_MANUAL_REVIEW_ASSET_ROOT",
                    "/workspace/validation-results",
                )
            ),
            data_dir=data_root / "manual-review",
            max_asset_bytes=maximum,
        )


@dataclass(frozen=True)
class ReviewContext:
    sources: dict[str, dict[str, Any]]
    assets: dict[str, dict[str, Any]]
    assets_by_decision: dict[str, dict[str, dict[str, Any]]]
    source_file_sha256: str
    asset_index_sha256: str


@dataclass(frozen=True)
class AssetContent:
    content: bytes
    media_type: str
    filename: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path, *, maximum: int) -> str:
    digest = hashlib.sha256()
    total = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ManualReviewError("artifact_unavailable")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(chunk)
            if total > maximum:
                raise ManualReviewError("artifact_unavailable")
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or total != before.st_size:
        raise ManualReviewError("artifact_unavailable")
    return digest.hexdigest()


def _read_bounded(path: Path, maximum: int = MAX_MANIFEST_BYTES) -> bytes:
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise ManualReviewError("artifact_unavailable")
            content = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if (
            len(content) > maximum
            or len(content) != before.st_size
            or identity_before != identity_after
        ):
            raise ManualReviewError("artifact_unavailable")
        return content
    except (OSError, ValueError) as exc:
        raise ManualReviewError("artifact_unavailable") from exc


def _safe_text(value: object, *, maximum: int = 240) -> str:
    if not isinstance(value, str):
        raise ManualReviewError("invalid_source_contract")
    stripped = value.strip()
    if not stripped or len(stripped) > maximum or CONTROL_RE.search(stripped):
        raise ManualReviewError("invalid_source_contract")
    return stripped


def _strict_nonnegative_int(value: object, state: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManualReviewError(state)
    return value


def _validate_range(value: object) -> dict[str, int | None]:
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        raise ManualReviewError("invalid_source_contract")
    minimum = _strict_nonnegative_int(value.get("min"), "invalid_source_contract")
    maximum = value.get("max")
    if maximum is not None:
        maximum = _strict_nonnegative_int(maximum, "invalid_source_contract")
        if maximum < minimum:
            raise ManualReviewError("invalid_source_contract")
    return {"min": minimum, "max": maximum}


def _source_projection(source: dict[str, Any]) -> dict[str, Any]:
    observation = source["observation"]
    window = source["window"]
    frame = source["frame"]
    return {
        "source_review_id": source["record_id"],
        "scene_id": source["scene_id"],
        "frame": {
            "index": frame["index"],
            "timestamp_seconds": frame["timestamp_seconds"],
        },
        "window": {
            "segment_label": window["segment_label"],
            "segment_role": window["segment_role"],
            "negative_window": window["negative_window"],
        },
        "observation": {
            "person_expected": observation["person_expected"],
            "visible_person_count_range": observation[
                "visible_person_count_range"
            ],
            "scorable_person_count_range": observation[
                "scorable_person_count_range"
            ],
            "dominant_scale": observation.get("dominant_scale"),
            "occlusion": observation.get("occlusion"),
            "top_view": observation.get("top_view") is True,
            "high_oblique": observation.get("high_oblique") is True,
            "medium_close": observation.get("medium_close") is True,
            "partial_body_only": observation.get("partial_body_only") is True,
        },
    }


def _load_sources(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    raw = _read_bounded(path)
    sources: dict[str, dict[str, Any]] = {}
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ManualReviewError("invalid_source_contract") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            source = strict_json_loads(line)
        except json.JSONDecodeError as exc:
            raise ManualReviewError("invalid_source_contract") from exc
        if not isinstance(source, dict):
            raise ManualReviewError("invalid_source_contract")
        if source.get("schema_version") != review_contract.SOURCE_SCHEMA_VERSION:
            raise ManualReviewError("invalid_source_contract")
        source_id = _safe_text(source.get("record_id"), maximum=180)
        scene_id = _safe_text(source.get("scene_id"), maximum=180)
        if any(character in source_id + scene_id for character in ("/", "\\", ":")):
            raise ManualReviewError("invalid_source_contract")
        if source_id in sources:
            raise ManualReviewError("invalid_source_contract")
        if not isinstance(source.get("sensitive"), bool):
            raise ManualReviewError("invalid_source_contract")
        frame = source.get("frame")
        observation = source.get("observation")
        window = source.get("window")
        if not isinstance(frame, dict) or not isinstance(observation, dict):
            raise ManualReviewError("invalid_source_contract")
        if window is None:
            # A materialized bundle may contain the intentionally minimal
            # source/frame/observation projection. Keep that distinction
            # explicit instead of inventing scene semantics.
            source = dict(source)
            window = {
                "segment_label": "sampled-frame",
                "segment_role": "unscored",
                "negative_window": False,
            }
            source["window"] = window
        elif not isinstance(window, dict):
            raise ManualReviewError("invalid_source_contract")
        _strict_nonnegative_int(frame.get("index"), "invalid_source_contract")
        timestamp = frame.get("timestamp_seconds")
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
            or timestamp < 0
        ):
            raise ManualReviewError("invalid_source_contract")
        if source["sensitive"]:
            # Sensitive records are deliberately excluded before they can enter
            # any projection or decision lookup.
            continue
        observation["visible_person_count_range"] = _validate_range(
            observation.get("visible_person_count_range")
        )
        observation["scorable_person_count_range"] = _validate_range(
            observation.get("scorable_person_count_range")
        )
        if observation.get("person_expected") not in review_contract.PERSON_EXPECTED:
            raise ManualReviewError("invalid_source_contract")
        scales = observation.get("scale_classes")
        if (
            not isinstance(scales, list)
            or len(scales) != len(set(scales))
            or not set(scales) <= review_contract.SCALE_CLASSES
        ):
            raise ManualReviewError("invalid_source_contract")
        dominant = observation.get("dominant_scale")
        if dominant is not None and dominant not in scales:
            raise ManualReviewError("invalid_source_contract")
        if observation.get("occlusion") not in review_contract.OCCLUSION_LEVELS:
            raise ManualReviewError("invalid_source_contract")
        for flag in ("top_view", "high_oblique", "medium_close", "partial_body_only"):
            if not isinstance(observation.get(flag), bool):
                raise ManualReviewError("invalid_source_contract")
        for key in ("segment_label", "segment_role"):
            _safe_text(window.get(key), maximum=240)
        if window.get("segment_role") not in review_contract.SEGMENT_ROLES:
            raise ManualReviewError("invalid_source_contract")
        if not isinstance(window.get("negative_window"), bool):
            raise ManualReviewError("invalid_source_contract")
        sources[source_id] = source
    if not sources:
        raise ManualReviewError("invalid_source_contract")
    return sources, _hash_bytes(raw)


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value:
        raise ManualReviewError("invalid_asset_index")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ManualReviewError("invalid_asset_index")
    if CONTROL_RE.search(value):
        raise ManualReviewError("invalid_asset_index")
    return value


def _load_asset_index(
    path: Path,
    sources: dict[str, dict[str, Any]],
    maximum: int,
    source_file_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]], str]:
    raw = _read_bounded(path)
    try:
        payload = strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManualReviewError("invalid_asset_index") from exc
    top_fields = {
        "schema_version",
        "status",
        "bundle_id",
        "decision_count",
        "asset_count",
        "profiles",
        "review_confidence_threshold",
        "sensitive_media_included",
        "source_records_sha256",
        "campaign_plan_sha256",
        "decisions",
        "assets",
        "input_provenance",
        "metric_guardrail",
    }
    threshold = payload.get("review_confidence_threshold") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != top_fields
        or payload.get("schema_version") != ASSET_SCHEMA_VERSION
        or payload.get("status") != "complete"
        or not isinstance(payload.get("bundle_id"), str)
        or not BOUND_ID_RE.fullmatch(payload["bundle_id"])
        or payload.get("decision_count") != 42
        or payload.get("asset_count") != 168
        or payload.get("profiles") != [640, 960]
        or not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
        or payload.get("sensitive_media_included") is not False
        or payload.get("source_records_sha256") != source_file_sha256
        or not isinstance(payload.get("campaign_plan_sha256"), str)
        or not SHA256_RE.fullmatch(payload["campaign_plan_sha256"])
        or not isinstance(payload.get("metric_guardrail"), str)
        or "not dense ground truth" not in payload["metric_guardrail"].lower()
    ):
        raise ManualReviewError("invalid_asset_index")
    provenance = payload.get("input_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {"videos", "jobs"}:
        raise ManualReviewError("invalid_asset_index")
    videos, jobs = provenance.get("videos"), provenance.get("jobs")
    if not isinstance(videos, dict) or len(videos) > 100 or not isinstance(jobs, list) or len(jobs) > 200:
        raise ManualReviewError("invalid_asset_index")
    for video_id, video in videos.items():
        if (
            not isinstance(video_id, str)
            or not video_id
            or not isinstance(video, dict)
            or set(video) != {"sha256", "size_bytes"}
            or not isinstance(video.get("sha256"), str)
            or not SHA256_RE.fullmatch(video["sha256"])
            or not isinstance(video.get("size_bytes"), int)
            or isinstance(video.get("size_bytes"), bool)
            or video["size_bytes"] < 1
        ):
            raise ManualReviewError("invalid_asset_index")
    job_fields = {
        "job_id",
        "scene_id",
        "model_input",
        "predictions_sha256",
        "gpu_guard_sha256",
        "run_manifest_sha256",
        "conversion_sha256",
    }
    for job in jobs:
        if (
            not isinstance(job, dict)
            or set(job) != job_fields
            or not isinstance(job.get("job_id"), str)
            or not job["job_id"]
            or not isinstance(job.get("scene_id"), str)
            or not job["scene_id"]
            or job.get("model_input") not in (640, 960)
            or any(
                not isinstance(job.get(field), str)
                or not SHA256_RE.fullmatch(job[field])
                for field in (
                    "predictions_sha256",
                    "gpu_guard_sha256",
                    "run_manifest_sha256",
                    "conversion_sha256",
                )
            )
        ):
            raise ManualReviewError("invalid_asset_index")
    assets = payload.get("assets")
    decision_contracts = payload.get("decisions")
    if (
        not isinstance(assets, dict)
        or len(assets) > 5000
        or not isinstance(decision_contracts, list)
    ):
        raise ManualReviewError("invalid_asset_index")
    required = {
        "decision_id",
        "source_review_id",
        "scene_id",
        "frame_index",
        "model_input",
        "kind",
        "relative_path",
        "sha256",
        "media_type",
        "size_bytes",
    }
    result: dict[str, dict[str, Any]] = {}
    by_decision: dict[str, dict[str, dict[str, Any]]] = {}
    for key, value in assets.items():
        if not isinstance(key, str) or not ASSET_ID_RE.fullmatch(key):
            raise ManualReviewError("invalid_asset_index")
        if not isinstance(value, dict) or set(value) != required:
            raise ManualReviewError("invalid_asset_index")
        source_id = value.get("source_review_id")
        if source_id not in sources:
            # This also fails closed if a manifest tries to reintroduce a
            # sensitive or otherwise unavailable source.
            raise ManualReviewError("invalid_asset_index")
        source = sources[source_id]
        profile = value.get("model_input")
        if profile not in (640, 960):
            raise ManualReviewError("invalid_asset_index")
        decision_id = f"{source_id}:{profile}"
        kind = value.get("kind")
        if (
            value.get("decision_id") != decision_id
            or value.get("scene_id") != source["scene_id"]
            or value.get("frame_index") != source["frame"]["index"]
            or kind not in ALLOWED_KINDS
            or value.get("media_type") not in ALLOWED_MEDIA_TYPES[kind]
        ):
            raise ManualReviewError("invalid_asset_index")
        relative_path = _validate_relative_path(value.get("relative_path"))
        digest = value.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ManualReviewError("invalid_asset_index")
        size = value.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= maximum:
            raise ManualReviewError("invalid_asset_index")
        item = dict(value) | {
            "asset_id": key,
            "relative_path": relative_path,
            "profile": profile,
            "frame": value["frame_index"],
            "size": size,
        }
        if kind in by_decision.setdefault(decision_id, {}):
            raise ManualReviewError("invalid_asset_index")
        by_decision[decision_id][kind] = item
        result[key] = item

    expected_decisions = {
        f"{source_id}:{profile}"
        for source_id in sources
        for profile in (640, 960)
    }
    if len(expected_decisions) != 42 or len(decision_contracts) != 42:
        raise ManualReviewError("invalid_asset_index")
    seen_decisions: set[str] = set()
    decision_fields = {
        "decision_id",
        "source_review_id",
        "scene_id",
        "frame_index",
        "model_input",
        "source_observation",
        "evidence",
    }
    for raw_decision in decision_contracts:
        if not isinstance(raw_decision, dict) or set(raw_decision) != decision_fields:
            raise ManualReviewError("invalid_asset_index")
        source_id = raw_decision.get("source_review_id")
        profile = raw_decision.get("model_input")
        if source_id not in sources or profile not in (640, 960):
            raise ManualReviewError("invalid_asset_index")
        source = sources[source_id]
        decision_id = f"{source_id}:{profile}"
        if (
            raw_decision.get("decision_id") != decision_id
            or decision_id in seen_decisions
            or raw_decision.get("scene_id") != source["scene_id"]
            or raw_decision.get("frame_index") != source["frame"]["index"]
            or raw_decision.get("source_observation") != source["observation"]
        ):
            raise ManualReviewError("invalid_asset_index")
        evidence = raw_decision.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != {
            "source_image",
            "overlay_image",
            "review_report",
            "predictions",
        }:
            raise ManualReviewError("invalid_asset_index")
        for kind, asset_id in evidence.items():
            if not isinstance(asset_id, str):
                raise ManualReviewError("invalid_asset_index")
            asset = result.get(asset_id)
            if (
                asset is None
                or asset["decision_id"] != decision_id
                or asset["kind"] != kind
            ):
                raise ManualReviewError("invalid_asset_index")
        if set(by_decision.get(decision_id, {})) != set(evidence):
            raise ManualReviewError("invalid_asset_index")
        seen_decisions.add(decision_id)
    if seen_decisions != expected_decisions:
        raise ManualReviewError("invalid_asset_index")
    return result, by_decision, _hash_bytes(raw)


def _resolve_asset(root: Path, relative_path: str) -> Path:
    """Resolve a regular file without permitting any symlink component."""

    try:
        if root.is_symlink() or not root.is_dir():
            raise ManualReviewError("asset_unavailable", status_code=404)
        resolved_root = root.resolve(strict=True)
        current = resolved_root
        for part in PurePosixPath(relative_path).parts:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ManualReviewError("asset_unavailable", status_code=404)
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
        if not resolved.is_file():
            raise ManualReviewError("asset_unavailable", status_code=404)
        return resolved
    except ManualReviewError:
        raise
    except (OSError, ValueError) as exc:
        raise ManualReviewError("asset_unavailable", status_code=404) from exc


def _verify_asset(config: ManualReviewConfig, asset: dict[str, Any]) -> Path:
    path = _resolve_asset(config.asset_root, asset["relative_path"])
    try:
        size = path.stat().st_size
        if size != asset["size"] or size > config.max_asset_bytes:
            raise ManualReviewError("asset_unavailable", status_code=404)
        if _hash_file(path, maximum=config.max_asset_bytes) != asset["sha256"]:
            raise ManualReviewError("asset_unavailable", status_code=404)
    except (OSError, ManualReviewError) as exc:
        raise ManualReviewError("asset_unavailable", status_code=404) from exc
    return path


def _verify_image_signature(path: Path, media_type: str) -> None:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
    except OSError as exc:
        raise ManualReviewError("asset_unavailable", status_code=404) from exc
    if media_type == "image/png" and prefix != b"\x89PNG\r\n\x1a\n":
        raise ManualReviewError("asset_unavailable", status_code=404)
    if media_type == "image/jpeg" and not prefix.startswith(b"\xff\xd8"):
        raise ManualReviewError("asset_unavailable", status_code=404)


def _source_row_sha256(source: dict[str, Any]) -> str:
    return _hash_bytes(_canonical_bytes(source))


def _evidence_assets(
    config: ManualReviewConfig,
    context: ReviewContext,
    decision_id: str,
) -> dict[str, dict[str, Any]]:
    assets = context.assets_by_decision.get(decision_id, {})
    if not BOUND_ASSET_KINDS <= set(assets):
        raise ManualReviewError("evidence_not_ready", status_code=409)
    selected = {kind: assets[kind] for kind in BOUND_ASSET_KINDS}
    for asset in selected.values():
        _verify_asset(config, asset)
    _verify_image_signature(
        _resolve_asset(config.asset_root, selected["overlay_image"]["relative_path"]),
        selected["overlay_image"]["media_type"],
    )
    _verify_image_signature(
        _resolve_asset(config.asset_root, selected["source_image"]["relative_path"]),
        selected["source_image"]["media_type"],
    )
    return selected


def _evidence_object(assets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "review_report_path": assets["review_report"]["relative_path"],
        "review_report_sha256": assets["review_report"]["sha256"],
        "overlay_image_path": assets["overlay_image"]["relative_path"],
        "overlay_image_sha256": assets["overlay_image"]["sha256"],
        "predictions_path": assets["predictions"]["relative_path"],
        "predictions_sha256": assets["predictions"]["sha256"],
    }


def _asset_projection(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": asset["asset_id"],
        "kind": asset["kind"],
        "media_type": asset["media_type"],
        "href": f"/api/manual-review/assets/{asset['asset_id']}",
    }


def _etag(revision: int) -> str:
    return f'"mr-r{revision}"'


def parse_if_match(value: str | None) -> int:
    if value is None:
        raise ManualReviewError("if_match_required", status_code=428)
    match = re.fullmatch(r'"mr-r(0|[1-9][0-9]{0,17})"', value.strip())
    if not match:
        raise ManualReviewError("invalid_if_match", status_code=400)
    return int(match.group(1))


@contextmanager
def _export_file_lock(data_dir: Path):
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        with (data_dir / ".export.lock").open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except ManualReviewError:
        raise
    except OSError as exc:
        raise ManualReviewError("decision_export_unavailable") from exc


class ManualReviewService:
    def __init__(self, config: ManualReviewConfig | None = None):
        self.config = config or ManualReviewConfig.from_env()

    @property
    def database_path(self) -> Path:
        return self.config.data_dir / "manual-review.sqlite3"

    def _context(self) -> ReviewContext:
        sources, source_hash = _load_sources(self.config.source_path)
        assets, by_decision, index_hash = _load_asset_index(
            self.config.asset_index_path,
            sources,
            self.config.max_asset_bytes,
            source_hash,
        )
        return ReviewContext(sources, assets, by_decision, source_hash, index_hash)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA synchronous=FULL")
            database_key = str(self.database_path.resolve())
            with _DATABASE_INIT_LOCK:
                schema_present = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'current_decisions'"
                ).fetchone()
                if database_key not in _INITIALIZED_DATABASES or schema_present is None:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("PRAGMA synchronous=FULL")
                    connection.executescript(
                        """
                CREATE TABLE IF NOT EXISTS current_decisions (
                    decision_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL CHECK (revision > 0),
                    row_json TEXT NOT NULL,
                    row_sha256 TEXT NOT NULL,
                    source_record_sha256 TEXT NOT NULL,
                    evidence_asset_ids_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision > 0),
                    recorded_at TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    event_sha256 TEXT NOT NULL,
                    previous_event_sha256 TEXT,
                    UNIQUE(decision_id, revision)
                );
                CREATE TABLE IF NOT EXISTS decision_ledger (
                    decision_id TEXT PRIMARY KEY,
                    first_event_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT OR IGNORE INTO decision_ledger
                    (decision_id, first_event_sha256, created_at)
                    SELECT decision_id, event_sha256, recorded_at
                    FROM audit_events WHERE revision = 1;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                BEFORE UPDATE ON audit_events BEGIN
                    SELECT RAISE(ABORT, 'audit events are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                BEFORE DELETE ON audit_events BEGIN
                    SELECT RAISE(ABORT, 'audit events are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS decision_ledger_no_update
                BEFORE UPDATE ON decision_ledger BEGIN
                    SELECT RAISE(ABORT, 'decision ledger is immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS decision_ledger_no_delete
                BEFORE DELETE ON decision_ledger BEGIN
                    SELECT RAISE(ABORT, 'decision ledger is immutable');
                END;
                        """
                    )
                    _INITIALIZED_DATABASES.add(database_key)
            return connection
        except (OSError, sqlite3.Error) as exc:
            raise ManualReviewError("decision_store_unavailable") from exc

    @staticmethod
    def _load_json(value: str, state: str) -> Any:
        try:
            return strict_json_loads(value)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManualReviewError(state) from exc

    def _verify_audit_chain(
        self,
        connection: sqlite3.Connection,
        context: ReviewContext,
        current_rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        """Verify the append-only log and its exact binding to current rows."""

        triggers = {
            row["name"]: (row["sql"] or "").lower()
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        normalized_triggers = {
            name: " ".join(sql.split()).rstrip(";") for name, sql in triggers.items()
        }
        if any(
            normalized_triggers.get(name) != sql
            for name, sql in AUDIT_TRIGGER_SQL.items()
        ):
            raise ManualReviewError("invalid_decision_store")

        count_row = connection.execute(
            "SELECT COUNT(*) AS event_count FROM audit_events"
        ).fetchone()
        if (
            count_row is None
            or not isinstance(count_row["event_count"], int)
            or count_row["event_count"] > MAX_AUDIT_EVENTS
        ):
            raise ManualReviewError("invalid_decision_store")
        audit_rows = connection.execute(
            "SELECT * FROM audit_events ORDER BY decision_id, revision"
        ).fetchall()
        if len(audit_rows) != count_row["event_count"]:
            raise ManualReviewError("invalid_decision_store")

        heads: dict[str, dict[str, Any]] = {}
        first_events: dict[str, dict[str, Any]] = {}
        next_revision: dict[str, int] = {}
        previous_hashes: dict[str, str | None] = {}
        verified_events: list[dict[str, Any]] = []
        for stored in audit_rows:
            event = self._load_json(stored["event_json"], "invalid_decision_store")
            if (
                not isinstance(event, dict)
                or set(event) != AUDIT_EVENT_FIELDS
                or event.get("schema_version") != AUDIT_SCHEMA_VERSION
            ):
                raise ManualReviewError("invalid_decision_store")
            if _canonical_bytes(event).decode("utf-8") != stored["event_json"]:
                raise ManualReviewError("invalid_decision_store")

            decision_id = event.get("decision_id")
            source_id, separator, profile_text = (
                decision_id.rpartition(":") if isinstance(decision_id, str) else ("", "", "")
            )
            revision = event.get("revision")
            if (
                not separator
                or source_id not in context.sources
                or profile_text not in {"640", "960"}
                or not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision != next_revision.get(decision_id, 1)
                or stored["decision_id"] != decision_id
                or stored["revision"] != revision
            ):
                raise ManualReviewError("invalid_decision_store")

            claimed_hash = event.get("event_sha256")
            unsigned_event = dict(event)
            unsigned_event.pop("event_sha256", None)
            if (
                not isinstance(claimed_hash, str)
                or not SHA256_RE.fullmatch(claimed_hash)
                or _hash_bytes(_canonical_bytes(unsigned_event)) != claimed_hash
                or stored["event_sha256"] != claimed_hash
            ):
                raise ManualReviewError("invalid_decision_store")

            expected_previous = previous_hashes.get(decision_id)
            if (
                event.get("previous_event_sha256") != expected_previous
                or stored["previous_event_sha256"] != expected_previous
            ):
                raise ManualReviewError("invalid_decision_store")

            recorded_at = event.get("recorded_at")
            reviewer_id = event.get("reviewer_id")
            try:
                parsed_time = datetime.fromisoformat(recorded_at)
            except (TypeError, ValueError) as exc:
                raise ManualReviewError("invalid_decision_store") from exc
            if (
                not isinstance(recorded_at, str)
                or len(recorded_at) > 80
                or CONTROL_RE.search(recorded_at)
                or parsed_time.tzinfo is None
                or stored["recorded_at"] != recorded_at
                or not isinstance(reviewer_id, str)
                or not reviewer_id.strip()
                or reviewer_id != reviewer_id.strip()
                or len(reviewer_id) > 120
                or CONTROL_RE.search(reviewer_id)
                or event.get("reviewer_type") not in review_contract.HUMAN_REVIEWER_TYPES
            ):
                raise ManualReviewError("invalid_decision_store")

            source = context.sources[source_id]
            source_sha = _source_row_sha256(source)
            sha_bindings = {
                "source_record_sha256": source_sha,
                "source_file_sha256": context.source_file_sha256,
                "asset_index_sha256": context.asset_index_sha256,
            }
            if any(event.get(key) != value for key, value in sha_bindings.items()):
                raise ManualReviewError("invalid_decision_store")

            asset_ids = event.get("evidence_asset_ids")
            indexed = context.assets_by_decision.get(decision_id, {})
            if (
                not isinstance(asset_ids, dict)
                or set(asset_ids) != BOUND_ASSET_KINDS
                or any(
                    not isinstance(asset_id, str)
                    or not ASSET_ID_RE.fullmatch(asset_id)
                    or kind not in indexed
                    or indexed[kind]["asset_id"] != asset_id
                    for kind, asset_id in asset_ids.items()
                )
            ):
                raise ManualReviewError("invalid_decision_store")

            decision_row = event.get("decision_row")
            decision_row_sha = event.get("decision_row_sha256")
            review = decision_row.get("review") if isinstance(decision_row, dict) else None
            if (
                not isinstance(decision_row, dict)
                or not isinstance(decision_row_sha, str)
                or not SHA256_RE.fullmatch(decision_row_sha)
                or _hash_bytes(_canonical_bytes(decision_row)) != decision_row_sha
                or decision_row.get("decision_id") != decision_id
                or decision_row.get("source_review_id") != source_id
                or decision_row.get("scene_id") != source["scene_id"]
                or decision_row.get("frame_index") != source["frame"]["index"]
                or decision_row.get("model_input") != int(profile_text)
                or not isinstance(review, dict)
                or review.get("reviewer_id") != reviewer_id
                or review.get("reviewer_type") != event.get("reviewer_type")
                or review.get("reviewed_at") != recorded_at
                or decision_row.get("overlay_evidence")
                != _evidence_object({kind: indexed[kind] for kind in BOUND_ASSET_KINDS})
            ):
                raise ManualReviewError("invalid_decision_store")

            next_revision[decision_id] = revision + 1
            previous_hashes[decision_id] = claimed_hash
            heads[decision_id] = event
            if revision == 1:
                first_events[decision_id] = event
            verified_events.append(event)

        ledger_rows = connection.execute(
            "SELECT * FROM decision_ledger ORDER BY decision_id"
        ).fetchall()
        ledger_by_id = {row["decision_id"]: row for row in ledger_rows}
        if (
            len(ledger_by_id) != len(ledger_rows)
            or set(ledger_by_id) != set(first_events)
        ):
            raise ManualReviewError("invalid_decision_store")
        for decision_id, stored in ledger_by_id.items():
            first = first_events[decision_id]
            if (
                stored["first_event_sha256"] != first["event_sha256"]
                or stored["created_at"] != first["recorded_at"]
            ):
                raise ManualReviewError("invalid_decision_store")

        current_by_id = {row["decision_id"]: row for row in current_rows}
        if len(current_by_id) != len(current_rows) or set(current_by_id) != set(heads):
            raise ManualReviewError("invalid_decision_store")
        for decision_id, stored in current_by_id.items():
            head = heads[decision_id]
            asset_ids = self._load_json(
                stored["evidence_asset_ids_json"], "invalid_decision_store"
            )
            row = self._load_json(stored["row_json"], "invalid_decision_store")
            if (
                stored["revision"] != head["revision"]
                or stored["row_sha256"] != head["decision_row_sha256"]
                or row != head["decision_row"]
                or _canonical_bytes(row).decode("utf-8") != stored["row_json"]
                or _canonical_bytes(asset_ids).decode("utf-8")
                != stored["evidence_asset_ids_json"]
                or stored["source_record_sha256"] != head["source_record_sha256"]
                or asset_ids != head["evidence_asset_ids"]
                or stored["updated_at"] != head["recorded_at"]
            ):
                raise ManualReviewError("invalid_decision_store")
        return verified_events

    def _validate_current_rows(
        self,
        context: ReviewContext,
        rows: list[sqlite3.Row],
    ) -> dict[str, sqlite3.Row]:
        result: dict[str, sqlite3.Row] = {}
        for stored in rows:
            decision_id = stored["decision_id"]
            source_id, separator, profile_text = decision_id.rpartition(":")
            if not separator or source_id not in context.sources or profile_text not in {"640", "960"}:
                raise ManualReviewError("invalid_decision_store")
            source = context.sources[source_id]
            if stored["source_record_sha256"] != _source_row_sha256(source):
                raise ManualReviewError("source_binding_changed")
            row = self._load_json(stored["row_json"], "invalid_decision_store")
            if _hash_bytes(_canonical_bytes(row)) != stored["row_sha256"]:
                raise ManualReviewError("invalid_decision_store")
            asset_ids = self._load_json(
                stored["evidence_asset_ids_json"], "invalid_decision_store"
            )
            if not isinstance(asset_ids, dict) or set(asset_ids) != BOUND_ASSET_KINDS:
                raise ManualReviewError("invalid_decision_store")
            selected: dict[str, dict[str, Any]] = {}
            for kind, asset_id in asset_ids.items():
                asset = context.assets.get(asset_id)
                if (
                    asset is None
                    or asset["kind"] != kind
                    or asset["decision_id"] != decision_id
                ):
                    raise ManualReviewError("evidence_binding_changed")
                _verify_asset(self.config, asset)
                selected[kind] = asset
            _verify_image_signature(
                _resolve_asset(
                    self.config.asset_root,
                    selected["overlay_image"]["relative_path"],
                ),
                selected["overlay_image"]["media_type"],
            )
            _verify_image_signature(
                _resolve_asset(
                    self.config.asset_root,
                    selected["source_image"]["relative_path"],
                ),
                selected["source_image"]["media_type"],
            )
            if row.get("overlay_evidence") != _evidence_object(selected):
                raise ManualReviewError("evidence_binding_changed")
            try:
                review_contract.validate_overlay_decisions(
                    [source],
                    [row],
                    workspace_root=self.config.asset_root,
                    require_profiles=(int(profile_text),),
                    require_complete=True,
                )
            except review_contract.ReviewValidationError as exc:
                raise ManualReviewError("invalid_decision_store") from exc
            result[decision_id] = stored
        return result

    def _current_rows(self, context: ReviewContext) -> dict[str, sqlite3.Row]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT * FROM current_decisions ORDER BY decision_id"
            ).fetchall()
            self._verify_audit_chain(connection, context, rows)
            result = self._validate_current_rows(context, rows)
            connection.commit()
            return result
        except ManualReviewError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise ManualReviewError("decision_store_unavailable") from exc
        finally:
            connection.close()

    def _baseline_row(
        self,
        context: ReviewContext,
        source: dict[str, Any],
        profile: int,
    ) -> dict[str, Any]:
        row = review_contract.make_overlay_template([source], profiles=(profile,))[0]
        decision_assets = context.assets_by_decision.get(row["decision_id"], {})
        if BOUND_ASSET_KINDS <= set(decision_assets):
            selected = {kind: decision_assets[kind] for kind in REQUIRED_EVIDENCE_KINDS}
            row["overlay_evidence"] = _evidence_object(selected)
            row["decision"]["status"] = "pending_review"
            row["decision"]["reasons"] = [
                "Bound overlay evidence is ready for human review."
            ]
        return row

    def _project_decision(
        self,
        context: ReviewContext,
        source: dict[str, Any],
        profile: int,
        stored: sqlite3.Row | None,
    ) -> dict[str, Any]:
        decision_id = f"{source['record_id']}:{profile}"
        if stored is None:
            row = self._baseline_row(context, source, profile)
            revision = 0
        else:
            row = self._load_json(stored["row_json"], "invalid_decision_store")
            revision = stored["revision"]
        indexed = context.assets_by_decision.get(decision_id, {})
        image_assets = [
            _asset_projection(indexed[kind])
            for kind in ("source_image", "overlay_image")
            if kind in indexed
        ]
        decision = row["decision"]
        return {
            "decision_id": decision_id,
            "model_input": profile,
            "status": decision["status"],
            "revision": revision,
            "etag": _etag(revision),
            "evidence_ready": BOUND_ASSET_KINDS <= set(indexed),
            "assets": image_assets,
            "decision": {
                "status": decision["status"],
                "detection_count_reviewed": decision[
                    "detection_count_reviewed"
                ],
                "visible_person_count_confirmed": decision[
                    "visible_person_count_confirmed"
                ],
                "scorable_person_count_confirmed": decision[
                    "scorable_person_count_confirmed"
                ],
                "true_positive_count": decision["true_positive_count"],
                "false_positive_count": decision["false_positive_count"],
                "false_negative_count": decision["false_negative_count"],
                "ignored_detection_count": decision[
                    "ignored_detection_count"
                ],
                "unscorable_visible_person_count": decision[
                    "unscorable_visible_person_count"
                ],
                "reasons": decision["reasons"],
            },
        }

    def queue(self) -> dict[str, Any]:
        context = self._context()
        current = self._current_rows(context)
        items: list[dict[str, Any]] = []
        counts = {"pending_overlay": 0, "pending_review": 0, "reviewed": 0, "ambiguous": 0, "excluded": 0}
        for source in context.sources.values():
            profiles = [
                self._project_decision(
                    context,
                    source,
                    profile,
                    current.get(f"{source['record_id']}:{profile}"),
                )
                for profile in (640, 960)
            ]
            for profile in profiles:
                counts[profile["status"]] = counts.get(profile["status"], 0) + 1
            items.append(_source_projection(source) | {"profiles": profiles})
        return {
            "schema_version": "deepsafe.admin-open-video-manual-review/v1",
            "metric_guardrail": "Sampled qualitative review only; not dense ground truth and not a dataset-level metric.",
            "source_count": len(items),
            "decision_count": len(items) * 2,
            "status_counts": counts,
            "items": items,
        }

    def _detail_payload(
        self,
        context: ReviewContext,
        current: dict[str, sqlite3.Row],
        decision_id: str,
    ) -> tuple[dict[str, Any], str]:
        source_id, separator, profile_text = decision_id.rpartition(":")
        if not separator or source_id not in context.sources or profile_text not in {"640", "960"}:
            raise ManualReviewError("decision_not_found", status_code=404)
        source = context.sources[source_id]
        profiles = [
            self._project_decision(
                context,
                source,
                profile,
                current.get(f"{source_id}:{profile}"),
            )
            for profile in (640, 960)
        ]
        selected = next(item for item in profiles if item["model_input"] == int(profile_text))
        return (
            {
                "schema_version": "deepsafe.admin-open-video-manual-review-detail/v1",
                "source": _source_projection(source),
                "selected": selected,
                "comparison_profiles": profiles,
                "metric_guardrail": "Sampled qualitative review only; not dense ground truth and not a dataset-level metric.",
            },
            selected["etag"],
        )

    def detail(self, decision_id: str) -> tuple[dict[str, Any], str]:
        context = self._context()
        current = self._current_rows(context)
        return self._detail_payload(context, current, decision_id)

    def asset(self, asset_id: str) -> AssetContent:
        if not ASSET_ID_RE.fullmatch(asset_id):
            raise ManualReviewError("asset_unavailable", status_code=404)
        context = self._context()
        asset = context.assets.get(asset_id)
        if asset is None or asset["kind"] not in IMAGE_KINDS:
            raise ManualReviewError("asset_unavailable", status_code=404)
        path = _verify_asset(self.config, asset)
        _verify_image_signature(path, asset["media_type"])
        try:
            content = _read_bounded(path, self.config.max_asset_bytes)
        except (OSError, ManualReviewError) as exc:
            raise ManualReviewError("asset_unavailable", status_code=404) from exc
        if len(content) != asset["size"] or _hash_bytes(content) != asset["sha256"]:
            raise ManualReviewError("asset_unavailable", status_code=404)
        if asset["media_type"] == "image/png" and not content.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise ManualReviewError("asset_unavailable", status_code=404)
        if asset["media_type"] == "image/jpeg" and not content.startswith(b"\xff\xd8"):
            raise ManualReviewError("asset_unavailable", status_code=404)
        extension = "png" if asset["media_type"] == "image/png" else "jpg"
        return AssetContent(
            content=content,
            media_type=asset["media_type"],
            filename=f"{asset['kind']}-{asset_id}.{extension}",
        )

    def put(
        self,
        decision_id: str,
        *,
        expected_revision: int,
        reviewer_id: str,
        reviewer_type: str,
        decision: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        context = self._context()
        source_id, separator, profile_text = decision_id.rpartition(":")
        if not separator or source_id not in context.sources or profile_text not in {"640", "960"}:
            raise ManualReviewError("decision_not_found", status_code=404)
        if (
            not isinstance(reviewer_id, str)
            or not reviewer_id.strip()
            or len(reviewer_id.strip()) > 120
            or CONTROL_RE.search(reviewer_id.strip())
        ):
            raise ManualReviewError("invalid_reviewer", status_code=422)
        reviewer_id = reviewer_id.strip()
        if reviewer_type not in review_contract.HUMAN_REVIEWER_TYPES:
            raise ManualReviewError("invalid_reviewer", status_code=422)
        if decision.get("status") not in TERMINAL_STATUSES:
            raise ManualReviewError("invalid_decision", status_code=422)
        reasons = decision.get("reasons")
        if not isinstance(reasons, list) or len(reasons) > 16:
            raise ManualReviewError("invalid_decision", status_code=422)
        cleaned_reasons: list[str] = []
        for reason in reasons:
            if (
                not isinstance(reason, str)
                or not reason.strip()
                or len(reason.strip()) > 500
                or CONTROL_RE.search(reason.strip())
            ):
                raise ManualReviewError("invalid_decision", status_code=422)
            cleaned_reasons.append(reason.strip())
        decision = dict(decision)
        decision["reasons"] = cleaned_reasons
        source = context.sources[source_id]
        profile = int(profile_text)
        selected_assets = _evidence_assets(self.config, context, decision_id)
        row = review_contract.make_overlay_template([source], profiles=(profile,))[0]
        row["overlay_evidence"] = _evidence_object(selected_assets)
        row["decision"] = decision
        recorded_at = datetime.now(timezone.utc).isoformat()
        row["review"] = {
            "reviewer_id": reviewer_id,
            "reviewer_type": reviewer_type,
            "reviewed_at": recorded_at,
        }
        try:
            review_contract.validate_overlay_decisions(
                [source],
                [row],
                workspace_root=self.config.asset_root,
                require_profiles=(profile,),
                require_complete=True,
            )
        except review_contract.ReviewValidationError as exc:
            raise ManualReviewError("decision_invariant_failed", status_code=422) from exc

        row_bytes = _canonical_bytes(row)
        row_json = row_bytes.decode("utf-8")
        row_sha = _hash_bytes(row_bytes)
        source_sha = _source_row_sha256(source)
        asset_ids = {
            kind: selected_assets[kind]["asset_id"]
            for kind in sorted(BOUND_ASSET_KINDS)
        }
        asset_ids_json = _canonical_bytes(asset_ids).decode("utf-8")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current_rows = connection.execute(
                "SELECT * FROM current_decisions ORDER BY decision_id"
            ).fetchall()
            self._verify_audit_chain(connection, context, current_rows)
            validated_current = self._validate_current_rows(context, current_rows)
            existing = validated_current.get(decision_id)
            current_revision = 0 if existing is None else existing["revision"]
            if current_revision != expected_revision:
                connection.rollback()
                raise ManualReviewError("stale_revision", status_code=409)
            if existing is not None and existing["source_record_sha256"] != source_sha:
                connection.rollback()
                raise ManualReviewError("source_binding_changed")
            revision = current_revision + 1
            previous = connection.execute(
                "SELECT event_sha256 FROM audit_events WHERE decision_id = ? ORDER BY revision DESC LIMIT 1",
                (decision_id,),
            ).fetchone()
            previous_hash = None if previous is None else previous["event_sha256"]
            event = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "decision_id": decision_id,
                "revision": revision,
                "recorded_at": recorded_at,
                "reviewer_id": reviewer_id,
                "reviewer_type": reviewer_type,
                "source_record_sha256": source_sha,
                "source_file_sha256": context.source_file_sha256,
                "asset_index_sha256": context.asset_index_sha256,
                "evidence_asset_ids": asset_ids,
                "decision_row_sha256": row_sha,
                "previous_event_sha256": previous_hash,
                "decision_row": row,
            }
            event_sha = _hash_bytes(_canonical_bytes(event))
            event["event_sha256"] = event_sha
            event_json = _canonical_bytes(event).decode("utf-8")
            if existing is None:
                connection.execute(
                    """INSERT INTO decision_ledger
                       (decision_id, first_event_sha256, created_at)
                       VALUES (?, ?, ?)""",
                    (decision_id, event_sha, recorded_at),
                )
                connection.execute(
                    """INSERT INTO current_decisions
                       (decision_id, revision, row_json, row_sha256,
                        source_record_sha256, evidence_asset_ids_json, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        decision_id,
                        revision,
                        row_json,
                        row_sha,
                        source_sha,
                        asset_ids_json,
                        recorded_at,
                    ),
                )
            else:
                connection.execute(
                    """UPDATE current_decisions
                       SET revision = ?, row_json = ?, row_sha256 = ?,
                           source_record_sha256 = ?, evidence_asset_ids_json = ?,
                           updated_at = ?
                       WHERE decision_id = ?""",
                    (
                        revision,
                        row_json,
                        row_sha,
                        source_sha,
                        asset_ids_json,
                        recorded_at,
                        decision_id,
                    ),
                )
            connection.execute(
                """INSERT INTO audit_events
                   (decision_id, revision, recorded_at, event_json,
                    event_sha256, previous_event_sha256)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    decision_id,
                    revision,
                    recorded_at,
                    event_json,
                    event_sha,
                    previous_hash,
                ),
            )
            committed_rows = connection.execute(
                "SELECT * FROM current_decisions ORDER BY decision_id"
            ).fetchall()
            self._verify_audit_chain(connection, context, committed_rows)
            committed_current = self._validate_current_rows(context, committed_rows)
            response_payload, response_etag = self._detail_payload(
                context, committed_current, decision_id
            )
            connection.commit()
        except ManualReviewError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise ManualReviewError("decision_store_unavailable") from exc
        finally:
            connection.close()

        exports_current = True
        try:
            self._export_jsonl(context)
        except ManualReviewError:
            # SQLite is authoritative. A derived JSONL snapshot failure must
            # not turn a durable commit into an apparent failed write that an
            # operator may retry with a stale ETag.
            exports_current = False
        response_payload["persistence"] = {
            "decision_committed": True,
            "exports_current": exports_current,
        }
        return response_payload, response_etag

    @staticmethod
    def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ManualReviewError("decision_export_unavailable") from exc

    def _export_jsonl(self, context: ReviewContext) -> None:
        # SQLite serializes commits; the paired process/file locks serialize
        # deterministic JSONL snapshots across threads and web workers.
        with _EXPORT_LOCK, _export_file_lock(self.config.data_dir):
            connection = self._connect()
            try:
                connection.execute("BEGIN")
                rows = connection.execute(
                    "SELECT * FROM current_decisions ORDER BY decision_id"
                ).fetchall()
                self._verify_audit_chain(connection, context, rows)
                current = self._validate_current_rows(context, rows)
                events = [
                    self._load_json(row["event_json"], "invalid_decision_store")
                    for row in connection.execute(
                        "SELECT event_json FROM audit_events ORDER BY id"
                    ).fetchall()
                ]
                connection.commit()
            except ManualReviewError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise ManualReviewError("decision_store_unavailable") from exc
            finally:
                connection.close()

            decisions: list[dict[str, Any]] = []
            for source in context.sources.values():
                for profile in (640, 960):
                    decision_id = f"{source['record_id']}:{profile}"
                    stored = current.get(decision_id)
                    decisions.append(
                        self._baseline_row(context, source, profile)
                        if stored is None
                        else self._load_json(stored["row_json"], "invalid_decision_store")
                    )
            try:
                review_contract.validate_overlay_decisions(
                    list(context.sources.values()),
                    decisions,
                    workspace_root=self.config.asset_root,
                    require_profiles=(640, 960),
                    require_complete=False,
                )
            except review_contract.ReviewValidationError as exc:
                raise ManualReviewError("decision_export_invalid") from exc
            self._atomic_jsonl(self.config.data_dir / "overlay-decisions-v1.jsonl", decisions)
            self._atomic_jsonl(self.config.data_dir / "audit-events-v1.jsonl", events)
