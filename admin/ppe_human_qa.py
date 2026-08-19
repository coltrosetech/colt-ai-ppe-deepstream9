"""Fail-closed PPE R6 human-QA administration service.

Only the immutable, curated R6 packet metadata and its rendered PNG tiles are
read.  Raw candidate/source images and labels are deliberately never opened.
Human decisions are represented by an append-only, hash-chained SQLite audit
log below ``DEEPSAFE_DATA``; current state is replayed from that log.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from validation.strict_json import loads as strict_json_loads


PACKET_RECEIPT_SHA256 = "87c62ea4ba3515ad549d30a32c9a628d68909f37ca3b372b06fddd44fa071325"
PACKET_RECEIPT_FILE = {
    "bytes": 4961,
    "sha256": "f34218294ada32965206326c508dbab7613a1233fac66d2be63605a0c3eafc55",
}
SAMPLES_FILE = {
    "bytes": 1_401_636,
    "sha256": "bfff611a251e56f43916a218adbc99ef63d3f11d30bcc2f0d49dce71698e7942",
}
MANIFEST_FILE = {
    "bytes": 105_808,
    "sha256": "0c8af4c5e13ada15fbfb63de6938c5c32ac8f6f96a4e990621dfc26079c18a0b",
}
SCHEMA_PINS = {
    "ppe-human-qa-adjudication-r6-v1.schema.json": {
        "bytes": 4813,
        "sha256": "2f0e8d8f4715dbb824b3e7e9fe8020fb1d440f937dee6076446fe61dfbaf6388",
    },
    "ppe-human-qa-packet-receipt-r6-v1.schema.json": {
        "bytes": 6702,
        "sha256": "17a16498bdc18b5c057b5650515c81114a827a08637bfb5c4d6995f3dec083f3",
    },
    "ppe-human-qa-sample-r6-v1.schema.json": {
        "bytes": 5737,
        "sha256": "aa62ec2caa8155d3a47b5c324be33b3a74ea8cb91f49890acf007f5e9cf1581d",
    },
}
PACKET_LOGICAL_PATH = (
    "validation/results/ppe/human-qa/"
    "mendeley-ppe-four-class-r6/receipt.json"
)
AUDIT_SCHEMA_VERSION = "deepsafe.ppe-human-qa-r6-audit/v1"
EXPORT_SCHEMA_VERSION = "deepsafe.ppe-human-qa-adjudication-r6/v1"
REVIEW_ID = "mendeley-ppe-four-class-r6-admin-review"
AUTHORIZATION_EFFECT = (
    "none_human_qa_only_new_exact_training_authorization_required"
)
EXPECTED_SAMPLES = 718
EXPECTED_POLICIES = 6
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_AUDIT_EVENTS = 100_000
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAMPLE_ID_RE = re.compile(r"^ppe-r6-[a-z0-9-]+-[0-9]{4}$")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_DATABASE_LOCK = threading.Lock()

POLICY_NAMES = (
    "head_zone_helmet_worn_candidate_policy",
    "retained_helmet_false_worn_risk",
    "hi_vis_quarantine_and_retained_semantics",
    "runtime_no_hi_vis_class_absent",
    "unresolved_person_ppe_absence_unknown",
    "dataset_rights_and_ultralytics_license_basis",
)
SAMPLE_DECISIONS = {
    "approve_sample_semantics",
    "reject_sample_semantics",
    "uncertain_needs_relabel",
}
REASON_CODES = {
    "visually_correct",
    "held_or_carried_helmet",
    "helmet_not_on_associated_head",
    "head_zone_policy_mismatch",
    "harness_not_hi_vis",
    "ordinary_garment_not_hi_vis",
    "no_helmet_box_wrong",
    "quarantine_should_be_retained",
    "retained_should_be_quarantined",
    "unknown_must_not_be_negative_ground_truth",
    "image_too_ambiguous",
    "other",
}
CATEGORIES = {
    "helmet_head_zone_boundary_retained",
    "helmet_head_zone_boundary_quarantined_below",
    "helmet_worn_candidate_random",
    "hi_vis_worn_candidate_random",
    "quarantine_reason_stratified",
    "no_helmet_explicit_random",
    "candidate_zero_label_context_all",
}
AUDIT_EVENT_FIELDS = {
    "schema_version",
    "sequence",
    "entity_type",
    "entity_id",
    "entity_revision",
    "recorded_at",
    "reviewer_identity",
    "packet_receipt_sha256",
    "packet_file_sha256",
    "decision_row",
    "decision_row_sha256",
    "previous_event_sha256",
    "event_sha256",
}

CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    entity_type TEXT NOT NULL CHECK (entity_type IN ('sample', 'policy')),
    entity_id TEXT NOT NULL,
    entity_revision INTEGER NOT NULL CHECK (entity_revision > 0),
    recorded_at TEXT NOT NULL,
    event_json TEXT NOT NULL,
    event_sha256 TEXT NOT NULL,
    previous_event_sha256 TEXT,
    UNIQUE(entity_type, entity_id, entity_revision)
)
"""
CREATE_NO_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS ppe_r6_audit_no_update
BEFORE UPDATE ON audit_events BEGIN
    SELECT RAISE(ABORT, 'PPE R6 audit events are immutable');
END
"""
CREATE_NO_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS ppe_r6_audit_no_delete
BEFORE DELETE ON audit_events BEGIN
    SELECT RAISE(ABORT, 'PPE R6 audit events are immutable');
END
"""


class PpeHumanQaError(RuntimeError):
    """Safe PPE review failure that can be mapped to an HTTP response."""

    def __init__(self, state: str, *, status_code: int = 503):
        super().__init__(state)
        self.state = state
        self.status_code = status_code


@dataclass(frozen=True)
class PpeHumanQaConfig:
    packet_root: Path
    schema_root: Path
    data_dir: Path
    max_tile_bytes: int

    @classmethod
    def from_env(cls) -> "PpeHumanQaConfig":
        try:
            maximum = int(
                os.getenv("DEEPSAFE_PPE_HUMAN_QA_MAX_TILE_BYTES", "4194304")
            )
        except ValueError as exc:
            raise PpeHumanQaError("invalid_configuration") from exc
        if not 1024 <= maximum <= 32 * 1024 * 1024:
            raise PpeHumanQaError("invalid_configuration")
        data_root = Path(os.getenv("DEEPSAFE_DATA", "/data"))
        return cls(
            packet_root=Path(
                os.getenv(
                    "DEEPSAFE_PPE_HUMAN_QA_PACKET",
                    "/workspace/validation-results/ppe/human-qa/"
                    "mendeley-ppe-four-class-r6",
                )
            ),
            schema_root=Path(
                os.getenv("DEEPSAFE_VALIDATION_SCHEMA_ROOT", "/app/validation/schemas")
            ),
            data_dir=data_root / "ppe-human-qa-r6",
            max_tile_bytes=maximum,
        )


@dataclass(frozen=True)
class PacketContext:
    receipt: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    samples_by_id: dict[str, dict[str, Any]]
    sample_hashes: dict[str, str]
    manifest_by_path: dict[str, dict[str, Any]]
    export_validator: Draft202012Validator


@dataclass(frozen=True)
class AuditState:
    heads: dict[tuple[str, str], dict[str, Any]]
    event_count: int
    reviewer_identity: str | None
    complete: bool
    completed_at: str | None
    chain_head_sha256: str | None


@dataclass(frozen=True)
class TileContent:
    content: bytes
    filename: str
    media_type: str = "image/png"


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PpeHumanQaError("invalid_json")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)


def _canonical_bytes(value: Any) -> bytes:
    _reject_nonfinite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or ".." in candidate.parts
        or "." in candidate.parts
        or candidate.as_posix() != relative
    ):
        raise PpeHumanQaError("packet_integrity_failed")
    try:
        resolved_root = root.resolve(strict=True)
        lexical = Path(os.path.abspath(resolved_root / Path(*candidate.parts)))
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise PpeHumanQaError("packet_integrity_failed") from exc
    if resolved != lexical or resolved_root not in resolved.parents:
        raise PpeHumanQaError("packet_integrity_failed")
    return resolved


def _stable_read(path: Path, *, expected: Mapping[str, Any], maximum: int) -> bytes:
    try:
        expected_bytes = expected.get("bytes")
        expected_sha = expected.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or expected_bytes > maximum
            or not isinstance(expected_sha, str)
            or not HEX_SHA256.fullmatch(expected_sha)
        ):
            raise PpeHumanQaError("packet_integrity_failed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_bytes
            ):
                raise PpeHumanQaError("packet_integrity_failed")
            content = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
        current = path.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (
            len(content) != expected_bytes
            or len(content) > maximum
            or identity
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or identity
            != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
            or after.st_nlink != 1
            or current.st_nlink != 1
            or _sha256(content) != expected_sha
        ):
            raise PpeHumanQaError("packet_integrity_failed")
        return content
    except PpeHumanQaError:
        raise
    except (OSError, ValueError) as exc:
        raise PpeHumanQaError("packet_integrity_failed") from exc


def _strict_object(payload: bytes, *, state: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PpeHumanQaError(state) from exc
    if not isinstance(value, dict):
        raise PpeHumanQaError(state)
    return value


def _canonical_jsonl(payload: bytes, *, state: str) -> list[dict[str, Any]]:
    if not payload.endswith(b"\n") or payload == b"\n":
        raise PpeHumanQaError(state)
    rows: list[dict[str, Any]] = []
    for line in payload[:-1].split(b"\n"):
        if not line:
            raise PpeHumanQaError(state)
        row = _strict_object(line, state=state)
        if _canonical_bytes(row) != line:
            raise PpeHumanQaError(state)
        rows.append(row)
    return rows


def _schema_validator(config: PpeHumanQaConfig, filename: str) -> Draft202012Validator:
    pin = SCHEMA_PINS[filename]
    path = _safe_path(config.schema_root, filename)
    payload = _stable_read(path, expected=pin, maximum=64 * 1024)
    schema = _strict_object(payload, state="schema_integrity_failed")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise PpeHumanQaError("schema_integrity_failed") from exc
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _validate_with(
    validator: Draft202012Validator, value: Mapping[str, Any], *, state: str
) -> None:
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        raise PpeHumanQaError(state)


def _unsigned_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(receipt))
    value.pop("receipt_sha256", None)
    return _sha256(_canonical_bytes(value))


def _load_packet(config: PpeHumanQaConfig) -> PacketContext:
    receipt_validator = _schema_validator(
        config, "ppe-human-qa-packet-receipt-r6-v1.schema.json"
    )
    sample_validator = _schema_validator(
        config, "ppe-human-qa-sample-r6-v1.schema.json"
    )
    export_validator = _schema_validator(
        config, "ppe-human-qa-adjudication-r6-v1.schema.json"
    )

    receipt_path = _safe_path(config.packet_root, "receipt.json")
    receipt_payload = _stable_read(
        receipt_path, expected=PACKET_RECEIPT_FILE, maximum=MAX_METADATA_BYTES
    )
    receipt = _strict_object(receipt_payload, state="packet_integrity_failed")
    if receipt_payload != _canonical_bytes(receipt) + b"\n":
        raise PpeHumanQaError("packet_integrity_failed")
    _validate_with(receipt_validator, receipt, state="packet_integrity_failed")
    if (
        receipt.get("receipt_sha256") != PACKET_RECEIPT_SHA256
        or _unsigned_receipt_sha256(receipt) != PACKET_RECEIPT_SHA256
        or receipt.get("selection", {}).get("samples")
        != {"path": "samples.jsonl", **SAMPLES_FILE}
        or receipt.get("artifacts", {}).get("manifest")
        != {"path": "artifact-manifest.jsonl", **MANIFEST_FILE}
    ):
        raise PpeHumanQaError("packet_integrity_failed")

    samples_payload = _stable_read(
        _safe_path(config.packet_root, "samples.jsonl"),
        expected=SAMPLES_FILE,
        maximum=MAX_METADATA_BYTES,
    )
    samples = _canonical_jsonl(samples_payload, state="packet_integrity_failed")
    if len(samples) != EXPECTED_SAMPLES:
        raise PpeHumanQaError("packet_integrity_failed")
    sample_ids: list[str] = []
    record_ids: list[str] = []
    sample_hashes: dict[str, str] = {}
    for sample in samples:
        _validate_with(sample_validator, sample, state="packet_integrity_failed")
        serialized = _canonical_bytes(sample)
        if b"development_holdout" in serialized.lower():
            raise PpeHumanQaError("excluded_payload_reference")
        sample_id = sample.get("sample_id")
        record_id = sample.get("record_id")
        if not isinstance(sample_id, str) or not isinstance(record_id, str):
            raise PpeHumanQaError("packet_integrity_failed")
        sample_ids.append(sample_id)
        record_ids.append(record_id)
        sample_hashes[sample_id] = _sha256(serialized)
    if len(set(sample_ids)) != EXPECTED_SAMPLES or len(set(record_ids)) != EXPECTED_SAMPLES:
        raise PpeHumanQaError("packet_integrity_failed")
    if Counter(row["category"] for row in samples) != Counter(
        receipt["selection"]["categories"]
    ) or Counter(row["role"] for row in samples) != Counter(
        receipt["selection"]["roles"]
    ):
        raise PpeHumanQaError("packet_integrity_failed")

    manifest_payload = _stable_read(
        _safe_path(config.packet_root, "artifact-manifest.jsonl"),
        expected=MANIFEST_FILE,
        maximum=MAX_METADATA_BYTES,
    )
    manifest = _canonical_jsonl(manifest_payload, state="packet_integrity_failed")
    if len(manifest) != 763:
        raise PpeHumanQaError("packet_integrity_failed")
    manifest_by_path: dict[str, dict[str, Any]] = {}
    for pin in manifest:
        if set(pin) != {"path", "bytes", "sha256"}:
            raise PpeHumanQaError("packet_integrity_failed")
        relative = pin.get("path")
        size = pin.get("bytes")
        digest = pin.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or not isinstance(digest, str)
            or not HEX_SHA256.fullmatch(digest)
            or not relative.startswith(("tiles/", "contact-sheets/"))
            or relative in manifest_by_path
        ):
            raise PpeHumanQaError("packet_integrity_failed")
        # Resolve without opening.  Symlink traversal or a missing artifact is
        # already enough to close the packet; bytes are re-hashed on serving.
        _safe_path(config.packet_root, relative)
        manifest_by_path[relative] = pin
    tile_paths: set[str] = set()
    for sample in samples:
        for kind in ("tile", "contact_sheet"):
            pin = sample["evidence"][kind]
            if manifest_by_path.get(pin["path"]) != pin:
                raise PpeHumanQaError("packet_integrity_failed")
        if sample["evidence"]["tile"]["path"] in tile_paths:
            raise PpeHumanQaError("packet_integrity_failed")
        tile_paths.add(sample["evidence"]["tile"]["path"])
    if len(tile_paths) != EXPECTED_SAMPLES:
        raise PpeHumanQaError("packet_integrity_failed")

    by_id = {row["sample_id"]: row for row in samples}
    return PacketContext(
        receipt=receipt,
        samples=tuple(samples),
        samples_by_id=by_id,
        sample_hashes=sample_hashes,
        manifest_by_path=manifest_by_path,
        export_validator=export_validator,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _valid_utc(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 40:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0


def _utc_value(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _clean_reviewer(value: object) -> str:
    if not isinstance(value, str):
        raise PpeHumanQaError("invalid_reviewer", status_code=422)
    cleaned = value.strip()
    if (
        len(cleaned) < 2
        or len(cleaned) > 200
        or CONTROL_RE.search(cleaned)
    ):
        raise PpeHumanQaError("invalid_reviewer", status_code=422)
    return cleaned


def _clean_notes(value: object, *, required: bool) -> str:
    if not isinstance(value, str) or len(value) > 4000 or CONTROL_RE.search(value):
        raise PpeHumanQaError("invalid_decision", status_code=422)
    cleaned = value.strip()
    if required and not cleaned:
        raise PpeHumanQaError("invalid_decision", status_code=422)
    return cleaned


def _validate_sample_decision(
    row: object, sample_id: str, context: PacketContext
) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != {
        "sample_id",
        "sample_row_sha256",
        "decision",
        "reason_code",
        "notes",
        "decided_at",
    }:
        raise PpeHumanQaError("invalid_audit_store")
    if (
        row.get("sample_id") != sample_id
        or row.get("sample_row_sha256") != context.sample_hashes.get(sample_id)
        or row.get("decision") not in SAMPLE_DECISIONS
        or row.get("reason_code") not in REASON_CODES
        or not _valid_utc(row.get("decided_at"))
    ):
        raise PpeHumanQaError("invalid_audit_store")
    cleaned_notes = _clean_notes(
        row.get("notes"), required=row.get("reason_code") == "other"
    )
    if cleaned_notes != row.get("notes"):
        raise PpeHumanQaError("invalid_audit_store")
    decision = row["decision"]
    reason = row["reason_code"]
    if decision == "approve_sample_semantics" and reason not in {
        "visually_correct",
        "unknown_must_not_be_negative_ground_truth",
    }:
        raise PpeHumanQaError("invalid_audit_store")
    if decision != "approve_sample_semantics" and reason == "visually_correct":
        raise PpeHumanQaError("invalid_audit_store")
    if (
        reason == "unknown_must_not_be_negative_ground_truth"
        and context.samples_by_id[sample_id]["candidate_state"]
        != "zero_label_unknown_context"
    ):
        raise PpeHumanQaError("invalid_audit_store")
    return row


def _validate_policy_decision(row: object) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != {"decision", "notes", "decided_at"}:
        raise PpeHumanQaError("invalid_audit_store")
    if row.get("decision") not in {"approve", "reject"} or not _valid_utc(
        row.get("decided_at")
    ):
        raise PpeHumanQaError("invalid_audit_store")
    if _clean_notes(row.get("notes"), required=True) != row.get("notes"):
        raise PpeHumanQaError("invalid_audit_store")
    return row


def _normalize_sql(value: str) -> str:
    return " ".join(value.lower().replace("if not exists", "").split())


def _verify_database_contract(connection: sqlite3.Connection) -> None:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise PpeHumanQaError("invalid_audit_store")
    objects = connection.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE type IN ('table', 'view', 'trigger') ORDER BY type, name"
    ).fetchall()
    tables = {row["name"] for row in objects if row["type"] == "table"}
    triggers = {
        row["name"]: row["sql"] for row in objects if row["type"] == "trigger"
    }
    views = {row["name"] for row in objects if row["type"] == "view"}
    if tables != {"audit_events"} or views or set(triggers) != {
        "ppe_r6_audit_no_update",
        "ppe_r6_audit_no_delete",
    }:
        raise PpeHumanQaError("invalid_audit_store")
    table_sql = next(
        row["sql"]
        for row in objects
        if row["type"] == "table" and row["name"] == "audit_events"
    )
    if _normalize_sql(table_sql) != _normalize_sql(CREATE_AUDIT_TABLE):
        raise PpeHumanQaError("invalid_audit_store")
    expected_triggers = {
        "ppe_r6_audit_no_update": CREATE_NO_UPDATE_TRIGGER,
        "ppe_r6_audit_no_delete": CREATE_NO_DELETE_TRIGGER,
    }
    for name, expected in expected_triggers.items():
        if _normalize_sql(triggers[name]) != _normalize_sql(expected):
            raise PpeHumanQaError("invalid_audit_store")
    columns = connection.execute("PRAGMA table_info(audit_events)").fetchall()
    expected_columns = [
        "sequence",
        "entity_type",
        "entity_id",
        "entity_revision",
        "recorded_at",
        "event_json",
        "event_sha256",
        "previous_event_sha256",
    ]
    if [row["name"] for row in columns] != expected_columns:
        raise PpeHumanQaError("invalid_audit_store")


def _etag(revision: int) -> str:
    return f'"ppe-r6-r{revision}"'


def parse_ppe_if_match(value: str | None) -> int:
    if value is None:
        raise PpeHumanQaError("if_match_required", status_code=428)
    match = re.fullmatch(r'"ppe-r6-r(0|[1-9][0-9]{0,17})"', value.strip())
    if not match:
        raise PpeHumanQaError("invalid_if_match", status_code=400)
    return int(match.group(1))


class PpeHumanQaService:
    def __init__(self, config: PpeHumanQaConfig | None = None):
        self.config = config or PpeHumanQaConfig.from_env()

    @property
    def database_path(self) -> Path:
        return self.config.data_dir / "adjudication.sqlite3"

    def _context(self) -> PacketContext:
        return _load_packet(self.config)

    def _connect(self) -> sqlite3.Connection:
        try:
            absolute_data_dir = Path(os.path.abspath(self.config.data_dir))
            current = Path(absolute_data_dir.anchor)
            for part in absolute_data_dir.parts[1:]:
                current /= part
                if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
                    raise PpeHumanQaError("audit_store_unavailable")
            self.config.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            info = self.config.data_dir.lstat()
            if not stat.S_ISDIR(info.st_mode) or self.config.data_dir.is_symlink():
                raise PpeHumanQaError("audit_store_unavailable")
            os.chmod(self.config.data_dir, 0o700)
            if os.path.lexists(self.database_path):
                before_open = self.database_path.lstat()
                if (
                    not stat.S_ISREG(before_open.st_mode)
                    or before_open.st_nlink != 1
                ):
                    raise PpeHumanQaError("audit_store_unavailable")
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar = Path(str(self.database_path) + suffix)
                if os.path.lexists(sidecar):
                    sidecar_info = sidecar.lstat()
                    if (
                        not stat.S_ISREG(sidecar_info.st_mode)
                        or sidecar_info.st_nlink != 1
                    ):
                        raise PpeHumanQaError("audit_store_unavailable")
            with _DATABASE_LOCK:
                connection = sqlite3.connect(self.database_path, timeout=10)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA busy_timeout=10000")
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute(CREATE_AUDIT_TABLE)
                connection.execute(CREATE_NO_UPDATE_TRIGGER)
                connection.execute(CREATE_NO_DELETE_TRIGGER)
                connection.commit()
                os.chmod(self.database_path, 0o600)
                db_info = self.database_path.lstat()
                if not stat.S_ISREG(db_info.st_mode) or db_info.st_nlink != 1:
                    connection.close()
                    raise PpeHumanQaError("audit_store_unavailable")
            _verify_database_contract(connection)
            return connection
        except PpeHumanQaError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PpeHumanQaError("audit_store_unavailable") from exc

    def _verified_tile(
        self, context: PacketContext, sample_id: str
    ) -> TileContent:
        sample = context.samples_by_id.get(sample_id)
        if sample is None:
            raise PpeHumanQaError("tile_unavailable", status_code=404)
        pin = sample["evidence"]["tile"]
        if (
            context.manifest_by_path.get(pin["path"]) != pin
            or pin["bytes"] > self.config.max_tile_bytes
        ):
            raise PpeHumanQaError("tile_unavailable", status_code=404)
        try:
            content = _stable_read(
                _safe_path(self.config.packet_root, pin["path"]),
                expected=pin,
                maximum=self.config.max_tile_bytes,
            )
        except PpeHumanQaError as exc:
            raise PpeHumanQaError("tile_unavailable", status_code=404) from exc
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise PpeHumanQaError("tile_unavailable", status_code=404)
        return TileContent(content=content, filename=f"{sample_id}.png")

    def _read_audit(
        self, connection: sqlite3.Connection, context: PacketContext
    ) -> AuditState:
        rows = connection.execute(
            "SELECT * FROM audit_events ORDER BY sequence"
        ).fetchall()
        if len(rows) > MAX_AUDIT_EVENTS:
            raise PpeHumanQaError("invalid_audit_store")
        heads: dict[tuple[str, str], dict[str, Any]] = {}
        revisions: dict[tuple[str, str], int] = {}
        previous: str | None = None
        reviewer: str | None = None
        complete = False
        completed_at: str | None = None
        previous_recorded_at: datetime | None = None
        for expected_sequence, stored in enumerate(rows, start=1):
            if complete:
                raise PpeHumanQaError("invalid_audit_store")
            try:
                event = strict_json_loads(stored["event_json"])
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise PpeHumanQaError("invalid_audit_store") from exc
            if (
                not isinstance(event, dict)
                or set(event) != AUDIT_EVENT_FIELDS
                or event.get("schema_version") != AUDIT_SCHEMA_VERSION
                or _canonical_bytes(event).decode("utf-8") != stored["event_json"]
            ):
                raise PpeHumanQaError("invalid_audit_store")
            entity_type = event.get("entity_type")
            entity_id = event.get("entity_id")
            revision = event.get("entity_revision")
            key = (entity_type, entity_id)
            if (
                event.get("sequence") != expected_sequence
                or stored["sequence"] != expected_sequence
                or entity_type not in {"sample", "policy"}
                or not isinstance(entity_id, str)
                or not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision != revisions.get(key, 0) + 1
                or stored["entity_type"] != entity_type
                or stored["entity_id"] != entity_id
                or stored["entity_revision"] != revision
            ):
                raise PpeHumanQaError("invalid_audit_store")
            if entity_type == "sample":
                if entity_id not in context.samples_by_id:
                    raise PpeHumanQaError("invalid_audit_store")
                try:
                    decision_row = _validate_sample_decision(
                        event.get("decision_row"), entity_id, context
                    )
                except PpeHumanQaError as exc:
                    raise PpeHumanQaError("invalid_audit_store") from exc
            else:
                if entity_id not in POLICY_NAMES:
                    raise PpeHumanQaError("invalid_audit_store")
                try:
                    decision_row = _validate_policy_decision(
                        event.get("decision_row")
                    )
                except PpeHumanQaError as exc:
                    raise PpeHumanQaError("invalid_audit_store") from exc
            row_sha = event.get("decision_row_sha256")
            if (
                not isinstance(row_sha, str)
                or _sha256(_canonical_bytes(decision_row)) != row_sha
                or event.get("packet_receipt_sha256") != PACKET_RECEIPT_SHA256
                or event.get("packet_file_sha256") != PACKET_RECEIPT_FILE["sha256"]
                or event.get("previous_event_sha256") != previous
                or stored["previous_event_sha256"] != previous
                or stored["recorded_at"] != event.get("recorded_at")
                or not _valid_utc(event.get("recorded_at"))
                or event.get("recorded_at") != decision_row.get("decided_at")
            ):
                raise PpeHumanQaError("invalid_audit_store")
            recorded_at_value = _utc_value(event["recorded_at"])
            if (
                previous_recorded_at is not None
                and recorded_at_value < previous_recorded_at
            ):
                raise PpeHumanQaError("invalid_audit_store")
            event_hash = event.get("event_sha256")
            unsigned = dict(event)
            unsigned.pop("event_sha256", None)
            if (
                not isinstance(event_hash, str)
                or not HEX_SHA256.fullmatch(event_hash)
                or _sha256(_canonical_bytes(unsigned)) != event_hash
                or stored["event_sha256"] != event_hash
            ):
                raise PpeHumanQaError("invalid_audit_store")
            try:
                actor = _clean_reviewer(event.get("reviewer_identity"))
            except PpeHumanQaError as exc:
                raise PpeHumanQaError("invalid_audit_store") from exc
            if actor != event.get("reviewer_identity"):
                raise PpeHumanQaError("invalid_audit_store")
            if reviewer is None:
                reviewer = actor
            elif actor != reviewer:
                raise PpeHumanQaError("invalid_audit_store")
            heads[key] = event
            revisions[key] = revision
            previous = event_hash
            previous_recorded_at = recorded_at_value
            sample_count = sum(1 for kind, _ in heads if kind == "sample")
            policy_count = sum(1 for kind, _ in heads if kind == "policy")
            if sample_count == EXPECTED_SAMPLES and policy_count == EXPECTED_POLICIES:
                complete = True
                completed_at = event["recorded_at"]
        # Every persisted sample decision remains bound to a currently exact
        # packet tile.  Undecided tiles are verified when they are served; a
        # decision can therefore never survive evidence drift unnoticed.
        for entity_type, entity_id in heads:
            if entity_type == "sample":
                try:
                    self._verified_tile(context, entity_id)
                except PpeHumanQaError as exc:
                    raise PpeHumanQaError("invalid_audit_store") from exc
        return AuditState(
            heads=heads,
            event_count=len(rows),
            reviewer_identity=reviewer,
            complete=complete,
            completed_at=completed_at,
            chain_head_sha256=previous,
        )

    def _state(self, context: PacketContext) -> AuditState:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            state = self._read_audit(connection, context)
            connection.commit()
            return state
        except PpeHumanQaError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise PpeHumanQaError("audit_store_unavailable") from exc
        finally:
            connection.close()

    @staticmethod
    def _sample_projection(
        context: PacketContext, state: AuditState, sample: Mapping[str, Any]
    ) -> dict[str, Any]:
        event = state.heads.get(("sample", sample["sample_id"]))
        revision = 0 if event is None else event["entity_revision"]
        decision = None if event is None else event["decision_row"]
        return {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "question": sample["question"],
            "role": sample["role"],
            "candidate_state": sample["candidate_state"],
            "semantic_class": sample["semantic_class"],
            "quarantine_reason": sample.get("quarantine_reason"),
            "association": {
                "vertical_fraction": sample["association"]["vertical_fraction"],
                "head_zone_threshold": sample["association"]["head_zone_threshold"],
            },
            "decision": decision,
            "revision": revision,
            "etag": _etag(revision),
            "tile_href": f"/api/ppe-human-qa-r6/tiles/{sample['sample_id']}",
        }

    @staticmethod
    def _policy_projection(state: AuditState, policy: str) -> dict[str, Any]:
        event = state.heads.get(("policy", policy))
        revision = 0 if event is None else event["entity_revision"]
        decision = (
            {"decision": "pending", "notes": "", "decided_at": None}
            if event is None
            else event["decision_row"]
        )
        return {
            "policy": policy,
            "decision": decision,
            "revision": revision,
            "etag": _etag(revision),
        }

    @staticmethod
    def _progress(state: AuditState) -> dict[str, Any]:
        sample_events = [
            event for (kind, _), event in state.heads.items() if kind == "sample"
        ]
        policy_events = [
            event for (kind, _), event in state.heads.items() if kind == "policy"
        ]
        return {
            "state": "complete" if state.complete else "in_progress",
            "expected_samples": EXPECTED_SAMPLES,
            "decided_samples": len(sample_events),
            "pending_samples": EXPECTED_SAMPLES - len(sample_events),
            "sample_decision_counts": dict(
                sorted(Counter(event["decision_row"]["decision"] for event in sample_events).items())
            ),
            "expected_policies": EXPECTED_POLICIES,
            "decided_policies": len(policy_events),
            "pending_policies": EXPECTED_POLICIES - len(policy_events),
            "completed_at": state.completed_at,
            "training_authorized": False,
            "production_ready": False,
            "new_exact_training_authorization_required": True,
            "authorization_effect": AUTHORIZATION_EFFECT,
        }

    def queue(
        self,
        *,
        offset: int = 0,
        limit: int = 24,
        status: str | None = None,
        category: str | None = None,
        role: str | None = None,
        decision: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 100
            or status not in {None, "pending", "decided"}
            or category not in ({None} | CATEGORIES)
            or role not in {None, "train", "calibration"}
            or decision not in ({None} | SAMPLE_DECISIONS)
        ):
            raise PpeHumanQaError("invalid_filter", status_code=422)
        if query is not None:
            query = query.strip().casefold()
            if len(query) > 100 or CONTROL_RE.search(query):
                raise PpeHumanQaError("invalid_filter", status_code=422)
        context = self._context()
        state = self._state(context)
        selected: list[dict[str, Any]] = []
        for sample in context.samples:
            event = state.heads.get(("sample", sample["sample_id"]))
            if status == "pending" and event is not None:
                continue
            if status == "decided" and event is None:
                continue
            if category is not None and sample["category"] != category:
                continue
            if role is not None and sample["role"] != role:
                continue
            if decision is not None and (
                event is None or event["decision_row"]["decision"] != decision
            ):
                continue
            if query and query not in sample["sample_id"].casefold():
                continue
            selected.append(self._sample_projection(context, state, sample))
        total = len(selected)
        return {
            "schema_version": "deepsafe.admin-ppe-human-qa-r6/v1",
            "packet_integrity": "exact_pins_verified_metadata_assets_verified_on_access",
            "reviewer_identity": state.reviewer_identity,
            "reviewer_identity_locked": state.reviewer_identity is not None,
            "progress": self._progress(state),
            "audit": {
                "append_only": True,
                "chain_verified": True,
                "event_count": state.event_count,
                "chain_head_present": state.chain_head_sha256 is not None,
            },
            "filters": {
                "categories": sorted(CATEGORIES),
                "roles": ["train", "calibration"],
                "statuses": ["pending", "decided"],
                "decisions": sorted(SAMPLE_DECISIONS),
            },
            "pagination": {
                "offset": offset,
                "limit": limit,
                "filtered_total": total,
                "next_offset": offset + limit if offset + limit < total else None,
                "previous_offset": max(0, offset - limit) if offset > 0 else None,
            },
            "policies": [self._policy_projection(state, item) for item in POLICY_NAMES],
            "items": selected[offset : offset + limit],
            "guardrail": (
                "Human QA only. Completion does not authorize training, export, "
                "evaluation, or production; a new exact training authorization is required."
            ),
        }

    def sample(self, sample_id: str) -> tuple[dict[str, Any], str]:
        if not SAMPLE_ID_RE.fullmatch(sample_id):
            raise PpeHumanQaError("sample_not_found", status_code=404)
        context = self._context()
        sample = context.samples_by_id.get(sample_id)
        if sample is None:
            raise PpeHumanQaError("sample_not_found", status_code=404)
        state = self._state(context)
        projection = self._sample_projection(context, state, sample)
        return (
            {
                "schema_version": "deepsafe.admin-ppe-human-qa-r6-sample/v1",
                "sample": projection,
                "review_complete": state.complete,
                "training_authorized": False,
                "production_ready": False,
                "new_exact_training_authorization_required": True,
            },
            projection["etag"],
        )

    def tile(self, sample_id: str) -> TileContent:
        if not SAMPLE_ID_RE.fullmatch(sample_id):
            raise PpeHumanQaError("tile_unavailable", status_code=404)
        context = self._context()
        return self._verified_tile(context, sample_id)

    def _build_export(
        self, context: PacketContext, state: AuditState
    ) -> dict[str, Any]:
        if state.reviewer_identity is None:
            raise PpeHumanQaError("reviewer_not_initialized", status_code=409)
        sample_decisions = [
            state.heads[("sample", sample["sample_id"])]["decision_row"]
            for sample in context.samples
            if ("sample", sample["sample_id"]) in state.heads
        ]
        policies = {
            policy: (
                state.heads[("policy", policy)]["decision_row"]
                if ("policy", policy) in state.heads
                else {"decision": "pending", "notes": "", "decided_at": None}
            )
            for policy in POLICY_NAMES
        }
        value = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "review_id": REVIEW_ID,
            "state": "complete" if state.complete else "in_progress",
            "reviewer_identity": state.reviewer_identity,
            "packet": {
                "path": PACKET_LOGICAL_PATH,
                "bytes": PACKET_RECEIPT_FILE["bytes"],
                "sha256": PACKET_RECEIPT_FILE["sha256"],
                "receipt_sha256": PACKET_RECEIPT_SHA256,
            },
            "policy_decisions": policies,
            "sample_decisions": sample_decisions,
            "completion": {
                "expected_samples": EXPECTED_SAMPLES,
                "decided_samples": len(sample_decisions),
                "all_samples_decided": len(sample_decisions) == EXPECTED_SAMPLES,
                "all_policies_decided": all(
                    item["decision"] != "pending" for item in policies.values()
                ),
                "completed_at": state.completed_at,
            },
            "training_authorized": False,
            "production_ready": False,
            "authorization_effect": AUTHORIZATION_EFFECT,
        }
        _validate_with(
            context.export_validator, value, state="adjudication_export_invalid"
        )
        return value

    def export(self) -> dict[str, Any]:
        context = self._context()
        state = self._state(context)
        return self._build_export(context, state)

    def _put(
        self,
        *,
        context: PacketContext,
        entity_type: str,
        entity_id: str,
        expected_revision: int,
        reviewer_identity: str,
        decision_row: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        reviewer_identity = _clean_reviewer(reviewer_identity)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            state = self._read_audit(connection, context)
            if state.complete:
                raise PpeHumanQaError("review_complete", status_code=409)
            if (
                state.reviewer_identity is not None
                and state.reviewer_identity != reviewer_identity
            ):
                raise PpeHumanQaError("reviewer_identity_locked", status_code=409)
            current = state.heads.get((entity_type, entity_id))
            revision = 0 if current is None else current["entity_revision"]
            if revision != expected_revision:
                raise PpeHumanQaError("stale_revision", status_code=412)
            sequence = state.event_count + 1
            next_revision = revision + 1
            event = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "sequence": sequence,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_revision": next_revision,
                "recorded_at": decision_row["decided_at"],
                "reviewer_identity": reviewer_identity,
                "packet_receipt_sha256": PACKET_RECEIPT_SHA256,
                "packet_file_sha256": PACKET_RECEIPT_FILE["sha256"],
                "decision_row": decision_row,
                "decision_row_sha256": _sha256(_canonical_bytes(decision_row)),
                "previous_event_sha256": state.chain_head_sha256,
            }
            event["event_sha256"] = _sha256(_canonical_bytes(event))
            event_json = _canonical_bytes(event).decode("utf-8")
            connection.execute(
                "INSERT INTO audit_events "
                "(sequence, entity_type, entity_id, entity_revision, recorded_at, "
                "event_json, event_sha256, previous_event_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sequence,
                    entity_type,
                    entity_id,
                    next_revision,
                    event["recorded_at"],
                    event_json,
                    event["event_sha256"],
                    state.chain_head_sha256,
                ),
            )
            replayed = self._read_audit(connection, context)
            self._build_export(context, replayed)
            connection.commit()
            return decision_row, next_revision
        except PpeHumanQaError:
            connection.rollback()
            raise
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PpeHumanQaError("stale_revision", status_code=412) from exc
        except sqlite3.Error as exc:
            connection.rollback()
            raise PpeHumanQaError("audit_store_unavailable") from exc
        finally:
            connection.close()

    def put_sample(
        self,
        sample_id: str,
        *,
        expected_revision: int,
        reviewer_identity: str,
        decision: str,
        reason_code: str,
        notes: str,
    ) -> tuple[dict[str, Any], str]:
        if not SAMPLE_ID_RE.fullmatch(sample_id):
            raise PpeHumanQaError("sample_not_found", status_code=404)
        context = self._context()
        sample = context.samples_by_id.get(sample_id)
        if sample is None:
            raise PpeHumanQaError("sample_not_found", status_code=404)
        self._verified_tile(context, sample_id)
        cleaned_notes = _clean_notes(notes, required=reason_code == "other")
        decided_at = _utc_now()
        row = {
            "sample_id": sample_id,
            "sample_row_sha256": context.sample_hashes[sample_id],
            "decision": decision,
            "reason_code": reason_code,
            "notes": cleaned_notes,
            "decided_at": decided_at,
        }
        try:
            _validate_sample_decision(row, sample_id, context)
        except PpeHumanQaError as exc:
            raise PpeHumanQaError("invalid_decision", status_code=422) from exc
        _, revision = self._put(
            context=context,
            entity_type="sample",
            entity_id=sample_id,
            expected_revision=expected_revision,
            reviewer_identity=reviewer_identity,
            decision_row=row,
        )
        return (
            {
                "sample_id": sample_id,
                "decision": row,
                "revision": revision,
                "etag": _etag(revision),
                "training_authorized": False,
                "production_ready": False,
                "new_exact_training_authorization_required": True,
            },
            _etag(revision),
        )

    def put_policy(
        self,
        policy: str,
        *,
        expected_revision: int,
        reviewer_identity: str,
        decision: str,
        notes: str,
    ) -> tuple[dict[str, Any], str]:
        if policy not in POLICY_NAMES:
            raise PpeHumanQaError("policy_not_found", status_code=404)
        context = self._context()
        row = {
            "decision": decision,
            "notes": _clean_notes(notes, required=True),
            "decided_at": _utc_now(),
        }
        try:
            _validate_policy_decision(row)
        except PpeHumanQaError as exc:
            raise PpeHumanQaError("invalid_decision", status_code=422) from exc
        _, revision = self._put(
            context=context,
            entity_type="policy",
            entity_id=policy,
            expected_revision=expected_revision,
            reviewer_identity=reviewer_identity,
            decision_row=row,
        )
        return (
            {
                "policy": policy,
                "decision": row,
                "revision": revision,
                "etag": _etag(revision),
                "training_authorized": False,
                "production_ready": False,
                "new_exact_training_authorization_required": True,
            },
            _etag(revision),
        )
