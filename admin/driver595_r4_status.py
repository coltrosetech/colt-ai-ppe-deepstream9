"""Fail-closed admin projection for the independent Driver 595 R4 rejection."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "validation/results/driver595-maintenance/r4-independent-review/review.json": (24462, "c1a9d83effeff5371115201b182929f6ac047908ad46894280dc4d0c74b83a21"),
    "validation/driver595_maintenance_r4_independent_review.py": (10934, "c74fbe217d63e0d93a3508a40a5b1e22e1107f790a7372397a937ff912b143e5"),
    "validation/schemas/driver595-maintenance-r4-independent-review.schema.json": (9638, "225e58e286c096ccb470929d98606831c0aee2202cd0d84d92a6188215213f24"),
    "validation/driver595_maintenance_r4.py": (68367, "e7d2bb6ebcc545d66fb87288f5f99ed9e86205259b8c9c0b2e273986795f81cf"),
    "validation/schemas/driver595-maintenance-plan-r4.schema.json": (15349, "dd6c8e2f52540b71203c86e49e757082a2276d757f881a5a8cf6cad7cfaf58fa"),
}
REVIEW_FINGERPRINT = "b25bcddb2ee181df783ee20d5e4dd671ca0fdb804b6d5233558482e643e61877"


class DriverR4StatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise DriverR4StatusError(message)


def _read_exact(relative: str) -> bytes:
    workspace = Path(os.getenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", ROOT))
    path = workspace / relative
    size, expected_sha = PINS[relative]
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _expect(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, f"unsafe evidence: {relative}")
        _expect(before.st_mode & 0o222 == 0 and before.st_size == size, f"mode/size differs: {relative}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        _expect((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), f"evidence changed: {relative}")
    finally:
        os.close(descriptor)
    _expect(digest.hexdigest() == expected_sha, f"hash differs: {relative}")
    return b"".join(chunks)


def _strict_json(raw: bytes) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            _expect(key not in value, f"duplicate review key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook, parse_constant=lambda token: (_ for _ in ()).throw(DriverR4StatusError(f"non-finite token: {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverR4StatusError("invalid review JSON") from exc
    _expect(isinstance(value, dict), "review root differs")
    return value


def _validate(review: dict[str, Any]) -> None:
    _expect(review.get("schema_version") == "deepsafe.driver595-maintenance-r4-independent-review/v1", "schema differs")
    _expect(review.get("decision") == "REJECT", "decision differs")
    _expect(review.get("severity_counts") == {"P0": 0, "P1": 7, "P2": 3}, "severity counts differ")
    findings = review.get("findings")
    _expect(isinstance(findings, list) and len(findings) == 10, "finding count differs")
    _expect(sum(item.get("severity") == "P1" and item.get("blocks_acceptance") is True for item in findings) == 7, "blocking findings differ")
    _expect(sum(item.get("severity") == "P2" and item.get("blocks_acceptance") is False for item in findings) == 3, "P2 findings differ")
    _expect(review.get("review_fingerprint_sha256") == REVIEW_FINGERPRINT, "declared fingerprint differs")
    unsigned = {key: item for key, item in review.items() if key != "review_fingerprint_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    _expect(hashlib.sha256(canonical).hexdigest() == REVIEW_FINGERPRINT, "fingerprint replay differs")
    authority = review.get("authority")
    _expect(isinstance(authority, dict) and all(authority.get(key) is False for key in (
        "subject_accepted", "prepared_plan_accepted", "cache_accepted", "download_authorized",
        "install_authorized", "reboot_authorized", "gpu_workload_authorized", "terminal_root_published",
    )), "authority overclaim")
    readiness = review.get("operational_readiness")
    _expect(isinstance(readiness, dict) and readiness.get("ready") is False, "operational readiness differs")
    _expect(readiness.get("general_apt_cache_exact_count") == 15 and readiness.get("required_deb_count") == 22 and readiness.get("missing_deb_count") == 7, "cache counts differ")
    _expect(readiness.get("target_initramfs_present") is False and readiness.get("target_595_modules_present") is False and readiness.get("reboot_allowed") is False, "boot readiness overclaim")
    replay = review.get("test_replay")
    _expect(isinstance(replay, dict) and replay.get("collected_case_count") == 39 and replay.get("passed_case_count") == 39 and replay.get("failed_case_count") == 0 and replay.get("collection_claim_matches") is False, "test replay differs")


def _unavailable() -> dict[str, Any]:
    return {
        "label": "NVIDIA 595 bakım planı R4 bağımsız karar",
        "state": "unavailable_integrity_error",
        "reason": "driver595_r4_reject_integrity_failed",
        "available": False,
        "subject_accepted": False,
        "operational_ready": False,
        "download_authorized": False,
        "install_authorized": False,
        "reboot_authorized": False,
        "gpu_workload_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["R4 reject kaydı exact-pin doğrulamasından geçmedi; bütün yetkiler kapalıdır."],
    }


def load_driver595_r4_status() -> dict[str, Any]:
    try:
        raw = {relative: _read_exact(relative) for relative in PINS}
        review = _strict_json(raw["validation/results/driver595-maintenance/r4-independent-review/review.json"])
        _validate(review)
    except (OSError, KeyError, TypeError, DriverR4StatusError):
        return _unavailable()

    readiness = review["operational_readiness"]
    return {
        "label": "NVIDIA 595 bakım planı R4 bağımsız karar",
        "state": "independent_reject_successor_required",
        "reason": "seven_p1_findings_block_r4",
        "available": True,
        "decision": "REJECT",
        "subject_accepted": False,
        "severity": {"p0": 0, "p1": 7, "p2": 3},
        "tests": {"subject_passed": 39, "review_passed": 15, "combined_passed": 54, "failed": 0, "claimed_subject_count": 44, "collected_subject_count": 39},
        "cache": {"exact_present": readiness["general_apt_cache_exact_count"], "required": readiness["required_deb_count"], "missing": readiness["missing_deb_count"], "dedicated_present": readiness["dedicated_cache_present"]},
        "target_initramfs_present": False,
        "target_modules_present": False,
        "operational_ready": False,
        "download_authorized": False,
        "install_authorized": False,
        "reboot_authorized": False,
        "gpu_workload_authorized": False,
        "successor": {"revision": "R5", "state": "in_progress", "required_acceptance_criteria": len(review["successor_acceptance_criteria"])},
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": [
            "Verifier PASS yalnız immutable REJECT kaydının bütünlüğünü gösterir; R4 kabul edilmemiştir.",
            "Paket indirme/kurma ve reboot yetkisi yoktur; R5 successor bağımsız kabul beklemektedir.",
        ],
    }
