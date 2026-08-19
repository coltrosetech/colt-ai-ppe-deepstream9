"""Fail-closed admin projection for the independent Driver 595 R7 acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = "validation/results/driver595-maintenance/r7-independent-review-r1/review.json"
PINS = {
    REVIEW_PATH: (32491, "1788fe97ae521515407c3fe3e97222bddbe83f1f2459b9a46668d52841545f3d"),
    "validation/driver595_live_qualification_r7_independent_review.py": (74474, "14c970357d815bf1be67d4f5c7a4a387441463c20b92d9afa32a6d63b9e7780e"),
    "validation/schemas/driver595-live-r7-independent-review.schema.json": (24998, "6fe63bbc2a7d3ceaad00043a36c94416f3b11bbbacd222bb8550b04c4921caea"),
    "tests/test_driver595_live_qualification_r7_independent_review.py": (16216, "4de964ace83263bf8ff8c4040125cb51c7e0c38aa9042eafbd4e5d481b7ffb1e"),
    "validation/results/driver595-maintenance/r7-live-candidate/candidate.json": (28267, "2b92f79cfefa233988535d575d934ef97ea1053fc3579eb0d3f1a8bce9c658e7"),
    "validation/results/driver595-maintenance/r7-live-candidate/handoff.json": (4585, "6041a786fbfbdc404453d8fc6437cb689699e830df325426588b863e6beed287"),
    "validation/driver595_live_qualification_r7.py": (51350, "7aa3eea8bf6e27ef806993cd7911a10c7394e91416680904cbc4b2bfa128e849"),
    "validation/schemas/driver595-live-qualification-r7.schema.json": (18333, "15ae065235375db356f1b42f2a8a7a9bd806eecdc57f5680bd88854d7292bce0"),
    "tests/test_driver595_live_qualification_r7.py": (11805, "5c7e49462f6e0557a9fc6a5931a2c6ef65653c74d63191ffcf527f17f9a056de"),
    "docs/deepstream91-driver595-live-qualification-r7.md": (2883, "318eacdbcecfe7586a9cc515a691a1cd5bbccb205187a32e4d80de6b68da07c2"),
}
REVIEW_FINGERPRINT = "ebf0cfafe76bb6165574575e86930dda3cb3e807997cc09a914dee4b288bdaf7"
EXPECTED_AUTHORITY = {
    "candidate_accepted_for_current_boot_only": True,
    "current_boot_runtime_prerequisite_accepted": True,
    "deepstream_runtime_authorized": False,
    "deepstream_runtime_validated": False,
    "download_authorized": False,
    "future_reboot_selection_accepted": False,
    "gpu_workload_authorized": False,
    "historical_install_authorized": False,
    "historical_install_proven": False,
    "independent_acceptance_present": True,
    "install_authorized": False,
    "production_authorized": False,
    "quality_validated": False,
    "reboot_authorized": False,
    "remove_authorized": False,
    "same_uid_mutation_resistance_claimed": False,
    "terminal_current_boot_prerequisite_receipt": True,
    "update_authorized": False,
}


class DriverR7StatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise DriverR7StatusError(message)


def _read_exact(relative: str) -> bytes:
    workspace = Path(os.getenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", ROOT))
    path = workspace / relative
    size, expected_sha = PINS[relative]
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _expect(stat.S_ISREG(before.st_mode), f"not regular: {relative}")
        _expect(before.st_nlink == 1, f"link count differs: {relative}")
        _expect(before.st_mode & 0o222 == 0, f"writable evidence: {relative}")
        _expect(before.st_size == size, f"size differs: {relative}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        _expect(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"evidence changed: {relative}",
        )
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

    def reject_constant(token: str) -> Any:
        raise DriverR7StatusError(f"non-finite token: {token}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverR7StatusError("invalid review JSON") from exc
    _expect(isinstance(value, dict), "review root differs")
    return value


def _projection(pin: dict[str, Any]) -> tuple[str, int, str, str, int]:
    return (
        pin.get("path"),
        pin.get("bytes"),
        pin.get("sha256"),
        pin.get("mode"),
        pin.get("nlink"),
    )


def _expected_projection(relative: str) -> tuple[str, int, str, str, int]:
    size, digest = PINS[relative]
    return (relative, size, digest, "0440", 1)


def _validate_pin_rows(rows: Any, expected_paths: set[str], label: str) -> None:
    _expect(isinstance(rows, list), f"{label} pins differ")
    mapped = {row.get("path"): row for row in rows if isinstance(row, dict)}
    _expect(set(mapped) == expected_paths, f"{label} paths differ")
    for relative in expected_paths:
        _expect(_projection(mapped[relative]) == _expected_projection(relative), f"{label} pin differs: {relative}")


def _validate(review: dict[str, Any]) -> str:
    _expect(review.get("schema_version") == "deepsafe.driver595-live-r7-independent-review/v1", "schema differs")
    _expect(review.get("decision") == "ACCEPT", "decision differs")
    _expect(review.get("severity_counts") == {"P0": 0, "P1": 0, "P2": 0}, "severity differs")
    _expect(review.get("findings") == [], "findings differ")
    _expect(review.get("review_fingerprint_sha256") == REVIEW_FINGERPRINT, "fingerprint differs")
    unsigned = {key: value for key, value in review.items() if key != "review_fingerprint_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    _expect(hashlib.sha256(canonical).hexdigest() == REVIEW_FINGERPRINT, "fingerprint replay differs")

    _expect(review.get("authority") == EXPECTED_AUTHORITY, "authority differs")
    scope = review.get("scope")
    _expect(isinstance(scope, dict), "scope differs")
    _expect(scope.get("accepted_claim") == "current boot and NVIDIA 595.71.05 host runtime prerequisite observation only", "scope claim differs")
    for key in ("author_subject_modified", "docker_used", "gpu_workload_performed", "host_packages_mutated", "model_loaded", "network_used", "reboot_performed", "sudo_used"):
        _expect(scope.get(key) is False, f"scope overclaim: {key}")
    boundaries = review.get("boundaries")
    _expect(isinstance(boundaries, dict) and boundaries and all(value is False for value in boundaries.values()), "boundary overclaim")
    comparison = review.get("candidate_live_comparison")
    _expect(isinstance(comparison, dict) and len(comparison) == 8 and all(value is True for value in comparison.values()), "live comparison differs")

    replay = review.get("test_replay")
    _expect(isinstance(replay, dict), "test replay differs")
    _expect(replay.get("author_collected") == 35 and replay.get("author_passed") == 35 and replay.get("author_failed") == 0, "author test replay differs")
    _expect(replay.get("independent_collected") == 45 and replay.get("independent_passed") == 45 and replay.get("independent_failed") == 0, "independent test replay differs")
    _expect(replay.get("author_frozen_candidate_cli_verified") is True and replay.get("author_live_collector_replayed") is True, "author replay flags differ")

    subject_paths = {
        "docs/deepstream91-driver595-live-qualification-r7.md",
        "tests/test_driver595_live_qualification_r7.py",
        "validation/driver595_live_qualification_r7.py",
        "validation/schemas/driver595-live-qualification-r7.schema.json",
    }
    review_paths = {
        "tests/test_driver595_live_qualification_r7_independent_review.py",
        "validation/driver595_live_qualification_r7_independent_review.py",
        "validation/schemas/driver595-live-r7-independent-review.schema.json",
    }
    _validate_pin_rows(review.get("subject_pins"), subject_paths, "subject")
    _validate_pin_rows(review.get("review_source_pins"), review_paths, "review")
    candidate = review.get("candidate")
    handoff = review.get("handoff")
    _expect(isinstance(candidate, dict) and _projection(candidate.get("artifact", {})) == _expected_projection("validation/results/driver595-maintenance/r7-live-candidate/candidate.json"), "candidate pin differs")
    _expect(isinstance(handoff, dict) and _projection(handoff.get("artifact", {})) == _expected_projection("validation/results/driver595-maintenance/r7-live-candidate/handoff.json"), "handoff pin differs")

    live = review.get("live_observation")
    _expect(isinstance(live, dict) and live.get("status") == "pass", "live observation differs")
    _expect(live.get("checks") == {"boot_mounts": True, "graphics": True, "initramfs": True, "modules": True, "packages": True, "runtime_libraries": True, "secure_boot": True}, "live checks differ")
    boot = live.get("boot_mounts")
    graphics = live.get("graphics")
    modules = live.get("modules")
    secure_boot = live.get("secure_boot")
    _expect(isinstance(boot, dict) and boot.get("status") == "pass" and boot.get("kernel") == "7.0.0-28-generic", "boot observation differs")
    boot_id = boot.get("boot_id")
    _expect(isinstance(boot_id, str) and candidate.get("boot_id") == boot_id, "boot identity differs")
    _expect(isinstance(graphics, dict) and graphics.get("status") == "pass", "graphics observation differs")
    _expect(graphics.get("driver_version") == "595.71.05" and graphics.get("cuda_driver_version") == "13.2", "driver observation differs")
    _expect(graphics.get("compute_before") == [] and graphics.get("compute_after") == [] and graphics.get("compute_xml") == [], "compute process evidence differs")
    gpus = graphics.get("gpus")
    _expect(isinstance(gpus, list) and len(gpus) == 1 and gpus[0].get("name") == "NVIDIA RTX A5000 Laptop GPU", "GPU identity differs")
    _expect(isinstance(modules, dict) and modules.get("status") == "pass" and modules.get("loaded_modules") == ["nvidia", "nvidia_drm", "nvidia_modeset", "nvidia_uvm"], "module observation differs")
    _expect(isinstance(secure_boot, dict) and secure_boot.get("status") == "pass" and secure_boot.get("state") == "SecureBoot enabled", "Secure Boot observation differs")
    _expect(review.get("next_gate") == {"driver_prerequisite_gate": "accepted_for_this_exact_boot", "required_next": "separately gated DeepStream 9.1 engine/runtime validation; this receipt grants no execution authority"}, "next gate differs")
    return boot_id


def _current_boot_id() -> str | None:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return None
    return value or None


def _unavailable(reason: str = "driver595_r7_accept_integrity_failed") -> dict[str, Any]:
    return {
        "label": "NVIDIA 595 R7 canlı sürücü yeterliliği",
        "state": "unavailable_integrity_error",
        "reason": reason,
        "available": False,
        "decision": None,
        "current_boot_prerequisite_accepted": False,
        "gpu_workload_authorized": False,
        "deepstream_runtime_authorized": False,
        "production_ready": False,
        "download_authorized": False,
        "install_authorized": False,
        "remove_authorized": False,
        "update_authorized": False,
        "reboot_authorized": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["R7 kabul zinciri exact-pin doğrulamasından geçmedi; bütün çalıştırma yetkileri kapalıdır."],
    }


def load_driver595_r7_status() -> dict[str, Any]:
    try:
        raw = {relative: _read_exact(relative) for relative in PINS}
        review = _strict_json(raw[REVIEW_PATH])
        receipt_boot_id = _validate(review)
    except (OSError, KeyError, TypeError, DriverR7StatusError):
        return _unavailable()

    current_boot_id = _current_boot_id()
    boot_current = current_boot_id == receipt_boot_id
    live = review["live_observation"]
    graphics = live["graphics"]
    gpu = graphics["gpus"][0]
    state = "current_boot_prerequisite_accepted" if boot_current else "stale_boot_identity"
    reason = "driver595_current_boot_prerequisite_only" if boot_current else "driver595_r7_boot_identity_changed"
    return {
        "label": "NVIDIA 595 R7 canlı sürücü yeterliliği",
        "state": state,
        "reason": reason,
        "available": True,
        "decision": "ACCEPT",
        "severity": {"p0": 0, "p1": 0, "p2": 0},
        "tests": {"author_passed": 35, "independent_passed": 45, "combined_passed": 80, "failed": 0},
        "receipt_boot_id": receipt_boot_id,
        "current_boot_id": current_boot_id,
        "current_boot_match": boot_current,
        "current_boot_prerequisite_accepted": boot_current,
        "driver": {"version": graphics["driver_version"], "cuda_driver": graphics["cuda_driver_version"]},
        "kernel": live["boot_mounts"]["kernel"],
        "gpu": {"name": gpu["name"], "uuid": gpu["uuid"], "compute_process_count": 0},
        "secure_boot": True,
        "checks": {"passed": 8, "total": 8},
        "next_gate": "DeepStream 9.1 GPU/runtime smoke",
        "gpu_workload_authorized": False,
        "deepstream_runtime_authorized": False,
        "deepstream_runtime_validated": False,
        "production_ready": False,
        "download_authorized": False,
        "install_authorized": False,
        "remove_authorized": False,
        "update_authorized": False,
        "reboot_authorized": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": [
            "ACCEPT yalnız bu exact boot üzerindeki host sürücü/runtime önkoşuludur.",
            "GPU, DeepStream, model, kalite ve production doğrulanmadı; bütün çalıştırma eylemleri kapalıdır.",
        ],
    }
