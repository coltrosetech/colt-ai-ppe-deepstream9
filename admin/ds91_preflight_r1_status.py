"""Fail-closed admin projection for the immutable DS9.1 preflight R1 rejection.

This reader is intentionally observational.  It cannot start Docker, a GPU
workload, an engine build, a benchmark, or a driver operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVIEW_RELATIVE = (
    "validation/results/deepstream91-full-stack-preflight-v1/"
    "independent-review-r1/review.json"
)
PINS: dict[str, tuple[int, str, int]] = {
    REVIEW_RELATIVE: (
        10684,
        "02e03cc67acb1aacc5ba89d26ebb7f705b430340091fb1b02689e650d2392e2e",
        0o440,
    ),
    "validation/deepstream91_full_stack_preflight_v1_independent_review.py": (
        34683,
        "a6f7b7268f924efa381c1fb8cd0d83862bace092317dd37746be4a386e5ec72e",
        0o440,
    ),
    "validation/schemas/deepstream91-full-stack-preflight-independent-review-v1.schema.json": (
        8256,
        "ef9fba8ed4bceb13eac9ef0ac0b3c4b798bb75b8ebd1b94223e3d02fac6b166f",
        0o440,
    ),
}
REVIEW_FINGERPRINT = (
    "702616fea56dc932140262152a32b9faf04507b1a5376747f77afce09781e27b"
)
EXPECTED_AUTHORITY = {
    "benchmark_authorized": False,
    "docker_gpu_authorized": False,
    "driver_download_install_remove_authorized": False,
    "endurance_authorized": False,
    "engine_build_authorized": False,
    "execution_ready": False,
    "gpu_workload_authorized": False,
    "reboot_authorized": False,
    "same_uid_mutation_resistance_claimed": False,
    "subject_accepted": False,
    "terminal_root_published": False,
}


class PreflightR1StatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightR1StatusError(message)


def _parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    _expect(not path.is_absolute(), f"absolute evidence path: {relative}")
    _expect(
        bool(path.parts)
        and all(part not in ("", ".", "..") for part in path.parts),
        f"unsafe evidence path: {relative}",
    )
    return path.parts


def _read_exact(relative: str) -> bytes:
    expected_size, expected_sha, expected_mode = PINS[relative]
    workspace = Path(os.getenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", ROOT))
    descriptors: list[int] = []
    try:
        root_fd = os.open(
            workspace,
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors.append(root_fd)
        parent_fd = root_fd
        parts = _parts(relative)
        for component in parts[:-1]:
            parent_fd = os.open(
                component,
                os.O_RDONLY
                | os.O_CLOEXEC
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            descriptors.append(parent_fd)

        final_name = parts[-1]
        file_fd = os.open(
            final_name,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        descriptors.append(file_fd)
        before = os.fstat(file_fd)
        _expect(stat.S_ISREG(before.st_mode), f"not regular: {relative}")
        _expect(before.st_nlink == 1, f"link count differs: {relative}")
        _expect(stat.S_IMODE(before.st_mode) == expected_mode, f"mode differs: {relative}")
        _expect(before.st_size == expected_size, f"size differs: {relative}")

        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)

        after = os.fstat(file_fd)
        named = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            stat.S_IFMT(value.st_mode),
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        _expect(identity(before) == identity(after), f"evidence changed: {relative}")
        _expect(identity(after) == identity(named), f"named evidence changed: {relative}")
        _expect(digest.hexdigest() == expected_sha, f"hash differs: {relative}")
        return b"".join(chunks)
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _strict_json(raw: bytes) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            _expect(key not in value, f"duplicate review key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PreflightR1StatusError(f"non-finite token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightR1StatusError("invalid review JSON") from exc
    _expect(isinstance(value, dict), "review root differs")
    return value


def _validate(review: dict[str, Any]) -> None:
    _expect(
        review.get("schema_version")
        == "deepsafe.deepstream91-full-stack-preflight-independent-review/v1",
        "schema differs",
    )
    _expect(
        review.get("review_id")
        == "ds91-full-stack-preflight-v1-independent-review-r1",
        "review id differs",
    )
    _expect(review.get("decision") == "REJECT", "decision differs")
    _expect(review.get("severity_counts") == {"P0": 0, "P1": 4, "P2": 1}, "severity differs")
    _expect(review.get("authority") == EXPECTED_AUTHORITY, "authority boundary differs")
    _expect(review.get("review_fingerprint_sha256") == REVIEW_FINGERPRINT, "fingerprint differs")
    unsigned = {
        key: item
        for key, item in review.items()
        if key != "review_fingerprint_sha256"
    }
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _expect(hashlib.sha256(canonical).hexdigest() == REVIEW_FINGERPRINT, "fingerprint replay differs")

    findings = review.get("findings")
    _expect(isinstance(findings, list) and len(findings) == 5, "finding count differs")
    _expect(
        [item.get("id") for item in findings]
        == [
            "DSPF-P1-001",
            "DSPF-P1-002",
            "DSPF-P1-003",
            "DSPF-P1-004",
            "DSPF-P2-001",
        ],
        "finding ids differ",
    )
    _expect(
        sum(item.get("severity") == "P1" and item.get("blocks_acceptance") is True for item in findings) == 4,
        "blocking P1 findings differ",
    )
    _expect(
        sum(item.get("severity") == "P2" and item.get("blocks_acceptance") is False for item in findings) == 1,
        "P2 findings differ",
    )

    baseline = review.get("baseline_replay")
    _expect(isinstance(baseline, dict), "baseline missing")
    _expect(
        baseline.get("subject_tests_collected") == 13
        and baseline.get("subject_tests_passed") == 13
        and baseline.get("exact_media_count") == 12
        and baseline.get("distinct_video_types") == 12
        and baseline.get("profiles") == [640, 960]
        and baseline.get("simulated_streams") == 12
        and baseline.get("measurement_seconds_per_profile") == 300
        and baseline.get("authorization_all_false") is True,
        "baseline projection differs",
    )
    adversarial = review.get("adversarial_replay")
    _expect(
        isinstance(adversarial, dict)
        and adversarial.get("tests_collected") == 30
        and adversarial.get("tests_passed") == 30,
        "adversarial replay differs",
    )


def _unavailable() -> dict[str, Any]:
    return {
        "label": "DeepStream 9.1 tam-yığın ön-uçuş R1",
        "state": "unavailable_integrity_error",
        "reason": "ds91_preflight_r1_reject_integrity_failed",
        "available": False,
        "decision": "UNAVAILABLE",
        "subject_accepted": False,
        "execution_ready": False,
        "engine_build_authorized": False,
        "benchmark_authorized": False,
        "gpu_workload_authorized": False,
        "endurance_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["R1 REJECT kanıtı exact-pin doğrulamasından geçmedi; bütün yetkiler kapalıdır."],
    }


def load_ds91_preflight_r1_status() -> dict[str, Any]:
    try:
        raw = {relative: _read_exact(relative) for relative in PINS}
        review = _strict_json(raw[REVIEW_RELATIVE])
        _validate(review)
    except (OSError, KeyError, TypeError, ValueError, PreflightR1StatusError):
        return _unavailable()

    baseline = review["baseline_replay"]
    return {
        "label": "DeepStream 9.1 tam-yığın ön-uçuş R1",
        "state": "independent_reject_successor_in_progress",
        "reason": "four_p1_findings_block_r1",
        "available": True,
        "decision": "REJECT",
        "subject_accepted": False,
        "execution_ready": False,
        "severity": {"p0": 0, "p1": 4, "p2": 1},
        "tests": {"subject_passed": 13, "independent_passed": 30, "failed": 0},
        "matrix": {
            "sources": baseline["exact_media_count"],
            "distinct_video_types": baseline["distinct_video_types"],
            "profiles": baseline["profiles"],
            "simulated_streams": baseline["simulated_streams"],
            "measurement_seconds_per_profile": baseline[
                "measurement_seconds_per_profile"
            ],
        },
        "engine_build_authorized": False,
        "benchmark_authorized": False,
        "gpu_workload_authorized": False,
        "endurance_authorized": False,
        "successor": {"revision": "R2", "state": "in_progress", "findings_to_close": 5},
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": [
            "R1 baseline medya matrisi geçse de bağımsız karar REJECT'tir; bu bir çalıştırma yetkisi değildir.",
            "R2; source-only manifest, dış otorite pinleri, strict kontrol şemaları, FD-bound okuma ve paket-güvenli giriş noktasıyla hazırlanıyor.",
        ],
    }
