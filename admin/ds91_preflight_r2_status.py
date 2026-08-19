"""Read-only admin projection of the narrowly accepted DS9.1 preflight R2."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVIEW_RELATIVE = (
    "validation/results/deepstream91-full-stack-preflight-r2/"
    "independent-review-r1/review.json"
)
PINS: dict[str, tuple[int, str, int]] = {
    REVIEW_RELATIVE: (
        10893,
        "afb8f8c7c5ff57913fc9c320058d305a36647003fe1d57e4eacae89aaeb7b5bc",
        0o440,
    ),
    "validation/deepstream91_full_stack_preflight_r2.py": (
        30155,
        "21cc7ed9abdb4037476fc0aada316c49e5ff26b42eb49330323521cab179f340",
        0o440,
    ),
    "validation/plans/deepstream91-full-stack-preflight-r2.json": (
        5057,
        "a7f70ee6c366f33acbf448b71594b47ff13c4d0f94e37722a8c58ea6cb17e368",
        0o440,
    ),
    "validation/schemas/deepstream91-full-stack-preflight-r2.schema.json": (
        9048,
        "849fb9152059ad5dcf504fdb57deefb58043c70b89995ff907e4553eda61b0c1",
        0o440,
    ),
    "validation/manifests/deepstream91-source-matrix-r2.json": (
        4436,
        "ce622a9925c548a360fe68e37f0ce8793dcb5d5ccfe2b2fe1c0f51e3e97677c5",
        0o440,
    ),
    "validation/schemas/deepstream91-source-matrix-r2.schema.json": (
        1693,
        "a13b58cf7b2dbc74ef877c9d0308a384405e7dcc4e673b58ef30f92db9b21a83",
        0o440,
    ),
    "validation/accepted-roots/ds91-engine-builder-r1c3-terminal/terminal-root-r1c3.json": (
        2926,
        "57805489b3c0238e9cc5b13802ef644a9f43346629b6d444a7f89de5a9d54a49",
        0o440,
    ),
    "validation/schemas/deepstream91-r1c3-accepted-root-boundary-r2.schema.json": (
        4474,
        "86659137d6fb7359859dad54fee03f166dee0c24ddca5489bd48bf9822bffef0",
        0o440,
    ),
    "validation/results/ds91-engine-builder/r1c3/candidate-receipt-r1c3.json": (
        11849,
        "6acdc00e90441d7f7e3f923b597365a74cc6df9bb13619291e08a6db36b6c75a",
        0o440,
    ),
    "validation/contracts/gpu-lease-v5.json": (
        3581,
        "039322d84b3ad495e6bf4aa16966c11c9f1dfda981908908bd7f77cc0b58b38d",
        0o440,
    ),
    "validation/schemas/gpu-lease-v5-locked-boundary-r2.schema.json": (
        6140,
        "dab835b334f2b720b63bd76ea7fbca98ead241e79192dd00645d0fd42ff2f86e",
        0o440,
    ),
    "validation/results/deepstream91-full-stack-preflight-v1/independent-review-r1/review.json": (
        10684,
        "02e03cc67acb1aacc5ba89d26ebb7f705b430340091fb1b02689e650d2392e2e",
        0o440,
    ),
}
REVIEW_FINGERPRINT = (
    "0a903948db385344c3372e2228580c3a0b4f589121c04f3a0c1366b2036077ef"
)
EXPECTED_REMAINING_BLOCKERS = [
    "driver595:independent_acceptance_and_operator_apply_pending",
    "runtime:ds91_gpu_qualification_missing",
    "runtime:ds91_fusion_and_parallel_app_missing",
    "profile:640:person_engine_config_parity_missing",
    "profile:640:pose_engine_config_parity_missing",
    "profile:640:ppe_engine_config_parity_missing",
    "profile:960:person_engine_config_parity_missing",
    "profile:960:pose_engine_config_parity_missing",
    "profile:960:ppe_engine_config_parity_missing",
    "profile:640:gpu_lease_v5_live_plan_missing",
    "profile:960:gpu_lease_v5_live_plan_missing",
]
EXPECTED_PERMISSIONS = {
    "benchmark_execution": False,
    "checkpoint_deserialization": False,
    "docker": False,
    "docker_gpu": False,
    "driver_download_install_remove": False,
    "endurance_execution": False,
    "engine_build": False,
    "gpu_workload": False,
    "installation": False,
    "model_load": False,
    "network": False,
    "reboot": False,
    "runtime_execution": False,
    "sudo": False,
}


class PreflightR2StatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightR2StatusError(message)


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
        parent_fd = os.open(
            workspace,
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors.append(parent_fd)
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
                PreflightR2StatusError(f"non-finite token: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightR2StatusError("invalid review JSON") from exc
    _expect(isinstance(value, dict), "review root differs")
    return value


def _validate(review: dict[str, Any]) -> None:
    _expect(
        review.get("schema_version")
        == "deepsafe.deepstream91-full-stack-preflight-independent-review/v2",
        "schema differs",
    )
    _expect(review.get("decision") == "ACCEPT", "decision differs")
    _expect(
        review.get("status")
        == "accepted_cpu_only_preflight_definition_not_runtime_qualification",
        "status differs",
    )
    _expect(review.get("severity_counts") == {"P0": 0, "P1": 0, "P2": 0}, "severity differs")
    _expect(review.get("permissions") == EXPECTED_PERMISSIONS, "permission boundary differs")
    _expect(
        review.get("post_review_remaining_blockers") == EXPECTED_REMAINING_BLOCKERS,
        "remaining blockers differ",
    )
    boundary = review.get("authority_boundary")
    _expect(
        boundary
        == {
            "model": "cooperative_same_uid",
            "malicious_same_uid_resistance_claimed": False,
            "self_fingerprint_is_authority": False,
            "independent_receipt_is_external_entrypoint": True,
            "receipt_file_mode": "0440",
            "receipt_parent_mode": "0550",
        },
        "authority boundary differs",
    )
    accepted = review.get("accepted_scope")
    _expect(isinstance(accepted, dict), "accepted scope missing")
    for key in (
        "r2_cpu_only_preflight_definition",
        "source_only_manifest_exact_boundary",
        "twelve_media_exact_pin_replay",
        "external_control_constants_in_frozen_verifier",
        "strict_prepared_closed_builder_boundary",
        "strict_locked_no_live_plan_lease_boundary",
        "descriptor_bound_workspace_reads",
        "clean_cwd_module_and_direct_python_entrypoints",
    ):
        _expect(accepted.get(key) is True, f"accepted CPU scope differs: {key}")
    for key in (
        "deepstream_runtime_qualified",
        "gpu_runtime_qualified",
        "inference_qualified",
        "engine_build_qualified",
        "benchmark_completed",
        "endurance_completed",
        "production_ready",
    ):
        _expect(accepted.get(key) is False, f"accepted scope overclaim: {key}")
    _expect(accepted.get("profiles_declared") == [640, 960], "profiles differ")
    _expect(accepted.get("simulated_streams_declared") == 12, "stream count differs")
    _expect(accepted.get("measurement_seconds_per_profile_declared") == 300, "duration differs")
    findings = review.get("r1_finding_disposition")
    _expect(isinstance(findings, list) and len(findings) == 5, "finding closure count differs")
    _expect(
        [item.get("finding_id") for item in findings]
        == ["DSPF-P1-001", "DSPF-P1-002", "DSPF-P1-003", "DSPF-P1-004", "DSPF-P2-001"]
        and all(item.get("status") == "fixed" and item.get("acceptance_blocker") is False for item in findings),
        "finding closure differs",
    )
    verification = review.get("verification")
    _expect(isinstance(verification, dict), "verification missing")
    _expect(
        verification.get("author_suite", {}).get("passed") == 39
        and verification.get("author_suite", {}).get("failed") == 0,
        "author replay differs",
    )
    _expect(
        verification.get("independent_suite", {}).get("passed") == 44
        and verification.get("independent_suite", {}).get("failed") == 0,
        "independent replay differs",
    )
    media = verification.get("normal_full_media_replay", {})
    _expect(
        media.get("exit_code") == 0
        and media.get("source_count") == 12
        and media.get("distinct_video_types") == 12
        and media.get("profiles") == [640, 960]
        and media.get("simulated_streams") == 12
        and media.get("measurement_seconds_per_profile") == 300
        and media.get("execution_ready") is False
        and media.get("gpu_or_docker_called") is False,
        "media replay differs",
    )
    resource = verification.get("resource_boundary", {})
    _expect(isinstance(resource, dict) and resource and all(value is False for value in resource.values()), "resource boundary differs")
    fingerprint = review.get("self_fingerprint")
    _expect(
        fingerprint
        == {
            "algorithm": "sha256-canonical-json-without-self_fingerprint",
            "canonical_sha256": REVIEW_FINGERPRINT,
        },
        "fingerprint declaration differs",
    )
    unsigned = {key: item for key, item in review.items() if key != "self_fingerprint"}
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _expect(hashlib.sha256(canonical).hexdigest() == REVIEW_FINGERPRINT, "fingerprint replay differs")


def _unavailable() -> dict[str, Any]:
    return {
        "label": "DeepStream 9.1 tam-yığın ön-uçuş R2",
        "state": "unavailable_integrity_error",
        "reason": "ds91_preflight_r2_accept_integrity_failed",
        "available": False,
        "decision": "UNAVAILABLE",
        "cpu_preflight_accepted": False,
        "execution_ready": False,
        "gpu_workload_authorized": False,
        "engine_build_authorized": False,
        "benchmark_authorized": False,
        "endurance_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["R2 ACCEPT zinciri exact-pin doğrulamasından geçmedi; bütün çalışma yetkileri kapalıdır."],
    }


def load_ds91_preflight_r2_status() -> dict[str, Any]:
    try:
        raw = {relative: _read_exact(relative) for relative in PINS}
        review = _strict_json(raw[REVIEW_RELATIVE])
        _validate(review)
    except (OSError, KeyError, TypeError, ValueError, PreflightR2StatusError):
        return _unavailable()

    return {
        "label": "DeepStream 9.1 tam-yığın ön-uçuş R2",
        "state": "accepted_cpu_preflight_runtime_closed",
        "reason": "runtime_and_module_artifacts_pending",
        "available": True,
        "decision": "ACCEPT",
        "cpu_preflight_accepted": True,
        "execution_ready": False,
        "severity": {"p0": 0, "p1": 0, "p2": 0},
        "tests": {"author_passed": 39, "independent_passed": 44, "failed": 0},
        "matrix": {
            "sources": 12,
            "distinct_video_types": 12,
            "profiles": [640, 960],
            "simulated_streams": 12,
            "measurement_seconds_per_profile": 300,
        },
        "r1_findings_fixed": 5,
        "remaining_blockers": len(EXPECTED_REMAINING_BLOCKERS),
        "gpu_workload_authorized": False,
        "engine_build_authorized": False,
        "benchmark_authorized": False,
        "endurance_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": [
            "ACCEPT yalnız exact-pinli CPU ön-uçuş tanımını kapsar; canlı DeepStream/GPU yeterliliği değildir.",
            "Person, pose ve PPE 640/960 parity, canlı GPU Lease V5 planları, benchmark ve endurance ayrı kapılardır.",
        ],
    }
