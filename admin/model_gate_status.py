"""Fail-closed admin projections for independently accepted model build gates.

These cards expose immutable review evidence only.  They deliberately provide
no endpoint or authority for Docker, GPU, model loading, TensorRT, DeepStream,
publication, quality, or production actions.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PERSON_REVIEW = (
    "validation/results/person/export/rtdetrv4-s-r-livit-person-r11/"
    "r14i-independent-review-r1/receipt.json"
)
PERSON_REVIEW_FINGERPRINT = (
    "4353d1e0ee00fd2201a130af40c201065650a1ca05517a4c7250eb24752787ba"
)
PERSON_PINS: dict[str, tuple[int, str, str]] = {
    PERSON_REVIEW: (
        17397,
        "4d6fe66f8ad7bfe9fd72fe2e271eade187c1bf24423b62cedc67cc1e81585dcc",
        "0440",
    ),
    "validation/person_rtdetrv4_tensorrt_r14i_independent_review.py": (
        71756,
        "f4b7caa61a8a14f049a3b4a61ddb14e1eab7d7c5e19e9a38757164730a2889e1",
        "0555",
    ),
    "validation/schemas/person-rtdetrv4-tensorrt-r14i-independent-review-v1.schema.json": (
        15202,
        "485cf702e14840cf84ff39a2e40187a37bbb4f78ca7668cc0224e0d42d8ef8bb",
        "0440",
    ),
    "tests/test_person_rtdetrv4_tensorrt_r14i_independent_review.py": (
        18263,
        "4e78df50427155af8cd49c5fd4dfd569e6079236d202b3cef6c8753d423efe12",
        "0440",
    ),
    "docs/person-rtdetrv4-tensorrt-r14i-independent-review.md": (
        4142,
        "ec3c535fab0b61d4da513238ddb3ada81dafd9f8b9e5388a8a49136d75485970",
        "0440",
    ),
    "validation/person_rtdetrv4_tensorrt_r14i.py": (
        50034,
        "d1fbadbf2f12f97ae02f4918219479b7429967eca9435bb3c159298b46de11c5",
        "0440",
    ),
    "validation/schemas/person-rtdetrv4-tensorrt-controller-r14i.schema.json": (
        24968,
        "53f7608b46c2b90b8e2e120c8033c6b784eb477df780bffbf303f74bd2f93d85",
        "0440",
    ),
    "tests/test_person_rtdetrv4_tensorrt_r14i.py": (
        21393,
        "37f553ad4217a163fa82a6a0dbc8a843090883e7d02735b3e933d9567270cf02",
        "0440",
    ),
    "docs/person-rtdetrv4-tensorrt-r14i.md": (
        5789,
        "246690ddbdf37e6ca3ca1382dd66cc23e316527cc0140e47bcb183c773ee5f4a",
        "0440",
    ),
    "models/person/export-lanes/rtdetrv4-s-r-livit-person-r11/tensorrt-controller-r14i.json": (
        19658,
        "0893572ab868799901e09ede7425c20d21d57c1cc6252b692533cea8fe08f106",
        "0440",
    ),
    "validation/plans/person-rtdetrv4-tensorrt-workload-template-r14i.json": (
        14244,
        "03ef8234efc65c73977ed4c12cb09332fcfe06d7451c50b2784657531bee138f",
        "0440",
    ),
    "validation/schemas/person-rtdetrv4-tensorrt-workload-template-r14i.schema.json": (
        13728,
        "cbc45ce830da03ccf6e281b8f7fed6d44fe05db11e4eda6e398f8ecd622f003b",
        "0440",
    ),
    "validation/results/person/export/rtdetrv4-s-r-livit-person-r11/r14i-author-handoff-r1/handoff.json": (
        6644,
        "5b026ca9b871bd1ee888e8c09f5aef40fb4534cd2478c651423fc00eed4e1e7c",
        "0440",
    ),
}

PPE_REVIEW = (
    "validation/results/ppe/models/safetyvision-yolov8s-v2-cpu-image-context-r5a3/"
    "phase-b-independent-review-a32-r1-001/receipt.json"
)
PPE_REVIEW_FINGERPRINT = (
    "ece6a92f476d59cd0bdf862bb82bc9153665e1960cf5165a6ad5aaf572e8d7f7"
)
PPE_PINS: dict[str, tuple[int, str, str]] = {
    PPE_REVIEW: (
        23805,
        "bbd5c98b02a9304b918c5cd056847f9a3c14ee7e36b6275096c4b6b8c2d015f2",
        "0440",
    ),
    "validation/ppe_safetyvision_r5_phase_b_a32_independent_review.py": (
        63676,
        "cfbbad6e35bab4bfa38c9cd460833e8fdc579981538dd29df451bc4877027851",
        "0555",
    ),
    "validation/schemas/ppe-safetyvision-r5-phase-b-a32-independent-review-v1.schema.json": (
        17819,
        "307ffed341c2956603eb728846dc735e2f0b0196d98388406c71361a7e414d98",
        "0440",
    ),
    "tests/test_ppe_safetyvision_r5_phase_b_a32_independent_review.py": (
        20147,
        "36fb755148eedbc74141a773d3d5ecdaf4c44031484e05d2e1afa71a33b349f5",
        "0440",
    ),
    "validation/ppe_safetyvision_r5_phase_a32_postpublication.py": (
        26639,
        "6820b6e7f68814279f6ef9858cb1c70abfb2f68c1c470f41d82bb7c5a7fc32b9",
        "0555",
    ),
    "validation/schemas/ppe-safetyvision-r5-phase-a32-postpublication-candidate-v1.schema.json": (
        11503,
        "60d0a692348c80024fe55e8b60a9aa91e24273fbcb7ec4a6b4eb0097720cd91a",
        "0440",
    ),
    "tests/test_ppe_safetyvision_r5_phase_a32_postpublication.py": (
        10373,
        "c4eb5139f3ed3c40b8b2c10f213c0b0e4e1ab1c3d4cbc15090d6463664f285de",
        "0440",
    ),
    "docs/ppe-safetyvision-r5-phase-a32-postpublication.md": (
        3649,
        "35b5d7fd14ff92365673c5fc8c88676bbc22ea306bad5f88502db0ab7ec64bdc",
        "0440",
    ),
    "validation/results/ppe/models/safetyvision-yolov8s-v2-cpu-image-context-r5a3/phase-a3.2-post-publication-candidate-001/handoff.json": (
        16107,
        "72e8c3836a3b7710f6628147d2a0798db09210d8c56ca320bfec9203c0349792",
        "0440",
    ),
}


class ModelGateStatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ModelGateStatusError(message)


def _workspace() -> Path:
    return Path(os.getenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", ROOT))


def _read_exact(
    relative: str, pins: dict[str, tuple[int, str, str]]
) -> bytes:
    expected_size, expected_sha, expected_mode = pins[relative]
    descriptor = os.open(
        _workspace() / relative,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _expect(stat.S_ISREG(before.st_mode), f"not regular: {relative}")
        _expect(before.st_nlink == 1, f"link count differs: {relative}")
        _expect(f"{stat.S_IMODE(before.st_mode):04o}" == expected_mode, f"mode differs: {relative}")
        _expect(before.st_size == expected_size, f"size differs: {relative}")
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
            f"evidence changed while reading: {relative}",
        )
    finally:
        os.close(descriptor)
    _expect(digest.hexdigest() == expected_sha, f"hash differs: {relative}")
    return b"".join(chunks)


def _strict_json(raw: bytes) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            _expect(key not in value, f"duplicate key: {key}")
            value[key] = item
        return value

    def reject_constant(token: str) -> Any:
        raise ModelGateStatusError(f"non-finite token: {token}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelGateStatusError("invalid JSON") from exc
    _expect(isinstance(value, dict), "JSON root differs")
    return value


def _validate_fingerprint(review: dict[str, Any], expected: str) -> None:
    _expect(review.get("review_fingerprint_sha256") == expected, "fingerprint differs")
    unsigned = {
        key: value
        for key, value in review.items()
        if key != "review_fingerprint_sha256"
    }
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    _expect(hashlib.sha256(canonical).hexdigest() == expected, "fingerprint replay differs")


def _pin_projection(row: Any) -> tuple[Any, Any, Any, Any]:
    _expect(isinstance(row, dict), "pin row differs")
    return row.get("path"), row.get("bytes"), row.get("sha256"), row.get("mode")


def _expected_projection(
    relative: str, pins: dict[str, tuple[int, str, str]]
) -> tuple[str, int, str, str]:
    size, digest, mode = pins[relative]
    return relative, size, digest, mode


def _validate_pin_rows(
    rows: Any,
    expected_paths: set[str],
    pins: dict[str, tuple[int, str, str]],
    label: str,
) -> None:
    _expect(isinstance(rows, list), f"{label} rows differ")
    mapped = {row.get("path"): row for row in rows if isinstance(row, dict)}
    _expect(set(mapped) == expected_paths, f"{label} paths differ")
    for relative in expected_paths:
        _expect(
            _pin_projection(mapped[relative]) == _expected_projection(relative, pins),
            f"{label} pin differs: {relative}",
        )


def _person_unavailable() -> dict[str, Any]:
    return {
        "label": "Kişi RT-DETRv4-S TensorRT R14I kapısı",
        "state": "unavailable_integrity_error",
        "reason": "person_r14i_independent_acceptance_integrity_failed",
        "available": False,
        "decision": None,
        "independent_static_acceptance": False,
        "engine_build_authorized": False,
        "gpu_workload_authorized": False,
        "deepstream_runtime_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["R14I exact-pin zinciri doğrulanamadı; bütün yürütme kapıları kapalıdır."],
    }


def _validate_person(review: dict[str, Any]) -> None:
    _expect(
        review.get("schema_version")
        == "deepsafe.person-rtdetrv4-tensorrt-r14i-independent-review/v1",
        "person schema differs",
    )
    _expect(review.get("decision") == "ACCEPT", "person decision differs")
    _expect(
        review.get("status")
        == "accepted_static_execution_closed_controller_definition_not_runtime_or_build_authorization",
        "person status differs",
    )
    _expect(review.get("severity_counts") == {"P0": 0, "P1": 0, "P2": 0}, "person severity differs")
    _expect(review.get("findings") == [], "person findings differ")
    _validate_fingerprint(review, PERSON_REVIEW_FINGERPRINT)

    review_paths = {
        "validation/person_rtdetrv4_tensorrt_r14i_independent_review.py",
        "validation/schemas/person-rtdetrv4-tensorrt-r14i-independent-review-v1.schema.json",
        "tests/test_person_rtdetrv4_tensorrt_r14i_independent_review.py",
        "docs/person-rtdetrv4-tensorrt-r14i-independent-review.md",
    }
    _validate_pin_rows(review.get("review_control_pins"), review_paths, PERSON_PINS, "person review")

    subject = review.get("subject")
    _expect(isinstance(subject, dict), "person subject differs")
    author_artifacts = subject.get("author_artifacts")
    _expect(isinstance(author_artifacts, dict), "person author artifacts differ")
    author_paths = {
        "validation/person_rtdetrv4_tensorrt_r14i.py",
        "validation/schemas/person-rtdetrv4-tensorrt-controller-r14i.schema.json",
        "tests/test_person_rtdetrv4_tensorrt_r14i.py",
        "docs/person-rtdetrv4-tensorrt-r14i.md",
        "models/person/export-lanes/rtdetrv4-s-r-livit-person-r11/tensorrt-controller-r14i.json",
        "validation/plans/person-rtdetrv4-tensorrt-workload-template-r14i.json",
        "validation/schemas/person-rtdetrv4-tensorrt-workload-template-r14i.schema.json",
    }
    _validate_pin_rows(list(author_artifacts.values()), author_paths, PERSON_PINS, "person author")
    handoff_path = (
        "validation/results/person/export/rtdetrv4-s-r-livit-person-r11/"
        "r14i-author-handoff-r1/handoff.json"
    )
    handoff = subject.get("handoff")
    _expect(isinstance(handoff, dict), "person handoff differs")
    _expect(
        _pin_projection(handoff.get("pin"))
        == _expected_projection(handoff_path, PERSON_PINS),
        "person handoff pin differs",
    )
    _expect(handoff.get("self_fingerprint_valid") is True, "person handoff fingerprint differs")
    _expect(handoff.get("terminal_accepted") is False, "person handoff overclaim")

    tests = review.get("independent_test_replay")
    _expect(
        isinstance(tests, dict)
        and tests.get("collected") == 67
        and tests.get("passed") == 67
        and tests.get("failed") == 0,
        "person tests differ",
    )
    scope = review.get("accepted_scope")
    _expect(isinstance(scope, dict), "person scope differs")
    _expect(scope.get("profiles") == [640, 960], "person profiles differ")
    _expect(scope.get("batch_min_opt_max") == [1, 12, 12], "person batch differs")
    _expect(scope.get("precision") == "fp16_tf32_off_int8_off", "person precision differs")
    _expect(scope.get("static_r14i_controller_definition") is True, "person static scope differs")
    for key in (
        "runtime_execution_authorized",
        "engine_build_authorized",
        "config_publication_authorized",
        "parity_authorized",
        "gpu_lease_activation_authorized",
        "production_ready",
    ):
        _expect(scope.get(key) is False, f"person scope overclaim: {key}")
    permissions = review.get("permissions")
    _expect(isinstance(permissions, dict) and permissions and all(value is False for value in permissions.values()), "person permissions overclaim")
    resource = review.get("resource_boundary")
    _expect(isinstance(resource, dict) and resource and all(value is False for value in resource.values()), "person resource overclaim")
    profiles = review.get("verification", {}).get("profiles", {})
    _expect(profiles.get("output_count") == 18, "person output count differs")
    profile_rows = profiles.get("profiles")
    _expect(isinstance(profile_rows, dict) and set(profile_rows) == {"640", "960"}, "person profile rows differ")
    _expect(all(row.get("all_outputs_absent") is True for row in profile_rows.values()), "person outputs unexpectedly present")


def load_person_r14i_status() -> dict[str, Any]:
    try:
        raw = {relative: _read_exact(relative, PERSON_PINS) for relative in PERSON_PINS}
        review = _strict_json(raw[PERSON_REVIEW])
        _validate_person(review)
        handoff_path = (
            "validation/results/person/export/rtdetrv4-s-r-livit-person-r11/"
            "r14i-author-handoff-r1/handoff.json"
        )
        handoff = _strict_json(raw[handoff_path])
        author_tests = handoff.get("author_tests")
        _expect(
            isinstance(author_tests, dict)
            and author_tests.get("collected") == 50
            and author_tests.get("passed") == 50
            and author_tests.get("failed") == 0,
            "person author tests differ",
        )
    except (OSError, KeyError, TypeError, ModelGateStatusError):
        return _person_unavailable()

    return {
        "label": "Kişi RT-DETRv4-S TensorRT R14I kapısı",
        "state": "independent_static_controller_accepted_execution_closed",
        "reason": "engine_build_runtime_parity_and_publication_gates_pending",
        "available": True,
        "decision": "ACCEPT",
        "severity": {"p0": 0, "p1": 0, "p2": 0},
        "independent_static_acceptance": True,
        "profiles": [640, 960],
        "batch": {"min": 1, "opt": 12, "max": 12},
        "precision": "FP16 (TF32 kapalı, INT8 kapalı)",
        "tests": {"author_passed": 50, "independent_passed": 67, "combined_passed": 117, "failed": 0},
        "planned_outputs": 18,
        "outputs_present": 0,
        "next_gate": "DeepStream 9.1 GPU runtime smoke kabulü; ardından ayrı Lease V5 engine-build planları",
        "engine_build_authorized": False,
        "gpu_workload_authorized": False,
        "deepstream_runtime_authorized": False,
        "parity_authorized": False,
        "config_publication_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "does_not_imply_product_readiness": True,
        "evidence": [],
        "caveats": [
            "ACCEPT yalnız 640/960 TensorRT kontrol tanımına aittir; engine veya inference üretilmedi.",
            "GPU, DeepStream, parity, config yayını ve production yetkileri kapalıdır.",
        ],
    }


EXPECTED_PPE_AUTHORITY = {
    "checkpoint_deserialization_or_export_authorized": False,
    "context_copy_authorized": False,
    "cpu_image_build_execution_authorized": False,
    "cpu_parity": False,
    "docker_authorized": False,
    "docker_review_completed": False,
    "gpu_authorized": False,
    "model_acceptance": False,
    "model_or_onnx_load_authorized": False,
    "phase_b_context_accepted": True,
    "production_ready": False,
    "quality_validated": False,
    "same_uid_kernel_immutability_claimed": False,
    "separate_exact_cpu_image_build_gate_entry_allowed": True,
    "subject_mutation_authorized": False,
    "tensorrt_or_deepstream_authorized": False,
}


def _ppe_unavailable() -> dict[str, Any]:
    return {
        "label": "PPE SafetyVision A3.2 build-context kapısı",
        "state": "unavailable_integrity_error",
        "reason": "ppe_a32_independent_acceptance_integrity_failed",
        "available": False,
        "decision": None,
        "phase_b_context_accepted": False,
        "cpu_image_build_authorized": False,
        "gpu_workload_authorized": False,
        "tensorrt_or_deepstream_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["PPE A3.2 exact-pin zinciri doğrulanamadı; bütün yürütme kapıları kapalıdır."],
    }


def _validate_ppe(review: dict[str, Any]) -> None:
    _expect(
        review.get("schema_version")
        == "deepsafe.ppe-safetyvision-phase-b-a32-independent-review/v1",
        "PPE schema differs",
    )
    _expect(review.get("decision") == "ACCEPT", "PPE decision differs")
    _expect(review.get("severity_counts") == {"P0": 0, "P1": 0, "P2": 0}, "PPE severity differs")
    _expect(review.get("findings") == [], "PPE findings differ")
    _validate_fingerprint(review, PPE_REVIEW_FINGERPRINT)
    _expect(review.get("authority") == EXPECTED_PPE_AUTHORITY, "PPE authority differs")

    review_paths = {
        "validation/ppe_safetyvision_r5_phase_b_a32_independent_review.py",
        "validation/schemas/ppe-safetyvision-r5-phase-b-a32-independent-review-v1.schema.json",
        "tests/test_ppe_safetyvision_r5_phase_b_a32_independent_review.py",
    }
    _validate_pin_rows(review.get("review_source_pins"), review_paths, PPE_PINS, "PPE review")
    replay = review.get("subject_replay")
    _expect(isinstance(replay, dict), "PPE subject replay differs")
    author_paths = {
        "validation/ppe_safetyvision_r5_phase_a32_postpublication.py",
        "validation/schemas/ppe-safetyvision-r5-phase-a32-postpublication-candidate-v1.schema.json",
        "tests/test_ppe_safetyvision_r5_phase_a32_postpublication.py",
        "docs/ppe-safetyvision-r5-phase-a32-postpublication.md",
    }
    _validate_pin_rows(replay.get("author_control_pins"), author_paths, PPE_PINS, "PPE author")
    candidate_path = (
        "validation/results/ppe/models/safetyvision-yolov8s-v2-cpu-image-context-r5a3/"
        "phase-a3.2-post-publication-candidate-001/handoff.json"
    )
    _expect(
        _pin_projection(replay.get("candidate_pin"))
        == _expected_projection(candidate_path, PPE_PINS),
        "PPE candidate pin differs",
    )
    _expect(replay.get("source_count") == 39, "PPE source count differs")
    _expect(replay.get("receipts_strict") is True and len(replay.get("receipt_pins", [])) == 6, "PPE receipt chain differs")
    surface = replay.get("execution_surface")
    _expect(isinstance(surface, dict), "PPE execution surface differs")
    for key in (
        "checkpoint_or_model_loaded",
        "gpu_called",
        "image_or_docker_called",
        "onnx_loaded",
        "tensorrt_or_deepstream_called",
    ):
        _expect(surface.get(key) is False, f"PPE execution overclaim: {key}")
    trees = replay.get("trees")
    _expect(isinstance(trees, dict) and trees.get("replayed") is True, "PPE tree replay differs")
    closure = trees.get("a3_symlink_closure")
    _expect(
        isinstance(closure, dict)
        and closure.get("verified") is True
        and closure.get("symlinks") == 47
        and closure.get("cycles") == 0
        and closure.get("dangling") == 0
        and closure.get("root_escapes") == 0,
        "PPE symlink closure differs",
    )
    mode = trees.get("a3_mode_inventory")
    _expect(
        isinstance(mode, dict)
        and mode.get("special_entries") == 0
        and mode.get("writable_directories_or_regular_files") == 0,
        "PPE mode inventory differs",
    )
    tests = review.get("test_replay")
    _expect(
        isinstance(tests, dict)
        and tests.get("independent_collected") == 54
        and tests.get("independent_passed") == 54
        and tests.get("independent_failed") == 0
        and tests.get("full_author_subject_replay") == "pass",
        "PPE tests differ",
    )
    author_runs = tests.get("author_runs")
    _expect(
        isinstance(author_runs, list)
        and len(author_runs) == 2
        and all(row.get("collected") == 25 and row.get("passed") == 25 and row.get("failed") == 0 for row in author_runs),
        "PPE author test runs differ",
    )
    idempotency = review.get("idempotency_replay")
    _expect(
        isinstance(idempotency, dict)
        and idempotency.get("runs") == 2
        and idempotency.get("same_projection") is True
        and idempotency.get("published_subject_deleted_or_republished") is False,
        "PPE idempotency differs",
    )
    next_gate = review.get("next_gate")
    _expect(
        next_gate
        == {
            "all_later_model_parity_gpu_tensorrt_deepstream_quality_production_gates_remain_closed": True,
            "allowed": True,
            "image_build_executes_here": False,
            "name": "separate exact CPU image build gate",
        },
        "PPE next gate differs",
    )


def load_ppe_a32_status() -> dict[str, Any]:
    try:
        raw = {relative: _read_exact(relative, PPE_PINS) for relative in PPE_PINS}
        review = _strict_json(raw[PPE_REVIEW])
        _validate_ppe(review)
    except (OSError, KeyError, TypeError, ModelGateStatusError):
        return _ppe_unavailable()

    trees = review["subject_replay"]["trees"]
    return {
        "label": "PPE SafetyVision A3.2 build-context kapısı",
        "state": "independent_phase_b_context_accepted_build_closed",
        "reason": "separate_exact_cpu_image_build_gate_pending",
        "available": True,
        "decision": "ACCEPT",
        "severity": {"p0": 0, "p1": 0, "p2": 0},
        "phase_b_context_accepted": True,
        "tests": {"author_runs": 2, "author_passed_each": 25, "independent_passed": 54, "failed": 0},
        "integrity": {
            "source_pins": 39,
            "strict_receipts": 6,
            "symlinks": 47,
            "cycles": 0,
            "dangling": 0,
            "root_escapes": 0,
            "writable_or_special_entries": 0,
            "tree_sha256": trees["a3_full_tree"]["tree_sha256"],
        },
        "next_gate": "Ayrı exact CPU image-build kapısı",
        "cpu_image_build_authorized": False,
        "context_copy_authorized": False,
        "model_or_onnx_load_authorized": False,
        "gpu_workload_authorized": False,
        "tensorrt_or_deepstream_authorized": False,
        "quality_validated": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "does_not_imply_product_readiness": True,
        "evidence": [],
        "caveats": [
            "ACCEPT yalnız dondurulmuş CPU build-context bütünlüğüne aittir; imaj build edilmedi.",
            "Model/ONNX yükleme, GPU, TensorRT, DeepStream, kalite ve production kapıları kapalıdır.",
        ],
    }
