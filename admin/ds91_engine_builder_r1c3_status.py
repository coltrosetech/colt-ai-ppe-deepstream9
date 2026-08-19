"""Fail-closed admin projection for the independently accepted R1C3 image."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = "sha256:5ebb850cb11c046cf8d924dcea0e052167911e8f9a28b04b384dfa18026bdd6c"
CONFIG_ID = "sha256:71f057c648640fb7d27d2097f9c3691d98fee4cafb084f6ddddf7f581bde967a"
PINS = {
    "validation/accepted-roots/ds91-engine-builder-r1c3-terminal/terminal-root-r1c3.json": (2926, "57805489b3c0238e9cc5b13802ef644a9f43346629b6d444a7f89de5a9d54a49"),
    "validation/results/ds91-engine-builder/r1c3-independent-review/terminal-review-r1c3.json": (5090, "446c435e2bc45b22d5e5334e0aa722aad1d7786f5d0e7085a2aff0a58c7d133e"),
    "validation/results/ds91-engine-builder/r1c3/candidate-receipt-r1c3.json": (11849, "6acdc00e90441d7f7e3f923b597365a74cc6df9bb13619291e08a6db36b6c75a"),
    "validation/accepted-roots/ds91-engine-builder-r1c3-plan/release-plan-r1c3.json": (20252, "c9e16441416898582c2044b96e700f95aa37301e9b018f42737ce19dd8873f59"),
    "deepstream/ds91-engine-builder-r1c3/acceptance_gate_r1c3.py": (13311, "9d7c020526d46cde6d5fbca7cae0ca2968dd40dec3382f60b3c293cd7f3dfb15"),
    "validation/ds91_engine_builder_r1c3_terminal_verify.py": (53933, "d98999d60a312721d25c5fd1808b777a7e52f1e1bf04db7983be2e7981cd9172"),
    "validation/schemas/ds91-engine-builder-independent-review-v1.schema.json": (10704, "6387999acd168aef78d9ac0c48399f5c3eef528cd9e769fa3906feb73681c4d6"),
    "validation/schemas/ds91-engine-builder-terminal-root-v1.schema.json": (5932, "348417c214d5ceeaf5543c9d3a860665c34e7324d64c45b0782d3e22f0a25b3a"),
}


class EngineBuilderStatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise EngineBuilderStatusError(message)


def _resolve(relative: str) -> Path:
    if relative.startswith("deepstream/"):
        return ROOT / relative
    workspace = Path(os.getenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", ROOT))
    return workspace / relative


def _read_exact(relative: str) -> bytes:
    path = _resolve(relative)
    expected_size, expected_sha = PINS[relative]
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _expect(stat.S_ISREG(before.st_mode), f"not regular: {relative}")
        _expect(before.st_nlink == 1, f"link count differs: {relative}")
        _expect(before.st_mode & 0o222 == 0, f"writable evidence: {relative}")
        _expect(before.st_size == expected_size, f"size differs: {relative}")
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
            f"evidence changed while reading: {relative}",
        )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    _expect(digest.hexdigest() == expected_sha, f"hash differs: {relative}")
    return raw


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            _expect(key not in value, f"duplicate key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EngineBuilderStatusError(f"non-finite token in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineBuilderStatusError(f"invalid JSON: {label}") from exc
    _expect(isinstance(value, dict), f"non-object JSON: {label}")
    return value


def _self_fingerprint(value: dict[str, Any], expected: str, label: str) -> None:
    fingerprint = value.get("self_fingerprint")
    _expect(isinstance(fingerprint, dict), f"fingerprint missing: {label}")
    _expect(
        fingerprint == {
            "algorithm": "sha256-canonical-json-without-self_fingerprint",
            "canonical_sha256": expected,
        },
        f"fingerprint declaration differs: {label}",
    )
    unsigned = {key: item for key, item in value.items() if key != "self_fingerprint"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    _expect(hashlib.sha256(raw).hexdigest() == expected, f"fingerprint replay differs: {label}")


def _pin(relative: str) -> dict[str, Any]:
    size, digest = PINS[relative]
    return {"path": relative, "bytes": size, "sha256": digest}


def _validate(root: dict[str, Any], review: dict[str, Any], candidate: dict[str, Any]) -> None:
    accepted_scope = {
        "terminal_accepted": True,
        "prepared_closed_engine_builder_image_identity": True,
        "evidence_chain": True,
        "gpu_runtime": False,
        "tensorrt_engine_build": False,
        "deepstream_runtime": False,
        "inference": False,
        "production": False,
    }
    qualification = {
        "deepstream_runtime_qualified": False,
        "gpu_runtime_qualified": False,
        "inference_qualified": False,
        "production_ready": False,
        "tensorrt_engine_qualified": False,
    }
    _expect(root.get("schema_version") == "deepsafe.ds91-engine-builder-terminal-accepted-root/v1", "root schema differs")
    _expect(root.get("status") == "accepted_terminal_prepared_closed_image_identity_and_evidence_only", "root status differs")
    _expect(root.get("accepted_scope") == accepted_scope, "root scope differs")
    _expect(root.get("qualification") == qualification, "root qualification differs")
    _expect(root.get("review") == _pin("validation/results/ds91-engine-builder/r1c3-independent-review/terminal-review-r1c3.json"), "root review pin differs")
    expected_subject = {
        "candidate": _pin("validation/results/ds91-engine-builder/r1c3/candidate-receipt-r1c3.json"),
        "gate": _pin("deepstream/ds91-engine-builder-r1c3/acceptance_gate_r1c3.py"),
        "release_plan": _pin("validation/accepted-roots/ds91-engine-builder-r1c3-plan/release-plan-r1c3.json"),
    }
    _expect(root.get("subject") == expected_subject, "root subject pins differ")
    _expect(root.get("authority_boundary", {}).get("model") == "cooperative_same_uid", "authority model differs")
    _expect(root.get("authority_boundary", {}).get("malicious_same_uid_resistance_claimed") is False, "authority overclaim")
    _self_fingerprint(root, "5c5a4868fb06bd5adda4d7b1f623eabe5d5e93b60dfe11bbbaa4f76fc48dc09e", "terminal root")

    _expect(review.get("schema_version") == "deepsafe.ds91-engine-builder-independent-terminal-review/v1", "review schema differs")
    _expect(review.get("status") == "terminal_accepted_prepared_closed_image_identity_and_evidence_only", "review status differs")
    _expect(review.get("scope") == accepted_scope and review.get("qualification") == qualification, "review scope differs")
    _expect(review.get("subject") == expected_subject, "review subject pins differ")
    _expect(review.get("findings") == {"acceptance_blockers": 0, "p0": 0, "p1": 0, "p2": 0}, "review findings differ")
    _expect(review.get("evidence", {}).get("image_id") == IMAGE_ID, "review image differs")
    _expect(review.get("evidence", {}).get("config") == CONFIG_ID, "review config differs")
    _expect(review.get("replays", {}).get("unit_tests") == {"r1c": 17, "r1c2": 7, "r1c3": 6, "total": 30, "failed": 0}, "unit replay differs")
    _expect(review.get("replays", {}).get("mutation_tests") == {"passed": 20, "failed": 0}, "mutation replay differs")
    _self_fingerprint(review, "317db743c54d4cfd7dc881672de6803e177ae7f9e993bb7f3d69011636b27dd0", "independent review")

    _expect(candidate.get("schema_version") == "deepsafe.ds91-engine-builder-candidate-receipt/v1c3", "candidate schema differs")
    _expect(candidate.get("status") == "candidate_passed_not_terminal" and candidate.get("revision") == 3, "candidate status differs")
    _expect(candidate.get("image") == {"config": CONFIG_ID, "id": IMAGE_ID, "manifest": IMAGE_ID, "publication_tag": "deepsafe-ds91-engine-builder:r1c3-prepared-closed"}, "candidate image differs")
    _expect(candidate.get("qualification") == {"candidate_only": True, "independent_review_required": True, "production_ready": False, "terminal_accepted": False}, "candidate boundary differs")
    _self_fingerprint(candidate, "2d8d010646b8906ae9e1d8a85c8c14bf0dec2e656a741cbde7e803cb463dcbfa", "candidate")


def _unavailable() -> dict[str, Any]:
    return {
        "label": "DeepStream 9.1 exact engine-builder R1C3",
        "state": "unavailable_integrity_error",
        "reason": "ds91_engine_builder_r1c3_integrity_failed",
        "available": False,
        "terminal_accepted": False,
        "prepared_closed_image_identity_accepted": False,
        "gpu_runtime_qualified": False,
        "tensorrt_engine_qualified": False,
        "deepstream_runtime_qualified": False,
        "inference_qualified": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["R1C3 dış accepted-root zinciri exact-pin doğrulamasından geçmedi."],
    }


def load_ds91_engine_builder_r1c3_status() -> dict[str, Any]:
    try:
        raw = {relative: _read_exact(relative) for relative in PINS}
        root = _strict_json(raw["validation/accepted-roots/ds91-engine-builder-r1c3-terminal/terminal-root-r1c3.json"], "terminal root")
        review = _strict_json(raw["validation/results/ds91-engine-builder/r1c3-independent-review/terminal-review-r1c3.json"], "independent review")
        candidate = _strict_json(raw["validation/results/ds91-engine-builder/r1c3/candidate-receipt-r1c3.json"], "candidate")
        _validate(root, review, candidate)
    except (OSError, KeyError, TypeError, EngineBuilderStatusError):
        return _unavailable()

    return {
        "label": "DeepStream 9.1 exact engine-builder R1C3",
        "state": "terminal_accepted_prepared_closed_identity_only",
        "reason": "gpu_engine_and_runtime_validation_pending",
        "available": True,
        "terminal_accepted": True,
        "prepared_closed_image_identity_accepted": True,
        "image": {"id": IMAGE_ID, "config": CONFIG_ID, "tag": candidate["image"]["publication_tag"], "size_bytes": review["live_observations"]["image_size"]},
        "tests": {"unit_passed": 30, "mutation_passed": 20, "gate_replays_passed": 6, "failed": 0, "p0": 0, "p1": 0, "p2": 0},
        "evidence_counts": {"author_files": 23, "raw_files": 18, "alias": 5, "retag": 4},
        "gpu_runtime_qualified": False,
        "tensorrt_engine_qualified": False,
        "deepstream_runtime_qualified": False,
        "inference_qualified": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": [
            "Kabul yalnız prepared-closed image kimliği ve kanıt zinciridir.",
            "TensorRT engine üretimi, GPU, DeepStream pipeline, inference ve ürün kabulü kapsam dışıdır.",
        ],
    }
