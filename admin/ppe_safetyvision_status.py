"""Read-only admin projection for the SafetyVision PPE challenger."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATUS = "acquired_static_verified_not_executed_not_accepted"
PINS = {
    "manifest": (
        4519,
        "6b38eec658952fc5d2327625c1fc631df2ba160e1c64ca72cc2ce8a8e8254e30",
    ),
    "validator": (
        10564,
        "35f49b46a7d788d3114a0f6aee047a0f5b4b3b1b2e6819875318dc5bd9ea5900",
    ),
    "receipt": (
        2421,
        "98da1f733539df308aae94e77c2d673696c52925d94ff3472024eed1e136ef6a",
    ),
}


class SafetyVisionStatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise SafetyVisionStatusError(message)


def _roots() -> tuple[Path, Path]:
    workspace = Path(os.getenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", ROOT))
    validation = Path(
        os.getenv("DEEPSAFE_VALIDATION_ROOT", workspace / "validation/results")
    )
    return workspace, validation


def _read_exact(path: Path, pin: tuple[int, str]) -> bytes:
    expected_bytes, expected_sha256 = pin
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        _expect(stat.S_ISREG(before.st_mode), f"not regular: {path}")
        _expect(before.st_nlink == 1, f"unexpected link count: {path}")
        _expect(before.st_mode & 0o222 == 0, f"writable evidence: {path}")
        _expect(before.st_size == expected_bytes, f"size drift: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        _expect(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"evidence changed while reading: {path}",
        )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    _expect(len(raw) == expected_bytes, f"byte count drift: {path}")
    _expect(hashlib.sha256(raw).hexdigest() == expected_sha256, f"hash drift: {path}")
    return raw


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _expect(key not in result, f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SafetyVisionStatusError(f"non-finite number in {label}: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetyVisionStatusError(f"invalid JSON: {label}") from exc
    _expect(isinstance(value, dict), f"non-object JSON: {label}")
    return value


def _validate(manifest: dict[str, Any], receipt: dict[str, Any]) -> None:
    _expect(
        manifest.get("schema_version")
        == "deepsafe.ppe-model-challenger-quarantine/v1",
        "manifest schema differs",
    )
    _expect(manifest.get("status") == STATUS, "manifest status differs")
    _expect(
        receipt.get("schema_version")
        == "deepsafe.ppe-model-challenger-admin-receipt/v1",
        "receipt schema differs",
    )
    _expect(receipt.get("status") == STATUS, "receipt status differs")
    source = receipt["source"]
    _expect(
        source
        == {
            "repository": "ayushgupta7777/safetyvision-yolov8",
            "commit": "56a71758b55f0e9f2b4b2d6b51a779a1f882da10",
            "license_id": "AGPL-3.0",
        },
        "source differs",
    )
    controls = receipt["control_pins"]
    _expect(
        controls["manifest"]
        == {
            "path": "data/manifests/ppe-safetyvision-yolov8s-v2-challenger-r1.json",
            "bytes": PINS["manifest"][0],
            "sha256": PINS["manifest"][1],
        },
        "manifest receipt pin differs",
    )
    _expect(
        controls["validator"]
        == {
            "path": "validation/ppe_safetyvision_challenger.py",
            "bytes": PINS["validator"][0],
            "sha256": PINS["validator"][1],
        },
        "validator receipt pin differs",
    )
    _expect(
        receipt["artifact_pins"]
        == {
            "onnx_640": {
                "bytes": 44_764_727,
                "sha256": "ea18ae903a566e8fa76f3ee1c503075522dca269269315e9c862efa170430b35",
            },
            "onnx_896": {
                "bytes": 44_926_046,
                "sha256": "b250353639e01800f9cbe79c6002b8b041bdae7560328b8e18ad4a42dc3844e1",
            },
        },
        "artifact receipt pins differ",
    )
    mapping = receipt["static_audit"]["runtime_class_mapping"]
    _expect(
        mapping == {"helmet": 3, "no_helmet": 7, "hi_vis": 12, "no_hi_vis": 9},
        "runtime mapping differs",
    )
    _expect(receipt["static_audit"]["onnx_checker_passed"] == 2, "ONNX checker differs")
    _expect(receipt["static_audit"]["unit_tests_passed"] == 10, "unit tests differ")
    readiness = receipt["readiness"]
    _expect(readiness["accepted_model"] is False, "model acceptance overclaim")
    for key in (
        "target_960_present",
        "dynamic_batch_12_present",
        "deepstream91_parser_parity_tested",
        "tensorrt_engine_built",
        "overhead_camera_qualified",
        "production_ready",
    ):
        _expect(readiness[key] is False, f"readiness overclaim: {key}")
    boundary = receipt["execution_boundary"]
    for key in (
        "cpu_inference_executed",
        "gpu_inference_executed",
        "tensorrt_or_deepstream_executed",
        "training_or_re_export_executed",
        "admin_execution_actions_available",
        "onnx_sha256_live_rechecked_by_admin",
    ):
        _expect(boundary[key] is False, f"execution overclaim: {key}")
    _expect(boundary["admin_read_only"] is True, "admin boundary differs")


def _unavailable() -> dict[str, Any]:
    return {
        "label": "SafetyVision PPE YOLOv8s challenger",
        "state": "unavailable_integrity_error",
        "reason": "ppe_challenger_evidence_integrity_failed",
        "available": False,
        "ready": False,
        "accepted_model": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["PPE challenger kanıtı exact-pin doğrulamasından geçmedi."],
    }


def load_safetyvision_challenger_status() -> dict[str, Any]:
    workspace, validation_root = _roots()
    try:
        manifest = _strict_object(
            _read_exact(
                workspace
                / "data/manifests/ppe-safetyvision-yolov8s-v2-challenger-r1.json",
                PINS["manifest"],
            ),
            "manifest",
        )
        _read_exact(
            workspace / "validation/ppe_safetyvision_challenger.py",
            PINS["validator"],
        )
        receipt = _strict_object(
            _read_exact(
                validation_root / "ppe/safetyvision-yolov8s-v2-challenger-r1/receipt.json",
                PINS["receipt"],
            ),
            "receipt",
        )
        _validate(manifest, receipt)
    except (OSError, KeyError, TypeError, SafetyVisionStatusError):
        return _unavailable()

    readiness = receipt["readiness"]
    metrics = manifest["author_reported_metrics_not_locally_reproduced"]
    return {
        "label": "SafetyVision PPE YOLOv8s challenger",
        "state": "challenger_static_verified",
        "reason": "fixed_batch1_no_960_overhead_and_runtime_validation_pending",
        "available": True,
        "ready": False,
        "accepted_model": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "model": {
            "family": manifest["source"]["model_family"],
            "repository": manifest["source"]["repository"],
            "commit": manifest["source"]["commit"],
            "license": manifest["source"]["license_id"],
            "runtime_class_mapping": receipt["static_audit"]["runtime_class_mapping"],
        },
        "artifacts": {
            "640": {"present": True, "fixed_batch": 1, **receipt["artifact_pins"]["onnx_640"]},
            "896": {"present": True, "fixed_batch": 1, **receipt["artifact_pins"]["onnx_896"]},
            "960": {"present": False},
        },
        "reported_metrics": {
            "locally_reproduced": False,
            "onnx_640_map50": metrics["onnx_640_map50"],
            "onnx_640_map50_95": metrics["onnx_640_map50_95"],
            "no_safety_vest_map50": metrics["pt_896_class_map50"]["NO-Safety Vest"],
        },
        "readiness": dict(readiness),
        "tests": {
            "onnx_checker_passed": receipt["static_audit"]["onnx_checker_passed"],
            "unit_passed": receipt["static_audit"]["unit_tests_passed"],
            "failed": 0,
        },
        "integrity": {
            "manifest_live_exact_pin_verified": True,
            "validator_live_exact_pin_verified": True,
            "receipt_live_exact_pin_verified": True,
            "onnx_sha256_verified_by_offline_receipt": True,
            "onnx_sha256_live_rechecked_by_admin": False,
        },
        "production_ready": False,
        "caveats": [
            "Dört hedef PPE sınıfı mevcut; iki ONNX exact SHA ve statik graph checker ile doğrulandı.",
            "Dosyalar fixed batch 1 ve 640/896'dır; dynamic batch 12 ve exact 960 henüz yoktur.",
            "Yazar üst/drone kamerayı kapsam dışı sayıyor; NO-Safety Vest en zayıf sınıftır.",
            "Yayımlanan metrikler yerelde yeniden üretilmedi; GPU, inference, TensorRT ve DeepStream çalıştırılmadı.",
        ],
        "evidence": [],
    }
