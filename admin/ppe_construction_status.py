"""Read-only admin projection for the quarantined Construction-PPE source."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PIN = (
    3535,
    "b0c46198c165dd8eec640dc080a3416ac9d5f89f5bab885d2bcac8677e8cc9f4",
)
VALIDATOR_PIN = (
    11220,
    "42e1f886c46a9b0d9cb1b0c02694248d2291ed755e3a83a9b1eb1c91e4ef1f84",
)
RECEIPT_PIN = (
    1963,
    "6385e2e25536677035cd5f51e8856c948318fda20ec6b3078af3484c3571710c",
)
STATUS = "quarantined_offline_diagnostic_only_not_training_or_final_gt_authorized"


class ConstructionPPEStatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ConstructionPPEStatusError(message)


def _roots() -> tuple[Path, Path]:
    workspace = Path(os.getenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", ROOT))
    validation = Path(
        os.getenv("DEEPSAFE_VALIDATION_ROOT", workspace / "validation/results")
    )
    return workspace, validation


def _read_exact(path: Path, pin: tuple[int, str]) -> bytes:
    expected_bytes, expected_sha256 = pin
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _expect(stat.S_ISREG(before.st_mode), f"not regular: {path}")
        _expect(before.st_nlink == 1, f"unexpected link count: {path}")
        _expect(before.st_size == expected_bytes, f"size drift: {path}")
        _expect(before.st_mode & 0o222 == 0, f"writable evidence: {path}")
        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining:
            block = os.read(descriptor, min(64 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        _expect(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"evidence changed while reading: {path}",
        )
    finally:
        os.close(descriptor)
    _expect(len(raw) == expected_bytes, f"byte count drift: {path}")
    _expect(hashlib.sha256(raw).hexdigest() == expected_sha256, f"hash drift: {path}")
    return raw


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _expect(key not in result, f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ConstructionPPEStatusError(f"non-finite number in {label}: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConstructionPPEStatusError(f"invalid JSON: {label}") from exc
    _expect(isinstance(value, dict), f"non-object JSON: {label}")
    return value


def _validate(manifest: dict[str, Any], receipt: dict[str, Any]) -> None:
    _expect(
        manifest.get("schema_version") == "deepsafe.ppe-construction-ppe-quarantine/v1",
        "manifest schema differs",
    )
    _expect(manifest.get("status") == STATUS, "manifest status differs")
    _expect(
        receipt.get("schema_version")
        == "deepsafe.ppe-construction-ppe-admin-receipt/v1",
        "receipt schema differs",
    )
    _expect(receipt.get("status") == STATUS, "receipt status differs")
    inputs = receipt["inputs"]
    _expect(
        inputs["manifest"]
        == {
            "path": "data/manifests/ppe-ultralytics-construction-ppe-quarantine-r1.json",
            "bytes": MANIFEST_PIN[0],
            "sha256": MANIFEST_PIN[1],
        },
        "manifest receipt pin differs",
    )
    _expect(
        inputs["validator"]
        == {
            "path": "validation/ppe_construction_ppe_quarantine.py",
            "bytes": VALIDATOR_PIN[0],
            "sha256": VALIDATOR_PIN[1],
        },
        "validator receipt pin differs",
    )
    _expect(
        inputs["archive"]
        == {
            key: manifest["archive"][key]
            for key in ("path", "bytes", "sha256")
        },
        "archive projection differs",
    )
    auth = manifest["authorization"]
    receipt_auth = receipt["authorization"]
    _expect(auth["offline_diagnostic_model_evaluation"] is True, "diagnostic gate differs")
    for key in ("training", "threshold_calibration", "independent_final_ground_truth", "product_acceptance"):
        _expect(auth[key] is False and receipt_auth[key] is False, f"authorization overclaim: {key}")
    _expect(
        manifest["split_leakage_diagnostic"]["published_random_split_is_independent_gt"]
        is False,
        "independent GT overclaim",
    )
    audit = receipt["audit"]
    dataset = manifest["extracted_dataset"]
    _expect(audit["decoded_image_count"] == dataset["decoded_image_count"] == 1416, "image count differs")
    _expect(audit["paired_box_count"] == sum(dataset["paired_box_counts"].values()) == 11521, "box count differs")
    _expect(audit["orphan_label_file_count"] == len(dataset["orphan_label_files"]) == 10, "orphan count differs")
    _expect(
        audit["cross_split_phash_pair_counts_by_max_hamming"]
        == manifest["split_leakage_diagnostic"]["cross_split_pair_counts_by_max_hamming"],
        "leakage projection differs",
    )
    _expect(receipt["tests"] == {"passed": 4, "failed": 0, "gpu_docker_deepstream_used": False}, "test receipt differs")
    readiness = receipt["model_readiness"]
    _expect(all(readiness[key] is False for key in readiness), "model readiness overclaim")
    boundary = receipt["admin_boundary"]
    _expect(boundary["read_only"] is True, "admin write boundary differs")
    _expect(boundary["execution_actions_available"] is False, "admin execution boundary differs")
    _expect(boundary["archive_sha256_live_rechecked_by_admin"] is False, "live archive claim differs")


def _unavailable() -> dict[str, Any]:
    return {
        "label": "Construction-PPE tanısal karantinası",
        "state": "unavailable_integrity_error",
        "reason": "construction_ppe_evidence_integrity_failed",
        "available": False,
        "ready": False,
        "training_authorized": False,
        "independent_final_ground_truth_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["Construction-PPE kartı exact-pin doğrulamasından geçmedi."],
    }


def load_construction_ppe_status() -> dict[str, Any]:
    workspace, validation_root = _roots()
    try:
        manifest = _strict_object(
            _read_exact(
                workspace
                / "data/manifests/ppe-ultralytics-construction-ppe-quarantine-r1.json",
                MANIFEST_PIN,
            ),
            "manifest",
        )
        _read_exact(workspace / "validation/ppe_construction_ppe_quarantine.py", VALIDATOR_PIN)
        receipt = _strict_object(
            _read_exact(
                validation_root / "ppe/construction-ppe-quarantine-r1/receipt.json",
                RECEIPT_PIN,
            ),
            "receipt",
        )
        _validate(manifest, receipt)
    except (OSError, KeyError, TypeError, ConstructionPPEStatusError):
        return _unavailable()

    dataset = manifest["extracted_dataset"]
    leakage = manifest["split_leakage_diagnostic"]
    return {
        "label": "Construction-PPE tanısal karantinası",
        "state": "quarantined_diagnostic_only",
        "reason": "split_leakage_no_hi_vis_class_and_model_artifacts_missing",
        "available": True,
        "ready": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "offline_diagnostic_model_evaluation_authorized": True,
        "training_authorized": False,
        "threshold_calibration_authorized": False,
        "independent_final_ground_truth_authorized": False,
        "production_ready": False,
        "dataset": {
            "id": manifest["source"]["dataset_id"],
            "license": dataset["license_id"],
            "images": dataset["decoded_image_count"],
            "paired_boxes": receipt["audit"]["paired_box_count"],
            "classes": dataset["class_names"],
            "target_class_mapping": {
                "helmet": 0,
                "no_helmet": 7,
                "hi_vis": 2,
                "no_hi_vis": None,
            },
        },
        "split_leakage": {
            "published_split_independent_gt": False,
            "cross_split_pairs_by_max_hamming": leakage[
                "cross_split_pair_counts_by_max_hamming"
            ],
            "source_asset_or_sequence_grouping_available": False,
        },
        "model_readiness": dict(receipt["model_readiness"]),
        "tests": dict(receipt["tests"]),
        "integrity": {
            "manifest_live_exact_pin_verified": True,
            "validator_live_exact_pin_verified": True,
            "compact_receipt_live_exact_pin_verified": True,
            "archive_sha256_verified_by_offline_audit": True,
            "archive_sha256_live_rechecked_by_admin": False,
        },
        "caveats": [
            "1.416 görüntü ve 11.521 eşlenmiş kutu tanısal inceleme için hazır; GPU veya inference çalıştırılmadı.",
            "Yayımlanan train/val/test bölümlerinde yakın-kopya dizi sızıntısı var; final GT veya eşik kalibrasyonu olarak kullanılamaz.",
            "Veri setinde no_hi_vis sınıfı yok; mevcut dört-sınıflı PPE runtime sözleşmesini tek başına karşılamaz.",
            "Kabul edilmiş PPE ağırlığı, ONNX, TensorRT 640/960 motoru ve DeepStream GIE config'i henüz yok.",
        ],
        "evidence": [],
    }
