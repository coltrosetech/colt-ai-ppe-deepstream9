"""Read-only admin projection for the exact DeepStream 9.1 native CPU build."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = ROOT / "deepstream/ds91-native"
BASE_DIGEST = "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994"
IMAGE_ID = "sha256:c4a7870f5e8e2cf06ed115a511aec2e92ac29d7c1452038accbdecca332d1215"
PINS = {
    "Dockerfile": (6497, "d4a029b545a1e4ea3549879aa60cc7fa40afedc22d83e4568928febf70f7fbb5"),
    "README.md": (2015, "0eeb37b5569639428c2c124f7d4a5f71bee1d9a43ea9a51cd0e78c3172d0d0b0"),
    "build-receipt-r1.json": (3348, "cde3638acf0ce48452197074fcf91340c001f78d02dff85f2f0feadceab2931f"),
    "build.sh": (2074, "8825863bb5111b3b4455ac8999f39a13ea03b1d499445f18d6fe35628c315c81"),
    "native-build-contract-v1.json": (2992, "5d7178a5471359f73587eb55520e91f6187f880f388e8a9e669fd90c4e567d92"),
    "test_static.py": (1306, "9e11bc469d7ab2797c596cd2120f450e7359f50cf84eff766f039758a58bef60"),
    "verify_static.py": (4003, "07e35aa42450a07ac18ad24dc7f00aa4581b8828c758f1107ade3a568b5a560c"),
    ".dockerignore": (46, "84cb98c6520ddfd9a66816abc34275eabef88660481d0fb60040afbfa475ab54"),
}


class DS91NativeStatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise DS91NativeStatusError(message)


def _read_exact(name: str) -> bytes:
    path = NATIVE_ROOT / name
    info = path.lstat()
    expected_bytes, expected_sha = PINS[name]
    _expect(stat.S_ISREG(info.st_mode), f"not regular: {name}")
    _expect(info.st_size == expected_bytes, f"size drift: {name}")
    raw = path.read_bytes()
    _expect(hashlib.sha256(raw).hexdigest() == expected_sha, f"hash drift: {name}")
    return raw


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            _expect(key not in value, f"duplicate key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DS91NativeStatusError(f"invalid JSON: {label}") from exc
    _expect(isinstance(value, dict), f"non-object JSON: {label}")
    return value


def _validate(contract: dict[str, Any], receipt: dict[str, Any]) -> None:
    _expect(contract.get("schema_version") == "deepsafe.ds91-native-build/v1", "contract schema differs")
    _expect(receipt.get("schema_version") == "deepsafe.ds91-native-build-receipt/v1", "receipt schema differs")
    _expect(receipt.get("status") == "passed_cpu_only", "receipt status differs")
    _expect(contract["base"] == receipt["base"], "base projection differs")
    _expect(contract["base"]["digest"] == BASE_DIGEST, "base digest differs")
    _expect(receipt["contract"] == {
        "path": "deepstream/ds91-native/native-build-contract-v1.json",
        "sha256": PINS["native-build-contract-v1.json"][1],
    }, "receipt contract pin differs")
    _expect(receipt["image"]["id"] == IMAGE_ID, "image id differs")
    _expect(receipt["image"]["architecture"] == "amd64", "image architecture differs")
    _expect(receipt["image"]["os"] == "linux", "image OS differs")
    boundary = receipt["execution_boundary"]
    _expect(boundary["network_for_run_layers"] == "none", "build network differs")
    _expect(all(boundary[key] is False for key in (
        "gpu_device_requested",
        "gpu_inference_executed",
        "deepstream_pipeline_executed",
        "tensorrt_engine_built",
        "parallel_infer_binary_executed",
    )), "CPU-only boundary overclaim")
    tests = receipt["tests"]
    expected_tests = {
        "static_policy": (2, 0),
        "person_parser_ctest": (1, 0),
        "fusion_pose_ppe_ctest": (6, 0),
        "artifact_integrity": (7, 0),
    }
    for name, (passed, failed) in expected_tests.items():
        _expect(tests[name] == {"passed": passed, "failed": failed}, f"test result differs: {name}")
    _expect(len(receipt["artifacts"]) == 7, "artifact inventory differs")
    qualification = receipt["qualification"]
    _expect(qualification["native_cpu_build_accepted"] is True, "CPU build acceptance differs")
    _expect(qualification["gpu_runtime_qualified"] is False, "GPU qualification overclaim")
    _expect(qualification["production_ready"] is False, "production readiness overclaim")


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "label": "DeepStream 9.1 native bileşen derlemesi",
        "state": "unavailable_integrity_error",
        "reason": reason,
        "available": False,
        "cpu_build_accepted": False,
        "gpu_runtime_qualified": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["Native build kanıtı exact-pin doğrulamasından geçmedi."],
    }


def load_deepstream91_native_status() -> dict[str, Any]:
    try:
        for name in PINS:
            _read_exact(name)
        contract = _strict_object(_read_exact("native-build-contract-v1.json"), "contract")
        receipt = _strict_object(_read_exact("build-receipt-r1.json"), "receipt")
        _validate(contract, receipt)
    except (OSError, KeyError, TypeError, DS91NativeStatusError):
        return _unavailable("native_build_evidence_unavailable")

    return {
        "label": "DeepStream 9.1 native bileşen derlemesi",
        "state": "passed_cpu_only",
        "reason": "gpu_runtime_and_engine_validation_pending",
        "available": True,
        "cpu_build_accepted": True,
        "gpu_runtime_qualified": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "runtime": {
            "deepstream": receipt["base"]["deepstream"],
            "cuda": receipt["base"]["cuda"],
            "tensorrt": receipt["base"]["tensorrt"],
            "gstreamer": receipt["base"]["gstreamer"],
            "base_digest": receipt["base"]["digest"],
        },
        "image": {
            "tag": receipt["image"]["tag"],
            "id": receipt["image"]["id"],
            "size_bytes": receipt["image"]["size_bytes"],
            "live_cache_checked_by_admin": False,
        },
        "tests": {
            "passed": sum(item["passed"] for item in receipt["tests"].values()),
            "failed": sum(item["failed"] for item in receipt["tests"].values()),
            "artifacts": len(receipt["artifacts"]),
        },
        "remaining_gates": list(receipt["qualification"]["remaining_gates"]),
        "caveats": [
            "Bu kart exact-pinli CPU build receipt'ini gösterir; admin Docker veya GPU çalıştırmaz.",
            "TensorRT engine, canlı DeepStream pipeline, 12 akış FPS ve kalite kabulü henüz yoktur.",
        ],
        "evidence": [],
    }
