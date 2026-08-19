from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "models/runtime/rtdetrv4-parser-ds9-r1/build-receipt.json"
RECEIPT_PIN = "05fc33c32ede4f0090b38f2232c36936b1ac86de1b4296f21b260054e82e8e5c"
RECEIPT_FILE_SHA = "0917d1d96b0757a3f7714b8adf8205a817e176c597b4c062a44a2a0fc402a405"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def _canonical_receipt(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256")
    raw = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _assert_pin(pin: dict) -> Path:
    path = ROOT / pin["path"]
    assert path.is_file()
    assert not path.is_symlink()
    assert path.stat().st_size == pin["bytes"]
    assert _digest(path) == pin["sha256"]
    return path


def test_build_receipt_and_all_sources_are_exact_pinned() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert _digest(RECEIPT) == RECEIPT_FILE_SHA
    assert receipt["receipt_sha256"] == RECEIPT_PIN
    assert _canonical_receipt(receipt) == RECEIPT_PIN
    for pin in receipt["inputs"].values():
        _assert_pin(pin)
    artifact = _assert_pin(receipt["artifact"])
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o440


def test_parser_abi_exports_only_versioned_contract_symbols() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    artifact = ROOT / receipt["artifact"]["path"]
    output = subprocess.run(
        ["readelf", "--dyn-syms", "--wide", str(artifact)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    for symbol in receipt["abi"]["symbols"]:
        assert symbol in output
    assert "NvDsInferParseCustomRTDETRv4Person@@" in output
    assert "NvDsInferInitializeInputLayers@@" in output


def test_person_parser_contract_is_ready_but_gpu_and_product_stay_closed() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    readiness = receipt["readiness"]
    assert readiness["parser_cpu_contract_ready"] is True
    assert readiness["onnx_profiles_present"] is True
    assert all(
        readiness[key] is False
        for key in (
            "tensorrt_engines_built",
            "gpu_integration_validated",
            "deepstream9_real_inference_validated",
            "real_image_parity_passed",
            "quality_passed",
            "capacity_passed",
            "production_ready",
        )
    )
    assert receipt["capabilities"]["person_only_coco_class_zero"] is True
    assert receipt["capabilities"]["max_batch_contract"] == 12


def test_reproducible_build_is_networkless_and_gpu_less() -> None:
    script = (
        ROOT / "models/person/postprocess/rtdetrv4_ds9/build-runtime.sh"
    ).read_text(encoding="utf-8")
    assert "--network=none" in script
    assert "--runtime=runc" in script
    assert "--gpus" not in script
    assert "ctest --output-on-failure" in script
    assert "ldd -r" in script
