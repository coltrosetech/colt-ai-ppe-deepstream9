from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "models/person/candidates/rtdetrv4-s/deepstream/config-contract.json"
)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def _fingerprint(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("fingerprint_sha256")
    raw = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _container_model_path(value: str) -> Path:
    assert value.startswith("/models/")
    return ROOT / "models" / value.removeprefix("/models/")


def test_config_contract_is_self_hashed_and_pins_live_artifacts() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["fingerprint_sha256"] == _fingerprint(contract)
    for profile in ("640", "960"):
        item = contract["profiles"][profile]
        config = ROOT / item["config"]
        onnx = ROOT / item["onnx"]
        assert config.stat().st_size == item["config_bytes"]
        assert _sha256(config) == item["config_sha256"]
        assert onnx.stat().st_size == item["onnx_bytes"]
        assert _sha256(onnx) == item["onnx_sha256"]
        assert not (ROOT / item["engine"]).exists()
        assert item["engine_present"] is False
    parser = contract["parser"]
    library = ROOT / parser["library"]
    receipt = ROOT / parser["build_receipt"]
    assert library.stat().st_size == parser["library_bytes"]
    assert _sha256(library) == parser["library_sha256"]
    assert receipt.stat().st_size == parser["build_receipt_bytes"]
    assert _sha256(receipt) == parser["build_receipt_sha256"]
    labels = ROOT / contract["labels"]["path"]
    assert labels.read_text(encoding="utf-8") == "person\n"
    assert _sha256(labels) == contract["labels"]["sha256"]


def test_each_nvinfer_config_is_person_only_profile_isolated_and_fail_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for profile in ("640", "960"):
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(ROOT / contract["profiles"][profile]["config"])
        prop = parser["property"]
        attrs = parser["class-attrs-all"]
        assert prop.getint("batch-size") == 12
        assert prop.getint("network-mode") == 2
        assert prop.getint("num-detected-classes") == 1
        assert prop.getint("interval") == 0
        assert prop.getint("cluster-mode") == 4
        assert prop.getboolean("maintain-aspect-ratio") is True
        assert prop.getboolean("symmetric-padding") is True
        assert prop["parse-bbox-func-name"] == "NvDsInferParseCustomRTDETRv4Person"
        assert prop["output-blob-names"] == "labels;boxes;scores"
        assert prop["custom-lib-path"].endswith("libdeepsafe_rtdetrv4_parser.so")
        assert f"/onnx/{profile}/" in prop["onnx-file"]
        assert f"/engines/{profile}/" in prop["model-engine-file"]
        assert _container_model_path(prop["onnx-file"]).is_file()
        assert not _container_model_path(prop["model-engine-file"]).exists()
        assert attrs.getfloat("pre-cluster-threshold") == 0.25
        assert attrs.getint("topk") == 300


def test_config_contract_does_not_overclaim_engine_or_runtime_readiness() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "planned_engine_missing_not_gpu_validated"
    assert contract["inference"]["threshold_requires_held_out_calibration"] is True
    gates = contract["gates"]
    assert gates["configs_present"] is True
    assert gates["parser_cpu_contract_ready"] is True
    assert gates["onnx_profiles_present"] is True
    assert all(
        gates[key] is False
        for key in (
            "engines_present",
            "gpu_integration_validated",
            "real_image_parity_passed",
            "quality_passed",
            "exact_25m_passed",
            "capacity_passed",
            "production_ready",
        )
    )
