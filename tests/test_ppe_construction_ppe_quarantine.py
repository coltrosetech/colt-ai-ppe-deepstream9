from __future__ import annotations

import copy
import json
import socket
import subprocess
from pathlib import Path

import pytest

from validation import ppe_construction_ppe_quarantine as quarantine


def test_checked_quarantine_replays_exact_bytes_and_leakage() -> None:
    report = quarantine.audit()
    assert report["decoded_image_count"] == 1416
    assert report["paired_box_count"] == 11521
    assert report["orphan_label_file_count"] == 10
    assert report["orphan_box_count"] == 93
    assert report["cross_split_phash_pair_counts_by_max_hamming"] == {
        "0": 2,
        "2": 16,
        "4": 38,
        "6": 92,
        "8": 193,
    }
    assert report["offline_diagnostic_model_evaluation_authorized"] is True
    assert report["training_authorized"] is False
    assert report["independent_final_ground_truth_authorized"] is False
    assert report["product_acceptance_authorized"] is False
    assert report["gpu_docker_deepstream_used"] is False


def test_manifest_never_promotes_reciprocal_random_split_to_product_gt(tmp_path: Path) -> None:
    value, _ = quarantine.load_manifest()
    for section, key in (
        ("authorization", "training"),
        ("authorization", "independent_final_ground_truth"),
        ("authorization", "product_acceptance"),
        ("split_leakage_diagnostic", "published_random_split_is_independent_gt"),
    ):
        tampered = copy.deepcopy(value)
        tampered[section][key] = True
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(quarantine.ConstructionPPEQuarantineError):
            quarantine.audit(path)


def test_auditor_exposes_no_network_process_or_write_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden side effect")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    report = quarantine.audit()
    assert report["gpu_docker_deepstream_used"] is False


def test_cli_prints_json_without_creating_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert quarantine.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"].startswith("quarantined_")
    assert report["product_acceptance_authorized"] is False
