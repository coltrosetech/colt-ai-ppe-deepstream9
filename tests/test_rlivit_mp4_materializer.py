import copy
import hashlib
import json
import os
import stat
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from validation import rlivit_materializer
from validation import rlivit_mp4_materializer as mp4
from validation.rlivit_contract import RLiViTError


FRAMES = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46]


def _png(width: int, height: int, value: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(
        b"\x00"
        + b"".join(
            bytes(
                (
                    (value + x * 7 + y * 11) % 256,
                    (value + 8 + x * 13 + y * 5) % 256,
                    (value + 19 + x * 3 + y * 17) % 256,
                )
            )
            for x in range(width)
        )
        for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _annotation(sequence: str, frame: int, width: int, height: int) -> str:
    return f"""<annotation verified="yes">
<size><width>{width}</width><height>{height}</height><depth>3</depth></size>
<sequence>{sequence}</sequence><positionInSequence>{frame:02d}</positionInSequence>
<object><name>person</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>{width - 1}</xmax><ymax>{height - 1}</ymax></bndbox></object>
</annotation>
"""


def _pin(path: Path, relative: str) -> dict:
    raw = path.read_bytes()
    return {
        "path": relative,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _build_fixture(tmp_path: Path) -> dict:
    project = tmp_path / "project"
    validation = project / "validation"
    validation.mkdir(parents=True)
    archive = project / "data/raw/r-livit/R-LiViT_RGB-T.zip"
    archive.parent.mkdir(parents=True)
    width, height = 64, 48
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("RLiViT/README.md", "Synthetic R-LiViT fixture.\n")
        zf.writestr("RLiViT/train.txt", "S001\n")
        zf.writestr("RLiViT/test.txt", "S002\n")
        zf.writestr(
            "RLiViT/sequences.xml",
            """<sequences>
<sequence><rgbt_seq_id>S001</rgbt_seq_id><daytime>day</daytime><location>L1</location></sequence>
<sequence><rgbt_seq_id>S002</rgbt_seq_id><daytime>night</daytime><location>L2</location></sequence>
</sequences>""",
        )
        for sequence_index, sequence in enumerate(("S001", "S002")):
            for frame_index, frame in enumerate(FRAMES):
                image = _png(width, height, 40 + sequence_index * 80 + frame_index)
                zf.writestr(f"RLiViT/rgb/{sequence}/{frame:02d}.png", image)
                zf.writestr(f"RLiViT/thermal/{sequence}/{frame:02d}.png", image)
                zf.writestr(
                    f"RLiViT/annotations/{sequence}/{frame:02d}.xml",
                    _annotation(sequence, frame, width, height),
                )
    archive_raw = archive.read_bytes()
    source_contract = {
        "schema_version": "deepsafe.rlivit-source-contract/v1",
        "dataset": {
            "dataset_id": "R-LiViT_RGB-T_fixture",
            "title": "R-LiViT fixture",
            "version": "1.0",
            "doi": "10.5281/zenodo.16356714",
            "zenodo_record_id": 16356714,
            "publication_date": "2025-07-23",
        },
        "archive": {
            "filename": archive.name,
            "size_bytes": len(archive_raw),
            "checksum": {
                "algorithm": "md5",
                "value": hashlib.md5(archive_raw, usedforsecurity=False).hexdigest(),
            },
            "content_url": "https://zenodo.org/api/records/16356714/files/R-LiViT_RGB-T.zip/content",
        },
        "license": {
            "id": "cc-by-4.0",
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Synthetic fixture",
        },
        "official_references": {
            "github_repository": "https://github.com/XITASO/r-livit",
            "github_commit": "1" * 40,
            "files": [{"path": "README.md", "size_bytes": 1, "sha256": "2" * 64}],
        },
        "archive_dataset_layout": {
            "root_directory": "RLiViT",
            "rgb_directory": "rgb",
            "thermal_directory": "thermal",
            "annotations_directory": "annotations",
            "sequence_metadata_file": "sequences.xml",
            "sequence_id_xml_element": "rgbt_seq_id",
            "train_split_file": "train.txt",
            "test_split_file": "test.txt",
            "readme_file": "README.md",
        },
        "dataset_expectations": {
            "sequence_count": 2,
            "train_sequence_count": 1,
            "test_sequence_count": 1,
            "annotated_frame_indices": FRAMES,
            "verified_frame_count": 24,
            "total_object_count": 24,
            "source_capture_fps": 5.0,
            "annotation_fps": 1.25,
            "rgb_dimensions": [width, height],
            "thermal_dimensions": [width, height],
            "supported_classes": [
                "person", "bicycle", "car", "motorcycle", "bus", "tramway", "truck", "escooter"
            ],
            "required_daytime_values": ["day", "night"],
            "minimum_distinct_locations": 2,
            "campaign_split": "test",
            "model_input_profiles": [640, 960],
        },
        "extraction_limits": {
            "max_entries": 200,
            "max_path_bytes": 256,
            "max_file_uncompressed_bytes": 1024 * 1024,
            "max_total_uncompressed_bytes": 32 * 1024 * 1024,
            "max_overall_expansion_ratio": 100.0,
            "allowed_compression_methods": [0, 8],
            "normalized_mtime_epoch": 0,
        },
        "provenance_notes": ["Synthetic fixture only."],
    }
    source_contract_path = validation / "rlivit-source-contract.json"
    source_contract_path.write_text(
        json.dumps(source_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    materialized = project / "data/derived/r-livit/materialized-v1"
    rlivit_materializer.materialize(
        archive_path=archive,
        output_root=materialized,
        contract_path=source_contract_path,
        download_complete_confirmed=True,
    )
    materialization = json.loads(
        (materialized / "evidence/materialization-receipt.json").read_text()
    )
    dataset = json.loads(
        (materialized / "derived/dataset-validation-receipt.json").read_text()
    )
    lineage = {
        "source_contract": _pin(
            source_contract_path, "validation/rlivit-source-contract.json"
        ),
        "materialization_receipt": _pin(
            materialized / "evidence/materialization-receipt.json",
            "evidence/materialization-receipt.json",
        ),
        "dataset_receipt": {
            **_pin(
                materialized / "derived/dataset-validation-receipt.json",
                "derived/dataset-validation-receipt.json",
            ),
            "fingerprint_sha256": dataset["fingerprint_sha256"],
        },
        "extraction_manifest": _pin(
            materialized / "evidence/extraction-manifest.json",
            "evidence/extraction-manifest.json",
        ),
        "source_license_receipt": _pin(
            materialized / "evidence/source-license-receipt.json",
            "evidence/source-license-receipt.json",
        ),
        "mp4_plan": _pin(
            materialized / "derived/plans/mp4-plan.json",
            "derived/plans/mp4-plan.json",
        ),
        "campaign_plan": _pin(
            materialized / "derived/plans/person-campaign-plan.json",
            "derived/plans/person-campaign-plan.json",
        ),
        "materialization_fingerprint_sha256": materialization["fingerprint_sha256"],
        "archive_sha256": materialization["archive"]["sha256"],
    }
    contract = {
        "schema_version": "deepsafe.rlivit-mp4-contract/v1",
        "dataset_id": "R-LiViT_RGB-T_fixture",
        "lineage": lineage,
        "campaign": {
            "split": "test",
            "sequence_count": 1,
            "profiles": [640, 960],
            "expected_jobs": 2,
            "required_gpu_job_status": "blocked_pending_mp4_gpu_reentry_and_model_binding",
            "source_frame_indices": FRAMES,
            "video_frame_indices": list(range(12)),
        },
        "video": {
            "width": width,
            "height": height,
            "frame_count": 12,
            "fps_numerator": 5,
            "fps_denominator": 4,
            "duration_numerator": 48,
            "duration_denominator": 5,
            "codec_name": "h264",
            "encoder": "libx264",
            "profile": "High",
            "level": 40,
            "pixel_format": "yuv420p",
            "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
            "codec_tag_string": "avc1",
        },
        "encoder": {
            "thread_count": 1,
            "preset": "slow",
            "crf": 12,
            "gop_size": 1,
            "b_frames": 0,
            "scene_cut_threshold": 0,
            "video_track_timescale": 12800,
            "software_only": True,
            "source_plan_thread_count": 1,
            "nice_adjustment": 10,
            "source_plan_override_reason": "Deterministic serial libx264 encoding uses exactly one encoder thread and niceness +10; workstation BIOS/EC and advertised hardware-critical temperatures own thermal protection.",
        },
        "quality": {
            "comparison_pixel_format": "yuv420p",
            "minimum_frame_psnr_y_db": 45.0,
            "minimum_frame_psnr_average_db": 45.0,
            "minimum_frame_ssim_y": 0.99,
            "minimum_frame_ssim_all": 0.99,
            "software_decode_and_filter_only": True,
        },
        "thermal": {
            "required_hwmon_names": ["coretemp", "dell_smm"],
            "enforcement_mode": "workstation_hardware_managed_record_only",
            "start_threshold_millidegrees_celsius": 99000,
            "abort_threshold_millidegrees_celsius": 100000,
            "minimum_plausible_millidegrees_celsius": -20000,
            "maximum_plausible_millidegrees_celsius": 150000,
            "monitor_interval_milliseconds": 25,
        },
        "execution": {
            "serial_jobs": True,
            "gpu_requested": False,
            "docker_requested": False,
            "inference_requested": False,
            "hide_nvidia_devices": True,
            "output_directory_name": "test-mp4-v1",
            "job_timeout_seconds": 60,
            "normalized_mtime_epoch": 0,
        },
    }
    contract_path = validation / "rlivit-mp4-contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hwmon = project / "sys/class/hwmon"
    core = hwmon / "hwmon0"
    dell = hwmon / "hwmon1"
    core.mkdir(parents=True)
    dell.mkdir(parents=True)
    (core / "name").write_text("coretemp\n")
    (core / "temp1_input").write_text("40000\n")
    (core / "temp1_label").write_text("Package id 0\n")
    (core / "temp1_crit").write_text("100000\n")
    (dell / "name").write_text("dell_smm\n")
    (dell / "temp1_input").write_text("41000\n")
    return {
        "project": project,
        "materialized": materialized,
        "contract": contract_path,
        "hwmon": hwmon,
        "public": validation / "results/rlivit/mp4-batch-receipt.json",
        "test_rgb": materialized / "archive/RLiViT/rgb/S002/02.png",
    }


def _run(fixture: dict, output: Path, **kwargs):
    return mp4.materialize_test_mp4s(
        project_root=fixture["project"],
        materialized_root=fixture["materialized"],
        output_root=output,
        contract_path=fixture["contract"],
        hwmon_root=fixture["hwmon"],
        public_receipt_path=kwargs.pop("public_receipt_path", fixture["public"]),
        **kwargs,
    )


def test_real_software_encode_is_deterministic_exact_resumable_and_tamper_evident(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    first = fixture["project"] / "runs/first/test-mp4-v1"
    second = fixture["project"] / "runs/second/test-mp4-v1"
    result = _run(fixture, first)
    assert result["status"] == "complete_verified_cpu_only"
    assert result["sequence_count"] == 1
    assert result["total_frames"] == 12
    assert result["encoder"] == {
        "name": "libx264",
        "thread_count": 1,
        "nice_adjustment": 10,
        "serial_jobs": True,
        "encoder_binding_fingerprint_sha256": result["encoder"][
            "encoder_binding_fingerprint_sha256"
        ],
    }
    receipt = json.loads((first / "jobs/S002/receipt.json").read_text())
    assert receipt["media"]["codec_name"] == "h264"
    assert receipt["media"]["profile"] == "High"
    assert receipt["media"]["pixel_format"] == "yuv420p"
    assert receipt["media"]["frame_count"] == 12
    assert receipt["media"]["r_frame_rate"] == "5/4"
    assert receipt["source_quality"]["status"] == "all_frames_pass_fixed_psnr_ssim_floors"
    assert receipt["gpu"]["requested"] is False
    assert receipt["gpu"]["device_fd_inspection_required"] is False
    assert receipt["gpu"]["device_fd_observation"] is None
    assert receipt["gpu"]["device_fd_policy"] == "not_required_pinned_software_media_command"
    assert [
        item["role"] for item in receipt["thermal"]["auxiliary_process_monitors"]
    ] == ["probe", "framemd5", "psnr", "ssim"]
    assert all(
        item["nvidia_device_fd_inspection_required"] is False
        and item["nvidia_device_fd_checks"] == 0
        and item["nvidia_device_fds_observed"] is None
        for item in [
            receipt["thermal"]["encoder_process_monitor"],
            *receipt["thermal"]["auxiliary_process_monitors"],
        ]
    )
    assert all(
        sample["maximum_millidegrees_celsius"]
        < sample["enforced_threshold_millidegrees_celsius"]
        for sample in receipt["thermal"]["samples"]
    )
    assert receipt["thermal"]["maximum_millidegrees_celsius"] == 41000
    first_hash = receipt["output"]["sha256"]

    resumed = _run(fixture, first)
    assert resumed["created_this_run"] == 0
    assert resumed["resumed_this_run"] == 1

    second_public = fixture["project"] / "runs/second/public.json"
    second_result = _run(fixture, second, public_receipt_path=second_public)
    second_receipt = json.loads((second / "jobs/S002/receipt.json").read_text())
    assert second_result["status"] == "complete_verified_cpu_only"
    assert second_receipt["output"]["sha256"] == first_hash

    admin = json.loads(fixture["public"].read_text())
    assert admin["schema_version"] == "deepsafe.rlivit-mp4-admin-receipt/v2"
    assert admin["thermal_policy_id"] == "workstation_managed"
    assert admin["sequences"] == {"complete": 1, "expected": 1}
    assert admin["gpu_jobs"]["blocked"] == 2
    private_batch_bytes = (first / "batch-receipt.json").read_bytes()
    assert admin["batch_receipt_pin"] == {
        "size_bytes": len(private_batch_bytes),
        "sha256": hashlib.sha256(private_batch_bytes).hexdigest(),
    }

    def assert_pathless(value):
        if isinstance(value, dict):
            assert not ({"path", "outputs", "job_receipts"} & set(value))
            for item in value.values():
                assert_pathless(item)
        elif isinstance(value, list):
            for item in value:
                assert_pathless(item)
        elif isinstance(value, str):
            assert not value.startswith("/")
            assert not any(token in value for token in ("jobs/", "derived/", "validation/"))

    assert_pathless(admin)

    with (first / "jobs/S002/video.mp4").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RLiViTError, match="output size/SHA-256 differs"):
        _run(fixture, first, verify_only=True)


def test_hardware_critical_temperature_is_recorded_not_software_blocked(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    output = fixture["project"] / "runs/hot/test-mp4-v1"
    core = fixture["hwmon"] / "hwmon0/temp1_input"
    core.write_text("100000\n")
    receipt = _run(fixture, output)
    assert receipt["thermal"]["hardware_critical_reached_samples"] > 0
    assert receipt["thermal"]["policy_threshold_reached_samples"] > 0
    assert receipt["thermal"]["all_jobs_below_limit"] is False

    core.write_text("40000\n")
    (fixture["hwmon"] / "hwmon1/temp1_input").unlink()
    with pytest.raises(RLiViTError, match="no numeric temperature inputs"):
        _run(
            fixture,
            fixture["project"] / "runs/missing/test-mp4-v1",
            public_receipt_path=fixture["project"] / "runs/missing/public.json",
        )


def test_tampered_rgb_is_rejected_before_transcode(tmp_path: Path, monkeypatch) -> None:
    fixture = _build_fixture(tmp_path)
    fixture["test_rgb"].write_bytes(fixture["test_rgb"].read_bytes() + b"tamper")

    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("encoder must not start after a source-pin mismatch")

    monkeypatch.setattr(mp4.subprocess, "Popen", forbidden_popen)
    with pytest.raises(RLiViTError, match="live size/SHA-256 differs"):
        _run(
            fixture,
            fixture["project"] / "runs/tampered/test-mp4-v1",
            public_receipt_path=fixture["project"] / "runs/tampered/public.json",
        )


@pytest.mark.parametrize("kind", ["partial_job", "stale_stage", "unexpected"])
def test_partial_stale_or_unexpected_output_is_never_accepted(
    tmp_path: Path, kind: str
) -> None:
    fixture = _build_fixture(tmp_path)
    output = fixture["project"] / f"runs/{kind}/test-mp4-v1"
    output.mkdir(parents=True)
    if kind == "partial_job":
        partial = output / "jobs/S002"
        partial.mkdir(parents=True)
        (partial / "video.mp4").write_bytes(b"partial")
    elif kind == "stale_stage":
        (output / ".staging-S002-crash").mkdir()
    else:
        (output / "foreign.txt").write_text("unexpected")
    with pytest.raises(RLiViTError):
        _run(
            fixture,
            output,
            public_receipt_path=fixture["project"] / f"runs/{kind}/public.json",
        )


def test_during_encode_thermal_spike_terminates_process(monkeypatch) -> None:
    contract = {
        "thermal": {
            "monitor_interval_milliseconds": 25,
            "start_threshold_millidegrees_celsius": 99000,
            "abort_threshold_millidegrees_celsius": 100000,
        },
        "encoder": {"nice_adjustment": 10},
        "execution": {"job_timeout_seconds": 60},
    }
    calls = 0

    def sampler(_manifest, _contract, *, phase, sample_index):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RLiViTError("thermal abort threshold reached during encode")
        return {
            "phase": phase,
            "sample_index": sample_index,
            "source_count": 1,
            "maximum_millidegrees_celsius": 40000,
            "readings": [{"millidegrees_celsius": 40000}],
        }

    class FakeProcess:
        pid = os.getpid()

        def __init__(self):
            self.terminated = False

        def poll(self):
            return 1 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 1

    process = FakeProcess()
    monkeypatch.setattr(mp4.os, "getpriority", lambda _which, _pid: 10)
    monkeypatch.setattr(mp4.time, "sleep", lambda _seconds: None)
    with pytest.raises(RLiViTError, match="during encode"):
        mp4.run_guarded_ffmpeg(
            ["ffmpeg", "dummy"],
            thermal_sources={},
            contract=contract,
            thermal_sampler=sampler,
            popen_factory=lambda *_args, **_kwargs: process,
        )
    assert process.terminated is True


def test_during_auxiliary_quality_spike_terminates_process(monkeypatch) -> None:
    contract = {
        "thermal": {
            "monitor_interval_milliseconds": 25,
            "start_threshold_millidegrees_celsius": 99000,
            "abort_threshold_millidegrees_celsius": 100000,
        },
        "encoder": {"nice_adjustment": 10},
    }
    calls = 0

    def sampler(_manifest, _contract, *, phase, sample_index):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RLiViTError("thermal abort threshold reached during PSNR")
        return {
            "phase": phase,
            "sample_index": sample_index,
            "source_count": 1,
            "enforced_threshold_millidegrees_celsius": 99000,
            "maximum_millidegrees_celsius": 40000,
            "readings": [{"millidegrees_celsius": 40000}],
        }

    class FakeProcess:
        pid = os.getpid()

        def __init__(self):
            self.terminated = False

        def poll(self):
            return 1 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 1

    process = FakeProcess()
    monkeypatch.setattr(mp4.os, "getpriority", lambda _which, _pid: 10)
    monkeypatch.setattr(mp4.time, "sleep", lambda _seconds: None)
    with pytest.raises(RLiViTError, match="during PSNR"):
        mp4._run_guarded_cpu_command(
            ["nice", "-n", "10", "ffmpeg", "dummy"],
            role="psnr",
            thermal_sources={},
            contract=contract,
            thermal_samples=[],
            capture_stdout=True,
            timeout_seconds=60,
            thermal_sampler=sampler,
            popen_factory=lambda *_args, **_kwargs: process,
        )
    assert process.terminated is True


def test_fast_pinned_software_process_does_not_require_proc_fd_inspection(monkeypatch) -> None:
    contract = {
        "thermal": {
            "monitor_interval_milliseconds": 25,
            "start_threshold_millidegrees_celsius": 99000,
            "abort_threshold_millidegrees_celsius": 100000,
        },
        "encoder": {"nice_adjustment": 10},
    }

    def sampler(_manifest, _contract, *, phase, sample_index):
        return {
            "phase": phase,
            "sample_index": sample_index,
            "source_count": 1,
            "enforced_threshold_millidegrees_celsius": (
                99000 if phase.startswith("pre_") else 100000
            ),
            "maximum_millidegrees_celsius": 40000,
            "readings": [{"millidegrees_celsius": 40000}],
        }

    class AlreadyExitedProcess:
        pid = os.getpid()

        def poll(self):
            return 0

        def communicate(self, timeout=None):
            return b"", b""

    monkeypatch.setattr(mp4.os, "getpriority", lambda _which, _pid: 10)
    result = mp4._run_guarded_cpu_command(
        ["nice", "-n", "10", "ffprobe", "dummy"],
        role="probe",
        thermal_sources={},
        contract=contract,
        thermal_samples=[],
        capture_stdout=True,
        timeout_seconds=60,
        thermal_sampler=sampler,
        popen_factory=lambda *_args, **_kwargs: AlreadyExitedProcess(),
    )
    assert result["monitor"]["nvidia_device_fd_inspection_required"] is False
    assert result["monitor"]["nvidia_device_fd_checks"] == 0
    assert result["monitor"]["nvidia_device_fds_observed"] is None
