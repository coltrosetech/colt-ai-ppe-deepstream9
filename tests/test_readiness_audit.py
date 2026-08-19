import hashlib
import json
import shutil
from pathlib import Path

import pytest

from validation import readiness_audit as audit


def _write(path: Path, content: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_json(path: Path, value: dict) -> Path:
    return _write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _pin(workspace: Path, path: Path, *, size_key: str = "bytes") -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.relative_to(workspace)),
        size_key: len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _prepare_scene(workspace: Path, readiness: Path) -> None:
    runs = []
    for scene_index in range(12):
        scene = f"scene-{scene_index:02d}"
        for profile in audit.PROFILES:
            _write(
                readiness / f"scene/{scene}/{profile}/deepstream.txt",
                f"scene={scene}\nprofile={profile}\n".encode(),
            )
            runs.append(
                {
                    "scene_id": scene,
                    "benchmark_type": f"camera-type-{scene_index:02d}",
                    "model_input_size": profile,
                    "status": "pending",
                    "fingerprint": None,
                    "status_path": str(
                        (readiness / f"scene/{scene}/{profile}/status.json").relative_to(
                            workspace
                        )
                    ),
                }
            )
    _write_json(
        readiness / "scene/matrix-summary.json",
        {
            "schema_version": "deepsafe.scene-benchmark-matrix/v1",
            "generated_at_utc": "2026-07-16T07:00:00Z",
            "selected_scenes": 12,
            "selected_sizes": [640, 960],
            "streams": 12,
            "duration_seconds_per_run": 300,
            "expected_runs": 24,
            "status_counts": {"pending": 24},
            "runs": runs,
        },
    )


def _prepare_caviar(workspace: Path, readiness: Path) -> None:
    sequences = []
    for index in range(8):
        identity = f"caviar-{index}"
        video = _write(workspace / f"inputs/caviar/{identity}.mp4", b"video-" + bytes([index]))
        ground_truth = _write(workspace / f"inputs/caviar/{identity}.xml", b"gt-" + bytes([index]))
        sequences.append(
            {
                "sequence_id": identity,
                "video": str(video.relative_to(workspace)),
                "ground_truth": str(ground_truth.relative_to(workspace)),
            }
        )
    jobs = []
    for sequence in sequences:
        for profile in audit.PROFILES:
            identity = sequence["sequence_id"]
            jobs.append(
                {
                    "job_id": f"{identity}:{profile}",
                    "sequence_id": identity,
                    "model_input": profile,
                    "run_root": str(
                        (
                            readiness
                            / f"caviar/results/{identity}/{profile}"
                        ).relative_to(workspace)
                    ),
                    "command": ["python", "-m", "validation.run_caviar"],
                    "status": "planned",
                }
            )
    _write_json(
        readiness / "caviar/batch-manifest.json",
        {
            "schema_version": "deepsafe.caviar-batch-plan/v1",
            "created_at": "2026-07-16T07:01:00Z",
            "status": "planned",
            "gpu_execution_requested": False,
            "campaign": {
                "sequence_count": 8,
                "model_input_sizes": [640, 960],
                "expected_jobs": 16,
                "gpu": 0,
            },
            "sequences": sequences,
            "jobs": jobs,
        },
    )
    _write_json(
        readiness / "caviar/batch-aggregate.json",
        {
            "generated_at": "2026-07-16T07:02:00Z",
            "completeness": {
                "expected_jobs": 16,
                "complete_jobs": 0,
                "pending_jobs": 16,
                "is_complete": False,
            },
            "results": [],
            "paired_profile_deltas": [],
        },
    )
    _write(readiness / "caviar/batch-aggregate.md")


def _prepare_open(workspace: Path, readiness: Path) -> None:
    catalog_path = workspace / "inputs/open/source-catalog.json"
    normalization_report = workspace / "inputs/open/normalization.tsv"
    scenes = []
    assets = []
    normalization_rows = ["id\tsource_frames\toutput_frames\tfps\toutput"]
    for index in range(12):
        identity = f"open-{index}"
        normalized = _write(
            workspace / f"inputs/open/normalized/{identity}.mp4",
            b"normalized-" + bytes([index]),
        )
        original = _write(
            workspace / f"inputs/open/original/{identity}.webm",
            b"original-" + bytes([index]),
        )
        normalized_pin = _pin(workspace, normalized)
        original_pin = _pin(workspace, original)
        asset_evidence = {
            "asset_page_url": f"https://example.test/{identity}",
            "license": {"spdx": "CC0-1.0"},
            "camera": {"view": f"fixture-{index}", "motion": "fixed"},
            "ground_truth": {"available": False},
        }
        assets.append({"id": identity, **asset_evidence})
        normalization_rows.append(
            "\t".join(
                [
                    identity,
                    "1",
                    "1",
                    "25/1",
                    str(normalized.relative_to(workspace)),
                ]
            )
        )
        scenes.append(
            {
                "id": identity,
                "video": {
                    "path": normalized_pin["path"],
                    "size_bytes": normalized_pin["bytes"],
                    "sha256": normalized_pin["sha256"],
                },
                "original_source": {
                    "path": original_pin["path"],
                    "size_bytes": original_pin["bytes"],
                    "sha256": original_pin["sha256"],
                },
                "normalization": {
                    "report": str(normalization_report.relative_to(workspace)),
                    "id": identity,
                    "source_frames": 1,
                    "output_frames": 1,
                    "frame_count_preserved": True,
                    "fps_fraction": "25/1",
                },
                "source_catalog": {
                    "manifest": str(catalog_path.relative_to(workspace)),
                    "asset_id": identity,
                    **asset_evidence,
                },
            }
        )
    _write(normalization_report, ("\n".join(normalization_rows) + "\n").encode())
    _write_json(catalog_path, {"assets": assets})
    jobs = []
    for scene in scenes:
        for profile in audit.PROFILES:
            identity = scene["id"]
            jobs.append(
                {
                    "job_id": f"{identity}:{profile}",
                    "scene_id": identity,
                    "model_input": profile,
                    "run_root": str(
                        (readiness / f"open/results/{identity}/{profile}").relative_to(
                            workspace
                        )
                    ),
                    "command": ["python", "-m", "validation.open_video_review"],
                    "status": "planned",
                }
            )
    _write_json(
        readiness / "open/campaign-plan.json",
        {
            "schema_version": "deepsafe.open-video-review-plan/v1",
            "created_at": "2026-07-16T07:03:00Z",
            "status": "planned",
            "gpu_execution_requested": False,
            "campaign": {
                "scene_count": 12,
                "expected_jobs": 24,
                "model_input_sizes": [640, 960],
                "gpu": 0,
            },
            "accuracy_guardrail": {
                "ground_truth": False,
                "forbidden_metrics": sorted(audit.OPEN_FORBIDDEN_METRICS),
                "allowed_result": "ranked qualitative review candidates only",
                "looped_frames": "forbidden",
            },
            "scenes": scenes,
            "jobs": jobs,
        },
    )


def _prepare_loaf(workspace: Path, readiness: Path) -> dict:
    def make_pin(relative: str, content: bytes) -> dict:
        return _pin(workspace, _write(workspace / relative, content))

    source_artifacts = {
        name: make_pin(f"inputs/loaf/source-{name}.json", name.encode())
        for name in ("ground_truth", "media_plan", "selection_report")
    }
    code = {
        name: make_pin(f"code/{name}.py", name.encode())
        for name in audit.LOAF_CODE_KEYS
    }
    source_contract = {
        "split": "val",
        "sequence_count": 8,
        "artifacts": source_artifacts,
    }
    generated_root = readiness / "loaf/jobs/generated-configs"
    campaign = {
        "split": "val",
        "sequence_count": 8,
        "model_input_sizes": [640, 960],
        "expected_jobs": 16,
        "gpu": 0,
        "generated_config_root": str(generated_root.relative_to(workspace)),
        "code": code,
    }
    sequences = []
    for index in range(8):
        identity = f"loaf-{index:04d}"
        sequences.append(
            {
                "sequence_id": identity,
                "video": make_pin(f"inputs/loaf/{identity}.mp4", f"video-{identity}".encode()),
                "ground_truth": make_pin(
                    f"inputs/loaf/{identity}.json", f"gt-{identity}".encode()
                ),
            }
        )
    profiles = {}
    runtime_names = (
        "config_infer_primary.txt",
        "yolo11s_b12_gpu0_fp16.engine",
        "labels.txt",
        "yolo11s.onnx",
        "yolo11s.onnx.data",
    )
    for profile in audit.PROFILES:
        runtime = {
            name: make_pin(
                f"models/person/{profile}/{name}", f"{profile}-{name}".encode()
            )
            for name in runtime_names
        }
        profiles[profile] = {"size": profile, "runtime_artifacts": runtime}
    jobs = []
    for sequence in sequences:
        identity = sequence["sequence_id"]
        for profile in audit.PROFILES:
            configs = {}
            for kind, filename in (("infer", "infer.txt"), ("deepstream", "deepstream.txt")):
                path = generated_root / identity / str(profile) / filename
                configs[kind] = _pin(
                    workspace,
                    _write(path, f"{identity}-{profile}-{kind}".encode()),
                )
            jobs.append(
                {
                    "job_id": f"{identity}:{profile}",
                    "sequence_id": identity,
                    "model_input": profile,
                    "run_root": str(
                        (
                            readiness / f"loaf/jobs/runs/{identity}/{profile}"
                        ).relative_to(workspace)
                    ),
                    "model_profile": profiles[profile],
                    "configs": configs,
                    "status": "planned",
                    "command": ["python", "-m", "validation.run_loaf"],
                }
            )
    projection = {
        "schema_version": "deepsafe.loaf-deepstream-batch-plan/v1",
        "source_contract": source_contract,
        "campaign": campaign,
        "sequences": sequences,
        "jobs": [
            {
                key: job[key]
                for key in (
                    "job_id",
                    "sequence_id",
                    "model_input",
                    "run_root",
                    "model_profile",
                    "configs",
                )
            }
            for job in jobs
        ],
    }
    plan = {
        **projection,
        "created_at": "2026-07-16T07:04:00Z",
        "status": "planned",
        "gpu_execution_requested": False,
        "plan_fingerprint": audit._canonical_sha(projection),
        "fingerprint_input": projection,
        "jobs": jobs,
    }
    _write_json(readiness / "loaf/dry-run-plan.json", plan)
    _write_json(
        readiness / "loaf/batch-aggregate.json",
        {
            "generated_at": "2026-07-16T07:05:00Z",
            "aggregation_status": "withheld_incomplete",
            "completeness": {
                "expected_jobs": 16,
                "complete_jobs": 0,
                "pending_jobs": 16,
                "is_complete": False,
            },
            "results": [],
            "profiles": {},
            "plan_fingerprint": plan["plan_fingerprint"],
        },
    )
    _write(readiness / "loaf/batch-aggregate.md")
    return plan


def _prepare_endurance(workspace: Path, readiness: Path) -> None:
    root = readiness / "endurance"
    sources = []
    source_pins = []
    for index in range(12):
        path = _write(workspace / f"inputs/endurance/source-{index}.mp4", bytes([index]))
        base_pin = _pin(workspace, path, size_key="size_bytes")
        metadata = {
            "camera_id": index,
            "scene_id": f"scene-{index}",
            "codec": "h264",
            "pixel_format": "yuv420p",
            "width": 640,
            "height": 360,
            "fps_fraction": "25/1",
            "duration_seconds": 1.0,
        }
        source_pins.append({**metadata, **base_pin})
        sources.append(
            {
                **metadata,
                "video_path": base_pin["path"],
                "video_size_bytes": base_pin["size_bytes"],
                "video_sha256": base_pin["sha256"],
            }
        )
    model_profiles = {}
    for profile in audit.PROFILES:
        infer = _write(
            workspace / f"models/person/{profile}/config_infer_primary.txt",
            f"{profile}-config_infer_primary.txt".encode(),
        )
        engine = _write(
            workspace / f"models/person/{profile}/yolo11s_b12_gpu0_fp16.engine",
            f"{profile}-yolo11s_b12_gpu0_fp16.engine".encode(),
        )
        infer_pin = _pin(workspace, infer, size_key="size_bytes")
        engine_pin = _pin(workspace, engine, size_key="size_bytes")
        model_profiles[str(profile)] = {
            "size": profile,
            "infer_config": infer_pin["path"],
            "engine": engine_pin["path"],
            "infer_config_sha256": infer_pin["sha256"],
            "engine_size_bytes": engine_pin["size_bytes"],
            "engine_sha256": engine_pin["sha256"],
            "person_only_classes": [0],
        }
    controls = [
        _pin(
            workspace,
            _write(workspace / f"controls/control-{index}.txt", bytes([index + 32])),
            size_key="size_bytes",
        )
        for index in range(8)
    ]
    input_pins = {
        "schema_version": "deepsafe.endurance-input-pins/v1",
        "source_media": source_pins,
        "model_profiles": model_profiles,
        "control_files": controls,
    }
    campaign = {
        "schema_version": "deepsafe.endurance-campaign/v1",
        "name": "fixture-seven-day",
        "duration_seconds": 604800,
        "segment_seconds": 21600,
        "streams": 12,
        "profiles": [640, 960],
        "profile_strategy": "alternate_segments",
        "sources": sources,
        "input_pins": input_pins,
        "power_safety": {"sample_interval_seconds": 1},
    }
    campaign["static_input_fingerprint"] = audit._supervisor_sha(
        {"campaign": campaign, "source_files": []}
    )
    campaign["execution_request"] = {"image": "deepsafe-deepstream:9.0", "gpu_index": 0}
    campaign["config_fingerprint"] = audit._supervisor_sha(
        {
            "campaign": {
                key: value for key, value in campaign.items() if key != "config_fingerprint"
            },
            "source_files": [],
        }
    )
    segments = [
        {
            "index": index,
            "segment_id": f"segment-{index:03d}-{audit.PROFILES[index % 2]}",
            "profile": audit.PROFILES[index % 2],
            "duration_seconds": 21600,
            "campaign_day": index // 4 + 1,
        }
        for index in range(28)
    ]
    plan = {
        "schema_version": "deepsafe.endurance-plan/v1",
        "config_fingerprint": campaign["config_fingerprint"],
        "static_input_fingerprint": campaign["static_input_fingerprint"],
        "input_pins_sha256": audit._supervisor_sha(input_pins),
        "execution_request": campaign["execution_request"],
        "power_safety_policy": campaign["power_safety"],
        "segments": segments,
    }
    _write_json(root / "campaign-resolved.json", campaign)
    _write_json(root / "plan.json", plan)
    _write(root / "generated/deepstream-12x-mixed-640.txt")
    _write(root / "generated/deepstream-12x-mixed-960.txt")
    _write(root / "supervisor.lock", b"")


def _prepare_all(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    implementation = workspace / "validation/readiness_audit.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(audit.__file__), implementation)
    _prepare_scene(workspace, readiness)
    _prepare_caviar(workspace, readiness)
    _prepare_open(workspace, readiness)
    _prepare_loaf(workspace, readiness)
    _prepare_endurance(workspace, readiness)
    return workspace, readiness


def test_build_receipt_verifies_preparation_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    workspace, readiness = _prepare_all(tmp_path)
    receipt = audit.build_receipt(
        workspace_root=workspace,
        readiness_root=readiness,
    )
    assert receipt["status"] == "verified_preparation_only"
    assert receipt["acceptance_effect"] == "preparation_only"
    assert receipt["gpu_started"] is False
    assert receipt["docker_started"] is False
    assert receipt["checks"]["loaf_ground_truth"]["readiness_config_pins"] == 32
    assert receipt["checks"]["seven_day_endurance"]["scheduled_seconds"] == 604800
    caviar = receipt["checks"]["caviar_ground_truth"]
    assert caviar["live_source_videos"] == 8
    assert caviar["live_ground_truth_files"] == 8
    assert len(caviar["video_source_pins"]) == 8
    assert len(caviar["ground_truth_source_pins"]) == 8
    open_video = receipt["checks"]["open_video_manual"]
    assert open_video["live_normalized_videos"] == 12
    assert open_video["live_original_sources"] == 12
    assert len(open_video["normalized_video_pins"]) == 12
    assert len(open_video["original_source_pins"]) == 12
    assert receipt["generated_at_utc"] == "2026-07-16T07:05:00Z"
    assert receipt["reproducibility"]["wall_clock_used"] is False
    assert receipt["self_code"] in receipt["artifact_hashes"]


def test_missing_artifacts_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(audit.AuditError, match="matrix-summary"):
        audit.build_receipt(workspace_root=tmp_path, readiness_root=tmp_path)


def test_tampered_live_loaf_pin_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    plan = _prepare_loaf(workspace, readiness)
    config = workspace / plan["jobs"][0]["configs"]["infer"]["path"]
    config.write_bytes(config.read_bytes() + b"tampered")
    with pytest.raises(audit.AuditError, match="live size differs"):
        audit._validate_loaf(audit.FileVerifier(workspace), readiness, [])


def test_shared_readiness_config_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    plan = _prepare_loaf(workspace, readiness)
    plan["jobs"][1]["configs"]["infer"] = plan["jobs"][0]["configs"]["infer"]
    _write_json(readiness / "loaf/dry-run-plan.json", plan)
    with pytest.raises(audit.AuditError, match="shared or non-canonical"):
        audit._validate_loaf(audit.FileVerifier(workspace), readiness, [])


def test_pseudo_complete_plan_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    _prepare_caviar(workspace, readiness)
    path = readiness / "caviar/batch-manifest.json"
    plan = json.loads(path.read_text())
    plan["jobs"][0]["status"] = "complete"
    _write_json(path, plan)
    with pytest.raises(audit.AuditError, match="pseudo-complete"):
        audit._validate_caviar(audit.FileVerifier(workspace), readiness, [])


def test_endurance_execution_state_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    _prepare_endurance(workspace, readiness)
    _write_json(readiness / "endurance/checkpoint.json", {"status": "complete"})
    with pytest.raises(audit.AuditError, match="execution-state"):
        audit._validate_endurance(audit.FileVerifier(workspace), readiness, [])


def test_endurance_plan_resolved_binding_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    _prepare_endurance(workspace, readiness)
    path = readiness / "endurance/plan.json"
    plan = json.loads(path.read_text())
    plan["execution_request"]["gpu_index"] = 1
    _write_json(path, plan)
    with pytest.raises(audit.AuditError, match="execution-request binding"):
        audit._validate_endurance(audit.FileVerifier(workspace), readiness, [])


def test_missing_caviar_source_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    _prepare_caviar(workspace, readiness)
    (workspace / "inputs/caviar/caviar-0.mp4").unlink()
    with pytest.raises(audit.AuditError, match="missing or unreadable"):
        audit._validate_caviar(audit.FileVerifier(workspace), readiness, [])


def test_missing_open_normalized_source_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    _prepare_open(workspace, readiness)
    (workspace / "inputs/open/normalized/open-0.mp4").unlink()
    with pytest.raises(audit.AuditError, match="missing or unreadable"):
        audit._validate_open(audit.FileVerifier(workspace), readiness, [])


def test_tampered_open_normalized_source_fails_declared_pin(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    _prepare_open(workspace, readiness)
    source = workspace / "inputs/open/normalized/open-0.mp4"
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(audit.AuditError, match="live size differs"):
        audit._validate_open(audit.FileVerifier(workspace), readiness, [])


def test_tampered_open_catalog_evidence_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    readiness = workspace / "current"
    _prepare_open(workspace, readiness)
    catalog_path = workspace / "inputs/open/source-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["assets"][0]["license"] = {"spdx": "tampered"}
    _write_json(catalog_path, catalog)
    with pytest.raises(audit.AuditError, match="source-catalog evidence is stale/tampered"):
        audit._validate_open(audit.FileVerifier(workspace), readiness, [])


def test_receipt_is_byte_deterministic_for_unchanged_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    workspace, readiness = _prepare_all(tmp_path)
    first = audit.build_receipt(workspace_root=workspace, readiness_root=readiness)
    second = audit.build_receipt(workspace_root=workspace, readiness_root=readiness)
    assert first == second
    assert (
        json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        == json.dumps(second, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    assert first["generated_at_utc"] == "2026-07-16T07:05:00Z"
    assert (
        first["reproducibility"]["timestamp_source"]
        == "latest_pinned_plan_intrinsic_timestamp"
    )


def test_source_date_epoch_controls_receipt_timestamp(tmp_path: Path) -> None:
    workspace, readiness = _prepare_all(tmp_path)
    receipt = audit.build_receipt(
        workspace_root=workspace,
        readiness_root=readiness,
        source_date_epoch=0,
    )
    assert receipt["generated_at_utc"] == "1970-01-01T00:00:00Z"
    assert receipt["reproducibility"]["timestamp_source"] == "SOURCE_DATE_EPOCH"
    assert receipt["reproducibility"]["source_date_epoch"] == 0


def test_publication_commits_and_verifies_one_generation(tmp_path: Path) -> None:
    workspace, readiness = _prepare_all(tmp_path)
    receipt = audit.build_receipt(
        workspace_root=workspace,
        readiness_root=readiness,
        source_date_epoch=0,
    )
    json_output = readiness / "summary.json"
    markdown_output = readiness / "summary.md"
    manifest_output = readiness / "summary.manifest.json"
    publication = audit.publish_receipt(
        receipt,
        workspace_root=workspace,
        json_output=json_output,
        markdown_output=markdown_output,
        manifest_output=manifest_output,
    )
    resolved_receipt, resolved_publication = audit.verify_published_receipt(
        workspace_root=workspace,
        manifest_path=manifest_output,
    )
    assert resolved_receipt == receipt
    assert resolved_publication == publication
    assert publication["schema_version"] == audit.PUBLICATION_SCHEMA
    assert publication["commit_semantics"].startswith("manifest-last")


def test_repeated_publication_is_byte_deterministic(tmp_path: Path) -> None:
    workspace, readiness = _prepare_all(tmp_path)
    receipt = audit.build_receipt(
        workspace_root=workspace,
        readiness_root=readiness,
        source_date_epoch=0,
    )
    json_output = readiness / "summary.json"
    markdown_output = readiness / "summary.md"
    manifest_output = readiness / "summary.manifest.json"
    first = audit.publish_receipt(
        receipt,
        workspace_root=workspace,
        json_output=json_output,
        markdown_output=markdown_output,
        manifest_output=manifest_output,
    )
    first_bytes = tuple(
        path.read_bytes() for path in (json_output, markdown_output, manifest_output)
    )
    second = audit.publish_receipt(
        receipt,
        workspace_root=workspace,
        json_output=json_output,
        markdown_output=markdown_output,
        manifest_output=manifest_output,
    )
    second_bytes = tuple(
        path.read_bytes() for path in (json_output, markdown_output, manifest_output)
    )
    assert first == second
    assert first_bytes == second_bytes


def test_partial_publication_failure_invalidates_all_projections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, readiness = _prepare_all(tmp_path)
    receipt = audit.build_receipt(
        workspace_root=workspace,
        readiness_root=readiness,
        source_date_epoch=0,
    )
    json_output = readiness / "summary.json"
    markdown_output = readiness / "summary.md"
    manifest_output = readiness / "summary.manifest.json"
    audit.publish_receipt(
        receipt,
        workspace_root=workspace,
        json_output=json_output,
        markdown_output=markdown_output,
        manifest_output=manifest_output,
    )
    real_replace = audit.os.replace

    def fail_markdown_replace(source: Path, target: Path) -> None:
        if Path(target) == markdown_output:
            raise OSError("injected Markdown publication failure")
        real_replace(source, target)

    monkeypatch.setattr(audit.os, "replace", fail_markdown_replace)
    with pytest.raises(audit.AuditError, match="receipt publication failed"):
        audit.publish_receipt(
            receipt,
            workspace_root=workspace,
            json_output=json_output,
            markdown_output=markdown_output,
            manifest_output=manifest_output,
        )
    assert not json_output.exists()
    assert not markdown_output.exists()
    assert not manifest_output.exists()
    assert list(readiness.glob(".*.stage-*")) == []


def test_failed_main_invalidates_stale_success(tmp_path: Path) -> None:
    workspace, readiness = _prepare_all(tmp_path)
    arguments = [
        "--workspace-root",
        str(workspace),
        "--readiness-root",
        str(readiness),
        "--source-date-epoch",
        "0",
    ]
    assert audit.main(arguments) == 0
    outputs = (
        readiness / "summary.json",
        readiness / "summary.md",
        readiness / "summary.manifest.json",
    )
    assert all(path.is_file() for path in outputs)
    source = workspace / "inputs/open/normalized/open-0.mp4"
    source.write_bytes(source.read_bytes() + b"tampered")
    assert audit.main(arguments) == 2
    assert all(not path.exists() for path in outputs)


def test_published_receipt_detects_live_caviar_source_tamper(tmp_path: Path) -> None:
    workspace, readiness = _prepare_all(tmp_path)
    receipt = audit.build_receipt(
        workspace_root=workspace,
        readiness_root=readiness,
        source_date_epoch=0,
    )
    manifest_output = readiness / "summary.manifest.json"
    audit.publish_receipt(
        receipt,
        workspace_root=workspace,
        json_output=readiness / "summary.json",
        markdown_output=readiness / "summary.md",
        manifest_output=manifest_output,
    )
    source = workspace / "inputs/caviar/caviar-0.xml"
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(audit.AuditError, match="live size differs"):
        audit.verify_published_receipt(
            workspace_root=workspace,
            manifest_path=manifest_output,
        )


def test_published_receipt_detects_markdown_tamper(tmp_path: Path) -> None:
    workspace, readiness = _prepare_all(tmp_path)
    receipt = audit.build_receipt(
        workspace_root=workspace,
        readiness_root=readiness,
        source_date_epoch=0,
    )
    markdown_output = readiness / "summary.md"
    manifest_output = readiness / "summary.manifest.json"
    audit.publish_receipt(
        receipt,
        workspace_root=workspace,
        json_output=readiness / "summary.json",
        markdown_output=markdown_output,
        manifest_output=manifest_output,
    )
    markdown_output.write_bytes(markdown_output.read_bytes() + b"tampered")
    with pytest.raises(audit.AuditError, match="live size differs"):
        audit.verify_published_receipt(
            workspace_root=workspace,
            manifest_path=manifest_output,
        )


def test_workspace_self_code_mismatch_fails_closed(tmp_path: Path) -> None:
    workspace, readiness = _prepare_all(tmp_path)
    implementation = workspace / "validation/readiness_audit.py"
    implementation.write_bytes(implementation.read_bytes() + b"# tampered\n")
    with pytest.raises(audit.AuditError, match="executing implementation"):
        audit.build_receipt(
            workspace_root=workspace,
            readiness_root=readiness,
            source_date_epoch=0,
        )
