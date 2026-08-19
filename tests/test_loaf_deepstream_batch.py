import json
from contextlib import nullcontext
from pathlib import Path

import pytest

import validation.run_loaf_batch as batch
from validation.run_loaf import docker_command


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "validation/results/loaf/val-20-25m/deepstream/dry-run-plan.json"
)


def workstation_policy():
    return batch.operating_policy_contract(batch.WORKSTATION_MANAGED_POLICY_ID)


def campaign_gpu_safety(*, mode=batch.DEFAULT_GPU_OPERATING_POLICY_MODE):
    policy_id = batch.REENTRY_POLICY_ID_BY_GPU_MODE[mode]
    return {
        "reentry_evidence": "reentry/evidence.json",
        "operating_policy_mode": mode,
        "reentry_operating_policy_id": policy_id,
        "reentry_operating_policy": batch.operating_policy_contract(policy_id),
        "temperature_power_slowdown_action": (
            batch.STATIC_SIGNAL_ACTION_BY_GPU_MODE[mode]
        ),
        "required_telemetry_failure_action": "abort",
    }


def record_only_diagnostic_event():
    return {
        "code": "power_limit_below_default",
        "operating_policy_mode": batch.DEFAULT_GPU_OPERATING_POLICY_MODE,
        "measurement_quality_signal": True,
        "disposition": "record_only_workstation_hardware_managed",
    }


def workstation_preflight_report(*, diagnostic_events=None):
    return {
        "status": "ok",
        "checked_at_utc": "now",
        "image_id": "sha256:" + "1" * 64,
        "power_safety_policy": {
            "operating_policy_mode": batch.DEFAULT_GPU_OPERATING_POLICY_MODE,
            "static_signal_action": "record_measurement_quality_diagnostic",
            "diagnostic_slowdown_flags": batch.DIAGNOSTIC_SLOWDOWN_FLAGS,
            "abort_slowdown_flags": [],
            "required_telemetry_failure_action": "safety_abort",
        },
        "safety_events": [],
        "diagnostic_events": list(diagnostic_events or []),
    }


def verified_workstation_reentry_receipt():
    policy = workstation_policy()
    return {
        "operating_policy": policy,
        "verification": {
            "status": "ready_for_operator_review",
            "operating_policy": policy,
        },
    }


def test_current_plan_has_exact_8_by_2_val_contract_and_pinned_configs():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert plan["schema_version"] == batch.PLAN_SCHEMA_VERSION
    assert plan["status"] in {
        "planned",
        "running",
        "execution-finished",
        "failed",
        "safety-abort",
    }
    assert plan["gpu_execution_requested"] is (plan["status"] != "planned")
    assert plan["campaign"]["split"] == "val"
    assert plan["campaign"]["sequence_count"] == 8
    assert plan["campaign"]["frame_count"] == 2948
    assert plan["campaign"]["model_input_sizes"] == [640, 960]
    assert plan["campaign"]["expected_jobs"] == 16
    assert len(plan["sequences"]) == 8
    assert len(plan["jobs"]) == 16
    assert {job["model_input"] for job in plan["jobs"]} == {640, 960}
    assert all("--execute" not in job["command"] for job in plan["jobs"])
    for job in plan["jobs"]:
        for config in job["configs"].values():
            path = ROOT / config["path"]
            assert path.is_file()
            assert batch.sha256_file(path) == config["sha256"]
            assert path.stat().st_size == config["bytes"]


def test_current_dry_plan_validates_against_published_schema():
    jsonschema = pytest.importorskip("jsonschema")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    schema = json.loads(
        (
            ROOT
            / "validation/schemas/loaf-deepstream-batch-plan-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(plan)


def test_custom_output_plan_cannot_overwrite_canonical_generated_configs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("validation.run_caviar.REPO_ROOT", tmp_path)

    def model_profile(size):
        return {
            "size": size,
            "runtime_artifacts": {
                "yolo11s_b12_gpu0_fp16.engine": {
                    "path": f"models/person/{size}/engine",
                    "bytes": 1,
                    "sha256": f"{size // 320}" * 64,
                },
                "labels.txt": {
                    "path": f"models/person/{size}/labels.txt",
                    "bytes": 1,
                    "sha256": "a" * 64,
                },
            },
        }

    monkeypatch.setattr(batch, "_extended_model_profile", model_profile)
    monkeypatch.setattr(
        batch,
        "_code_contract",
        lambda: {"job_runner": {"sha256": "b" * 64}},
    )
    sequence = {
        "sequence_id": "loaf-0001",
        "video": {"path": "inputs/video.mp4", "bytes": 1, "sha256": "c" * 64},
        "ground_truth": {"path": "inputs/gt.json", "bytes": 1, "sha256": "d" * 64},
        "frame_mapping": {"frame_count": 1},
    }
    source_contract = {"distance_band_m": [20, 25]}
    reentry = tmp_path / "reentry/evidence.json"
    canonical_output = (
        tmp_path / "validation/results/loaf/val-20-25m/deepstream"
    )
    canonical = batch.build_plan(
        [sequence],
        source_contract,
        output_root=canonical_output,
        reentry_evidence=reentry,
    )
    canonical_bytes = {
        config["path"]: (tmp_path / config["path"]).read_bytes()
        for job in canonical["jobs"]
        for config in job["configs"].values()
    }

    readiness_output = (
        tmp_path / "validation/results/readiness-audit/current/loaf/jobs"
    )
    with pytest.raises(ValueError, match="requires its generated config root"):
        batch.build_plan(
            [sequence],
            source_contract,
            output_root=readiness_output,
            generated_config_root=(
                tmp_path / "validation/generated/loaf-val-20-25m"
            ),
            reentry_evidence=reentry,
        )

    readiness = batch.build_plan(
        [sequence],
        source_contract,
        output_root=readiness_output,
        reentry_evidence=reentry,
    )
    legacy = batch.build_plan(
        [sequence],
        source_contract,
        output_root=tmp_path / "validation/results/loaf-legacy",
        reentry_evidence=reentry,
        gpu_operating_policy_mode=batch.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE,
    )
    assert canonical["campaign"]["generated_config_root"] == (
        "validation/generated/loaf-val-20-25m"
    )
    assert readiness["campaign"]["generated_config_root"] == (
        "validation/results/readiness-audit/current/loaf/jobs/generated-configs"
    )
    canonical_safety = canonical["campaign"]["gpu_safety"]
    assert canonical_safety["operating_policy_mode"] == (
        batch.DEFAULT_GPU_OPERATING_POLICY_MODE
    )
    assert canonical_safety["reentry_operating_policy_id"] == (
        batch.WORKSTATION_MANAGED_POLICY_ID
    )
    assert canonical_safety["reentry_operating_policy"] == workstation_policy()
    assert canonical_safety["temperature_power_slowdown_action"] == "record_only"
    assert canonical_safety["required_telemetry_failure_action"] == "abort"
    assert legacy["campaign"]["gpu_safety"]["operating_policy_mode"] == (
        batch.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
    )
    assert legacy["campaign"]["gpu_safety"]["reentry_operating_policy_id"] == (
        batch.LEGACY_STRICT_PHYSICAL_POLICY_ID
    )
    assert legacy["campaign"]["gpu_safety"][
        "reentry_operating_policy"
    ] == batch.operating_policy_contract(batch.LEGACY_STRICT_PHYSICAL_POLICY_ID)
    assert legacy["campaign"]["gpu_safety"][
        "temperature_power_slowdown_action"
    ] == "software_abort"
    assert {
        config["path"]
        for job in canonical["jobs"]
        for config in job["configs"].values()
    }.isdisjoint(
        {
            config["path"]
            for job in readiness["jobs"]
            for config in job["configs"].values()
        }
    )
    for relative, content in canonical_bytes.items():
        path = tmp_path / relative
        assert path.read_bytes() == content
    for plan in (canonical, readiness):
        for job in plan["jobs"]:
            for config in job["configs"].values():
                path = tmp_path / config["path"]
                assert batch.sha256_file(path) == config["sha256"]
                assert path.stat().st_size == config["bytes"]


def test_current_sequence_views_are_scoped_and_preserve_all_frame_ids():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    total_frames = total_targets = 0
    for sequence in plan["sequences"]:
        payload = json.loads(
            (ROOT / sequence["ground_truth"]["path"]).read_text(encoding="utf-8")
        )
        sequence_id = sequence["sequence_id"]
        images = payload["images"]
        assert {image["sequence_id"] for image in images} == {sequence_id}
        assert [image["frame_id"] for image in images] == list(range(len(images)))
        assert len(images) == sequence["frame_mapping"]["frame_count"]
        image_ids = {image["id"] for image in images}
        assert all(annotation["image_id"] in image_ids for annotation in payload["annotations"])
        total_frames += len(images)
        total_targets += sequence["frame_mapping"]["target_annotations"]
    assert total_frames == 2948
    assert total_targets == 7539


def test_docker_command_has_one_environment_pair_before_workspace_mount(tmp_path, monkeypatch):
    monkeypatch.setattr("validation.run_loaf.REPO_ROOT", tmp_path)
    config = tmp_path / "config.txt"
    config.write_text("[application]\n", encoding="utf-8")
    command = docker_command(
        image="deepstream:test",
        gpu=2,
        container_name="loaf-test",
        deepstream_config=config,
    )
    assert command.count("-e") == 1
    environment_index = command.index("-e")
    assert command[environment_index + 1] == (
        "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility"
    )
    assert command[environment_index + 2] == "-v"
    assert command.index("deepstream:test") > command.index("-w")


def test_aggregate_withholds_every_metric_and_never_merges_when_incomplete(
    monkeypatch,
):
    plan = {
        "plan_fingerprint": "f" * 64,
        "campaign": {"model_input_sizes": [640, 960]},
        "sequences": [{"sequence_id": "a"}],
        "jobs": [
            {"job_id": "a:640", "sequence_id": "a", "model_input": 640},
            {"job_id": "a:960", "sequence_id": "a", "model_input": 960},
        ],
    }
    monkeypatch.setattr(
        batch, "inspect_job", lambda job, sequence, plan: {"state": "missing", "reasons": []}
    )
    monkeypatch.setattr(
        batch,
        "_merge_profile",
        lambda *args, **kwargs: pytest.fail("profile merge must be withheld"),
    )
    aggregate = batch.aggregate_plan(plan)
    assert aggregate["aggregation_status"] == "withheld_incomplete"
    assert aggregate["completeness"]["complete_jobs"] == 0
    assert aggregate["results"] == []
    assert aggregate["profiles"] == {}
    assert aggregate["paired_profile_comparison"] is None


def test_test_unseen_execution_is_rejected_before_discovery(monkeypatch, capsys):
    monkeypatch.setattr(
        batch,
        "discover_sequences",
        lambda **kwargs: pytest.fail("discovery must not run after leakage gate fails"),
    )
    result = batch.main(["--execute", "--split", "test-unseen"])
    assert result == 2
    assert "test-unseen execution is refused by default" in capsys.readouterr().err


def test_default_cli_is_gpu_and_docker_inert():
    args = batch.build_parser().parse_args([])
    assert args.execute is False
    assert args.dry_run is False
    assert args.permit_test_unseen_execution is False
    assert args.gpu_operating_policy_mode == batch.DEFAULT_GPU_OPERATING_POLICY_MODE
    legacy = batch.build_parser().parse_args(
        ["--gpu-operating-policy-mode", "legacy_strict"]
    )
    assert legacy.gpu_operating_policy_mode == (
        batch.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
    )


def test_legacy_strict_preflight_policy_contract_remains_selectable():
    report = workstation_preflight_report()
    policy = report["power_safety_policy"]
    policy["operating_policy_mode"] = batch.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
    policy["static_signal_action"] = "safety_abort"
    policy["abort_slowdown_flags"] = batch.DIAGNOSTIC_SLOWDOWN_FLAGS
    binding = batch._preflight_policy_binding(
        report,
        operating_policy=batch.operating_policy_contract(
            batch.LEGACY_STRICT_PHYSICAL_POLICY_ID
        ),
        operating_policy_mode=batch.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE,
    )
    assert binding["status"] == "exact_match"
    assert binding["temperature_power_slowdown_action"] == "software_abort"


def test_execute_plan_holds_shared_lock_and_preflights_before_launch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    events = []
    sequence = {"sequence_id": "loaf-x"}
    job = {
        "job_id": "loaf-x:640",
        "sequence_id": "loaf-x",
        "model_input": 640,
        "run_root": "out/runs/loaf-x/640",
        "command": ["worker"],
        "status": "planned",
    }
    plan = {
        "campaign": {
            "gpu": 0,
            "container_image": "image",
            "output_root": "out",
            "ds9_runtime_compatibility": {
                "receipt": "compat/receipt.json",
            },
            "gpu_safety": campaign_gpu_safety(),
        },
        "sequences": [sequence],
        "jobs": [job],
    }
    monkeypatch.setattr(
        batch,
        "gpu_lock",
        lambda gpu: events.append("lock") or nullcontext({"gpu": gpu}),
    )
    monkeypatch.setattr(
        batch,
        "prevalidate_runtime_compatibility",
        lambda *args, **kwargs: events.append("compatibility-prevalidate")
        or {"status": "production_ready"},
    )
    inspections = iter(
        [
            {"state": "missing", "reasons": []},
            {"state": "complete", "reasons": []},
        ]
    )
    monkeypatch.setattr(batch, "inspect_job", lambda *args: next(inspections))
    def fake_preflight(**kwargs):
        events.append("preflight")
        assert kwargs["operating_policy_mode"] == (
            batch.DEFAULT_GPU_OPERATING_POLICY_MODE
        )
        return workstation_preflight_report(
            diagnostic_events=[record_only_diagnostic_event()]
        )

    monkeypatch.setattr(batch, "preflight", fake_preflight)
    monkeypatch.setattr(
        batch,
        "require_runtime_compatibility",
        lambda *args, **kwargs: events.append("compatibility-image")
        or {"status": "production_ready"},
    )
    monkeypatch.setattr(
        "validation.gpu_reentry_evidence.require_reentry_evidence",
        lambda *args, **kwargs: events.append("reentry")
        or verified_workstation_reentry_receipt(),
    )

    class Completed:
        returncode = 0

    monkeypatch.setattr(
        batch.subprocess,
        "run",
        lambda *args, **kwargs: events.append("launch") or Completed(),
    )
    result = batch.execute_plan(
        plan,
        tmp_path / "plan.json",
        retry_incomplete=False,
        rerun_complete=False,
        continue_on_error=False,
        max_jobs=None,
        max_temperature_c=86,
        allow_non_performance_profile=False,
        power_limit_drop_tolerance_w=5,
        slowdown_consecutive_samples=2,
    )
    assert result == 0
    assert events == [
        "compatibility-prevalidate",
        "lock",
        "reentry",
        "preflight",
        "compatibility-image",
        "launch",
    ]
    assert job["power_safety_preflight"]["policy_binding_status"] == "exact_match"
    assert job["power_safety_preflight"]["diagnostic_event_count"] == 1


def test_execute_plan_reentry_failure_never_reaches_preflight_or_launch(
    tmp_path, monkeypatch
):
    from validation.gpu_reentry_evidence import ReentryEvidenceError

    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    events = []
    plan = {
        "campaign": {
            "gpu": 0,
            "container_image": "image",
            "output_root": "out",
            "ds9_runtime_compatibility": {
                "receipt": "compat/receipt.json",
            },
            "gpu_safety": campaign_gpu_safety(),
        },
        "sequences": [{"sequence_id": "loaf-x"}],
        "jobs": [
            {
                "job_id": "loaf-x:640",
                "sequence_id": "loaf-x",
                "model_input": 640,
                "run_root": "out/runs/loaf-x/640",
                "command": ["worker"],
                "status": "planned",
            }
        ],
    }
    monkeypatch.setattr(
        batch,
        "gpu_lock",
        lambda gpu: events.append("lock") or nullcontext({"gpu": gpu}),
    )
    monkeypatch.setattr(
        batch,
        "prevalidate_runtime_compatibility",
        lambda *args, **kwargs: events.append("compatibility-prevalidate")
        or {"status": "production_ready"},
    )
    monkeypatch.setattr(
        batch,
        "inspect_job",
        lambda *args: {"state": "missing", "reasons": []},
    )
    monkeypatch.setattr(
        "validation.gpu_reentry_evidence.require_reentry_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ReentryEvidenceError("physical declaration incomplete")
        ),
    )
    monkeypatch.setattr(
        batch,
        "preflight",
        lambda **kwargs: pytest.fail("preflight must not run when re-entry is blocked"),
    )
    monkeypatch.setattr(
        batch.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("worker must not launch"),
    )

    result = batch.execute_plan(
        plan,
        tmp_path / "plan.json",
        retry_incomplete=False,
        rerun_complete=False,
        continue_on_error=False,
        max_jobs=None,
        max_temperature_c=86,
        allow_non_performance_profile=False,
        power_limit_drop_tolerance_w=5,
        slowdown_consecutive_samples=2,
    )
    assert result == 2
    assert events == ["compatibility-prevalidate", "lock"]
    assert plan["jobs"][0]["status"] == "safety-abort-reentry"
    artifact = json.loads(
        (tmp_path / "out/preflight/loaf-x-640.json").read_text(encoding="utf-8")
    )
    assert artifact["status"] == "reentry_blocked"
    assert artifact["gpu_process_started"] is False


def test_execute_plan_rejects_verified_reentry_policy_that_differs_from_plan(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    events = []
    job = {
        "job_id": "loaf-x:640",
        "sequence_id": "loaf-x",
        "model_input": 640,
        "run_root": "out/runs/loaf-x/640",
        "command": ["worker"],
        "status": "planned",
    }
    plan = {
        "campaign": {
            "gpu": 0,
            "container_image": "image",
            "output_root": "out",
            "ds9_runtime_compatibility": {"receipt": "compat/receipt.json"},
            "gpu_safety": campaign_gpu_safety(),
        },
        "sequences": [{"sequence_id": "loaf-x"}],
        "jobs": [job],
    }
    monkeypatch.setattr(
        batch,
        "prevalidate_runtime_compatibility",
        lambda *args, **kwargs: events.append("compatibility-prevalidate")
        or {"status": "production_ready"},
    )
    monkeypatch.setattr(
        batch,
        "gpu_lock",
        lambda gpu: events.append("lock") or nullcontext({"gpu": gpu}),
    )
    monkeypatch.setattr(
        batch, "inspect_job", lambda *args: {"state": "missing", "reasons": []}
    )
    legacy_policy = batch.operating_policy_contract(
        batch.LEGACY_STRICT_PHYSICAL_POLICY_ID
    )
    monkeypatch.setattr(
        "validation.gpu_reentry_evidence.require_reentry_evidence",
        lambda *args, **kwargs: events.append("reentry")
        or {
            "operating_policy": legacy_policy,
            "verification": {
                "status": "ready_for_operator_review",
                "operating_policy": legacy_policy,
            },
        },
    )
    monkeypatch.setattr(
        batch,
        "preflight",
        lambda **kwargs: pytest.fail("mismatched policy must block preflight"),
    )
    monkeypatch.setattr(
        batch.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("mismatched policy must block launch"),
    )

    result = batch.execute_plan(
        plan,
        tmp_path / "plan.json",
        retry_incomplete=False,
        rerun_complete=False,
        continue_on_error=False,
        max_jobs=None,
        max_temperature_c=86,
        allow_non_performance_profile=False,
        power_limit_drop_tolerance_w=5,
        slowdown_consecutive_samples=2,
    )

    assert result == 2
    assert events == ["compatibility-prevalidate", "lock", "reentry"]
    assert job["status"] == "safety-abort-reentry"
    assert "differs from campaign plan" in job["error"]


def test_canonical_frame_key_digest_is_order_and_duplicate_independent():
    expected = batch.canonical_frame_key_sha256([("b", 0), ("a", 1), ("a", 0)])
    assert expected == batch.canonical_frame_key_sha256(
        [("a", 0), ("a", 1), ("b", 0), ("a", 0)]
    )


def test_inspect_job_requires_manifest_hashes_and_exact_prediction_frame_map(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    sequence_id = "loaf-x"
    frame_digest = batch.canonical_frame_key_sha256(
        [(sequence_id, 0), (sequence_id, 1)]
    )
    sequence = {
        "sequence_id": sequence_id,
        "video": {"path": "video.mp4", "sha256": "1" * 64},
        "ground_truth": {"path": "gt.json", "sha256": "2" * 64},
        "frame_mapping": {
            "frame_count": 2,
            "width": 512,
            "height": 512,
            "frame_key_sha256": frame_digest,
        },
    }
    engine = {"path": "engine", "sha256": "3" * 64}
    labels = {"path": "labels", "sha256": "4" * 64}
    job = {
        "job_id": f"{sequence_id}:640",
        "sequence_id": sequence_id,
        "model_input": 640,
        "run_root": "runs/x/640",
        "configs": {
            "deepstream": {"path": "app.txt", "sha256": "5" * 64},
            "infer": {"path": "infer.txt", "sha256": "6" * 64},
        },
        "model_profile": {
            "runtime_artifacts": {
                "yolo11s_b12_gpu0_fp16.engine": engine,
                "labels.txt": labels,
            }
        },
    }
    code_names = (
        "job_runner",
        "config_renderer",
        "kitti_converter",
        "evaluation_cli",
        "evaluation_readers",
        "evaluation_metrics",
        "active_gpu_guard",
        "gpu_reentry_evidence",
    )
    plan = {
        "plan_fingerprint": "7" * 64,
        "campaign": {
            "split": "val",
            "gpu_safety": campaign_gpu_safety(),
            "code": {
                name: {"sha256": f"{index:x}" * 64}
                for index, name in enumerate(code_names, 8)
            },
        },
    }
    run_root = tmp_path / job["run_root"]
    run_root.mkdir(parents=True)
    records = [
        {
            "schema_version": "deepsafe.person-detections/v1",
            "sequence_id": sequence_id,
            "frame_index": index,
            "image_width": 512,
            "image_height": 512,
            "model_id": "yolo11s-640-fp16",
            "detections": [],
        }
        for index in range(2)
    ]
    (run_root / "predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (run_root / "conversion.json").write_text(
        json.dumps(
            {
                "sequence_id": sequence_id,
                "decoded_frame_files": 2,
                "exported_frame_records": 2,
                "skipped_unannotated_frames": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_root / "evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.person-evaluation/v1",
                "diagnostics": {
                    "prediction_only_frames": [],
                    "ground_truth_only_frame_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_root / "deepstream.log").write_text("ok\n", encoding="utf-8")
    output_hashes = {
        name: batch.sha256_file(run_root / name)
        for name in (
            "predictions.jsonl",
            "conversion.json",
            "evaluation.json",
            "deepstream.log",
        )
    }
    manifest = {
        "schema_version": "deepsafe.loaf-deepstream-run/v1",
        "status": "complete",
        "split": "val",
        "sequence_id": sequence_id,
        "model_input": 640,
        "plan_fingerprint": plan["plan_fingerprint"],
        "video": {
            "path": sequence["video"]["path"],
            "sha256": sequence["video"]["sha256"],
            "exact_decoded_frame_count": 2,
        },
        "ground_truth": sequence["ground_truth"],
        "configs": job["configs"],
        "model_profile": {
            "engine": engine,
            "labels": labels,
        },
        "code_hashes": {
            name: plan["campaign"]["code"][name]["sha256"] for name in code_names
        },
        "output_sha256": output_hashes,
        "gpu_safety": {
            "status": "complete",
            "execution_boundary": "validation.gpu_guarded_process/v1",
        },
    }
    (run_root / "run-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    policy = workstation_policy()
    diagnostic_event = record_only_diagnostic_event()
    diagnostics = {
        "preflight_static_signals": [],
        "runtime_monitor_events": [diagnostic_event],
        "runtime_monitor_event_counts": {diagnostic_event["code"]: 1},
    }
    safety = run_root / "safety/gpu-guard-report.json"
    safety.parent.mkdir(parents=True)
    safety.write_text(
        json.dumps(
            {
                "schema_version": batch.GPU_GUARD_REPORT_SCHEMA_VERSION,
                "status": "complete",
                "operating_policy_id": batch.WORKSTATION_MANAGED_POLICY_ID,
                "operating_policy": policy,
                "policy": {
                    "operating_policy_id": batch.WORKSTATION_MANAGED_POLICY_ID,
                    "temperature_power_slowdown_action": "record_only",
                },
                "preflight": {
                    "power_safety_policy": {
                        "operating_policy_mode": (
                            batch.DEFAULT_GPU_OPERATING_POLICY_MODE
                        )
                    }
                },
                "diagnostics": diagnostics,
            }
        ),
        encoding="utf-8",
    )
    guard_receipt = run_root / "safety/gpu-guard-artifact-receipt.json"
    guard_receipt.write_text(
        json.dumps(
            {
                "schema_version": batch.GPU_GUARD_RECEIPT_SCHEMA_VERSION,
                "guard_status": "complete",
                "operating_policy_id": batch.WORKSTATION_MANAGED_POLICY_ID,
                "operating_policy": policy,
                "diagnostics": diagnostics,
                "safety_event": {"present": False, "disposition": None},
                "record_only_diagnostic_event": {
                    "present": True,
                    "disposition": "record_only",
                },
            }
        ),
        encoding="utf-8",
    )

    assert batch.inspect_job(job, sequence, plan) == {
        "state": "complete",
        "reasons": [],
    }
    mismatched_receipt = json.loads(guard_receipt.read_text(encoding="utf-8"))
    mismatched_receipt["operating_policy_id"] = (
        batch.LEGACY_STRICT_PHYSICAL_POLICY_ID
    )
    guard_receipt.write_text(json.dumps(mismatched_receipt), encoding="utf-8")
    inspection = batch.inspect_job(job, sequence, plan)
    assert inspection["state"] == "conflict"
    assert any(
        "receipt operating-policy ID differs" in reason
        for reason in inspection["reasons"]
    )
    guard_receipt.write_text(
        json.dumps({**mismatched_receipt, "operating_policy_id": policy["id"]}),
        encoding="utf-8",
    )
    records[1]["frame_index"] = 2
    (run_root / "predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    inspection = batch.inspect_job(job, sequence, plan)
    assert inspection["state"] == "conflict"
    assert any("frame keys" in reason for reason in inspection["reasons"])
