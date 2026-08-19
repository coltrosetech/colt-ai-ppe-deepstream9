from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from admin import validation as admin_validation
from validation import product_finalization_v2 as finalizer
from validation.finalize_validation import FinalizationError


SOURCE_ROOT = Path(finalizer.__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sign(value: dict, field: str = "fingerprint_sha256") -> dict:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = hashlib.sha256(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def _pin(project: Path, relative: str | Path) -> dict:
    relative = Path(relative)
    content = (project / relative).read_bytes()
    return {
        "path": relative.as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _kind_policy(kind: str) -> dict:
    quality: dict
    if kind == "pose":
        quality = {
            "metric_family": "OKS",
            "minimum_ground_truth_keypoints": 1,
            "minimum_metric_value": 0.8,
        }
    elif kind == "ppe":
        quality = {
            "minimum_ground_truth_instances_by_class": {
                "helmet": 1,
                "hi_vis": 1,
            },
            "minimum_attribute_metrics": {
                "helmet": {"minimum_precision": 0.85, "minimum_recall": 0.9},
                "hi_vis": {"minimum_precision": 0.85, "minimum_recall": 0.9},
            },
            "minimum_ground_truth_events": 1,
            "minimum_event_precision": 0.85,
            "minimum_event_recall": 0.9,
            "maximum_event_latency_p95_ms": 2000,
            "maximum_false_transition_rate": 0.1,
            "maximum_false_safe_rate": 0.1,
        }
    else:
        quality = {
            "minimum_metadata_fusion_match_rate": 0.95,
            "maximum_unmatched_metadata_rate": 0.05,
            "maximum_dropped_frame_rate": 0.05,
            "maximum_fatal_error_count": 0,
        }
    return _sign(
        {
            "schema_version": f"deepsafe.{kind}-product-acceptance-policy/v1",
            "policy_id": f"fixture-{kind}-policy-v1",
            "status": "approved",
            "approval_strictly_before_campaign": True,
            "profiles": [
                {
                    "model_input": profile,
                    "minimum_aggregate_fps": 1.0,
                    "minimum_per_stream_fps": 0.01,
                }
                for profile in (640, 960)
            ],
            "quality": quality,
        }
    )


def _copy_fixed_sources(project: Path) -> None:
    generated = {
        finalizer.FREEZE_RELATIVE_PATH,
        finalizer.EXECUTION_RELATIVE_PATH,
        finalizer.HUMAN_QA_RELATIVE_PATH,
        finalizer.SOURCE_MATRIX_RELATIVE_PATH,
        *finalizer.KIND_POLICY_PATHS.values(),
        *finalizer.ACCEPTANCE_PATHS.values(),
        *finalizer.RAW_REPLAY_PATHS.values(),
        Path("validation/results/campaign-report/report.json"),
    }
    for relative in finalizer.FIXED_INPUT_PATHS:
        if relative in generated:
            continue
        source = SOURCE_ROOT / relative
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _full_stack_plan(profile: int) -> dict:
    source_ids = list(range(12))
    return {
        "schema_version": "deepsafe.deepstream-full-stack-plan/v1",
        "mode": "authorize_launch",
        "execution_ready": True,
        "launch_authorized": True,
        "readiness_blockers": [],
        "profile": {
            "id": str(profile),
            "input_width": profile,
            "input_height": profile,
        },
        "runtime": {"active_deepstream_version": "9.0.0"},
        "sources": {
            "count": 12,
            "batch_size": 12,
            "max_batch_size": 12,
            "ids": source_ids,
            "uris": [f"file:///fixture/source-{index}.mp4" for index in source_ids],
        },
        "topology": {
            "branches": [
                {"role": role, "source_ids": source_ids}
                for role in ("person", "pose", "ppe")
            ]
        },
    }


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    results = project / "validation/results"
    results.mkdir(parents=True)
    _copy_fixed_sources(project)

    for kind, relative in finalizer.KIND_POLICY_PATHS.items():
        _write_json(project / relative, _kind_policy(kind))
    for kind in finalizer.REQUIRED_KINDS:
        _write_json(
            project / finalizer.ACCEPTANCE_PATHS[kind],
            {"schema_version": f"fixture-{kind}-acceptance"},
        )
        _write_json(
            project / finalizer.RAW_REPLAY_PATHS[kind],
            {"schema_version": f"fixture-{kind}-raw-replay"},
        )
    _write_json(
        project / "validation/results/campaign-report/report.json",
        {"schema_version": "fixture-campaign"},
    )

    source_rows = []
    for source_id in range(12):
        media = Path(f"data/final-acceptance/source-{source_id}.mp4")
        media_path = project / media
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(f"fixture-video-{source_id}\n".encode())
        views = ["wide"]
        if source_id == 0:
            views = ["medium_close"]
        elif source_id == 1:
            views = ["overhead_security_camera"]
        source_rows.append(
            {
                "source_id": source_id,
                "video_type": f"video_type_{min(source_id, 9)}",
                "view_types": views,
                "plan_uri": f"file:///fixture/source-{source_id}.mp4",
                "media_pin": _pin(project, media),
            }
        )
    source_matrix = _sign(
        {
            "schema_version": finalizer.SOURCE_MATRIX_SCHEMA,
            "state": "frozen_before_measurement",
            "source_count": 12,
            "minimum_distinct_video_types": 10,
            "required_view_types": ["medium_close", "overhead_security_camera"],
            "sources": source_rows,
        }
    )
    _write_json(project / finalizer.SOURCE_MATRIX_RELATIVE_PATH, source_matrix)

    plan_pins: dict[int, dict] = {}
    for profile, relative in finalizer.PLAN_PATHS.items():
        _write_json(project / relative, _full_stack_plan(profile))
        plan_pins[profile] = _pin(project, relative)

    frozen_artifacts = []
    for role in finalizer.REQUIRED_FROZEN_ARTIFACT_ROLES:
        if role == "source_matrix":
            relative = finalizer.SOURCE_MATRIX_RELATIVE_PATH
        else:
            relative = Path(f"models/final-freeze/{role}.bin")
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"frozen:{role}\n".encode())
        frozen_artifacts.append({"role": role, "pin": _pin(project, relative)})

    policies = [
        {
            "id": "three_module",
            "pin": _pin(project, finalizer.COMBINED_POLICY_PATH),
        },
        *[
            {"id": kind, "pin": _pin(project, finalizer.KIND_POLICY_PATHS[kind])}
            for kind in finalizer.REQUIRED_KINDS
        ],
    ]
    freeze = _sign(
        {
            "schema_version": finalizer.FREEZE_SCHEMA,
            "state": "frozen_before_measurement",
            "freeze_id": "fixture-freeze-v1",
            "frozen_at_utc": "2026-07-18T00:00:00+00:00",
            "runtime": {
                "deepstream_version": "9.0.0",
                "modules": ["person", "pose", "ppe"],
                "simulated_streams": 12,
            },
            "policies": policies,
            "profiles": [
                {
                    "model_input": profile,
                    "simulated_streams": 12,
                    "measurement_seconds": 300,
                    "plan_pin": plan_pins[profile],
                }
                for profile in (640, 960)
            ],
            "frozen_artifacts": frozen_artifacts,
        }
    )
    _write_json(project / finalizer.FREEZE_RELATIVE_PATH, freeze)

    profile_runs = []
    for profile, start, finish in (
        (640, "2026-07-18T00:01:00+00:00", "2026-07-18T00:06:05+00:00"),
        (960, "2026-07-18T00:07:00+00:00", "2026-07-18T00:12:05+00:00"),
    ):
        telemetry = Path(
            f"validation/results/product-validation/three-module/run-{profile}/telemetry.json"
        )
        _write_json(project / telemetry, {"profile": profile, "rows": [1, 2, 3]})
        profile_runs.append(
            {
                "model_input": profile,
                "status": "complete",
                "simulated_streams": 12,
                "started_at_utc": start,
                "finished_at_utc": finish,
                "elapsed_ms": 300000,
                "plan_pin": plan_pins[profile],
                "telemetry_pin": _pin(project, telemetry),
                "runtime_health": {
                    "xid": 0,
                    "oom": 0,
                    "fatal": 0,
                    "unexpected_restart": 0,
                },
            }
        )
    execution = _sign(
        {
            "schema_version": finalizer.EXECUTION_SCHEMA,
            "state": "complete",
            "execution_id": "fixture-execution-v1",
            "freeze_pin": _pin(project, finalizer.FREEZE_RELATIVE_PATH),
            "runtime": {
                "deepstream_version": "9.0.0",
                "gpu_inference_executed": True,
                "modules_enabled_together": ["person", "pose", "ppe"],
            },
            "profile_runs": profile_runs,
            "acceptance_evidence": [
                {
                    "kind": kind,
                    "acceptance_receipt_pin": _pin(
                        project, finalizer.ACCEPTANCE_PATHS[kind]
                    ),
                    "raw_replay_pin": _pin(
                        project, finalizer.RAW_REPLAY_PATHS[kind]
                    ),
                }
                for kind in finalizer.REQUIRED_KINDS
            ],
        }
    )
    _write_json(project / finalizer.EXECUTION_RELATIVE_PATH, execution)

    reviewed = Path(
        "validation/results/product-validation/three-module/review/overlay-index.json"
    )
    _write_json(project / reviewed, {"reviewed_profiles": [640, 960]})
    human_qa = _sign(
        {
            "schema_version": finalizer.HUMAN_QA_SCHEMA,
            "state": "complete",
            "decision": "pass",
            "reviewed_by_role": "human_operator",
            "review_identity": "fixture-operator-self-assertion",
            "reviewed_at_utc": "2026-07-18T00:13:00+00:00",
            "execution_id": "fixture-execution-v1",
            "profiles": [640, 960],
            "modules": ["person", "pose", "ppe"],
            "external_identity_attestation_present": False,
            "limitations_acknowledged": [
                "hash_binding_is_not_external_identity_attestation",
                "human_review_does_not_replace_ground_truth_metrics",
            ],
            "reviewed_artifact_pins": [_pin(project, reviewed)],
        }
    )
    _write_json(project / finalizer.HUMAN_QA_RELATIVE_PATH, human_qa)
    runtime_lock = tmp_path / "runtime/product-finalizer-v2.lock"
    runtime_lock.parent.mkdir()
    return project, results, runtime_lock


def _fake_runner(
    command: list[str], _cwd: Path, _timeout: int
) -> subprocess.CompletedProcess[str]:
    project = Path(command[command.index("--project-root") + 1])
    outputs = {
        "validation/results/objective-completion/current/report.json": {
            "schema_version": "deepsafe.validation-objective-completion/v1",
            "fingerprint_sha256": "1" * 64,
        },
        "validation/results/product-readiness/current/report.json": {
            "schema_version": "deepsafe.product-readiness/v1",
            "fingerprint_sha256": "2" * 64,
        },
    }
    for relative, value in outputs.items():
        _write_json(project / relative, value)
    for relative, text in (
        ("validation/results/objective-completion/current/report.md", "# objective\n"),
        ("validation/results/product-readiness/current/report.md", "# product\n"),
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")


def _fake_bundle(config: finalizer.FinalizerConfig) -> dict:
    outputs = []
    for artifact_id, relative, media_type in finalizer.OUTPUT_SPECS:
        content = (config.project_root / relative).read_bytes()
        outputs.append(
            {
                "id": artifact_id,
                "path": relative,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "media_type": media_type,
            }
        )
    return {
        "outputs": outputs,
        "semantics": {
            "objective_evidence_complete": True,
            "objective_passed_gate_count": 6,
            "product_status": "ready",
            "product_ready_required": True,
            "all_six_product_gates_passed": True,
            "three_modules_enabled_together": True,
            "profiles": [640, 960],
            "simulated_streams": 12,
            "minimum_elapsed_ms_per_profile": 300000,
            "human_visual_qa_bound": True,
            "physical_execution_proof_role": "machine_local_hash_bound_evidence_not_external_attestation",
            "objective_fingerprint_sha256": "1" * 64,
            "product_fingerprint_sha256": "2" * 64,
        },
    }


def _run(project: Path, results: Path, lock: Path, **kwargs: object) -> finalizer.ProductOutcome:
    return finalizer.finalize_product(
        project_root=project,
        results_root=results,
        runtime_lock=lock,
        python=Path(sys.executable),
        runner=kwargs.pop("runner", _fake_runner),  # type: ignore[arg-type]
        bundle_verifier=kwargs.pop("bundle_verifier", _fake_bundle),  # type: ignore[arg-type]
        **kwargs,
    )


def _rebind_source_matrix(project: Path, matrix: dict) -> None:
    matrix_path = project / finalizer.SOURCE_MATRIX_RELATIVE_PATH
    _write_json(matrix_path, _sign(matrix))
    freeze_path = project / finalizer.FREEZE_RELATIVE_PATH
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_row = next(
        row for row in freeze["frozen_artifacts"] if row["role"] == "source_matrix"
    )
    source_row["pin"] = _pin(project, finalizer.SOURCE_MATRIX_RELATIVE_PATH)
    _write_json(freeze_path, _sign(freeze))
    execution_path = project / finalizer.EXECUTION_RELATIVE_PATH
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["freeze_pin"] = _pin(project, finalizer.FREEZE_RELATIVE_PATH)
    _write_json(execution_path, _sign(execution))


def test_missing_freeze_is_pending_and_never_publishes_marker(tmp_path: Path) -> None:
    project, results, lock = _project(tmp_path)
    (project / finalizer.FREEZE_RELATIVE_PATH).unlink()

    outcome = _run(project, results, lock)

    assert outcome.status == "waiting"
    assert outcome.reason == finalizer.FREEZE_RELATIVE_PATH.as_posix()
    assert outcome.mutated is False
    assert not (project / finalizer.RECEIPT_RELATIVE_PATH).exists()


def test_human_qa_is_required_and_not_synthesized(tmp_path: Path) -> None:
    project, results, lock = _project(tmp_path)
    (project / finalizer.HUMAN_QA_RELATIVE_PATH).unlink()

    outcome = _run(project, results, lock)

    assert outcome.status == "waiting"
    assert outcome.reason == finalizer.HUMAN_QA_RELATIVE_PATH.as_posix()
    assert not (project / finalizer.RECEIPT_RELATIVE_PATH).exists()


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("short_run", "execution_bundle"),
        ("eleven_streams", "execution_bundle"),
        ("missing_module", "execution_bundle"),
        ("pre_freeze_run", "execution_profile_run_invalid"),
    ],
)
def test_scope_overclaims_fail_closed(
    tmp_path: Path, mutation: str, expected_reason: str
) -> None:
    project, results, lock = _project(tmp_path)
    path = project / finalizer.EXECUTION_RELATIVE_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "short_run":
        value["profile_runs"][0]["elapsed_ms"] = 299999
    elif mutation == "eleven_streams":
        value["profile_runs"][0]["simulated_streams"] = 11
    elif mutation == "missing_module":
        value["runtime"]["modules_enabled_together"] = ["person", "pose"]
    else:
        value["profile_runs"][0]["started_at_utc"] = "2026-07-17T23:58:00+00:00"
        value["profile_runs"][0]["finished_at_utc"] = "2026-07-18T00:03:05+00:00"
    _write_json(path, _sign(value))

    outcome = _run(project, results, lock)

    assert outcome.status == "failed"
    assert expected_reason in outcome.reason
    assert not (project / finalizer.RECEIPT_RELATIVE_PATH).exists()


def test_acceptance_raw_replay_tamper_fails_before_generators(tmp_path: Path) -> None:
    project, results, lock = _project(tmp_path)
    raw = project / finalizer.RAW_REPLAY_PATHS["ppe"]
    raw.write_text('{"tampered":true}\n', encoding="utf-8")

    def forbidden_runner(*_args: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("tampered evidence must fail before report generation")

    outcome = _run(project, results, lock, runner=forbidden_runner)

    assert outcome.status == "failed"
    assert outcome.reason == "acceptance_pair_pin_mismatch"
    assert outcome.mutated is False


@pytest.mark.parametrize("coverage", ["video_types", "overhead_view"])
def test_source_matrix_requires_ten_types_and_security_camera_views(
    tmp_path: Path, coverage: str
) -> None:
    project, results, lock = _project(tmp_path)
    path = project / finalizer.SOURCE_MATRIX_RELATIVE_PATH
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if coverage == "video_types":
        for row in matrix["sources"]:
            row["video_type"] = f"video_type_{row['source_id'] % 9}"
    else:
        for row in matrix["sources"]:
            row["view_types"] = [
                "medium_close" if row["source_id"] == 0 else "wide"
            ]
    _rebind_source_matrix(project, matrix)

    outcome = _run(project, results, lock)

    assert outcome.status == "failed"
    assert outcome.reason in {
        "source_matrix_video_type_coverage_invalid",
        "source_matrix_view_coverage_invalid",
    }
    assert not (project / finalizer.RECEIPT_RELATIVE_PATH).exists()


def test_hardlinked_frozen_model_artifact_is_rejected(tmp_path: Path) -> None:
    project, results, lock = _project(tmp_path)
    artifact = project / "models/final-freeze/person_engine_640.bin"
    (project / "models/final-freeze/person_engine_640-copy.bin").hardlink_to(artifact)

    outcome = _run(project, results, lock)

    assert outcome.status == "failed"
    assert outcome.reason == "freeze_artifact_pin_mismatch"
    assert not (project / finalizer.RECEIPT_RELATIVE_PATH).exists()


def test_product_not_ready_can_never_publish_v2_marker(tmp_path: Path) -> None:
    project, results, lock = _project(tmp_path)

    def not_ready(_config: finalizer.FinalizerConfig) -> dict:
        raise FinalizationError("product_not_ready")

    outcome = _run(project, results, lock, bundle_verifier=not_ready)

    assert outcome.status == "failed"
    assert outcome.reason == "product_not_ready"
    assert not (project / finalizer.RECEIPT_RELATIVE_PATH).exists()


def test_success_is_separate_v2_commit_and_idempotent(tmp_path: Path) -> None:
    project, results, lock = _project(tmp_path)

    first = _run(project, results, lock)

    assert first.status == "complete"
    assert first.reason == "finalized"
    receipt_path = project / finalizer.RECEIPT_RELATIVE_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == finalizer.RECEIPT_SCHEMA
    assert receipt["semantics"]["product_status"] == "ready"
    assert receipt["semantics"]["product_ready_required"] is True
    assert receipt["lineage"]["profiles"] == [
        {"model_input": 640, "simulated_streams": 12, "elapsed_ms": 300000},
        {"model_input": 960, "simulated_streams": 12, "elapsed_ms": 300000},
    ]
    assert receipt["lock_contract"]["v1_receipt_rewritten"] is False
    assert finalizer._canonical_fingerprint_valid(receipt)

    def forbidden_runner(*_args: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("idempotent finalization cannot rerun generators")

    second = _run(project, results, lock, runner=forbidden_runner)
    assert second.status == "complete"
    assert second.idempotent is True
    assert second.mutated is False


def test_live_pin_change_makes_committed_marker_stale(tmp_path: Path) -> None:
    project, results, lock = _project(tmp_path)
    assert _run(project, results, lock).status == "complete"
    reviewed = project / (
        "validation/results/product-validation/three-module/review/overlay-index.json"
    )
    reviewed.write_text('{"changed":true}\n', encoding="utf-8")

    status = finalizer.inspect_committed_receipt(
        project_root=project,
        results_root=results,
        bundle_verifier=_fake_bundle,
    )

    assert status["state"] == "stale_lineage"
    assert status["committed"] is False
    assert status["reason"] == "stale_lineage"


def test_receipt_path_substitution_is_rejected(tmp_path: Path) -> None:
    project, results, lock = _project(tmp_path)
    other = results / "product-finalization/v2/other.json"

    outcome = finalizer.finalize_product(
        project_root=project,
        results_root=results,
        runtime_lock=lock,
        receipt_path=other,
        runner=_fake_runner,
        bundle_verifier=_fake_bundle,
    )

    assert outcome.status == "failed"
    assert outcome.reason == "receipt_path_not_canonical"
    assert not other.exists()


def test_pre_run_plan_is_schema_valid_and_self_hash_bound() -> None:
    plan_path = SOURCE_ROOT / (
        "validation/inputs/product-acceptance/product-finalization-v2.pre-run-plan.json"
    )
    schema_path = SOURCE_ROOT / (
        "validation/schemas/product-finalization-pre-run-plan-v2.schema.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    finalizer._schema_validate(plan, schema, source="pre_run_plan")
    assert finalizer._canonical_fingerprint_valid(plan)
    assert plan["state"] == "prepared_waiting_for_real_evidence"


def test_admin_v2_projection_is_read_only_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, results, lock = _project(tmp_path)
    assert _run(project, results, lock).status == "complete"
    monkeypatch.setattr(
        admin_validation,
        "_product_v2_report_semantics_valid",
        lambda _reader, _receipt, _reports: True,
    )
    reader = admin_validation.ArtifactReader(
        root=results,
        workspace_root=project,
        schema_root=project / "validation/schemas",
    )

    projection = admin_validation._product_finalization_v2(reader)

    assert projection["reason"] is None, (
        projection["reason"], projection["verified_input_count"]
    )
    assert projection["state"] == "complete", projection
    assert projection["committed"] is True
    assert projection["product_status"] == "ready"
    assert projection["read_only"] is True
    assert projection["execution_actions_available"] is False
    assert projection["raw_download_allowed"] is False
    assert projection["verified_input_count"] == len(finalizer.FIXED_INPUT_PATHS)
    assert projection["committed_input_count"] > projection["verified_input_count"]


def test_admin_html_has_separate_v2_read_only_card() -> None:
    html = (SOURCE_ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert 'id="productFinalizationV2Card"' in html
    assert "renderProductFinalizationV2(payload.product_finalization_v2)" in html
    assert "ürün ready/finalized değildir" in html


def test_admin_v2_projection_marks_fixed_input_tamper_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, results, lock = _project(tmp_path)
    assert _run(project, results, lock).status == "complete"
    monkeypatch.setattr(
        admin_validation,
        "_product_v2_report_semantics_valid",
        lambda _reader, _receipt, _reports: True,
    )
    qa = project / finalizer.HUMAN_QA_RELATIVE_PATH
    qa.write_text('{"tampered":true}\n', encoding="utf-8")
    reader = admin_validation.ArtifactReader(
        root=results,
        workspace_root=project,
        schema_root=project / "validation/schemas",
    )

    projection = admin_validation._product_finalization_v2(reader)

    assert projection["state"] == "stale_lineage"
    assert projection["committed"] is False
    assert projection["reason"] == "stale_lineage"


def test_admin_v2_receipt_is_not_raw_downloadable() -> None:
    spec = admin_validation.ARTIFACTS["product_finalization_v2_receipt"]
    assert spec.raw_download_allowed is False
