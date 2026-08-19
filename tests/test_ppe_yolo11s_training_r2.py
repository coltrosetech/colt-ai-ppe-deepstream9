from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from ppe_dataset.internal_evaluation_authorization_r2 import verify_subset_receipt
from validation import ppe_yolo11s_training_r2 as lane


ROOT = Path(__file__).resolve().parents[1]
PLAN = lane.PLAN_PATH
PLAN_SHA256 = "d0d2a0b239c0575e8b7ff46b470b18c9fba5e568a85863a862ae399d45db7a27"
PLAN_FINGERPRINT = "4d2c089624eaf53f8a8b33ef326b20e3a16dc8ee4af56d84fb12577c98a11118"
LICENSE_FINGERPRINT = "54169756f0fdb23a57897e8bd993df43caf5f9828f5fd828a5be9deb4666bf6a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def plan() -> dict[str, object]:
    return lane.load_plan(
        expected_plan_sha256=PLAN_SHA256,
        execute=False,
    )


def test_execution_plan_schema_fingerprint_and_external_pin() -> None:
    schema = json.loads(lane.PLAN_SCHEMA.read_text(encoding="utf-8"))
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert not list(
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).iter_errors(value)
    )
    assert _sha256(PLAN) == PLAN_SHA256
    assert value["fingerprint_sha256"] == PLAN_FINGERPRINT
    assert lane.canonical_fingerprint(value) == PLAN_FINGERPRINT
    assert value["execution_history"] == {
        "docker_build": False,
        "gpu_query": False,
        "gpu_execution": False,
        "smoke_train": False,
        "baseline_calibration": False,
        "full_train_150e": False,
        "resume": False,
        "development_holdout_open": False,
        "final_test": False,
        "export": False,
    }


def test_authorized_subset_replay_keeps_holdout_payload_unopened(
    plan: dict[str, object],
) -> None:
    dataset = plan["dataset"]
    assert isinstance(dataset, dict)
    receipt = dataset["authorization_receipt"]
    assert isinstance(receipt, dict)
    result = verify_subset_receipt(
        ROOT / str(receipt["path"]),
        expected_receipt_sha256=str(receipt["receipt_sha256"]),
        root=ROOT,
    )
    assert result["authorized_roles"] == ["train", "calibration"]
    assert result["images"] == 2327
    assert result["bbox_rows"] == 16058
    assert result["development_holdout_payload_files_opened"] == 0


def test_checkpoint_is_independent_single_link_read_only(
    plan: dict[str, object],
) -> None:
    model = plan["model"]
    assert isinstance(model, dict)
    checkpoint = model["checkpoint"]
    assert isinstance(checkpoint, dict)
    path = ROOT / str(checkpoint["path"])
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode)
    assert not stat.S_ISLNK(info.st_mode)
    assert info.st_nlink == 1
    assert stat.S_IMODE(info.st_mode) == 0o440
    assert info.st_size == checkpoint["bytes"]
    assert _sha256(path) == checkpoint["sha256"]


def test_default_render_never_invokes_subprocess_or_gpu_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("render path attempted a subprocess")

    monkeypatch.setattr(lane.subprocess, "run", forbidden)
    assert lane.main(["--expected-plan-sha256", PLAN_SHA256]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["status"] == "authorized_inert_render_no_execution"
    assert rendered["execution_performed"] is False
    assert rendered["docker_invoked"] is False
    assert rendered["gpu_queried"] is False
    assert rendered["development_holdout_mounted"] is False


@pytest.mark.parametrize(
    ("flag", "selected"),
    [
        ("--build-image", "build_image"),
        ("--smoke-train", "smoke_train"),
        ("--baseline-calibration", "baseline_calibration"),
        ("--full-train-150e", "full_train_150e"),
    ],
)
def test_selected_mode_without_execute_is_inert(
    flag: str,
    selected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        lane.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inert mode invoked subprocess")
        ),
    )
    arguments = ["--expected-plan-sha256", PLAN_SHA256, flag]
    if selected != "build_image":
        arguments += ["--run-id", lane.MODE_PREFIXES[selected] + "render"]
    assert lane.main(arguments) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == {
        "status": "rendered_not_executed",
        "selected": selected,
        "command": rendered["command"],
        "execution_performed": False,
        "docker_invoked": False,
        "gpu_queried": False,
    }
    assert "--execute" in rendered["command"]


def test_build_render_is_digest_pinned_and_offline(plan: dict[str, object]) -> None:
    attempt = lane.BUILD_ROOT / "image-build-r2-render"
    command = lane.render_build_command(
        plan, attempt_dir=attempt, image_reference=lane.IMAGE_REFERENCE
    )
    assert command[:2] == ["docker", "build"]
    assert "--pull=false" in command
    assert "--network=none" in command
    assert "--platform=linux/amd64" in command
    assert all("--gpus" not in item for item in command)
    dockerfile = lane.DOCKERFILE.read_text(encoding="utf-8")
    assert f"FROM {lane.BASE_IMAGE}" in dockerfile
    assert "ultralytics.__version__)')\" = \"8.4.99\"" in dockerfile
    assert all(
        token not in dockerfile
        for token in ("pip install", "apt-get", "curl ", "wget ")
    )


@pytest.mark.parametrize("mode", lane.MODES)
def test_runtime_render_has_only_authorized_dataset_mounts_and_gpu_lease(
    mode: str, plan: dict[str, object]
) -> None:
    run_id = lane.MODE_PREFIXES[mode] + "render"
    output = lane.RUNS_ROOT / run_id
    image_id = "sha256:" + "1" * 64
    docker = lane.render_runtime_command(
        plan=plan,
        mode=mode,
        run_id=run_id,
        image_id=image_id,
        output_dir=output,
    )
    rendered = "\n".join(docker)
    assert docker[:2] == ["docker", "run"]
    assert "--network=none" in docker
    assert "--read-only" in docker
    assert "--pull=never" in docker
    assert "--gpus=device=0" in docker
    assert "images/train" in rendered
    assert "labels/train" in rendered
    assert "images/calibration" in rendered
    assert "labels/calibration" in rendered
    assert "development_holdout" not in rendered
    assert "/test" not in rendered
    leased = lane.render_lease_command(plan, docker)
    assert leased[1:4] == ["-m", "validation.gpu_lease", "run"]
    assert leased[-len(docker) :] == docker


def test_execute_requires_both_exact_acceptance_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lane.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejected execution reached subprocess")
        ),
    )
    with pytest.raises(lane.PpeYolo11sR2Error, match="exact plan and license"):
        lane.load_plan(
            expected_plan_sha256=PLAN_SHA256,
            execute=True,
            accepted_fingerprint=PLAN_FINGERPRINT,
            accepted_license_fingerprint="0" * 64,
        )


def test_plan_rejects_wrong_external_file_hash() -> None:
    with pytest.raises(lane.PpeYolo11sR2Error, match="external execution plan"):
        lane.load_plan(expected_plan_sha256="0" * 64)


def test_scope_stays_internal_evaluation_only(plan: dict[str, object]) -> None:
    authorization = plan["authorization"]
    production = plan["production"]
    resume = plan["resume_contract"]
    assert isinstance(authorization, dict)
    assert isinstance(production, dict)
    assert isinstance(resume, dict)
    assert authorization["scope_label"] == "AGPL-3.0_internal_evaluation_only"
    assert not any(authorization["explicitly_not_authorized"].values())
    assert production["production_ready"] is False
    assert production["commercially_cleared"] is False
    assert len(production["blockers"]) >= 7
    assert resume["accepted_prior_statuses"] == ["passed", "failed"]
    assert resume["new_output_directory_required"] is True
    assert resume["in_place_resume_forbidden"] is True
