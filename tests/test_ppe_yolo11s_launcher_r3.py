from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from validation import ppe_yolo11s_launcher_r3 as launcher


ROOT = Path(__file__).resolve().parents[1]


def _forbidden(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("blocked/inert R3 path reached an execution backend")


def test_decision_schema_and_canonical_constants_are_exact() -> None:
    schema = json.loads(launcher.DECISION_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert launcher.GATE_PATH == (
        ROOT
        / "validation/authorizations/ppe-yolo11s-semantic-launch-gate-r3.json"
    )
    assert (
        hashlib.sha256(launcher.GATE_PATH.read_bytes()).hexdigest()
        == launcher.EXPECTED_GATE_FILE_SHA256
    )
    assert launcher.GATE_PATH.stat().st_size == launcher.EXPECTED_GATE_BYTES == 3966
    gate = json.loads(launcher.GATE_PATH.read_text(encoding="utf-8"))
    assert gate["fingerprint_sha256"] == launcher.EXPECTED_GATE_FINGERPRINT
    assert stat.S_IMODE(launcher.GATE_PATH.stat().st_mode) == 0o440
    assert launcher.r2_build_backend.PLAN_PATH.read_bytes()
    assert (
        hashlib.sha256(launcher.r2_build_backend.PLAN_PATH.read_bytes()).hexdigest()
        == launcher.EXPECTED_R2_PLAN_SHA256
    )


def test_policy_snapshot_holds_gate_and_every_exact_input() -> None:
    with launcher.verified_policy() as (gate, verification, snapshot):
        assert verification["valid"] is True
        assert gate["fingerprint_sha256"] == launcher.EXPECTED_GATE_FINGERPRINT
        assert len(snapshot.pins) == 6
        assert snapshot.pins[0].path == launcher.GATE_PATH
        assert {pin.path.relative_to(ROOT).as_posix() for pin in snapshot.pins[1:]} == {
            value["path"] for value in gate["inputs"].values()
        }
        snapshot.assert_current()


def test_held_pin_detects_same_content_path_replacement(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    payload = b'{"allowed":false}\n'
    path.write_bytes(payload)
    held = launcher.hold_exact_pin(
        path,
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    try:
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(payload)
        os.replace(replacement, path)
        with pytest.raises(
            launcher.PpeYolo11sR3LauncherError,
            match="path identity changed",
        ):
            held.assert_current()
    finally:
        held.close()


@pytest.mark.parametrize("flag", [action.flag for action in launcher.ACTIONS])
def test_inert_render_is_schema_valid_and_never_reaches_backend(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.r2_build_backend, "load_plan", _forbidden)
    monkeypatch.setattr(launcher.r2_build_backend, "execute_build", _forbidden)
    monkeypatch.setattr(launcher.r2_build_backend, "execute_runtime", _forbidden)
    result = launcher.run([flag])
    schema = json.loads(launcher.DECISION_SCHEMA.read_text(encoding="utf-8"))
    assert not list(
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).iter_errors(result)
    )
    assert result["execution"] == {
        "performed": False,
        "docker_invoked": False,
        "gpu_queried": False,
        "training_executed": False,
        "evaluation_executed": False,
        "export_executed": False,
    }
    assert isinstance(result["execute_argv"], list)
    assert result["execute_argv"][1:3] == ["-m", launcher.MODULE]
    assert "--execute" in result["execute_argv"]


@pytest.mark.parametrize(
    "flag",
    [
        "--smoke-train",
        "--baseline-calibration",
        "--calibration",
        "--evaluation",
        "--full-train-150e",
        "--resume",
        "--export",
    ],
)
def test_every_data_or_model_execute_mode_is_gate_blocked_before_r2(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(launcher.r2_build_backend, "load_plan", _forbidden)
    monkeypatch.setattr(launcher.r2_build_backend, "execute_build", _forbidden)
    monkeypatch.setattr(launcher.r2_build_backend, "execute_runtime", _forbidden)
    result = launcher.main(
        [
            flag,
            "--execute",
            "--accept-gate-file-sha256",
            launcher.EXPECTED_GATE_FILE_SHA256,
            "--accept-gate-fingerprint",
            launcher.EXPECTED_GATE_FINGERPRINT,
        ]
    )
    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "launch blocked for" in captured.err


def test_gate_decision_precedes_argument_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    original = launcher.semantic_gate.check_mode

    def observed(gate: dict, mode: str) -> dict:
        calls.append(mode)
        return original(gate, mode)

    monkeypatch.setattr(launcher.semantic_gate, "check_mode", observed)
    monkeypatch.setattr(launcher.r2_build_backend, "load_plan", _forbidden)
    assert launcher.main(["--resume", "--execute"]) == 2
    assert calls == ["resume"]
    assert "launch blocked for resume" in capsys.readouterr().err


def test_even_an_unexpected_allow_cannot_reach_historical_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        launcher.semantic_gate,
        "check_mode",
        lambda _gate, mode: {"mode": mode, "allowed": True},
    )
    monkeypatch.setattr(launcher.r2_build_backend, "load_plan", _forbidden)
    monkeypatch.setattr(launcher.r2_build_backend, "execute_runtime", _forbidden)
    result = launcher.main(
        [
            "--smoke-train",
            "--execute",
            "--accept-gate-file-sha256",
            launcher.EXPECTED_GATE_FILE_SHA256,
            "--accept-gate-fingerprint",
            launcher.EXPECTED_GATE_FINGERPRINT,
        ]
    )
    assert result == 2
    assert "execution backend is intentionally absent" in capsys.readouterr().err


def test_image_build_dispatch_is_narrow_and_argv_fields_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    fake_plan = {"fingerprint_sha256": launcher.EXPECTED_R2_PLAN_FINGERPRINT}

    def fake_load_plan(**kwargs: object) -> dict[str, str]:
        calls.append(("load_plan", kwargs))
        return fake_plan

    def fake_execute_build(**kwargs: object) -> dict[str, object]:
        calls.append(("execute_build", kwargs))
        return {"status": "passed", "execution": {"docker_build_executed": False}}

    monkeypatch.setattr(launcher.r2_build_backend, "load_plan", fake_load_plan)
    monkeypatch.setattr(launcher.r2_build_backend, "execute_build", fake_execute_build)
    monkeypatch.setattr(launcher.r2_build_backend, "execute_runtime", _forbidden)
    result = launcher.run(
        [
            "--build-image",
            "--execute",
            "--accept-gate-file-sha256",
            launcher.EXPECTED_GATE_FILE_SHA256,
            "--accept-gate-fingerprint",
            launcher.EXPECTED_GATE_FINGERPRINT,
            "--accept-plan-fingerprint",
            launcher.EXPECTED_R2_PLAN_FINGERPRINT,
            "--accept-license-decision-fingerprint",
            launcher.EXPECTED_R2_LICENSE_FINGERPRINT,
            "--build-attempt-id",
            "image-build-r3-test",
        ]
    )
    assert result["status"] == "passed"
    assert calls == [
        (
            "load_plan",
            {
                "expected_plan_sha256": launcher.EXPECTED_R2_PLAN_SHA256,
                "execute": True,
                "accepted_fingerprint": launcher.EXPECTED_R2_PLAN_FINGERPRINT,
                "accepted_license_fingerprint": launcher.EXPECTED_R2_LICENSE_FINGERPRINT,
            },
        ),
        (
            "execute_build",
            {"plan": fake_plan, "build_attempt_id": "image-build-r3-test"},
        ),
    ]


def test_no_alternate_path_or_opaque_argument_forwarding_is_accepted() -> None:
    parser = launcher._parser()
    for arguments in (
        ["--gate", "/tmp/alternate.json"],
        ["--plan", "/tmp/alternate.json"],
        ["--smoke-train", "--", "--shell-fragment"],
    ):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(arguments)
        assert exc.value.code == 2
    with pytest.raises(
        launcher.PpeYolo11sR3LauncherError,
        match="valid only with --execute",
    ):
        launcher.run(["--smoke-train", "--build-attempt-id", "unexpected"])


def test_source_never_calls_r2_cli_or_runtime_and_has_no_shell() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "r2_build_backend.main" not in source
    assert "r2_build_backend.execute_runtime" not in source
    assert "shell=True" not in source
    assert "subprocess" not in source
