from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from validation import reproduction_runner as runner


ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE = "validation.report_campaign"


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in runner.REJECTED_AMBIENT_ENVIRONMENT:
        environment.pop(name, None)
    return environment


@contextmanager
def _local_directory() -> Iterator[Path]:
    directory = Path(tempfile.mkdtemp(prefix=".reproduction-runner-test.", dir=ROOT))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _wrapper_command(trace: Path, *target_arguments: str) -> list[str]:
    test_harness = (
        "import sys; from validation import reproduction_runner as r; "
        "raise SystemExit(r.main(sys.argv[1:], _test_skip_launch_contract=True))"
    )
    return [
        sys.executable,
        "-c",
        test_harness,
        "--trace",
        str(trace),
        TARGET_MODULE,
        *target_arguments,
    ]


def _production_command(*arguments: str, flags: tuple[str, ...] = ("-I", "-S")) -> list[str]:
    return [
        "/proc/self/exe",
        *flags,
        os.fspath(ROOT / "validation" / "reproduction_runner.py"),
        *arguments,
    ]


def _trusted_environment() -> dict[str, str]:
    return dict(runner.TRUSTED_LAUNCH_ENVIRONMENT)


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=_clean_environment() if environment is None else environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def _canonical_sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_success_preserves_target_streams_and_emits_complete_stable_trace() -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        direct = _run([sys.executable, "-m", TARGET_MODULE, "--help"])
        wrapped = _run(_wrapper_command(trace_path, "--help"))

        assert direct.returncode == 0
        assert wrapped.returncode == direct.returncode
        assert wrapped.stdout == direct.stdout
        assert wrapped.stderr == direct.stderr == ""
        assert trace_path.is_file()
        assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600
        assert trace_path.stat().st_nlink == 1

        trace = json.loads(trace_path.read_bytes())
        assert trace["schema_version"] == runner.TRACE_SCHEMA
        assert trace["target"] == {"module": TARGET_MODULE, "argv": ["--help"]}
        fingerprint = trace.pop("fingerprint_sha256")
        assert fingerprint == _canonical_sha256(trace)

        pins = trace["loaded_project_modules"]
        assert pins == sorted(pins, key=lambda item: (item["path"], item["module_names"]))
        by_path = {pin["path"]: pin for pin in pins}
        assert "validation/__init__.py" in by_path
        assert "validation/report_campaign.py" in by_path
        assert "validation/reproduction_runner.py" in by_path
        assert TARGET_MODULE in by_path["validation/report_campaign.py"]["module_names"]
        assert runner.WRAPPER_MODULE in by_path["validation/reproduction_runner.py"]["module_names"]
        for relative, pin in by_path.items():
            source = ROOT / relative
            assert source.is_file()
            assert not source.is_symlink()
            assert source.stat().st_nlink == 1
            content = source.read_bytes()
            assert pin["size_bytes"] == len(content)
            assert pin["sha256"] == hashlib.sha256(content).hexdigest()

        runtime = trace["runtime"]
        assert runtime["implementation"] == sys.implementation.name
        assert runtime["cache_tag"] == sys.implementation.cache_tag
        assert runtime["version"] == {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
            "releaselevel": sys.version_info.releaselevel,
            "serial": sys.version_info.serial,
        }
        executable = runtime["executable"]
        executable_path = Path(executable["resolved_path"])
        assert executable["source"] == "/proc/self/exe"
        assert os.fspath(executable_path) == os.readlink("/proc/self/exe")
        binary = Path("/proc/self/exe").read_bytes()
        assert executable["size_bytes"] == len(binary)
        assert executable["sha256"] == hashlib.sha256(binary).hexdigest()


def test_nonzero_target_exit_is_exact_and_never_emits_trace() -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        target_arguments = ("--definitely-not-a-report-option",)
        direct = _run([sys.executable, "-m", TARGET_MODULE, *target_arguments])
        wrapped = _run(_wrapper_command(trace_path, *target_arguments))

        assert direct.returncode == 2
        assert wrapped.returncode == direct.returncode
        assert wrapped.stdout == direct.stdout
        assert wrapped.stderr == direct.stderr
        assert not trace_path.exists()


def test_identical_clean_runs_emit_byte_identical_traces() -> None:
    with _local_directory() as directory:
        first = directory / "first.json"
        second = directory / "second.json"

        first_result = _run(_wrapper_command(first, "--help"))
        second_result = _run(_wrapper_command(second, "--help"))

        assert first_result.returncode == second_result.returncode == 0
        assert first_result.stdout == second_result.stdout
        assert first_result.stderr == second_result.stderr == ""
        assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("name", runner.REJECTED_AMBIENT_ENVIRONMENT)
def test_rejects_ambient_reproducibility_environment_before_target(name: str) -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        environment = _clean_environment()
        environment[name] = ""
        result = _run(_wrapper_command(trace_path, "--help"), environment=environment)

        assert result.returncode == runner.WRAPPER_FAILURE_EXIT
        assert result.stdout == ""
        assert "ambient_environment_rejected" in result.stderr
        assert name in result.stderr
        assert not trace_path.exists()


def test_trace_conflict_is_rejected_before_target_and_not_overwritten() -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        sentinel = b"existing-owner\n"
        trace_path.write_bytes(sentinel)

        result = _run(_wrapper_command(trace_path, "--help"))

        assert result.returncode == runner.WRAPPER_FAILURE_EXIT
        assert result.stdout == ""
        assert "trace_conflict" in result.stderr
        assert trace_path.read_bytes() == sentinel


def test_trace_parent_symlink_and_outside_root_are_rejected_before_target(tmp_path: Path) -> None:
    with _local_directory() as directory:
        real_parent = directory / "real"
        real_parent.mkdir()
        linked_parent = directory / "linked"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        symlink_trace = linked_parent / "trace.json"

        linked = _run(_wrapper_command(symlink_trace, "--help"))
        outside = _run(_wrapper_command(tmp_path / "outside.json", "--help"))

        assert linked.returncode == runner.WRAPPER_FAILURE_EXIT
        assert linked.stdout == ""
        assert "local_path_symlink_rejected" in linked.stderr
        assert not (real_parent / "trace.json").exists()
        assert outside.returncode == runner.WRAPPER_FAILURE_EXIT
        assert outside.stdout == ""
        assert "path_outside_project_root" in outside.stderr
        assert not (tmp_path / "outside.json").exists()


def test_target_source_symlink_and_hardlink_are_rejected_without_execution() -> None:
    with _local_directory() as directory:
        package_name = directory.name.removeprefix(".").replace(".", "_")
        package = directory / package_name
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        source = package / "source.py"
        source.write_text("raise SystemExit('must not execute')\n", encoding="utf-8")
        linked_target = package / "linked_target.py"
        linked_target.symlink_to(source.name)
        hard_target = package / "hard_target.py"
        os.link(source, hard_target)

        sys.path.insert(0, os.fspath(directory))
        try:
            for target in (linked_target, hard_target):
                module = f"{package_name}.{target.stem}"
                trace_path = directory / f"{target.stem}.json"
                with pytest.raises(runner.ReproductionRunnerError) as raised:
                    runner.run_requested_module(
                        module,
                        (),
                        trace_path,
                        project_root=ROOT,
                        allowlisted_modules={module},
                        environment={},
                    )
                expected = (
                    "local_path_symlink_rejected"
                    if target.is_symlink()
                    else "file_hardlink_rejected"
                )
                assert raised.value.code == expected
                assert not trace_path.exists()
        finally:
            sys.path.remove(os.fspath(directory))
            sys.modules.pop(package_name, None)


def test_atomic_publish_loses_conflict_race_without_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        sentinel = b"racing-owner\n"
        real_link = runner._link_fd_noreplace

        def race(
            source_fd: int,
            target_name: str,
            *,
            target_dir_fd: int,
        ) -> None:
            trace_path.write_bytes(sentinel)
            real_link(
                source_fd,
                target_name,
                target_dir_fd=target_dir_fd,
            )

        monkeypatch.setattr(runner, "_link_fd_noreplace", race)
        with pytest.raises(runner.ReproductionRunnerError) as raised:
            runner.run_requested_module(
                TARGET_MODULE,
                ("--help",),
                trace_path,
                project_root=ROOT,
                allowlisted_modules={TARGET_MODULE},
                environment={},
            )

        assert raised.value.code == "trace_conflict"
        assert trace_path.read_bytes() == sentinel
        assert not list(directory.glob(".*.tmp"))


@pytest.mark.parametrize(
    "fault_stage",
    (
        "temporary_created",
        "content_written",
        "content_fsynced",
        "temporary_verified",
        "parent_verified",
        "parent_fsynced",
        "ready_to_link",
    ),
)
def test_every_injected_precommit_fault_removes_temporary_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
) -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"

        def inject(stage: str) -> None:
            if stage == fault_stage:
                raise RuntimeError(f"injected:{stage}")

        monkeypatch.setattr(runner, "_publication_precommit_hook", inject)
        with pytest.raises(RuntimeError, match=f"injected:{fault_stage}"):
            runner._atomic_publish_trace(trace_path, b"{}\n", project_root=ROOT)

        assert not trace_path.exists()
        assert not list(directory.glob(".*.tmp"))


def test_fd_bound_link_is_final_integrity_check_and_publication_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        events: list[str] = []
        real_stat = runner.os.stat
        real_fstat = runner.os.fstat
        real_fsync = runner.os.fsync
        real_link = runner._link_fd_noreplace

        def recording_stat(*args: object, **kwargs: object) -> os.stat_result:
            events.append("stat")
            return real_stat(*args, **kwargs)

        def recording_fstat(descriptor: int) -> os.stat_result:
            events.append("fstat")
            return real_fstat(descriptor)

        def recording_fsync(descriptor: int) -> None:
            events.append("fsync")
            real_fsync(descriptor)

        def recording_link(
            source_fd: int,
            target_name: str,
            *,
            target_dir_fd: int,
        ) -> None:
            metadata = os.fstat(source_fd)
            assert stat.S_ISREG(metadata.st_mode)
            assert metadata.st_nlink == 0
            events.append("link")
            real_link(
                source_fd,
                target_name,
                target_dir_fd=target_dir_fd,
            )

        monkeypatch.setattr(runner.os, "stat", recording_stat)
        monkeypatch.setattr(runner.os, "fstat", recording_fstat)
        monkeypatch.setattr(runner.os, "fsync", recording_fsync)
        monkeypatch.setattr(runner, "_link_fd_noreplace", recording_link)

        runner._atomic_publish_trace(trace_path, b"{}\n", project_root=ROOT)

        assert trace_path.read_bytes() == b"{}\n"
        assert events.count("link") == 1
        assert events[-1] == "link"


def test_publication_never_exposes_a_named_temporary_and_keeps_fd_open_through_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        real_link = runner._link_fd_noreplace

        def inspect_commit(
            source_fd: int,
            target_name: str,
            *,
            target_dir_fd: int,
        ) -> None:
            assert not list(directory.glob(".*.tmp"))
            metadata = os.fstat(source_fd)
            assert stat.S_ISREG(metadata.st_mode)
            assert metadata.st_nlink == 0
            real_link(source_fd, target_name, target_dir_fd=target_dir_fd)

        monkeypatch.setattr(runner, "_link_fd_noreplace", inspect_commit)
        runner._atomic_publish_trace(trace_path, b"{}\n", project_root=ROOT)

        assert trace_path.read_bytes() == b"{}\n"
        assert not list(directory.glob(".*.tmp"))


def test_runtime_pin_ignores_spoofed_sys_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = Path("/proc/self/exe").read_bytes()
    monkeypatch.setattr(runner.sys, "executable", "/attacker/spoofed-python")

    pin = runner._PythonRuntimePin()
    try:
        runtime = pin.verify_and_metadata()
    finally:
        pin.close()

    executable = runtime["executable"]
    assert executable == {
        "source": "/proc/self/exe",
        "resolved_path": os.readlink("/proc/self/exe"),
        "size_bytes": len(binary),
        "sha256": hashlib.sha256(binary).hexdigest(),
    }


def test_production_cli_accepts_exact_env_i_direct_script_with_isolation() -> None:
    result = _run(
        _production_command("--help"),
        environment=_trusted_environment(),
    )

    assert result.returncode == 0
    assert "usage: reproduction_runner.py" in result.stdout
    assert result.stderr == ""


def test_production_cli_rejects_direct_script_without_required_flags() -> None:
    result = _run(
        _production_command("--help", flags=()),
        environment=_trusted_environment(),
    )

    assert result.returncode == runner.WRAPPER_FAILURE_EXIT
    assert result.stdout == ""
    assert "trusted_launch_flags_missing" in result.stderr


def test_production_cli_rejects_module_mode() -> None:
    command = [
        "/proc/self/exe",
        "-S",
        "-m",
        "validation.reproduction_runner",
        "--help",
    ]
    result = _run(command, environment=_trusted_environment())

    assert result.returncode == runner.WRAPPER_FAILURE_EXIT
    assert result.stdout == ""
    assert "trusted_launch_not_direct_script" in result.stderr


def test_isolated_cli_rejects_pythonpath_without_loading_sitecustomize(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "sitecustomize-ran"
    (tmp_path / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({os.fspath(sentinel)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    environment = _trusted_environment()
    environment["PYTHONPATH"] = os.fspath(tmp_path)

    result = _run(_production_command("--help"), environment=environment)

    assert result.returncode == runner.WRAPPER_FAILURE_EXIT
    assert result.stdout == ""
    assert "ambient_environment_rejected" in result.stderr
    assert "PYTHONPATH" in result.stderr
    assert not sentinel.exists()


def test_production_cli_rejects_any_extra_environment_entry() -> None:
    environment = _trusted_environment()
    environment["UNEXPECTED"] = "1"

    result = _run(_production_command("--help"), environment=environment)

    assert result.returncode == runner.WRAPPER_FAILURE_EXIT
    assert result.stdout == ""
    assert "launch_environment_not_sanitized" in result.stderr
    assert "UNEXPECTED" in result.stderr


def test_bounded_trace_failure_leaves_no_artifact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        monkeypatch.setattr(runner, "MAX_TRACE_BYTES", 1)

        with pytest.raises(runner.ReproductionRunnerError) as raised:
            runner.run_requested_module(
                TARGET_MODULE,
                ("--help",),
                trace_path,
                project_root=ROOT,
                allowlisted_modules={TARGET_MODULE},
                environment={},
            )

        assert raised.value.code == "trace_size_limit_exceeded"
        assert "usage:" in capsys.readouterr().out
        assert not trace_path.exists()


def test_module_allowlist_is_enforced_before_resolution_or_execution() -> None:
    with _local_directory() as directory:
        trace_path = directory / "trace.json"
        with pytest.raises(runner.ReproductionRunnerError) as raised:
            runner.run_requested_module(
                "validation.not_a_generator",
                (),
                trace_path,
                project_root=ROOT,
                allowlisted_modules=runner.ALLOWLISTED_MODULES,
                environment={},
            )
        assert raised.value.code == "module_not_allowlisted"
        assert not trace_path.exists()
