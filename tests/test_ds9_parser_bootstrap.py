import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from validation import ds9_parser_bootstrap as bootstrap


DIGEST_HEX = "0123456789abcdef" * 4
BASE_REF = f"{bootstrap.BASE_TAG}@sha256:{DIGEST_HEX}"
IMAGE_ID = "sha256:" + "1234567890abcdef" * 4


def _root(tmp_path: Path) -> Path:
    for relative in (
        bootstrap.DOCKERFILE,
        bootstrap.DOCKERIGNORE,
        bootstrap.SOURCE_PATCH,
        bootstrap.CONTROLLER,
        bootstrap.RUNTIME_CONTROLLER,
        bootstrap.CONTROL_MANIFEST,
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == bootstrap.SOURCE_PATCH:
            path.write_bytes((bootstrap.PROJECT_ROOT / relative).read_bytes())
        else:
            path.write_text(f"fixture={relative}\n", encoding="utf-8")
    return tmp_path


def _thermal(
    value: float = 60.0,
    *,
    gpu_index: int = 0,
    gpu_name: str = "Fixture GPU",
    slowdown: bool = False,
):
    slowdown_flags = {
        name: slowdown if name == "clock_event_sw_thermal_slowdown" else False
        for name in bootstrap.BUILD_SLOWDOWN_FLAG_NAMES
    }
    return {
        "sampled_at_utc": "2026-07-16T10:00:00Z",
        "temperatures_c": {
            "cpu_package_c": value,
            f"gpu_{gpu_index}_c": value,
        },
        "max_temperature_c": value,
        "source_manifest": {
            "available": True,
            "columns": ["timestamp", "cpu_package_c"],
        },
        "gpu_telemetry": {
            "sample_timestamp": "2026/07/16 10:00:00.000",
            "gpu_index": gpu_index,
            "gpu_name": gpu_name,
            "temperature_c": value,
            "power_draw_w": 40.0,
            "power_limits_w": {
                "requested": 115.0,
                "current": 115.0,
                "default": 115.0,
            },
            "power_limit_telemetry_complete": True,
            "pstate": "P8",
            "clock_event_reasons_active_mask": "0x0000000000000000",
            "clock_event_sw_power_cap": False,
            "slowdown_flags": slowdown_flags,
            "dangerous_slowdown_active": slowdown,
        },
    }


def _source():
    return bootstrap.source_lineage_contract()


def _discovery_runner(root: Path, parser_bytes: bytes = b"real parser elf\x00"):
    def run(command, **kwargs):
        destination = next(
            value.split("dest=", 1)[1]
            for value in command
            if value.startswith("type=local,dest=")
        )
        export = root / destination / "parser"
        export.mkdir(parents=True, exist_ok=True)
        (export / "libnvdsinfer_custom_impl_Yolo.so").write_bytes(parser_bytes)
        digest = hashlib.sha256(parser_bytes).hexdigest()
        (export / "parser.sha256").write_text(digest + "\n", encoding="ascii")
        (export / "source-lineage.json").write_text(
            json.dumps(_source(), sort_keys=True) + "\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "audited build log\n", "")

    return run


def _make_discovery(root: Path, nonce: str = "a" * 64):
    session = root / f"validation/results/ds9-runtime-compatibility/parser/{nonce}"
    receipt = bootstrap.execute_discovery(
        base_ref=BASE_REF,
        image_tag="deepsafe-deepstream:9.0",
        session_root=session,
        project_root=root,
        runner=_discovery_runner(root),
        thermal_probe=lambda: _thermal(),
    )
    path = session / "pass-1-discovery/discovery-receipt.json"
    return session, path, receipt


def test_discovery_plan_has_no_expected_parser_sha_and_no_subprocess(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry plan must not call Docker"),
    )
    monkeypatch.setattr(
        bootstrap.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("dry plan must not launch Docker"),
    )
    plan = bootstrap.make_plan(
        base_ref=BASE_REF,
        image_tag="deepsafe-deepstream:9.0",
        session_root=root / "validation/results/ds9-runtime-compatibility/parser/plan",
        project_root=root,
    )
    command = plan["pass_1"]["command"]
    assert plan["pass_1"]["accepts_expected_parser_sha"] is False
    assert not any("PARSER_SHA256" in value for value in command)
    assert plan["inputs"]["dockerignore"]["path"] == ".dockerignore"
    assert plan["inputs"]["source_patch"]["sha256"] == hashlib.sha256(
        (root / bootstrap.SOURCE_PATCH).read_bytes()
    ).hexdigest()
    assert plan["docker_called"] is False


def test_checked_in_kernel_patch_is_exact_and_instruments_immediate_launch_site():
    patch_path = bootstrap.PROJECT_ROOT / bootstrap.SOURCE_PATCH
    content = patch_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == bootstrap.SOURCE_PATCH_SHA256
    text = content.decode("utf-8")
    launch_tail = text.index("perClassPreclusterThreshold.data()));")
    proof = text.index("recordCudaKernelProof", launch_tail)
    copy = text.index("thrust::copy(objects.begin()", proof)
    assert launch_tail < proof < copy
    assert text[launch_tail:proof] == (
        "perClassPreclusterThreshold.data()));\n \n+  if (!"
    )
    calls = [
        text.index("cudaGetLastError()"),
        text.index("cudaDeviceSynchronize()"),
        text.index("cudaFuncGetAttributes(&attributes, decodeTensorYoloCuda)"),
    ]
    assert calls == sorted(calls)
    for token in ("O_CREAT", "O_EXCL", "O_NOFOLLOW", "fchmod(descriptor, 0440)"):
        assert token in text


def test_sm86_make_override_is_exact_and_does_not_rewrite_upstream_makefile():
    source = bootstrap.source_lineage_contract()
    assert source["schema_version"] == "deepsafe.ds9-parser-source-lineage/v3"
    assert source["build_makefile_path"] == bootstrap.PARSER_BUILD_MAKEFILE_PATH
    assert source["build_makefile_sha256"] == bootstrap.PARSER_BUILD_MAKEFILE_SHA256
    assert source["build_makefile_modified"] is False
    assert source["cuda_cubin_architecture"] == "sm_86"
    assert source["cuda_ptx_architecture"] == "compute_86"
    assert source["cuda_gencode_flags"] == [
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_86,code=compute_86",
    ]
    assert hashlib.sha256(source["build_command"].encode("utf-8")).hexdigest() == (
        source["build_command_sha256"]
    )
    assert " CUDA_VER=13.1 " in source["build_command"]
    assert "env CUDA_VER" not in source["build_command"]
    assert source["build_command"].count("-gencode=") == 2
    assert source["post_link_tool_path"] == bootstrap.PARSER_POST_LINK_TOOL_PATH
    assert source["post_link_tool_sha256"] == bootstrap.PARSER_POST_LINK_TOOL_SHA256
    assert source["post_link_tool_version"] == bootstrap.PARSER_POST_LINK_TOOL_VERSION
    assert source["post_link_command"] == bootstrap.PARSER_POST_LINK_COMMAND
    assert hashlib.sha256(source["post_link_command"].encode("utf-8")).hexdigest() == (
        source["post_link_command_sha256"]
    )
    assert source["post_link_removed_sections"] == [".symtab", ".strtab"]
    assert source["post_link_retained_sections"] == [".dynsym", ".dynstr"]

    dockerfile = (bootstrap.PROJECT_ROOT / bootstrap.DOCKERFILE).read_text(
        encoding="utf-8"
    )
    assert (
        f'sha256sum {bootstrap.PARSER_BUILD_MAKEFILE_PATH} | cut -d\' \' -f1)" '
        f'= "{bootstrap.PARSER_BUILD_MAKEFILE_SHA256}"'
    ) in dockerfile
    assert "git add nvdsinfer_custom_impl_Yolo/Makefile" not in dockerfile
    assert "make -C nvdsinfer_custom_impl_Yolo -j2 CUDA_VER=13.1" in dockerfile
    assert dockerfile.count("-gencode=arch=compute_86,code=sm_86") >= 4
    assert dockerfile.count("-gencode=arch=compute_86,code=compute_86") >= 4
    assert (
        f'sha256sum {bootstrap.PARSER_POST_LINK_TOOL_PATH} | cut -d\' \' -f1)" '
        f'= "{bootstrap.PARSER_POST_LINK_TOOL_SHA256}"'
    ) in dockerfile
    assert bootstrap.PARSER_POST_LINK_TOOL_VERSION in dockerfile
    assert bootstrap.PARSER_POST_LINK_COMMAND in dockerfile
    strip_index = dockerfile.index(bootstrap.PARSER_POST_LINK_COMMAND)
    parser_digest_index = dockerfile.index(
        "sha256sum nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
    )
    assert strip_index < parser_digest_index
    for section in (*bootstrap.PARSER_POST_LINK_REMOVED_SECTIONS, *bootstrap.PARSER_POST_LINK_RETAINED_SECTIONS):
        assert section in dockerfile


@pytest.mark.parametrize(
    "value",
    [
        f"{bootstrap.BASE_TAG}@sha256:" + "0" * 64,
        f"{bootstrap.BASE_TAG}@sha256:" + "f" * 64,
        bootstrap.BASE_TAG,
        f"{bootstrap.BASE_TAG}@sha256:<registry-digest>",
    ],
)
def test_placeholder_or_tag_only_base_is_rejected(tmp_path, value):
    root = _root(tmp_path)
    with pytest.raises(bootstrap.ParserBootstrapError):
        bootstrap.make_plan(
            base_ref=value,
            image_tag="deepsafe-deepstream:9.0",
            session_root=root / "validation/results/parser/test",
            project_root=root,
        )


def test_default_snapshot_records_read_only_gpu_power_and_slowdown(monkeypatch):
    from validation.scene_benchmark import run_matrix

    manifest = {
        "available": True,
        "columns": ["timestamp", "cpu_package_c"],
    }
    gpu = {name: "0" for name in run_matrix.GPU_CSV_HEADER}
    gpu.update(
        {
            "timestamp": "2026/07/16 10:00:00.000",
            "gpu_index": "0",
            "gpu_name": "Fixture GPU",
            "temperature_c": "64",
            "power_draw_w": "43.5",
            "power_requested_limit_w": "115",
            "power_current_limit_w": "110",
            "power_default_limit_w": "115",
            "pstate": "P8",
            "clock_event_reasons_active_mask": "0x0000000000000040",
            "clock_event_sw_power_cap": "Not Active",
            "clock_event_sw_thermal_slowdown": "Active",
            "clock_event_hw_slowdown": "Not Active",
            "clock_event_hw_thermal_slowdown": "Not Active",
            "clock_event_hw_power_brake_slowdown": "Not Active",
        }
    )
    monkeypatch.setattr(
        run_matrix, "discover_platform_thermal_sources", lambda: manifest
    )
    monkeypatch.setattr(
        run_matrix,
        "read_platform_thermal_row",
        lambda observed, sampled: ([sampled, "55"], []),
    )
    monkeypatch.setattr(
        run_matrix,
        "query_gpu_row",
        lambda index: [gpu[name] for name in run_matrix.GPU_CSV_HEADER],
    )

    snapshot = bootstrap.platform_thermal_snapshot()
    assert snapshot["temperatures_c"] == {
        "cpu_package_c": 55.0,
        "gpu_0_c": 64.0,
    }
    assert snapshot["max_temperature_c"] == 64.0
    assert snapshot["gpu_telemetry"]["power_draw_w"] == 43.5
    assert snapshot["gpu_telemetry"]["power_limits_w"] == {
        "requested": 115.0,
        "current": 110.0,
        "default": 115.0,
    }
    assert snapshot["gpu_telemetry"]["dangerous_slowdown_active"] is True


def test_high_temperature_is_recorded_but_does_not_block_runner(tmp_path):
    root = _root(tmp_path)
    called = False

    def runner(*args, **kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess(args[0], 0, "build complete\n", "")

    report = bootstrap._run_logged(
        ["docker", "build"],
        log_path=root / "log.txt",
        thermal_path=root / "thermal.json",
        project_root=root,
        runner=runner,
        thermal_probe=lambda: _thermal(95.0, slowdown=True),
    )
    assert called is True
    assert report["status"] == "complete"
    assert report["max_observed_temperature_c"] == 95.0
    assert report["policy"]["id"] == "workstation_managed"
    assert (
        report["policy"]["temperature_threshold_enforcement"]
        == "informational_only"
    )
    assert report["samples"][0]["gpu_telemetry"]["dangerous_slowdown_active"] is True
    assert "preflight_must_be_below_c" not in report["policy"]
    assert "runtime_abort_at_or_above_c" not in report["policy"]
    replayed = bootstrap._validate_thermal_report(
        root / "thermal.json", project_root=root
    )
    assert replayed["max_observed_temperature_c"] == 95.0


def test_malformed_preflight_telemetry_blocks_before_runner(tmp_path):
    root = _root(tmp_path)
    called = False
    malformed = _thermal()
    malformed["gpu_telemetry"]["slowdown_flags"].pop(
        "clock_event_hw_slowdown"
    )

    def runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not launch")

    with pytest.raises(bootstrap.ParserBootstrapError, match="unavailable or malformed"):
        bootstrap._run_logged(
            ["docker", "build"],
            log_path=root / "log.txt",
            thermal_path=root / "thermal.json",
            project_root=root,
            runner=runner,
            thermal_probe=lambda: malformed,
        )
    assert called is False
    report = json.loads((root / "thermal.json").read_text())
    assert report["status"] == "telemetry_unavailable_before_start"
    assert report["samples"] == []


def test_high_runtime_temperature_and_slowdown_do_not_abort_process(tmp_path, monkeypatch):
    root = _root(tmp_path)
    samples = iter(
        [
            _thermal(60.0),
            _thermal(95.0, slowdown=True),
            _thermal(97.0, slowdown=True),
            _thermal(99.0, slowdown=True),
        ]
    )
    signals = []

    class Process:
        pid = 4242
        returncode = None
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls >= 3:
                self.returncode = 0
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    process = Process()
    monkeypatch.setattr(bootstrap.os, "getpgid", lambda pid: pid)

    def killpg(pid, selected):
        signals.append((pid, selected))
        process.returncode = -selected

    monkeypatch.setattr(bootstrap.os, "killpg", killpg)
    report = bootstrap._run_logged(
        ["docker", "build"],
        log_path=root / "log.txt",
        thermal_path=root / "thermal.json",
        project_root=root,
        runner=None,
        thermal_probe=lambda: next(samples),
        popen_factory=lambda *args, **kwargs: process,
        sleeper=lambda seconds: None,
    )
    assert signals == []
    assert report["status"] == "complete"
    assert report["max_observed_temperature_c"] == 99.0
    assert len(report["samples"]) == 4


def test_live_telemetry_loss_still_aborts_build_process_group(tmp_path, monkeypatch):
    root = _root(tmp_path)
    probe_calls = 0
    signals = []

    def thermal_probe():
        nonlocal probe_calls
        probe_calls += 1
        if probe_calls == 1:
            return _thermal(60.0)
        raise RuntimeError("nvidia-smi telemetry lost")

    class Process:
        pid = 4242
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    process = Process()
    monkeypatch.setattr(bootstrap.os, "getpgid", lambda pid: pid)

    def killpg(pid, selected):
        signals.append((pid, selected))
        process.returncode = -selected

    monkeypatch.setattr(bootstrap.os, "killpg", killpg)
    with pytest.raises(bootstrap.ParserBootstrapError, match="telemetry loss"):
        bootstrap._run_logged(
            ["docker", "build"],
            log_path=root / "log.txt",
            thermal_path=root / "thermal.json",
            project_root=root,
            runner=None,
            thermal_probe=thermal_probe,
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda seconds: None,
        )
    assert signals == [(4242, signal.SIGTERM)]
    report = json.loads((root / "thermal.json").read_text())
    assert report["status"] == "telemetry_abort"
    assert "nvidia-smi telemetry lost" in report["abort_reason"]


def test_gpu_identity_change_is_rejected_after_build(tmp_path):
    root = _root(tmp_path)
    samples = iter([_thermal(), _thermal(gpu_name="Different GPU")])

    with pytest.raises(bootstrap.ParserBootstrapError, match="malformed evidence"):
        bootstrap._run_logged(
            ["docker", "build"],
            log_path=root / "log.txt",
            thermal_path=root / "thermal.json",
            project_root=root,
            runner=lambda command, **kwargs: subprocess.CompletedProcess(
                command, 0, "build complete\n", ""
            ),
            thermal_probe=lambda: next(samples),
        )
    report = json.loads((root / "thermal.json").read_text())
    assert report["status"] == "telemetry_abort"
    assert "GPU identity changed" in report["abort_reason"]


def test_discovery_measures_binary_and_receipt_replays(tmp_path):
    root = _root(tmp_path)
    _session, path, receipt = _make_discovery(root)
    assert receipt["expected_parser_sha_input"] is None
    assert receipt["measured_parser_sha256"] == receipt["artifacts"]["parser"]["sha256"]
    assert stat_mode(path) == 0o440
    replayed, pin = bootstrap.validate_discovery_receipt(path, project_root=root)
    assert replayed == receipt
    assert pin["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert replayed["thermal_policy"]["docker_gpu_device_request"] is False


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


def test_discovery_artifact_tamper_is_rejected(tmp_path):
    root = _root(tmp_path)
    _session, path, receipt = _make_discovery(root)
    parser = root / receipt["artifacts"]["parser"]["path"]
    parser.write_bytes(parser.read_bytes() + b"tamper")
    with pytest.raises(bootstrap.ParserBootstrapError, match="artifact changed"):
        bootstrap.validate_discovery_receipt(path, project_root=root)


def test_discovery_replays_parser_digest_content_not_only_its_pin(tmp_path):
    root = _root(tmp_path)
    _session, path, receipt = _make_discovery(root)
    digest_path = root / receipt["artifacts"]["parser_digest"]["path"]
    digest_path.write_text("9" * 64 + "\n", encoding="ascii")
    receipt["artifacts"]["parser_digest"] = bootstrap.file_pin(
        digest_path, project_root=root
    )
    os.chmod(path, 0o600)
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o440)
    with pytest.raises(bootstrap.ParserBootstrapError, match="digest content differs"):
        bootstrap.validate_discovery_receipt(path, project_root=root)


def test_production_pass_uses_measured_sha_and_raw_image_id(tmp_path):
    root = _root(tmp_path)
    session, discovery_path, discovery = _make_discovery(root)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "build"]:
            iid = root / command[command.index("--iidfile") + 1]
            iid.parent.mkdir(parents=True, exist_ok=True)
            iid.write_text(IMAGE_ID + "\n", encoding="ascii")
            return subprocess.CompletedProcess(command, 0, "pass two log\n", "")
        pins = bootstrap._input_pins(root)
        labels = {
            "com.deepsafe.build-lineage.schema": bootstrap.BUILD_LINEAGE_SCHEMA_VERSION,
            "com.deepsafe.deepstream-yolo.parser-sha256": discovery[
                "measured_parser_sha256"
            ],
            "com.deepsafe.deepstream.base-ref": BASE_REF,
            "com.deepsafe.deepstream.base-digest": "sha256:" + DIGEST_HEX,
            "com.deepsafe.deepstream-yolo.patch-sha256": bootstrap.SOURCE_PATCH_SHA256,
            "com.deepsafe.deepstream-yolo.upstream-source-sha256": bootstrap.UPSTREAM_SOURCE_SHA256,
            "com.deepsafe.deepstream-yolo.patched-source-sha256": bootstrap.PATCHED_SOURCE_SHA256,
            "com.deepsafe.deepstream-yolo.patched-tree": bootstrap.PATCHED_SOURCE_TREE,
            "com.deepsafe.deepstream-yolo.parser-build-makefile-sha256": bootstrap.PARSER_BUILD_MAKEFILE_SHA256,
            "com.deepsafe.deepstream-yolo.parser-cuda-cubin-architecture": bootstrap.PARSER_CUDA_CUBIN_ARCHITECTURE,
            "com.deepsafe.deepstream-yolo.parser-cuda-ptx-architecture": bootstrap.PARSER_CUDA_PTX_ARCHITECTURE,
            "com.deepsafe.deepstream-yolo.parser-cuda-gencode-flags": ";".join(
                bootstrap.PARSER_CUDA_GENCODE_FLAGS
            ),
            "com.deepsafe.deepstream-yolo.parser-build-command-sha256": bootstrap.PARSER_BUILD_COMMAND_SHA256,
            "com.deepsafe.deepstream-yolo.parser-post-link-tool-path": bootstrap.PARSER_POST_LINK_TOOL_PATH,
            "com.deepsafe.deepstream-yolo.parser-post-link-tool-sha256": bootstrap.PARSER_POST_LINK_TOOL_SHA256,
            "com.deepsafe.deepstream-yolo.parser-post-link-tool-version": bootstrap.PARSER_POST_LINK_TOOL_VERSION,
            "com.deepsafe.deepstream-yolo.parser-post-link-command": bootstrap.PARSER_POST_LINK_COMMAND,
            "com.deepsafe.deepstream-yolo.parser-post-link-command-sha256": bootstrap.PARSER_POST_LINK_COMMAND_SHA256,
            "com.deepsafe.deepstream-yolo.parser-post-link-removed-sections": ";".join(
                bootstrap.PARSER_POST_LINK_REMOVED_SECTIONS
            ),
            "com.deepsafe.deepstream-yolo.parser-post-link-retained-sections": ";".join(
                bootstrap.PARSER_POST_LINK_RETAINED_SECTIONS
            ),
            "com.deepsafe.cuda-kernel-proof.schema": bootstrap.KERNEL_PROOF_SCHEMA_VERSION,
            "com.deepsafe.dockerignore.sha256": pins["dockerignore"]["sha256"],
            "com.deepsafe.runtime-compatibility-controller.sha256": pins[
                "runtime_controller"
            ]["sha256"],
            "com.deepsafe.runtime-control-manifest.sha256": pins[
                "runtime_control_manifest"
            ]["sha256"],
        }
        value = {
            "Id": IMAGE_ID,
            "Architecture": "amd64",
            "Os": "linux",
            "RepoDigests": [],
            "Config": {"Labels": labels},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps([value]), "")

    receipt = bootstrap.execute_production(
        discovery_receipt=discovery_path,
        image_tag="deepsafe-deepstream:9.0",
        project_root=root,
        runner=runner,
        thermal_probe=lambda: _thermal(),
    )
    parser_arg = (
        "DEEPSTREAM_YOLO_PARSER_SHA256=" + discovery["measured_parser_sha256"]
    )
    assert parser_arg in receipt["command"]
    assert (
        "DEEPSTREAM_YOLO_PATCH_SHA256=" + bootstrap.SOURCE_PATCH_SHA256
    ) in receipt["command"]
    assert receipt["parser_sha_source"] == receipt["discovery_receipt"]
    assert receipt["resolved_image_id"] == IMAGE_ID
    receipt_path = session / "pass-2-production/production-build-receipt.json"
    replayed, _ = bootstrap.validate_production_receipt(
        receipt_path,
        project_root=root,
        resolved_image_id=IMAGE_ID,
        parser_sha256=discovery["measured_parser_sha256"],
    )
    assert replayed == receipt
    assert len(calls) == 2


def test_production_receipt_must_equal_discovery_lineage(tmp_path):
    root = _root(tmp_path)
    session, discovery_path, discovery = _make_discovery(root)

    def runner(command, **kwargs):
        if command[:2] == ["docker", "build"]:
            iid = root / command[command.index("--iidfile") + 1]
            iid.parent.mkdir(parents=True, exist_ok=True)
            iid.write_text(IMAGE_ID + "\n", encoding="ascii")
            return subprocess.CompletedProcess(command, 0, "build\n", "")
        pins = bootstrap._input_pins(root)
        labels = {
            "com.deepsafe.build-lineage.schema": bootstrap.BUILD_LINEAGE_SCHEMA_VERSION,
            "com.deepsafe.deepstream-yolo.parser-sha256": discovery["measured_parser_sha256"],
            "com.deepsafe.deepstream.base-ref": BASE_REF,
            "com.deepsafe.deepstream.base-digest": "sha256:" + DIGEST_HEX,
            "com.deepsafe.deepstream-yolo.patch-sha256": bootstrap.SOURCE_PATCH_SHA256,
            "com.deepsafe.deepstream-yolo.upstream-source-sha256": bootstrap.UPSTREAM_SOURCE_SHA256,
            "com.deepsafe.deepstream-yolo.patched-source-sha256": bootstrap.PATCHED_SOURCE_SHA256,
            "com.deepsafe.deepstream-yolo.patched-tree": bootstrap.PATCHED_SOURCE_TREE,
            "com.deepsafe.deepstream-yolo.parser-build-makefile-sha256": bootstrap.PARSER_BUILD_MAKEFILE_SHA256,
            "com.deepsafe.deepstream-yolo.parser-cuda-cubin-architecture": bootstrap.PARSER_CUDA_CUBIN_ARCHITECTURE,
            "com.deepsafe.deepstream-yolo.parser-cuda-ptx-architecture": bootstrap.PARSER_CUDA_PTX_ARCHITECTURE,
            "com.deepsafe.deepstream-yolo.parser-cuda-gencode-flags": ";".join(
                bootstrap.PARSER_CUDA_GENCODE_FLAGS
            ),
            "com.deepsafe.deepstream-yolo.parser-build-command-sha256": bootstrap.PARSER_BUILD_COMMAND_SHA256,
            "com.deepsafe.deepstream-yolo.parser-post-link-tool-path": bootstrap.PARSER_POST_LINK_TOOL_PATH,
            "com.deepsafe.deepstream-yolo.parser-post-link-tool-sha256": bootstrap.PARSER_POST_LINK_TOOL_SHA256,
            "com.deepsafe.deepstream-yolo.parser-post-link-tool-version": bootstrap.PARSER_POST_LINK_TOOL_VERSION,
            "com.deepsafe.deepstream-yolo.parser-post-link-command": bootstrap.PARSER_POST_LINK_COMMAND,
            "com.deepsafe.deepstream-yolo.parser-post-link-command-sha256": bootstrap.PARSER_POST_LINK_COMMAND_SHA256,
            "com.deepsafe.deepstream-yolo.parser-post-link-removed-sections": ";".join(
                bootstrap.PARSER_POST_LINK_REMOVED_SECTIONS
            ),
            "com.deepsafe.deepstream-yolo.parser-post-link-retained-sections": ";".join(
                bootstrap.PARSER_POST_LINK_RETAINED_SECTIONS
            ),
            "com.deepsafe.cuda-kernel-proof.schema": bootstrap.KERNEL_PROOF_SCHEMA_VERSION,
            "com.deepsafe.dockerignore.sha256": pins["dockerignore"]["sha256"],
            "com.deepsafe.runtime-compatibility-controller.sha256": pins["runtime_controller"]["sha256"],
            "com.deepsafe.runtime-control-manifest.sha256": pins["runtime_control_manifest"]["sha256"],
        }
        image = {
            "Id": IMAGE_ID,
            "Architecture": "amd64",
            "Os": "linux",
            "RepoDigests": [],
            "Config": {"Labels": labels},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps([image]), "")

    bootstrap.execute_production(
        discovery_receipt=discovery_path,
        image_tag="deepsafe-deepstream:9.0",
        project_root=root,
        runner=runner,
        thermal_probe=lambda: _thermal(),
    )
    receipt_path = session / "pass-2-production/production-build-receipt.json"
    payload = json.loads(receipt_path.read_text())
    payload["source"] = {**payload["source"], "instrumentation_schema": "forged"}
    os.chmod(receipt_path, 0o600)
    receipt_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.chmod(receipt_path, 0o440)
    with pytest.raises(bootstrap.ParserBootstrapError, match="lineage differs: source"):
        bootstrap.validate_production_receipt(receipt_path, project_root=root)


def test_bootstrap_receipts_validate_published_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    root = _root(tmp_path)
    _session, _path, discovery = _make_discovery(root)
    schema = json.loads(
        (
            bootstrap.PROJECT_ROOT
            / "validation/schemas/ds9-parser-bootstrap-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(discovery)
