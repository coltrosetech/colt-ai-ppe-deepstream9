from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import jsonschema
import pytest

from validation import driver595_maintenance_r4 as r4


def _action_fixture(
    *,
    replace: tuple[str, str] | None = None,
    extra: tuple[str, ...] = (),
    origin: str = "Ubuntu:24.04/noble-updates, Ubuntu:24.04/noble-security",
) -> str:
    lines = ["NOTE: This is only a simulation!"]
    for package, (version, architecture) in sorted(r4.EXPECTED_REMOVE_VERSIONS.items()):
        lines.append(f"Remv {package} [{version}]")
    for package, (version, architecture) in sorted(r4.EXPECTED_INSTALL_VERSIONS.items()):
        lines.append(f"Inst {package} ({version} {origin} [{architecture}])")
    for package, (version, architecture) in sorted(r4.EXPECTED_CONFIGURE_VERSIONS.items()):
        lines.append(f"Conf {package} ({version} {origin} [{architecture}])")
    lines.extend(extra)
    lines.append(
        f"0 upgraded, {len(r4.EXPECTED_INSTALL_VERSIONS)} newly installed, "
        f"{len(r4.EXPECTED_REMOVE_VERSIONS)} to remove and 147 not upgraded."
    )
    output = "\n".join(lines)
    if replace:
        output = output.replace(*replace)
    return output


def _pin(path: Path) -> dict[str, object]:
    return r4.file_pin(path)


def _static_plan(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("validator", "schema", "documentation"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = _pin(path)
    state_basis = {"fixture": "static-plan"}
    state_id = r4.sha256_bytes(r4.canonical_json(state_basis))[:20]
    cache_dir = f"/var/cache/deepsafe-driver595-r4/{state_id}/archives"
    plan: dict[str, object] = {
        "schema_version": r4.SCHEMA_VERSION,
        "plan_id": f"driver595-r4-{state_id}",
        "status": "prepared_read_only_pending_independent_review_cache_and_authorization",
        "authority": {
            "this_plan_executes_work": False,
            "downloads_performed": False,
            "install_enabled": False,
            "external_authorization_required": True,
            "external_authorization_receipt": None,
            "independent_review_required": True,
        },
        "target": {
            "driver_version": r4.TARGET_DRIVER,
            "driver_deb_version": r4.TARGET_DRIVER_DEB,
            "target_kernel": r4.TARGET_KERNEL,
            "target_kernel_deb_version": r4.TARGET_KERNEL_DEB,
            "fallback_kernel": r4.FALLBACK_KERNEL,
            "fallback_kernel_deb_version": r4.FALLBACK_KERNEL_DEB,
            "gpu_name": r4.TARGET_GPU,
            "architecture": r4.ARCH,
        },
        "transaction_contract": {
            "install": [
                {"package": p, "version": v, "architecture": a}
                for p, (v, a) in sorted(r4.EXPECTED_INSTALL_VERSIONS.items())
            ],
            "remove": [
                {"package": p, "version": v, "architecture": a}
                for p, (v, a) in sorted(r4.EXPECTED_REMOVE_VERSIONS.items())
            ],
            "configure": [
                {"package": p, "version": v, "architecture": a}
                for p, (v, a) in sorted(r4.EXPECTED_CONFIGURE_VERSIONS.items())
            ],
            "forbidden_installed_packages": ["nvidia-dkms-595"],
            "autoremove_allowed": False,
            "package_download_allowed_before_external_authorization": False,
        },
        "future_commands": {
            "simulation_argv": r4.apt_simulation_argv(),
            "cache_dir": cache_dir,
            "cache_download_argv_after_authorization_only": r4.future_cache_download_argv(cache_dir),
            "install_argv_after_authorization_only": r4.future_install_argv(cache_dir),
        },
        "initial_state": {"state_basis": state_basis},
        "source_pins": files,
        "toolchain": r4.collect_toolchain(),
    }
    plan["plan_fingerprint_sha256"] = r4.plan_fingerprint(plan)
    return plan


def _pass_status() -> dict[str, str]:
    return {"status": "pass"}


def test_contract_counts_and_cli_are_execution_closed() -> None:
    assert len(r4.EXPECTED_INSTALL_VERSIONS) == 22
    assert len(r4.EXPECTED_REMOVE_VERSIONS) == 16
    assert len(r4.EXPECTED_CONFIGURE_VERSIONS) == 26
    assert "apply" not in r4.CLI_MODES
    assert "install" not in r4.CLI_MODES
    assert "download" not in r4.CLI_MODES
    assert all(Path(path).is_absolute() for path in r4.BIN.values())
    assert "LD_LIBRARY_PATH" not in r4.SAFE_ENV
    assert "APT_CONFIG" not in r4.SAFE_ENV


@pytest.mark.parametrize(
    "argv",
    [
        [r4.BIN["apt_get"], "install", "nvidia-driver-595"],
        [r4.BIN["apt_get"], "-s", "--download-only", "install", "x"],
        [r4.BIN["dpkg"], "--configure", "-a"],
        [r4.BIN["grub_editenv"], "/boot/grub/grubenv", "set", "next_entry=x"],
        [r4.BIN["mokutil"], "--import", "/tmp/key.der"],
        [r4.BIN["dkms"], "remove", "nvidia/595", "--all"],
        [r4.BIN["nvidia_smi"], "--gpu-reset"],
        [r4.BIN["dpkg_deb"], "--extract", "/tmp/x.deb", "/tmp/out", "--field", "Package"],
        ["apt-get", "-s", "install", "x"],
    ],
)
def test_read_only_command_guard_rejects_mutation_or_relative_path(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        r4._validate_read_only_argv(argv)


def test_run_uses_minimal_environment_timeout_and_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_run(argv, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "ok")

    monkeypatch.setattr(r4.subprocess, "run", fake_run)
    result = r4._run([r4.BIN["uname"], "-r"], timeout=7)
    assert result.returncode == 0
    assert observed["env"] == r4.SAFE_ENV
    assert observed["cwd"] == "/"
    assert observed["timeout"] == 7
    assert observed["stdin"] is subprocess.DEVNULL


def test_exact_inst_remove_conf_multiset_passes() -> None:
    report = r4.parse_apt_actions(_action_fixture())
    assert report["status"] == "pass"
    assert len(report["actions"]) == 64
    assert report["missing_actions"] == []
    assert report["unexpected_actions"] == []


def test_wrong_version_is_blocked_even_when_names_match() -> None:
    report = r4.parse_apt_actions(
        _action_fixture(replace=(r4.TARGET_DRIVER_DEB, "0.0.0-attacker"))
    )
    assert report["status"] == "blocked"
    assert report["checks"]["exact_action_projection"] is False


def test_unexpected_conf_is_blocked() -> None:
    report = r4.parse_apt_actions(
        _action_fixture(
            extra=(
                "Conf unrelated-critical-package "
                "(999 Ubuntu:24.04/noble-updates [amd64])",
            )
        )
    )
    assert report["status"] == "blocked"
    assert ["Conf", "unrelated-critical-package"] in report["unexpected_actions"]


def test_missing_conf_is_blocked() -> None:
    package = "linux-image-7.0.0-28-generic"
    version, architecture = r4.EXPECTED_CONFIGURE_VERSIONS[package]
    line = (
        f"Conf {package} ({version} Ubuntu:24.04/noble-updates, "
        f"Ubuntu:24.04/noble-security [{architecture}])\n"
    )
    report = r4.parse_apt_actions(_action_fixture().replace(line, ""))
    assert report["status"] == "blocked"
    assert ["Conf", package] in report["missing_actions"]


def test_untrusted_or_empty_origin_is_blocked() -> None:
    report = r4.parse_apt_actions(_action_fixture(origin="Evil:stable/repo"))
    assert report["status"] == "blocked"
    assert report["checks"]["trusted_nonempty_origins"] is False


def test_duplicate_action_is_blocked_by_multiset_count() -> None:
    package, (version, architecture) = next(iter(r4.EXPECTED_INSTALL_VERSIONS.items()))
    duplicate = (
        f"Inst {package} ({version} Ubuntu:24.04/noble-updates [{architecture}])"
    )
    report = r4.parse_apt_actions(_action_fixture(extra=(duplicate,)))
    assert report["status"] == "blocked"
    assert report["checks"]["exact_action_count"] is False


@pytest.mark.parametrize("status", ["ii ", "hi ", "ri ", "pi "])
def test_current_state_character_detects_every_installed_variant(status: str) -> None:
    assert r4.dpkg_status_is_installed(status) is True


@pytest.mark.parametrize("status", [None, "un ", "pn ", "rc "])
def test_noninstalled_states_are_not_misclassified(status: str | None) -> None:
    assert r4.dpkg_status_is_installed(status) is False


def test_held_installed_old_driver_blocks_initial_projection() -> None:
    entries = [
        {"package": package, "status": "ii ", "version": version, "architecture": arch}
        for package, (version, arch) in r4.EXPECTED_REMOVE_VERSIONS.items()
    ]
    entries[0]["status"] = "hi "
    entries.extend(
        {
            "package": package,
            "status": status,
            "version": r4.EXPECTED_CONFIGURE_VERSIONS[package][0],
            "architecture": r4.EXPECTED_CONFIGURE_VERSIONS[package][1],
        }
        for package, status in r4.EXPECTED_INITIAL_INCOMPLETE.items()
    )
    projection = {
        "status": "pass",
        "entries": entries,
        "incomplete": [
            entry for entry in entries if entry["package"] in r4.EXPECTED_INITIAL_INCOMPLETE
        ],
    }
    report = r4.validate_initial_dpkg_projection(projection)
    assert report["status"] == "blocked"
    assert report["checks"]["old_590_exact_installed"] is False


def test_held_installed_595_dkms_blocks_initial_projection() -> None:
    entries = [
        {"package": package, "status": "ii ", "version": version, "architecture": arch}
        for package, (version, arch) in r4.EXPECTED_REMOVE_VERSIONS.items()
    ]
    entries.extend(
        {
            "package": package,
            "status": status,
            "version": r4.EXPECTED_CONFIGURE_VERSIONS[package][0],
            "architecture": r4.EXPECTED_CONFIGURE_VERSIONS[package][1],
        }
        for package, status in r4.EXPECTED_INITIAL_INCOMPLETE.items()
    )
    entries.append(
        {
            "package": "nvidia-dkms-595",
            "status": "hi ",
            "version": r4.TARGET_DRIVER_DEB,
            "architecture": "amd64",
        }
    )
    projection = {
        "status": "pass",
        "entries": entries,
        "incomplete": [
            entry for entry in entries if entry["package"] in r4.EXPECTED_INITIAL_INCOMPLETE
        ],
    }
    report = r4.validate_initial_dpkg_projection(projection)
    assert report["status"] == "blocked"
    assert report["checks"]["595_dkms_current_state_not_installed"] is False


def test_apt_candidate_drift_blocks_even_with_matching_index_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = "nvidia-driver-595"
    version = r4.TARGET_DRIVER_DEB
    show = "\n".join(
        [
            f"Package: {package}",
            f"Version: {version}",
            "Architecture: amd64",
            "Filename: pool/multiverse/n/nvidia/pkg.deb",
            "Size: 123",
            "SHA256: " + "a" * 64,
            "",
        ]
    )

    def fake_run(argv, **kwargs):
        output = show if argv[1] == "show" else "  Candidate: 595.99.99-evil\n"
        return r4.CommandResult(tuple(argv), 0, output)

    monkeypatch.setattr(r4, "_run", fake_run)
    report = r4.apt_candidate_record(package, version, "amd64")
    assert report["status"] == "blocked"
    assert report["candidate_version"] == "595.99.99-evil"


def test_plan_fingerprint_and_source_pin_tamper_block(tmp_path: Path) -> None:
    plan = _static_plan(tmp_path)
    assert r4.verify_plan_static(plan)["status"] == "pass"
    plan["authority"]["install_enabled"] = True  # type: ignore[index]
    assert r4.verify_plan_static(plan)["status"] == "blocked"
    plan = _static_plan(tmp_path / "fresh")
    source = Path(plan["source_pins"]["validator"]["path"])  # type: ignore[index]
    source.write_bytes(b"changed")
    assert r4.verify_plan_static(plan)["status"] == "blocked"


def test_recomputed_self_fingerprint_cannot_change_exact_transaction(tmp_path: Path) -> None:
    plan = _static_plan(tmp_path)
    plan["transaction_contract"]["install"][0]["version"] = "evil"  # type: ignore[index]
    plan["plan_fingerprint_sha256"] = r4.plan_fingerprint(plan)
    report = r4.verify_plan_static(plan)
    assert report["status"] == "blocked"
    assert report["checks"]["self_fingerprint"] is True
    assert report["checks"]["transaction_contract_exact"] is False


def test_verify_cache_requires_exact_set_hash_size_and_deb_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _static_plan(tmp_path / "plan")
    cache = tmp_path / "cache"
    cache.mkdir()
    payload = b"exact-deb"
    deb = cache / "pkg_1_amd64.deb"
    deb.write_bytes(payload)
    plan["initial_state"]["apt"] = {  # type: ignore[index]
            "candidates": [
                {
                    "package": "pkg",
                    "version": "1",
                    "architecture": "amd64",
                    "filename": "pool/main/p/pkg/pkg_1_amd64.deb",
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ]
    }
    plan["plan_fingerprint_sha256"] = r4.plan_fingerprint(plan)

    def fake_run(argv, **kwargs):
        assert argv[0] == r4.BIN["dpkg_deb"]
        return r4.CommandResult(
            tuple(argv), 0, "Package: pkg\nVersion: 1\nArchitecture: amd64\n"
        )

    monkeypatch.setattr(r4, "_run", fake_run)
    monkeypatch.setattr(
        r4,
        "_trusted_cache_permissions",
        lambda cache_dir, paths: {"status": "pass", "checks": {}},
    )
    assert r4.verify_cache(plan, cache)["status"] == "pass"
    (cache / "unexpected.deb").write_bytes(b"x")
    report = r4.verify_cache(plan, cache)
    assert report["status"] == "blocked"
    assert report["checks"]["exact_deb_set"] is False
    (cache / "unexpected.deb").unlink()
    deb.write_bytes(b"tampered")
    report = r4.verify_cache(plan, cache)
    assert report["status"] == "blocked"
    assert report["details"][deb.name]["checks"]["sha256"] is False


def test_user_owned_or_outside_var_cache_is_not_a_trusted_execution_cache(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    deb = cache / "x.deb"
    deb.write_bytes(b"x")
    report = r4._trusted_cache_permissions(cache, [deb])
    assert report["status"] == "blocked"
    assert report["checks"]["under_root_owned_cache_root"] is False


def test_module_set_requires_consistent_exact_signer_and_sig_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_module(kernel: str, module: str):
        return {
            "status": "pass",
            "fields": {
                "signer": "signer-a" if module != "nvidia_uvm" else "signer-b",
                "sig_key": "key-a",
            },
        }

    monkeypatch.setattr(r4, "verify_module", fake_module)
    report = r4.verify_modules_for_kernel(r4.TARGET_KERNEL)
    assert report["status"] == "blocked"
    assert report["checks"]["one_exact_signer_per_kernel"] is False


def test_initramfs_requires_parse_kernel_metadata_nvme_and_ext4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = r4.TARGET_KERNEL
    root = f"usr/lib/modules/{kernel}"
    output = "\n".join(
        [
            "init",
            f"{root}/modules.dep",
            f"{root}/modules.alias",
            f"{root}/kernel/drivers/nvme/host/nvme.ko.zst",
            f"{root}/kernel/drivers/nvme/host/nvme-core.ko.zst",
            f"{root}/kernel/fs/ext4/ext4.ko.zst",
        ]
    )
    monkeypatch.setattr(r4, "_boot_file", lambda path: {"status": "pass"})
    monkeypatch.setattr(
        r4,
        "verify_kernel_boot_config",
        lambda kernel: {
            "status": "pass",
            "selected_values": {"CONFIG_EXT4_FS": "y"},
        },
    )
    monkeypatch.setattr(
        r4,
        "_run",
        lambda argv, **kwargs: r4.CommandResult(tuple(argv), 0, output),
    )
    assert r4.verify_initramfs(kernel)["status"] == "pass"
    output = output.replace(f"{root}/kernel/drivers/nvme/host/nvme.ko.zst\n", "")
    report = r4.verify_initramfs(kernel)
    assert report["status"] == "blocked"
    assert report["checks"]["has_nvme_host_driver"] is False


GRUB_FIXTURE = """
menuentry 'Ubuntu' --class ubuntu 'gnulinux-simple-root' {
    linux /boot/vmlinuz-7.0.0-28-generic root=/dev/nvme1n1p1
    initrd /boot/initrd.img-7.0.0-28-generic
}
submenu 'Advanced options for Ubuntu' 'gnulinux-advanced-root' {
    menuentry 'Ubuntu, with Linux 7.0.0-28-generic' 'gnulinux-7-root' {
        linux /boot/vmlinuz-7.0.0-28-generic root=/dev/nvme1n1p1
        initrd /boot/initrd.img-7.0.0-28-generic
    }
    menuentry 'Ubuntu, with Linux 6.17.0-35-generic' 'gnulinux-6-root' {
        linux /boot/vmlinuz-6.17.0-35-generic root=/dev/nvme1n1p1
        initrd /boot/initrd.img-6.17.0-35-generic
    }
}
"""


def test_grub_parser_proves_default_target_and_fallback() -> None:
    entries = r4.parse_grub_entries(GRUB_FIXTURE)
    default = r4._resolve_grub_default("0", entries)
    assert default is not None
    assert default["linux"] == f"/boot/vmlinuz-{r4.TARGET_KERNEL}"
    fallback = r4._resolve_grub_default("gnulinux-6-root", entries)
    assert fallback is not None
    assert fallback["initrd"] == f"/boot/initrd.img-{r4.FALLBACK_KERNEL}"


def test_grub_unsupported_numeric_submenu_selector_blocks_closed() -> None:
    entries = r4.parse_grub_entries(GRUB_FIXTURE)
    assert r4._resolve_grub_default("1>0", entries) is None


def test_pre_reboot_forbidden_held_package_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _static_plan(tmp_path)
    monkeypatch.setattr(r4, "_verify_package_exact", lambda *args: _pass_status())
    monkeypatch.setattr(
        r4,
        "_dpkg_entry_live",
        lambda package: {
            "package": package,
            "status": "hi ",
            "version": "x",
            "architecture": "amd64",
        },
    )
    monkeypatch.setattr(
        r4,
        "collect_dpkg_projection",
        lambda: {"status": "pass", "incomplete": []},
    )
    monkeypatch.setattr(r4, "verify_modules_for_kernel", lambda kernel: _pass_status())
    monkeypatch.setattr(r4, "verify_initramfs", lambda kernel: _pass_status())
    monkeypatch.setattr(r4, "_boot_file", lambda path: _pass_status())
    monkeypatch.setattr(r4, "verify_grub", _pass_status)

    def fake_run(argv, **kwargs):
        if argv[0] == r4.BIN["mokutil"]:
            return r4.CommandResult(tuple(argv), 0, "SecureBoot enabled\n")
        if argv[0] == r4.BIN["dkms"]:
            return r4.CommandResult(tuple(argv), 0, "")
        if argv[0] == r4.BIN["findmnt"]:
            return r4.CommandResult(tuple(argv), 0, "/dev/nvme1n1p1 ext4\n")
        raise AssertionError(argv)

    monkeypatch.setattr(r4, "_run", fake_run)
    report = r4.verify_pre_reboot(plan)
    assert report["status"] == "blocked"
    assert report["checks"]["old_590_and_595_dkms_current_state_absent"] is False


def test_future_commands_are_data_and_install_is_no_download(tmp_path: Path) -> None:
    cache = "/var/cache/deepsafe-driver595-r4/" + "a" * 20 + "/archives"
    download = r4.future_cache_download_argv(cache)
    install = r4.future_install_argv(cache)
    assert "--download-only" in download
    assert "--no-download" in install
    assert "--no-install-recommends" in install
    with pytest.raises(ValueError):
        r4._validate_read_only_argv(download)
    with pytest.raises(ValueError):
        r4._validate_read_only_argv(install)


def test_schema_accepts_real_read_only_prepared_plan() -> None:
    plan = r4.prepare_plan()
    schema = json.loads(r4.PLAN_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(plan)
    assert plan["status"] == (
        "prepared_read_only_pending_independent_review_cache_and_authorization"
    )
    assert r4.plan_fingerprint(plan) == plan["plan_fingerprint_sha256"]
    assert plan["authority"]["this_plan_executes_work"] is False
    assert plan["authority"]["downloads_performed"] is False
    assert plan["authority"]["install_enabled"] is False
