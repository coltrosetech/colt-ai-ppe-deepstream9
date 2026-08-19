from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from validation import driver595_maintenance_r5 as r5


def _apt_fixture(
    origin: str = "Ubuntu:24.04/noble-updates, Ubuntu:24.04/noble-security",
) -> str:
    lines = ["NOTE: This is only a simulation!"]
    for package, (version, _) in sorted(r5.EXPECTED_REMOVE_VERSIONS.items()):
        lines.append(f"Remv {package} [{version}]")
    for package, (version, architecture) in sorted(
        r5.EXPECTED_INSTALL_VERSIONS.items()
    ):
        lines.append(f"Inst {package} ({version} {origin} [{architecture}])")
    for package, (version, architecture) in sorted(
        r5.EXPECTED_CONFIGURE_VERSIONS.items()
    ):
        lines.append(f"Conf {package} ({version} {origin} [{architecture}])")
    lines.append(
        f"0 upgraded, {len(r5.EXPECTED_INSTALL_VERSIONS)} newly installed, "
        f"{len(r5.EXPECTED_REMOVE_VERSIONS)} to remove and 147 not upgraded."
    )
    return "\n".join(lines)


def _policy_fixture(component: str = "multiverse") -> str:
    return "\n".join(
        [
            "  Candidate: 1",
            f" 500 http://archive.ubuntu.com/ubuntu noble-updates/{component} amd64 Packages",
            f" 500 http://security.ubuntu.com/ubuntu noble-security/{component} amd64 Packages",
        ]
    )


@pytest.fixture(scope="session")
def real_plan() -> dict[str, object]:
    plan = r5.prepare_plan()
    assert plan["status"] == (
        "candidate_read_only_pending_independent_acceptance_and_authorization"
    )
    assert r5.validate_plan_schema(plan)["status"] == "pass"
    return plan


def _resign(plan: dict[str, object]) -> dict[str, object]:
    plan["plan_fingerprint_sha256"] = r5.plan_fingerprint(plan)
    return plan


def _rebind_snapshot(plan: dict[str, object]) -> dict[str, object]:
    initial = plan["initial_state"]
    assert isinstance(initial, dict)
    snapshot = initial["snapshot"]
    stability = initial["snapshot_stability"]
    assert isinstance(snapshot, dict)
    assert isinstance(stability, dict)
    snapshot["snapshot_sha256"] = r5.snapshot_digest(snapshot)
    stability["before_sha256"] = r5.snapshot_digest(snapshot)
    stability["after_sha256"] = r5.snapshot_digest(snapshot)
    source_pins = plan["source_pins"]
    toolchain = plan["toolchain"]
    assert isinstance(source_pins, dict)
    assert isinstance(toolchain, dict)
    basis = r5._state_basis(snapshot, stability, source_pins, toolchain)
    initial["state_basis"] = basis
    state_id = r5.sha256_bytes(r5.canonical_json(basis))[:20]
    cache_dir = f"/var/cache/deepsafe-driver595-r5/{state_id}/archives"
    plan["plan_id"] = f"driver595-r5-{state_id}"
    plan["future_commands"] = {
        "simulation_argv": r5.apt_simulation_argv(),
        "cache_dir": cache_dir,
        "cache_download_argv_after_separate_authorization_only": r5.future_cache_download_argv(
            cache_dir
        ),
        "install_argv_after_separate_authorization_only": r5.future_install_argv(
            cache_dir
        ),
    }
    plan["status"] = r5._candidate_status(snapshot, stability)
    return _resign(plan)


def test_contract_counts_and_cli_are_execution_closed() -> None:
    assert (len(r5.EXPECTED_INSTALL_VERSIONS), len(r5.EXPECTED_REMOVE_VERSIONS)) == (
        22,
        16,
    )
    assert len(r5.EXPECTED_CONFIGURE_VERSIONS) == 26
    assert {"apply", "install", "download", "reboot"}.isdisjoint(r5.CLI_MODES)
    assert r5.EXPECTED_AUTHORITY["this_plan_executes_work"] is False
    assert r5.EXPECTED_AUTHORITY["download_authorized"] is False
    assert r5.EXPECTED_AUTHORITY["install_authorized"] is False
    assert r5.EXPECTED_AUTHORITY["reboot_authorized"] is False


@pytest.mark.parametrize(
    "argv",
    [
        [r5.BIN["apt_get"], "install", "nvidia-driver-595"],
        r5.future_cache_download_argv(
            "/var/cache/deepsafe-driver595-r5/" + "a" * 20 + "/archives"
        ),
        r5.future_install_argv(
            "/var/cache/deepsafe-driver595-r5/" + "a" * 20 + "/archives"
        ),
        [r5.BIN["dpkg"], "--configure", "-a"],
        [r5.BIN["dpkg_query"], "-W", "--admindir=/tmp/evil"],
        [r5.BIN["modinfo"], "--help"],
        [r5.BIN["grub_editenv"], "/boot/grub/grubenv", "set", "next_entry=x"],
        [r5.BIN["mokutil"], "--import", "/tmp/key.der"],
        [r5.BIN["dkms"], "remove", "nvidia/595", "--all"],
        [r5.BIN["nvidia_smi"], "--gpu-reset"],
        [r5.BIN["dpkg_deb"], "--extract", "/tmp/x.deb", "/tmp/out"],
        ["apt-get", "-s", "install", "x"],
        [
            r5.BIN["dpkg_query"],
            "-S",
            "/usr/lib/modules/7.0.0-28-generic/kernel/nvidia-595/../evil.ko",
        ],
    ],
)
def test_exact_read_only_command_grammar_rejects_every_other_argv(
    argv: list[str],
) -> None:
    with pytest.raises(ValueError):
        r5._validate_read_only_argv(argv)


def test_run_uses_clean_environment_root_cwd_closed_stdin_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "ok")

    monkeypatch.setattr(r5.subprocess, "run", fake_run)
    result = r5._run([r5.BIN["uname"], "-r"], timeout=7)
    assert result.returncode == 0
    assert observed["env"] == r5.SAFE_ENV
    assert observed["cwd"] == "/"
    assert observed["stdin"] is subprocess.DEVNULL
    assert observed["timeout"] == 7
    assert observed["close_fds"] is True


def test_exact_apt_action_projection_and_origins_pass() -> None:
    report = r5.parse_apt_actions(_apt_fixture())
    assert report["status"] == "pass"
    assert len(report["actions"]) == 64
    assert report["checks"]["exact_fullmatch_origins"] is True


@pytest.mark.parametrize(
    "origin",
    [
        "Ubuntu:24.04/noble-evil-attacker",
        "Ubuntu:24.04/noble-updates-evil",
        "LP-PPA-attacker:24.04/noble",
        "file:/tmp/repo noble/main",
        "Ubuntu:24.04/noble-security, Ubuntu:24.04/noble-updates",
        "Ubuntu:24.04/noble-updates",
        "",
    ],
)
def test_spoofed_empty_reordered_or_incomplete_apt_origins_block(origin: str) -> None:
    report = r5.parse_apt_actions(_apt_fixture(origin))
    assert report["status"] == "blocked"
    assert report["checks"]["exact_fullmatch_origins"] is False or report["malformed_actions"]


def test_wrong_apt_version_blocks_with_exact_names() -> None:
    output = _apt_fixture().replace(r5.TARGET_DRIVER_DEB, "595.71.05-attacker")
    assert r5.parse_apt_actions(output)["status"] == "blocked"


def test_exact_archive_and_security_policy_projection_passes() -> None:
    report = r5.parse_policy_sources(_policy_fixture(), "nvidia-driver-595")
    assert report["status"] == "pass"
    assert len(report["sources"]) == 2


def test_exact_deb822_signed_by_authority_passes() -> None:
    text = r5.OFFICIAL_SOURCE_FILE.read_text(encoding="utf-8")
    assert r5.parse_official_deb822_source(text)["status"] == "pass"


@pytest.mark.parametrize(
    "old,new",
    [
        (
            "/usr/share/keyrings/ubuntu-archive-keyring.gpg",
            "/tmp/attacker-keyring.gpg",
        ),
        ("archive.ubuntu.com", "evil.example"),
        ("noble-updates", "noble-evil"),
        ("Types: deb", "Types: deb deb-src"),
        ("Components: main", "Unknown: injected\nComponents: main"),
    ],
)
def test_alternate_deb822_keyring_uri_suite_type_or_field_blocks(
    old: str, new: str
) -> None:
    text = r5.OFFICIAL_SOURCE_FILE.read_text(encoding="utf-8").replace(old, new)
    assert r5.parse_official_deb822_source(text)["status"] == "blocked"


@pytest.mark.parametrize(
    "old,new",
    [
        ("archive.ubuntu.com", "ppa.launchpadcontent.net/attacker/ppa/ubuntu"),
        ("http://archive.ubuntu.com", "file:/tmp"),
        ("noble-updates", "noble-evil"),
        ("noble-security", "noble"),
        (" 500 ", " 1001 "),
    ],
)
def test_foreign_policy_uri_suite_file_index_or_priority_blocks(
    old: str, new: str
) -> None:
    report = r5.parse_policy_sources(
        _policy_fixture().replace(old, new), "nvidia-driver-595"
    )
    assert report["status"] == "blocked"


@pytest.mark.parametrize(
    "extra",
    [
        " 500 file:/tmp/repo noble/multiverse amd64 Packages",
        " 500 http://ppa.launchpad.net/attacker/ppa/ubuntu noble/multiverse amd64 Packages",
        " 500 https://evil.example/ubuntu noble-updates/multiverse amd64 Packages",
    ],
)
def test_extra_local_ppa_or_foreign_policy_source_blocks_even_with_officials(
    extra: str,
) -> None:
    report = r5.parse_policy_sources(
        _policy_fixture() + "\n" + extra, "nvidia-driver-595"
    )
    assert report["status"] == "blocked"
    assert report["malformed_package_lines"] or len(report["sources"]) != 2


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}',
        b"\xef\xbb\xbf{}",
        b'{"value":NaN}',
        b"\xff",
        b"[]",
    ],
)
def test_strict_raw_plan_loader_rejects_duplicate_bom_nan_non_utf8_or_nonobject(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "plan.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        r5.load_plan_raw(path)


def test_real_read_only_candidate_is_schema_and_static_valid_but_not_accepted(
    real_plan: dict[str, object]
) -> None:
    schema = r5.validate_plan_schema(real_plan)
    static = r5.verify_candidate_static(real_plan)
    assert schema == {"status": "pass", "errors": []}
    assert static["status"] == "pass"
    assert static["candidate_accepted"] is False
    assert static["install_authorized"] is False
    assert static["reboot_authorized"] is False


def _mutate_missing_prepared(plan: dict[str, object]) -> None:
    plan.pop("prepared_at_utc")


def _mutate_extra_root(plan: dict[str, object]) -> None:
    plan["attacker"] = True


def _mutate_authority(plan: dict[str, object]) -> None:
    authority = plan["authority"]
    assert isinstance(authority, dict)
    authority["install_authorized"] = True


def _mutate_phase(plan: dict[str, object]) -> None:
    contract = plan["execution_contract"]
    assert isinstance(contract, dict)
    contract["phase_order"] = ["install_first"]


def _mutate_live_check(plan: dict[str, object]) -> None:
    initial = plan["initial_state"]
    assert isinstance(initial, dict)
    snapshot = initial["snapshot"]
    assert isinstance(snapshot, dict)
    live = snapshot["live_baseline"]
    assert isinstance(live, dict)
    checks = live["checks"]
    assert isinstance(checks, dict)
    checks["secure_boot_enabled"] = False


def _mutate_state_basis(plan: dict[str, object]) -> None:
    initial = plan["initial_state"]
    assert isinstance(initial, dict)
    basis = initial["state_basis"]
    assert isinstance(basis, dict)
    basis["snapshot_sha256"] = "0" * 64


def _mutate_predecessor_count(plan: dict[str, object]) -> None:
    evidence = plan["predecessor_test_evidence"]
    assert isinstance(evidence, dict)
    evidence["frozen_r4_collected"] = 44


def _mutate_nested_extra(plan: dict[str, object]) -> None:
    initial = plan["initial_state"]
    assert isinstance(initial, dict)
    snapshot = initial["snapshot"]
    assert isinstance(snapshot, dict)
    candidates = snapshot["candidates"]
    assert isinstance(candidates, list)
    candidates[0]["ignored_attacker_field"] = True


@pytest.mark.parametrize(
    "mutator",
    [
        _mutate_missing_prepared,
        _mutate_extra_root,
        _mutate_authority,
        _mutate_phase,
        _mutate_live_check,
        _mutate_state_basis,
        _mutate_predecessor_count,
        _mutate_nested_extra,
    ],
    ids=lambda item: item.__name__.removeprefix("_mutate_"),
)
def test_self_resigned_schema_or_contract_mutations_block(
    real_plan: dict[str, object], mutator: Callable[[dict[str, object]], None]
) -> None:
    plan = copy.deepcopy(real_plan)
    mutator(plan)
    _resign(plan)
    assert r5.verify_candidate_static(plan)["status"] == "blocked"


def test_arbitrary_consistent_module_key_is_semantically_blocked_after_rebind(
    real_plan: dict[str, object]
) -> None:
    plan = copy.deepcopy(real_plan)
    initial = plan["initial_state"]
    assert isinstance(initial, dict)
    snapshot = initial["snapshot"]
    assert isinstance(snapshot, dict)
    live = snapshot["live_baseline"]
    assert isinstance(live, dict)
    refs = live["module_trust_references"]
    assert isinstance(refs, dict)
    for ref in refs.values():
        ref["fields"]["sig_key"] = "DE:AD:BE:EF"
    _rebind_snapshot(plan)
    result = r5.verify_candidate_static(plan)
    assert result["status"] == "blocked"
    assert result["snapshot_validation"]["checks"]["module_authorities_exact"] is False


def test_external_raw_hash_and_fingerprint_match_is_integrity_only(
    real_plan: dict[str, object]
) -> None:
    raw = r5.canonical_json(real_plan)
    raw_sha = hashlib.sha256(raw).hexdigest()
    result = r5.verify_plan_static(
        real_plan,
        raw_sha256=raw_sha,
        expected_raw_sha256=raw_sha,
        expected_fingerprint=str(real_plan["plan_fingerprint_sha256"]),
    )
    assert result["status"] == "pass"
    assert result["candidate_accepted"] is False
    assert result["external_independent_acceptance_receipt_present"] is False
    assert result["install_authorized"] is False


def test_self_signed_plan_with_external_raw_hash_mismatch_blocks(
    real_plan: dict[str, object]
) -> None:
    raw_sha = hashlib.sha256(r5.canonical_json(real_plan)).hexdigest()
    result = r5.verify_plan_static(
        real_plan,
        raw_sha256=raw_sha,
        expected_raw_sha256="0" * 64,
        expected_fingerprint=str(real_plan["plan_fingerprint_sha256"]),
    )
    assert result["status"] == "blocked"
    assert result["checks"]["raw_plan_sha_exact"] is False


def test_current_verifier_blocks_double_snapshot_live_drift(
    real_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = copy.deepcopy(real_plan["initial_state"]["snapshot"])
    drift = copy.deepcopy(expected)
    drift["live_baseline"]["running_kernel"] = r5.TARGET_KERNEL
    drift["snapshot_sha256"] = r5.snapshot_digest(drift)
    values = iter([expected, drift])
    monkeypatch.setattr(r5, "collect_planning_snapshot", lambda: next(values))
    raw_sha = hashlib.sha256(r5.canonical_json(real_plan)).hexdigest()
    result = r5.verify_plan_current(
        real_plan,
        raw_sha256=raw_sha,
        expected_raw_sha256=raw_sha,
        expected_fingerprint=str(real_plan["plan_fingerprint_sha256"]),
    )
    assert result["status"] == "blocked"
    assert result["checks"]["current_double_snapshot_stable"] is False


def _module_observation(
    *, kernel: str = r5.TARGET_KERNEL, module: str = "nvidia"
) -> tuple[dict[str, object], dict[str, str], list[str], str]:
    authority = r5.TRUSTED_KERNEL_MODULE_AUTHORITIES[kernel]
    trust = {"status": "pass", "fields": copy.deepcopy(authority["fields"])}
    fields = copy.deepcopy(authority["fields"])
    fields["name"] = module
    fields["version"] = r5.TARGET_DRIVER
    basename = r5.MODULE_FILE_BASENAMES[module]
    resolved = f"/usr/lib/modules/{kernel}/kernel/nvidia-595/{basename}.zst"
    fields["filename"] = resolved
    owners = [f"linux-modules-nvidia-595-{kernel}"]
    return trust, fields, owners, resolved


def test_exact_nvidia_module_observation_passes() -> None:
    trust, fields, owners, resolved = _module_observation()
    report = r5.validate_nvidia_module_observation(
        r5.TARGET_KERNEL,
        "nvidia",
        trust,
        fields=fields,
        fields_ok=True,
        owners=owners,
        owner_package={"status": "pass"},
        resolved_filename=resolved,
        module_file={"status": "pass"},
    )
    assert report["status"] == "pass"


@pytest.mark.parametrize(
    "case",
    [
        "wrong-vermagic",
        "attacker-signer",
        "attacker-key",
        "wrong-sig-id",
        "wrong-hash-algorithm",
        "wrong-driver-version",
        "wrong-module-name",
        "dkms-path",
        "cross-kernel-path",
        "multiple-owners",
        "unverified-owner-package",
        "untrusted-module-file",
    ],
)
def test_nvidia_module_wrong_abi_trust_path_owner_or_integrity_blocks(case: str) -> None:
    trust, fields, owners, resolved = _module_observation()
    owner_package = {"status": "pass"}
    module_file = {"status": "pass"}
    if case == "wrong-vermagic":
        fields["vermagic"] = r5.FALLBACK_KERNEL + " SMP"
    elif case == "attacker-signer":
        fields["signer"] = "attacker"
    elif case == "attacker-key":
        fields["sig_key"] = "DE:AD:BE:EF"
    elif case == "wrong-sig-id":
        fields["sig_id"] = "X.509"
    elif case == "wrong-hash-algorithm":
        fields["sig_hashalgo"] = "sha256"
    elif case == "wrong-driver-version":
        fields["version"] = "590.48.01"
    elif case == "wrong-module-name":
        fields["name"] = "nvidia_uvm"
    elif case == "dkms-path":
        resolved = f"/usr/lib/modules/{r5.TARGET_KERNEL}/updates/dkms/nvidia.ko.zst"
    elif case == "cross-kernel-path":
        resolved = f"/usr/lib/modules/{r5.FALLBACK_KERNEL}/kernel/nvidia-595/nvidia.ko.zst"
    elif case == "multiple-owners":
        owners.append("attacker-package")
    elif case == "unverified-owner-package":
        owner_package = {"status": "blocked"}
    else:
        module_file = {"status": "blocked"}
    report = r5.validate_nvidia_module_observation(
        r5.TARGET_KERNEL,
        "nvidia",
        trust,
        fields=fields,
        fields_ok=True,
        owners=owners,
        owner_package=owner_package,
        resolved_filename=resolved,
        module_file=module_file,
    )
    assert report["status"] == "blocked"


def _initramfs_listing(kernel: str) -> list[str]:
    root = f"usr/lib/modules/{kernel}"
    return [
        "init",
        root,
        f"{root}/modules.dep",
        f"{root}/modules.alias",
        f"{root}/kernel/drivers/nvme/host/nvme.ko.zst",
        f"{root}/kernel/drivers/nvme/host/nvme-core.ko.zst",
    ]


def _run_initramfs_fixture(
    monkeypatch: pytest.MonkeyPatch,
    lines: list[str],
    *,
    ext4: str = "y",
) -> dict[str, object]:
    monkeypatch.setattr(r5, "_boot_file", lambda path: {"status": "pass"})
    monkeypatch.setattr(
        r5,
        "verify_kernel_boot_config",
        lambda kernel: {
            "status": "pass",
            "selected_values": {"CONFIG_EXT4_FS": ext4},
        },
    )
    output = "\n".join(lines)
    monkeypatch.setattr(
        r5,
        "_run",
        lambda argv, **kwargs: r5.CommandResult(tuple(argv), 0, output),
    )
    return r5.verify_initramfs(r5.TARGET_KERNEL)


def test_exact_initramfs_member_fullmatches_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_initramfs_fixture(
        monkeypatch, _initramfs_listing(r5.TARGET_KERNEL)
    )["status"] == "pass"


@pytest.mark.parametrize(
    "case",
    [
        "ko-attacker",
        "ko-not-a-module",
        "wrong-kernel-root",
        "duplicate-selected",
        "missing-modules-dep",
        "missing-nvme",
        "missing-ext4-module",
    ],
)
def test_initramfs_ambiguous_wrong_duplicate_or_missing_content_blocks(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    lines = _initramfs_listing(r5.TARGET_KERNEL)
    ext4 = "y"
    root = f"usr/lib/modules/{r5.TARGET_KERNEL}"
    if case == "ko-attacker":
        lines[4] += ".attacker"
    elif case == "ko-not-a-module":
        lines[5] += ".not-a-module"
    elif case == "wrong-kernel-root":
        lines.append(
            f"usr/lib/modules/{r5.FALLBACK_KERNEL}/kernel/drivers/nvme/host/nvme.ko.zst"
        )
    elif case == "duplicate-selected":
        lines.append(lines[4])
    elif case == "missing-modules-dep":
        lines.remove(f"{root}/modules.dep")
    elif case == "missing-nvme":
        lines.remove(f"{root}/kernel/drivers/nvme/host/nvme.ko.zst")
    else:
        ext4 = "m"
    assert _run_initramfs_fixture(monkeypatch, lines, ext4=ext4)["status"] == "blocked"


GRUB_FIXTURE = """
menuentry 'Ubuntu' --class ubuntu --id 'gnulinux-simple-root' {
    linux /boot/vmlinuz-7.0.0-28-generic root=/dev/nvme1n1p1
    initrd /boot/initrd.img-7.0.0-28-generic
}
submenu 'Advanced options for Ubuntu' --id 'gnulinux-advanced-root' {
    menuentry 'Ubuntu, with Linux 7.0.0-28-generic' --id 'gnulinux-7-root' {
        linux /boot/vmlinuz-7.0.0-28-generic root=/dev/nvme1n1p1
        initrd /boot/initrd.img-7.0.0-28-generic
    }
    menuentry 'Ubuntu, with Linux 6.17.0-35-generic' --id 'gnulinux-6-root' {
        linux /boot/vmlinuz-6.17.0-35-generic root=/dev/nvme1n1p1
        initrd /boot/initrd.img-6.17.0-35-generic
    }
}
"""


@pytest.mark.parametrize(
    "selector,entry_id",
    [
        ("0", "gnulinux-simple-root"),
        ("1>0", "gnulinux-7-root"),
        ("1>1", "gnulinux-6-root"),
        ("gnulinux-6-root", "gnulinux-6-root"),
        (
            "Advanced options for Ubuntu>Ubuntu, with Linux 6.17.0-35-generic",
            "gnulinux-6-root",
        ),
        ("gnulinux-advanced-root>gnulinux-7-root", "gnulinux-7-root"),
    ],
)
def test_grub_numeric_submenu_id_title_saved_selector_resolution(
    selector: str, entry_id: str
) -> None:
    entry = r5._resolve_grub_default(selector, r5.parse_grub_entries(GRUB_FIXTURE))
    assert entry is not None
    assert entry["id"] == entry_id


def test_grub_duplicate_id_or_title_is_ambiguous_and_blocks_resolution() -> None:
    duplicate = GRUB_FIXTURE.replace(
        "gnulinux-6-root", "gnulinux-7-root"
    ).replace(
        "Ubuntu, with Linux 6.17.0-35-generic",
        "Ubuntu, with Linux 7.0.0-28-generic",
    )
    entries = r5.parse_grub_entries(duplicate)
    assert r5._resolve_grub_default("gnulinux-7-root", entries) is None
    assert r5._resolve_grub_default("Ubuntu, with Linux 7.0.0-28-generic", entries) is None


@pytest.mark.parametrize(
    "text",
    [
        "if true; then GRUB_DEFAULT=attacker; fi",
        "source /tmp/attacker",
        "GRUB_DEFAULT=$(cat /tmp/attacker)",
        "eval GRUB_DEFAULT=attacker",
    ],
)
def test_dynamic_or_indirect_grub_dropin_shell_blocks(text: str) -> None:
    assert r5._inspect_grub_defaults_text(text, strict_dropin=True)["status"] == "blocked"


def test_grub_dropin_last_literal_assignment_wins() -> None:
    base = r5._inspect_grub_defaults_text("GRUB_DEFAULT=0", strict_dropin=False)
    dropin = r5._inspect_grub_defaults_text(
        "GRUB_DEFAULT='gnulinux-6-root'", strict_dropin=True
    )
    effective = [*base["assignments"], *dropin["assignments"]][-1]
    assert effective == "gnulinux-6-root"
    entry = r5._resolve_grub_default(effective, r5.parse_grub_entries(GRUB_FIXTURE))
    assert entry is not None
    assert entry["linux"] == f"/boot/vmlinuz-{r5.FALLBACK_KERNEL}"


def _run_grub_generated_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg: str = GRUB_FIXTURE,
    effective_default: str = "saved",
    grubenv: str = "saved_entry=gnulinux-simple-root\n",
) -> dict[str, object]:
    defaults = {"status": "pass", "effective_default": effective_default}
    monkeypatch.setattr(r5, "collect_grub_defaults_closure", lambda: defaults)
    monkeypatch.setattr(r5, "_trusted_regular_file", lambda path: {"status": "pass"})
    monkeypatch.setattr(r5.Path, "read_text", lambda self, **kwargs: cfg)
    monkeypatch.setattr(
        r5,
        "_run",
        lambda argv, **kwargs: r5.CommandResult(tuple(argv), 0, grubenv),
    )
    return r5.verify_grub_generated(defaults)


def test_grub_saved_default_and_absent_next_entry_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_grub_generated_fixture(monkeypatch)
    assert report["status"] == "pass"
    assert report["checks"]["default_exact_target"] is True


def test_grub_one_shot_next_entry_away_from_target_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_grub_generated_fixture(
        monkeypatch,
        grubenv=(
            "saved_entry=gnulinux-simple-root\n"
            "next_entry=gnulinux-6-root\n"
        ),
    )
    assert report["status"] == "blocked"
    assert report["checks"]["next_absent_or_target"] is False


def test_grub_missing_fallback_entry_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = GRUB_FIXTURE.replace(
        "    menuentry 'Ubuntu, with Linux 6.17.0-35-generic' --id 'gnulinux-6-root' {\n"
        "        linux /boot/vmlinuz-6.17.0-35-generic root=/dev/nvme1n1p1\n"
        "        initrd /boot/initrd.img-6.17.0-35-generic\n"
        "    }\n",
        "",
    )
    report = _run_grub_generated_fixture(monkeypatch, cfg=cfg)
    assert report["status"] == "blocked"
    assert report["checks"]["fallback_entry_present"] is False


def test_cache_argument_must_equal_exact_plan_state_path(
    real_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r5, "verify_plan_static", lambda *args, **kwargs: {"status": "pass"})
    monkeypatch.setattr(r5, "verify_plan_current", lambda *args, **kwargs: {"status": "pass"})
    wrong = Path("/var/cache/deepsafe-driver595-r5/" + "0" * 20 + "/archives")
    result = r5.verify_cache(
        real_plan,
        wrong,
        raw_sha256="a" * 64,
        expected_raw_sha256="a" * 64,
        expected_fingerprint=str(real_plan["plan_fingerprint_sha256"]),
    )
    assert result["status"] == "blocked"
    assert result["checks"]["cache_argument_equals_plan_exact_path"] is False


def test_trusted_regular_file_rejects_external_hardlink(
    tmp_path: Path,
) -> None:
    first = tmp_path / "package.deb"
    second = tmp_path / "external-link.deb"
    first.write_bytes(b"package")
    os.link(first, second)
    report = r5._trusted_regular_file(first)
    assert first.stat().st_nlink == 2
    assert report["status"] == "blocked"
    assert report["checks"]["single_link"] is False


@pytest.mark.parametrize("name,check", [("partial", "partial_empty"), ("lock", "lock_absent_or_trusted")])
def test_cache_broken_auxiliary_symlink_blocks(
    real_plan: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    check: str,
) -> None:
    cache = tmp_path / "archives"
    cache.mkdir()
    (cache / name).symlink_to(tmp_path / "missing-target")
    monkeypatch.setattr(r5, "verify_plan_static", lambda *args, **kwargs: {"status": "pass"})
    monkeypatch.setattr(r5, "verify_plan_current", lambda *args, **kwargs: {"status": "pass"})
    result = r5.verify_cache(real_plan, cache)
    assert result["status"] == "blocked"
    assert result["checks"][check] is False


def test_schema_invalid_cache_plan_blocks_before_field_access() -> None:
    result = r5.verify_cache({}, Path("/tmp/not-a-cache"), reverify_current=False)
    assert result["status"] == "blocked"
    assert result["checks"] == {"strict_schema": False}


def test_pre_reboot_without_external_hashes_never_authorizes_reboot(
    real_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r5, "_verify_package_exact", lambda *args, **kwargs: {"status": "pass"})
    monkeypatch.setattr(r5, "_dpkg_entry_live", lambda package: None)
    monkeypatch.setattr(
        r5,
        "collect_dpkg_projection",
        lambda: {"status": "pass", "incomplete": []},
    )
    monkeypatch.setattr(r5, "verify_modules_for_kernel", lambda *args: {"status": "pass"})
    monkeypatch.setattr(r5, "verify_initramfs", lambda *args: {"status": "pass"})
    monkeypatch.setattr(r5, "_boot_file", lambda *args: {"status": "pass"})
    refs = real_plan["initial_state"]["snapshot"]["live_baseline"][
        "module_trust_references"
    ]
    monkeypatch.setattr(
        r5, "collect_module_trust_reference", lambda kernel: copy.deepcopy(refs[kernel])
    )
    monkeypatch.setattr(r5, "verify_grub_generated", lambda *args: {"status": "pass"})

    def fake_run(argv, **kwargs):
        if argv[0] == r5.BIN["mokutil"]:
            output = "SecureBoot enabled\n"
        elif argv[0] == r5.BIN["dkms"]:
            output = ""
        elif argv[0] == r5.BIN["uname"]:
            output = r5.FALLBACK_KERNEL + "\n"
        else:
            output = (
                "/dev/nvme1n1p1 ext4 rw,relatime 259:6 "
                "ef2c0d87-cd28-4bdb-8c16-58650fa2b31c\n"
            )
        return r5.CommandResult(tuple(argv), 0, output)

    monkeypatch.setattr(r5, "_run", fake_run)
    result = r5.verify_pre_reboot(real_plan)
    assert result["status"] == "blocked"
    assert result["checks"]["static_external_plan_hash_gate"] is False
    assert result["reboot_authorized"] is False
    assert result["gpu_workload_authorized"] is False


def test_post_reboot_rechecks_installed_invariants_on_target_not_fallback(
    real_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_pre(plan, **kwargs):
        observed.update(kwargs)
        return {"status": "pass"}

    def fake_run(argv, **kwargs):
        output = (
            r5.TARGET_KERNEL + "\n"
            if argv[0] == r5.BIN["uname"]
            else f"{r5.TARGET_GPU}, {r5.TARGET_DRIVER}\n"
        )
        return r5.CommandResult(tuple(argv), 0, output)

    monkeypatch.setattr(r5, "validate_plan_schema", lambda plan: {"status": "pass", "errors": []})
    monkeypatch.setattr(r5, "verify_pre_reboot", fake_pre)
    monkeypatch.setattr(r5, "_run", fake_run)
    monkeypatch.setattr(
        r5.Path,
        "read_text",
        lambda self, **kwargs: f"NVIDIA UNIX {r5.TARGET_DRIVER}",
    )
    result = r5.verify_post_reboot(real_plan)
    assert observed["_expected_running_kernel"] == r5.TARGET_KERNEL
    assert result["status"] == "pass"
    assert result["gpu_workload_authorized"] is False


def test_predecessor_evidence_uses_mechanical_39_not_claimed_44(
    real_plan: dict[str, object]
) -> None:
    assert real_plan["predecessor_test_evidence"] == {
        "frozen_r4_collected": 39,
        "frozen_r4_passed": 39,
        "claimed_44_rejected": True,
    }
