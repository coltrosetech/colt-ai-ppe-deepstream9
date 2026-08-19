from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import jsonschema
import pytest

from validation import driver595_live_qualification_r7_independent_review as review


@pytest.fixture(scope="session")
def frozen_inputs() -> dict[str, object]:
    value = review.verify_frozen_inputs()
    assert value["status"] == "pass"
    return value


@pytest.fixture(scope="session")
def live_snapshot() -> dict[str, object]:
    value = review.collect_live()
    assert value["status"] == "pass"
    return value


@pytest.fixture(scope="session")
def receipt() -> dict[str, object]:
    value, _ = review.load_json_strict(
        review.DEFAULT_REVIEW,
        expected_uid=review.os.getuid(),
        parent_uids={0, review.os.getuid()},
    )
    return value


def _resign(value: dict[str, object]) -> dict[str, object]:
    value["review_fingerprint_sha256"] = review.review_fingerprint(value)
    return value


def test_target_and_acceptance_authority_are_current_boot_only() -> None:
    assert review.TARGET_KERNEL == "7.0.0-28-generic"
    assert review.TARGET_DRIVER == "595.71.05"
    assert review.TARGET_GPU == "NVIDIA RTX A5000 Laptop GPU"
    assert review.ACCEPT_AUTHORITY == {
        "candidate_accepted_for_current_boot_only": True,
        "current_boot_runtime_prerequisite_accepted": True,
        "independent_acceptance_present": True,
        "terminal_current_boot_prerequisite_receipt": True,
        "historical_install_proven": False,
        "historical_install_authorized": False,
        "download_authorized": False,
        "install_authorized": False,
        "remove_authorized": False,
        "update_authorized": False,
        "reboot_authorized": False,
        "future_reboot_selection_accepted": False,
        "gpu_workload_authorized": False,
        "deepstream_runtime_authorized": False,
        "deepstream_runtime_validated": False,
        "quality_validated": False,
        "production_authorized": False,
        "same_uid_mutation_resistance_claimed": False,
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["/usr/bin/apt-get", "install", "nvidia-driver-595"],
        ["/usr/bin/apt-get", "download", "nvidia-driver-595"],
        ["/usr/bin/dpkg", "--install", "payload.deb"],
        ["/usr/sbin/update-initramfs", "-u"],
        ["/usr/sbin/update-grub"],
        ["/usr/sbin/reboot"],
        ["/usr/bin/nvidia-smi", "--gpu-reset"],
        ["/usr/bin/docker", "run", "--gpus", "all", "image"],
        ["/usr/bin/python3", "model.py"],
    ],
)
def test_command_grammar_rejects_mutation_network_container_and_workload(
    argv: list[str],
) -> None:
    assert review._allowed_command(argv) is False
    with pytest.raises(ValueError, match="outside read-only grammar"):
        review._run(argv)


def test_review_schema_is_valid_draft_2020_12() -> None:
    schema = json.loads(review.REVIEW_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_duplicate_and_nonfinite_json_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"a":1,"a":2}', object_pairs_hook=review._no_duplicate_object)
    with pytest.raises(ValueError, match="non-finite"):
        json.loads('{"a":NaN}', parse_constant=review._reject_constant)


def test_reviewer_source_has_no_author_import_or_author_execution() -> None:
    source = inspect.getsource(review)
    assert "from validation import driver595_live_qualification_r7" not in source
    assert "import driver595_live_qualification_r7" not in source
    assert "-m validation.driver595_live_qualification_r7" not in source
    assert "subprocess.run" in source


def test_frozen_candidate_and_handoff_exact_replay(frozen_inputs: dict[str, object]) -> None:
    assert frozen_inputs["status"] == "pass"
    assert all(frozen_inputs["candidate_checks"].values())
    assert all(frozen_inputs["handoff_checks"].values())
    assert frozen_inputs["candidate_pin"]["sha256"] == review.EXPECTED_CANDIDATE_PIN[2]
    assert frozen_inputs["handoff_pin"]["sha256"] == review.EXPECTED_HANDOFF_PIN[2]


def test_external_subject_pin_authority_is_exact(frozen_inputs: dict[str, object]) -> None:
    candidate = frozen_inputs["candidate"]
    assert review._candidate_subject_projection(candidate) == review.EXPECTED_SUBJECT_PINS
    for relative, expected in review.EXPECTED_SUBJECT_PINS.items():
        assert review._pin_projection(review.artifact_pin(relative)) == expected


def test_module_signature_parser_requires_one_terminal_der_signature() -> None:
    signature = b"0" + b"x" * 63
    header = bytes([0, 0, 2, 0, 0, 0, 0, 0]) + len(signature).to_bytes(4, "big")
    fixture = b"ELF" + signature + header + review.MODULE_SIGNATURE_MARKER
    content, observed, projection = review.split_module_signature(fixture)
    assert content == b"ELF"
    assert observed == signature
    assert projection["id_type"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        review.split_module_signature(review.MODULE_SIGNATURE_MARKER + fixture)


def test_initramfs_parser_is_strict_about_kernel_root_and_duplicates() -> None:
    root = f"usr/lib/modules/{review.TARGET_KERNEL}"
    valid = "\n".join(
        [
            ".",
            "init",
            root,
            f"{root}/modules.dep",
            f"{root}/modules.alias",
            f"{root}/kernel/drivers/nvme/host/nvme.ko.zst",
            f"{root}/kernel/drivers/nvme/host/nvme-core.ko.zst",
        ]
    ) + "\n"
    assert review.inspect_initramfs_listing(valid)["status"] == "pass"
    assert review.inspect_initramfs_listing(valid + "usr/lib/modules/evil/x.ko\n")["status"] == "blocked"
    assert review.inspect_initramfs_listing(valid + f"{root}/modules.dep\n")["status"] == "blocked"


def test_mountinfo_and_dpkg_duplicate_rows_are_rejected() -> None:
    mount = "33 2 259:6 / / rw - ext4 /dev/nvme1n1p1 rw\n"
    with pytest.raises(ValueError, match="duplicate mount"):
        review.parse_mountinfo(mount + mount)
    package = "installed\tpkg\t1\tamd64\n"
    with pytest.raises(ValueError, match="duplicate dpkg"):
        review.parse_dpkg_rows(package + package)


def test_gpu_xml_rejects_unknown_process_type() -> None:
    raw = b"""<nvidia_smi_log><driver_version>595.71.05</driver_version><cuda_version>13.2</cuda_version>
    <gpu id='00000000:01:00.0'><product_name>NVIDIA RTX A5000 Laptop GPU</product_name>
    <uuid>GPU-x</uuid><pci><pci_bus_id>00000000:01:00.0</pci_bus_id></pci>
    <processes><process_info><pid>1</pid><type>X</type><process_name>x</process_name></process_info></processes></gpu></nvidia_smi_log>"""
    with pytest.raises(ValueError, match="malformed GPU process"):
        review.parse_gpu_xml(raw)


def test_independent_live_snapshot_all_sections_pass(live_snapshot: dict[str, object]) -> None:
    assert live_snapshot["status"] == "pass"
    assert all(live_snapshot["checks"].values())
    assert all(review.evaluate_live_snapshot(live_snapshot).values())


def test_live_boot_secure_boot_and_mount_identity(live_snapshot: dict[str, object]) -> None:
    boot = live_snapshot["boot_mounts"]
    secure = live_snapshot["secure_boot"]
    assert boot["boot_id"] == review.EXPECTED_BOOT_ID
    assert boot["kernel"] == review.TARGET_KERNEL
    assert boot["root_mount"]["mount_source"] == review.TARGET_ROOT_SOURCE
    assert boot["efi_mount"]["mount_source"] == review.TARGET_EFI_SOURCE
    assert secure["status"] == "pass"
    assert secure["mok"]["sha256"] == review.MOK_SHA256


def test_live_exact_five_signed_modules_and_four_loaded(live_snapshot: dict[str, object]) -> None:
    modules = live_snapshot["modules"]
    assert modules["entries"] == sorted(review.MODULES.values())
    assert modules["loaded_modules"] == sorted(review.REQUIRED_LOADED_MODULES)
    assert set(modules["rows"]) == set(review.MODULES)
    assert all(row["status"] == "pass" for row in modules["rows"].values())
    assert all(row["checks"]["cms_exact_mok"] for row in modules["rows"].values())


def test_live_packages_dkms_runtime_links_and_real_owners(live_snapshot: dict[str, object]) -> None:
    packages = live_snapshot["packages"]
    libraries = live_snapshot["runtime_libraries"]
    assert packages["status"] == "pass"
    assert set(packages["core"]) == set(review.CORE_PACKAGES)
    assert libraries["status"] == "pass"
    assert len(libraries["rows"]) == 4
    assert all(row["checks"]["owner_exact"] for row in libraries["rows"])


def test_live_initramfs_has_root_support_but_no_future_reboot_acceptance(
    live_snapshot: dict[str, object],
) -> None:
    initramfs = live_snapshot["initramfs"]
    assert initramfs["status"] == "pass"
    assert initramfs["config_values"] == {
        "CONFIG_EXT4_FS": "y",
        "CONFIG_NVME_CORE": "m",
        "CONFIG_BLK_DEV_NVME": "m",
    }
    assert initramfs["listing"]["wrong_kernel_roots"] == []
    assert initramfs["future_reboot_selection_accepted"] is False


def test_live_xorg_is_healthy_and_compute_is_empty(live_snapshot: dict[str, object]) -> None:
    graphics = live_snapshot["graphics"]
    assert graphics["status"] == "pass"
    assert graphics["driver_version"] == review.TARGET_DRIVER
    assert graphics["xorg"][0]["exe"] == "/usr/lib/xorg/Xorg"
    assert graphics["compute_before"] == []
    assert graphics["compute_after"] == []
    assert graphics["compute_xml"] == []


def test_frozen_candidate_matches_separate_live_collector(
    frozen_inputs: dict[str, object], live_snapshot: dict[str, object]
) -> None:
    comparison = review.compare_candidate_live(frozen_inputs["candidate"], live_snapshot)
    assert set(comparison) == {
        "boot_identity", "kernel_cmdline_mounts", "secure_boot_and_mok", "modules",
        "packages", "runtime_libraries", "initramfs", "graphics",
    }
    assert all(comparison.values())


def test_real_receipt_verifies_as_narrow_accept(receipt: dict[str, object]) -> None:
    result = review.verify_review(receipt)
    assert result["verification_status"] == "pass"
    assert result["decision"] == "ACCEPT"
    assert result["current_boot_runtime_prerequisite_accepted"] is True
    for field in (
        "historical_install_proven", "download_authorized", "install_authorized",
        "remove_authorized", "update_authorized", "reboot_authorized",
        "future_reboot_selection_accepted", "gpu_workload_authorized",
        "deepstream_runtime_authorized", "deepstream_runtime_validated",
        "production_authorized",
    ):
        assert result[field] is False


@pytest.mark.parametrize(
    "field",
    [
        "install_authorized", "remove_authorized", "update_authorized",
        "reboot_authorized", "gpu_workload_authorized", "deepstream_runtime_authorized",
    ],
)
def test_authority_escalation_is_rejected_even_after_refingerprint(
    receipt: dict[str, object], field: str
) -> None:
    value = copy.deepcopy(receipt)
    value["authority"][field] = True
    result = review.verify_review(_resign(value))
    assert result["verification_status"] == "blocked"
    assert result["install_authorized"] is False
    assert result["gpu_workload_authorized"] is False


def test_candidate_or_handoff_pin_drift_is_rejected(receipt: dict[str, object]) -> None:
    for section in ("candidate", "handoff"):
        value = copy.deepcopy(receipt)
        value[section]["artifact"]["sha256"] = "0" * 64
        result = review.verify_review(_resign(value))
        assert result["verification_status"] == "blocked"


def test_subject_or_review_source_pin_drift_is_rejected(receipt: dict[str, object]) -> None:
    for section in ("subject_pins", "review_source_pins"):
        value = copy.deepcopy(receipt)
        value[section][0]["sha256"] = "f" * 64
        result = review.verify_review(_resign(value))
        assert result["verification_status"] == "blocked"


def test_decision_control_or_comparison_tamper_is_rejected(receipt: dict[str, object]) -> None:
    mutations = [
        ("decision", None),
        ("controls", "same_boot_identity"),
        ("candidate_live_comparison", "modules"),
    ]
    for section, key in mutations:
        value = copy.deepcopy(receipt)
        if key is None:
            value[section] = "REJECT"
        else:
            value[section][key] = False
        assert review.verify_review(_resign(value))["verification_status"] == "blocked"


def test_recorded_stale_boot_or_module_tamper_is_rejected(receipt: dict[str, object]) -> None:
    value = copy.deepcopy(receipt)
    value["live_observation"]["boot_mounts"]["boot_id"] = "00000000-0000-0000-0000-000000000000"
    assert review.verify_review(_resign(value))["verification_status"] == "blocked"
    value = copy.deepcopy(receipt)
    value["live_observation"]["modules"]["rows"]["nvidia"]["file"]["sha256"] = "0" * 64
    assert review.verify_review(_resign(value))["verification_status"] == "blocked"


def test_recorded_owner_initramfs_or_compute_tamper_is_rejected(receipt: dict[str, object]) -> None:
    mutations = []
    owner = copy.deepcopy(receipt)
    owner["live_observation"]["runtime_libraries"]["rows"][0]["checks"]["owner_exact"] = False
    mutations.append(owner)
    initramfs = copy.deepcopy(receipt)
    initramfs["live_observation"]["initramfs"]["config_values"]["CONFIG_EXT4_FS"] = "m"
    mutations.append(initramfs)
    compute = copy.deepcopy(receipt)
    compute["live_observation"]["graphics"]["compute_after"] = ["123, evil, GPU-x"]
    mutations.append(compute)
    for value in mutations:
        assert review.verify_review(_resign(value))["verification_status"] == "blocked"


def test_unknown_root_and_nested_authority_fields_are_rejected(receipt: dict[str, object]) -> None:
    value = copy.deepcopy(receipt)
    value["unexpected"] = True
    value["authority"]["unexpected"] = False
    result = review.verify_review(_resign(value))
    assert result["verification_status"] == "blocked"
    assert result["schema_error_count"] >= 2


def test_nested_tamper_without_refingerprint_is_rejected(receipt: dict[str, object]) -> None:
    value = copy.deepcopy(receipt)
    value["next_gate"]["required_next"] = "attacker"
    result = review.verify_review(value)
    assert result["verification_status"] == "blocked"
    assert any("fingerprint" in failure for failure in result["failures"])


def test_severity_projection_must_remain_zero(receipt: dict[str, object]) -> None:
    value = copy.deepcopy(receipt)
    value["severity_counts"]["P2"] = 1
    assert review.verify_review(_resign(value))["verification_status"] == "blocked"


def test_live_replay_passes_with_independent_same_snapshot(
    receipt: dict[str, object], live_snapshot: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(review, "collect_live", lambda: copy.deepcopy(live_snapshot))
    result = review.verify_review(receipt, replay_live=True)
    assert result["verification_status"] == "pass"
    assert result["live_replay"]["status"] == "pass"


def test_live_replay_rejects_boot_change(
    receipt: dict[str, object], live_snapshot: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = copy.deepcopy(live_snapshot)
    changed["boot_mounts"]["boot_id"] = "00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr(review, "collect_live", lambda: changed)
    result = review.verify_review(receipt, replay_live=True)
    assert result["verification_status"] == "blocked"
    assert result["live_replay"]["status"] == "blocked"


def test_receipt_declares_exact_author_and_independent_replay_counts(receipt: dict[str, object]) -> None:
    replay = receipt["test_replay"]
    assert (replay["author_collected"], replay["author_passed"], replay["author_failed"]) == (35, 35, 0)
    assert (replay["independent_collected"], replay["independent_passed"], replay["independent_failed"]) == (
        review.INDEPENDENT_TEST_COUNT,
        review.INDEPENDENT_TEST_COUNT,
        0,
    )


def test_receipt_fingerprint_is_canonical_and_exact(receipt: dict[str, object]) -> None:
    assert receipt["review_fingerprint_sha256"] == review.review_fingerprint(receipt)
    assert len(review.canonical_json(receipt)) > 1000
