from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import jsonschema
import pytest

from validation import driver595_live_qualification_r7 as r7


@pytest.fixture(scope="session")
def live_candidate() -> dict[str, object]:
    value = r7.collect_candidate()
    assert value["status"] == "live_candidate_pass_pending_independent_review"
    assert value["evaluation"]["status"] == "pass"
    return value


def _reevaluate(value: dict[str, object]) -> dict[str, object]:
    value["evaluation"] = r7.evaluate_candidate(value)
    value["receipt_fingerprint_sha256"] = r7.receipt_fingerprint(value)
    return value


def test_constants_pin_current_host_target_not_historical_preinstall() -> None:
    assert r7.TARGET_DRIVER == "595.71.05"
    assert r7.TARGET_KERNEL == "7.0.0-28-generic"
    assert r7.TARGET_GPU == "NVIDIA RTX A5000 Laptop GPU"
    assert r7.CANDIDATE_AUTHORITY == {
        "observation_only": True,
        "independent_acceptance_present": False,
        "historical_install_authorized": False,
        "download_authorized": False,
        "install_authorized": False,
        "remove_authorized": False,
        "update_authorized": False,
        "reboot_authorized": False,
        "gpu_workload_authorized": False,
        "deepstream_runtime_authorized": False,
        "terminal_root_published": False,
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["/usr/bin/apt-get", "install", "nvidia-driver-595"],
        ["/usr/bin/apt-get", "download", "nvidia-driver-595"],
        ["/usr/bin/dpkg", "--install", "attacker.deb"],
        ["/usr/sbin/update-initramfs", "-u"],
        ["/usr/sbin/update-grub"],
        ["/usr/sbin/reboot"],
        ["/usr/bin/nvidia-smi", "--gpu-reset"],
        ["/usr/bin/docker", "run", "nvidia/cuda"],
        ["/usr/bin/python3", "model.py"],
    ],
)
def test_read_only_command_grammar_rejects_mutation_and_workload(argv: list[str]) -> None:
    assert r7._allowed_command(argv) is False
    with pytest.raises(ValueError):
        r7._run(argv)


def test_schema_is_valid_draft_2020_12() -> None:
    value = json.loads(r7.SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(value)


def test_duplicate_json_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"authority":{},"authority":{}}', object_pairs_hook=r7._no_duplicate_object)


def test_dpkg_parser_rejects_duplicate_package() -> None:
    row = "installed\tpkg\t1\tamd64\n"
    with pytest.raises(ValueError, match="duplicate package"):
        r7.parse_dpkg_rows(row + row)


def test_mountinfo_parser_rejects_duplicate_mount_id() -> None:
    row = "33 2 259:6 / / rw - ext4 /dev/nvme1n1p1 rw\n"
    with pytest.raises(ValueError, match="duplicate mount ID"):
        r7.parse_mountinfo(row + row)


def test_gpu_xml_rejects_unknown_process_type() -> None:
    raw = b"""<nvidia_smi_log><driver_version>595.71.05</driver_version>
    <gpu id='00000000:01:00.0'><product_name>NVIDIA RTX A5000 Laptop GPU</product_name>
    <uuid>GPU-x</uuid><pci><pci_bus_id>00000000:01:00.0</pci_bus_id></pci>
    <processes><process_info><pid>1</pid><type>X</type><process_name>x</process_name></process_info></processes>
    </gpu></nvidia_smi_log>"""
    with pytest.raises(ValueError, match="invalid GPU process"):
        r7.parse_gpu_xml(raw)


def test_module_signature_parser_accepts_one_terminal_marker_and_rejects_two() -> None:
    signature = b"0" + b"x" * 63
    header = bytes([0, 0, 2, 0, 0, 0, 0, 0]) + len(signature).to_bytes(4, "big")
    fixture = b"ELF-content" + signature + header + r7.MODULE_SIGNATURE_MARKER
    content, parsed, projection = r7.split_module_signature(fixture)
    assert content == b"ELF-content"
    assert parsed == signature
    assert projection["id_type"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        r7.split_module_signature(r7.MODULE_SIGNATURE_MARKER + fixture)


def test_initramfs_parser_rejects_wrong_kernel_root_and_duplicate_required_member() -> None:
    root = f"usr/lib/modules/{r7.TARGET_KERNEL}"
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
    assert r7.inspect_initramfs_listing(valid)["status"] == "pass"
    wrong = valid + "usr/lib/modules/attacker/kernel/x.ko.zst\n"
    assert r7.inspect_initramfs_listing(wrong)["status"] == "blocked"
    duplicate = valid + f"{root}/modules.dep\n"
    assert r7.inspect_initramfs_listing(duplicate)["status"] == "blocked"


def test_live_candidate_has_exact_current_boot_and_runtime(live_candidate: dict[str, object]) -> None:
    evidence = live_candidate["evidence"]
    assert evidence["boot_mounts"]["uname_release"] == r7.TARGET_KERNEL
    assert evidence["graphics"]["query_rows"] == [
        f"0, {r7.TARGET_GPU}, {r7.TARGET_UUID}, {r7.TARGET_DRIVER}"
    ]
    assert evidence["graphics"]["compute_processes"] == []
    assert evidence["modules"]["status"] == "pass"
    assert evidence["runtime_libraries"]["status"] == "pass"


def test_live_candidate_secure_boot_and_all_five_module_crypto_pass(
    live_candidate: dict[str, object],
) -> None:
    evidence = live_candidate["evidence"]
    assert evidence["secure_boot"]["status"] == "pass"
    assert evidence["secure_boot"]["mok"]["sha256"] == r7.MOK_DER_SHA256
    rows = evidence["modules"]["modules"]
    assert set(rows) == set(r7.MODULES)
    assert all(row["crypto"]["status"] == "pass" for row in rows.values())


def test_live_candidate_target_initramfs_is_strict_but_reboot_stays_unaccepted(
    live_candidate: dict[str, object],
) -> None:
    evidence = live_candidate["evidence"]
    assert evidence["initramfs"]["status"] == "pass"
    assert evidence["initramfs"]["listing"]["wrong_kernel_roots"] == []
    assert evidence["boot_mounts"]["grub"]["future_reboot_selection_accepted"] is False
    assert live_candidate["authority"]["reboot_authorized"] is False


def test_dormant_590_packages_are_visible_but_not_promoted(
    live_candidate: dict[str, object],
) -> None:
    packages = live_candidate["evidence"]["packages"]
    assert packages["legacy_590_installed_but_not_selected"]
    target = [line for line in packages["dkms_lines"] if f", {r7.TARGET_KERNEL}, " in line]
    assert target == [f"nvidia/{r7.TARGET_DRIVER}, {r7.TARGET_KERNEL}, x86_64: installed"]
    assert r7.evaluate_candidate(live_candidate)["status"] == "pass"


def test_stale_boot_is_rejected_even_after_refingerprinting(
    live_candidate: dict[str, object],
) -> None:
    value = copy.deepcopy(live_candidate)
    value["evidence"]["boot_mounts"]["uname_release"] = "6.17.0-35-generic"
    _reevaluate(value)
    assert value["evaluation"]["checks"]["boot_mounts"] is False
    assert value["evaluation"]["status"] == "blocked"


def test_loaded_module_version_skew_is_rejected(
    live_candidate: dict[str, object],
) -> None:
    value = copy.deepcopy(live_candidate)
    value["evidence"]["modules"]["modules"]["nvidia"]["loaded_version"] = "590.48.01"
    _reevaluate(value)
    assert value["evaluation"]["checks"]["modules"] is False


def test_unsigned_module_projection_is_rejected(
    live_candidate: dict[str, object],
) -> None:
    value = copy.deepcopy(live_candidate)
    crypto = value["evidence"]["modules"]["modules"]["nvidia_uvm"]["crypto"]
    crypto["checks"]["detached_pkcs7_valid_with_exact_mok"] = False
    _reevaluate(value)
    assert value["evaluation"]["checks"]["modules"] is False


def test_wrong_mok_key_is_rejected(live_candidate: dict[str, object]) -> None:
    value = copy.deepcopy(live_candidate)
    value["evidence"]["secure_boot"]["mok"]["sha256"] = "f" * 64
    _reevaluate(value)
    assert value["evaluation"]["checks"]["secure_boot"] is False


def test_wrong_initramfs_kernel_is_rejected(live_candidate: dict[str, object]) -> None:
    value = copy.deepcopy(live_candidate)
    value["evidence"]["initramfs"]["path"] = "/boot/initrd.img-6.17.0-35-generic"
    _reevaluate(value)
    assert value["evaluation"]["checks"]["initramfs"] is False


def test_current_590_runtime_smuggling_is_rejected(live_candidate: dict[str, object]) -> None:
    value = copy.deepcopy(live_candidate)
    row = value["evidence"]["runtime_libraries"]["libraries"][
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    ]
    row["resolved"] = "/usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01"
    _reevaluate(value)
    assert value["evaluation"]["checks"]["runtime_libraries"] is False


def test_approval_substitution_never_authorizes_new_action(
    live_candidate: dict[str, object],
) -> None:
    value = copy.deepcopy(live_candidate)
    value["evidence"]["boundaries"]["user_approval"][
        "new_maintenance_action_authorized"
    ] = True
    value["authority"]["install_authorized"] = True
    _reevaluate(value)
    assert value["evaluation"]["checks"]["approval_not_substituted"] is False
    assert value["evaluation"]["checks"]["authority_exact_nonoperational"] is False


def test_cache_receipt_substitution_is_rejected(live_candidate: dict[str, object]) -> None:
    value = copy.deepcopy(live_candidate)
    value["evidence"]["boundaries"]["cache"]["accepted"] = True
    _reevaluate(value)
    assert value["evaluation"]["checks"]["cache_not_promoted"] is False


def test_historical_r6_preinstall_evidence_cannot_be_promoted(
    live_candidate: dict[str, object],
) -> None:
    value = copy.deepcopy(live_candidate)
    value["evidence"]["boundaries"]["historical_r6"][
        "accepted_terminal_root_present"
    ] = True
    _reevaluate(value)
    assert value["evaluation"]["checks"]["historical_r6_not_promoted"] is False


def test_receipt_fingerprint_detects_nested_tamper(live_candidate: dict[str, object]) -> None:
    value = copy.deepcopy(live_candidate)
    before = value["receipt_fingerprint_sha256"]
    value["evidence"]["graphics"]["query_rows"][0] = "attacker"
    assert r7.receipt_fingerprint(value) != before


def test_schema_rejects_unknown_root_and_nested_authority_fields(
    live_candidate: dict[str, object],
) -> None:
    schema = json.loads(r7.SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    root_extra = copy.deepcopy(live_candidate)
    root_extra["attacker"] = True
    assert list(validator.iter_errors(root_extra))
    nested_extra = copy.deepcopy(live_candidate)
    nested_extra["authority"]["attacker"] = True
    assert list(validator.iter_errors(nested_extra))


def test_source_does_not_import_or_execute_historical_r6() -> None:
    source = inspect.getsource(r7)
    assert "from validation import driver595_maintenance_r6" not in source
    assert "import driver595_maintenance_r6" not in source
    assert "subprocess" in source


def test_candidate_validator_passes_only_as_nonterminal_author_candidate(
    live_candidate: dict[str, object],
) -> None:
    report = r7.validate_candidate(live_candidate)
    assert report["status"] == "pass"
    assert report["independent_acceptance_present"] is False
    assert report["install_authorized"] is False
    assert report["remove_authorized"] is False
    assert report["reboot_authorized"] is False
    assert report["gpu_workload_authorized"] is False


def test_subject_path_set_is_exact() -> None:
    assert r7.SUBJECT_PATHS == {
        "validation/driver595_live_qualification_r7.py",
        "validation/schemas/driver595-live-qualification-r7.schema.json",
        "tests/test_driver595_live_qualification_r7.py",
        "docs/deepstream91-driver595-live-qualification-r7.md",
    }
