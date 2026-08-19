import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import validation.gpu_reentry_evidence as reentry


NOW = datetime(2026, 7, 16, 3, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declaration(*, confirmed=True):
    payload = reentry.declaration_template()
    if not confirmed:
        return payload
    timestamp = NOW.isoformat()
    payload.update({"declared_by": "test-operator", "declared_at_utc": timestamp})
    payload["bios_thermal_management"].update(
        {"observed_value": "UltraPerformance", "observed_at_utc": timestamp}
    )
    payload["adapter"].update(
        {
            "original_dell_adapter_confirmed": True,
            "direct_barrel_connection_confirmed": True,
            "rated_output_w": 240,
            "output_voltage_v": 19.5,
            "output_current_a": 12.3,
        }
    )
    payload["cooling_and_epsa"].update(
        {
            "air_inlet_and_exhaust_clear_confirmed": True,
            "machine_elevated_for_airflow_confirmed": True,
            "epsa_thermal_test_completed": True,
            "epsa_result": "pass",
            "checked_at_utc": timestamp,
        }
    )
    return payload


def _valid_evidence(
    tmp_path,
    monkeypatch,
    *,
    confirmed=True,
    operating_policy_id=reentry.LEGACY_STRICT_PHYSICAL_POLICY_ID,
):
    project = tmp_path / "project"
    project.mkdir()
    guard_paths = ("validation/guard-a.py", "validation/guard-b.py")
    for relative in guard_paths:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")

    incident_paths = {
        "validation/results/incident/gpu.csv": b"gpu evidence\n",
        "validation/results/incident/deepstream.log": b"log evidence\n",
        "validation/results/incident/status.json": b"{}\n",
    }
    incident_hashes = {}
    for relative, content in incident_paths.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        incident_hashes[relative] = _hash(path)
    incident_doc = "docs/incident.md"
    doc_path = project / incident_doc
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("incident\n", encoding="utf-8")

    monkeypatch.setattr(reentry, "GUARD_PATHS", guard_paths)
    monkeypatch.setattr(reentry, "INCIDENT_SOURCE_HASHES", incident_hashes)
    monkeypatch.setattr(reentry, "INCIDENT_DOCUMENT_PATH", incident_doc)

    declaration = _declaration(confirmed=confirmed)
    declaration_path = project / "validation/results/reentry/operator.json"
    declaration_path.parent.mkdir(parents=True, exist_ok=True)
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")

    snapshot = {
        "timestamp": "2026/07/16 06:00:00.000",
        "gpu_index": "0",
        "gpu_name": "NVIDIA RTX A5000 Laptop GPU",
        "gpu_utilization_percent": "0",
        "memory_utilization_percent": "0",
        "memory_used_mib": "512",
        "memory_total_mib": "16384",
        "temperature_c": "48",
        "power_draw_w": "18.00",
        "sm_clock_mhz": "210",
        "memory_clock_mhz": "405",
        "power_requested_limit_w": "[N/A]",
        "power_current_limit_w": "115.00",
        "power_default_limit_w": "115.00",
        "pstate": "P8",
        "clock_event_reasons_active_mask": "0x0000000000000000",
        "clock_event_sw_power_cap": "Not Active",
        "clock_event_sw_thermal_slowdown": "Not Active",
        "clock_event_hw_slowdown": "Not Active",
        "clock_event_hw_thermal_slowdown": "Not Active",
        "clock_event_hw_power_brake_slowdown": "Not Active",
    }
    assessment = reentry.assess_gpu_safety(
        snapshot,
        power_limit_drop_tolerance_w=reentry.POWER_LIMIT_DROP_TOLERANCE_W,
    )
    evidence = {
        "schema_version": reentry.SCHEMA_VERSION,
        "collected_at_utc": NOW.isoformat(),
        "operating_policy": reentry.operating_policy_contract(operating_policy_id),
        "collection_policy": {"operating_policy_id": operating_policy_id},
        "gpu_index": 0,
        "gpu_identity": {
            "available": True,
            "fields": {
                "index": "0",
                "name": "NVIDIA RTX A5000 Laptop GPU",
                "driver_version": "590.48.01",
                "uuid": "GPU-test",
                "pci.bus_id": "00000000:01:00.0",
                "memory.total": "16384",
            },
        },
        "compute_processes": {
            "available": True,
            "processes": [],
            "count": 0,
            "error": None,
            "read_only_command": [
                "nvidia-smi",
                "-i",
                "0",
                "--query-compute-apps=pid,process_name,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
        },
        "idle_gpu_telemetry": {
            "available": True,
            "samples": [
                {"snapshot": dict(snapshot), "safety_assessment": dict(assessment)},
                {"snapshot": dict(snapshot), "safety_assessment": dict(assessment)},
            ],
        },
        "bios_thermal_management": {
            "path": str(reentry.THERMAL_MANAGEMENT_PATH),
            "readable": False,
            "value": None,
            "error": "PermissionError: root-only test fixture",
            "evidence_missing_command": reentry.THERMAL_MANAGEMENT_SUDO_COMMAND,
        },
        "linux_power_profiles": {
            "powerprofilesctl": {"available": True, "value": "performance"},
            "platform_profile": {"readable": True, "value": "performance"},
        },
        "ac_power": {"available": True, "mains_online": True},
        "platform_thermal": {
            "available": True,
            "values": {
                "dell_fan1_rpm": 1800,
                "dell_fan2_rpm": 1820,
                "thermal_tvga_c": 48,
                "thermal_tcpu_c": 52,
                "thermal_tskn_c": 45,
            },
        },
        "xid_current_boot": {
            "available": True,
            "source": "journalctl",
            "count": 0,
            "lines": [],
            "read_only_command": [
                "journalctl",
                "--quiet",
                "-k",
                "-b",
                "--no-pager",
                "--grep",
                "NVRM: Xid",
            ],
            "errors": [],
        },
        "operator_declaration_file": {
            "path": str(declaration_path.relative_to(project)),
            "present": True,
            "sha256": _hash(declaration_path),
            "error": None,
        },
        "operator_declaration": declaration,
        "guard_code": reentry._file_records(project, guard_paths),
        "incident_provenance": reentry.collect_incident_provenance(project),
    }
    return project, declaration_path, evidence


def _strict_dell_mapping(tmp_path, monkeypatch):
    source_root = tmp_path / "strict-dell-sources"
    source_root.mkdir(exist_ok=True)
    module_root = source_root / "dell_pc"
    module_root.mkdir(exist_ok=True)
    paths = {
        "SYS_VENDOR_PATH": source_root / "sys_vendor",
        "KERNEL_OSRELEASE_PATH": source_root / "osrelease",
        "DELL_PC_MODULE_SRCVERSION_PATH": module_root / "srcversion",
        "DELL_PC_MODULE_INITSTATE_PATH": module_root / "initstate",
        "DELL_PLATFORM_PROFILE_PROVIDER_NAME_PATH": source_root / "provider_name",
        "DELL_PLATFORM_PROFILE_PROVIDER_PROFILE_PATH": source_root / "provider_profile",
        "PLATFORM_PROFILE_PATH": source_root / "acpi_platform_profile",
        "THERMAL_MANAGEMENT_POSSIBLE_VALUES_PATH": source_root / "possible_values",
    }
    values = {
        "SYS_VENDOR_PATH": "Dell Inc.\n",
        "KERNEL_OSRELEASE_PATH": "6.17.0-35-generic\n",
        "DELL_PC_MODULE_SRCVERSION_PATH": "370814FE904C30776223695\n",
        "DELL_PC_MODULE_INITSTATE_PATH": "live\n",
        "DELL_PLATFORM_PROFILE_PROVIDER_NAME_PATH": "dell-pc\n",
        "DELL_PLATFORM_PROFILE_PROVIDER_PROFILE_PATH": "performance\n",
        "PLATFORM_PROFILE_PATH": "performance\n",
        "THERMAL_MANAGEMENT_POSSIBLE_VALUES_PATH": (
            "Optimized;Cool;Quiet;UltraPerformance;\n"
        ),
    }
    for name, path in paths.items():
        path.write_text(values[name], encoding="utf-8")
        monkeypatch.setattr(reentry, name, path)
    monkeypatch.setattr(reentry, "DELL_PC_MODULE_ROOT", module_root)
    return reentry.collect_dell_platform_profile_mapping()


def _bios_gate(verification):
    return next(
        item
        for item in verification["gates"]
        if item["id"] == "bios_thermal_management"
    )


def test_complete_current_bundle_is_only_ready_for_operator_review(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(tmp_path, monkeypatch)

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "ready_for_operator_review"
    assert verification["all_required_evidence_present"] is True
    assert verification["sustained_load_authorized"] is False
    assert all(gate["passed"] for gate in verification["gates"])


def test_null_template_never_fabricates_operator_confirmations(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(
        tmp_path, monkeypatch, confirmed=False
    )

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "blocked"
    assert {
        "bios_thermal_management",
        "original_240w_adapter_declaration",
        "cooling_and_epsa_acknowledgement",
    }.issubset(verification["failed_gate_ids"])
    template = reentry.declaration_template()
    assert template["declared_by"] is None
    assert template["adapter"]["original_dell_adapter_confirmed"] is None
    assert template["cooling_and_epsa"]["epsa_result"] is None
    assert verification["operating_policy"]["id"] == (
        reentry.LEGACY_STRICT_PHYSICAL_POLICY_ID
    )
    by_id = {gate["id"]: gate for gate in verification["gates"]}
    assert all(by_id[gate_id]["required"] for gate_id in reentry.LEGACY_REQUIRED_GATE_IDS)


def test_legacy_v1_bundle_without_explicit_policy_keeps_strict_compatible_semantics(
    tmp_path, monkeypatch
):
    project, _, evidence = _valid_evidence(tmp_path, monkeypatch, confirmed=True)
    evidence.pop("operating_policy")
    evidence.pop("collection_policy")

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "ready_for_operator_review"
    assert verification["operating_policy"]["id"] == (
        reentry.LEGACY_STRICT_PHYSICAL_POLICY_ID
    )
    assert all(gate["required"] for gate in verification["gates"])


def test_strict_dell_platform_mapping_can_satisfy_only_the_bios_gate(
    tmp_path, monkeypatch
):
    project, _, evidence = _valid_evidence(
        tmp_path, monkeypatch, confirmed=False
    )
    mapping = _strict_dell_mapping(tmp_path, monkeypatch)
    evidence["bios_thermal_management"]["dell_platform_profile_mapping"] = mapping

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    gate = _bios_gate(verification)
    assert mapping["strict_conditions_met"] is True
    assert mapping["failed_conditions"] == []
    assert (
        mapping["schema_version"]
        == reentry.DELL_PLATFORM_PROFILE_MAPPING_SCHEMA_VERSION
    )
    assert mapping["mapping_source"] == reentry.DELL_PLATFORM_PROFILE_MAPPING_SOURCE
    assert mapping["mapping_source"]["source_commit"] == (
        "e5f0a698b34ed76002dc5cff3804a61c80233a7a"
    )
    assert mapping["observations"]["provider_name"]["path"] == str(
        reentry.DELL_PLATFORM_PROFILE_PROVIDER_NAME_PATH
    )
    assert gate["passed"] is True
    assert "strict Dell mapping" in gate["detail"]
    assert "bios_thermal_management" not in verification["failed_gate_ids"]
    assert {
        "original_240w_adapter_declaration",
        "cooling_and_epsa_acknowledgement",
    }.issubset(verification["failed_gate_ids"])


def test_generic_linux_performance_profiles_never_satisfy_bios_gate(
    tmp_path, monkeypatch
):
    project, _, evidence = _valid_evidence(
        tmp_path, monkeypatch, confirmed=False
    )
    assert evidence["linux_power_profiles"]["powerprofilesctl"]["value"] == "performance"
    assert evidence["linux_power_profiles"]["platform_profile"]["value"] == "performance"

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert _bios_gate(verification)["passed"] is False
    assert "bios_thermal_management" in verification["failed_gate_ids"]


@pytest.mark.parametrize(
    ("record_name", "field", "bad_value"),
    (
        ("sys_vendor", "value", "Example Computer Inc."),
        ("kernel_osrelease", "value", "6.18.0-generic"),
        ("module_srcversion", "value", "UNREVIEWEDMODULEBUILD"),
        ("module_initstate", "value", "coming"),
        ("provider_name", "value", "generic-platform-profile"),
        ("provider_profile", "value", "balanced"),
        ("acpi_platform_profile", "value", "balanced"),
        (
            "wmi_possible_values",
            "value",
            "Optimized;Cool;Quiet;NotUltraPerformance;",
        ),
        ("provider_name", "path", "/tmp/untrusted-provider-name"),
    ),
)
def test_dell_fallback_requires_every_exact_source_condition(
    tmp_path, monkeypatch, record_name, field, bad_value
):
    project, _, evidence = _valid_evidence(
        tmp_path, monkeypatch, confirmed=False
    )
    mapping = _strict_dell_mapping(tmp_path, monkeypatch)
    mapping["observations"][record_name][field] = bad_value
    # A forged derived status must not bypass independent raw-source checks.
    mapping["strict_conditions_met"] = True
    mapping["failed_conditions"] = []
    evidence["bios_thermal_management"]["dell_platform_profile_mapping"] = mapping

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert _bios_gate(verification)["passed"] is False
    assert "bios_thermal_management" in verification["failed_gate_ids"]


def test_dell_fallback_rejects_absent_dell_pc_module(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(
        tmp_path, monkeypatch, confirmed=False
    )
    mapping = _strict_dell_mapping(tmp_path, monkeypatch)
    mapping["observations"]["module_name"].update(
        {
            "readable": False,
            "value": None,
            "error": "FileNotFoundError: /sys/module/dell_pc is missing",
        }
    )
    mapping["strict_conditions_met"] = True
    mapping["failed_conditions"] = []
    evidence["bios_thermal_management"]["dell_platform_profile_mapping"] = mapping

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    gate = _bios_gate(verification)
    assert gate["passed"] is False
    assert "module_name" in gate["detail"] or "module" in gate["detail"]


def test_dell_fallback_requires_exact_mapping_source_contract(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(
        tmp_path, monkeypatch, confirmed=False
    )
    mapping = _strict_dell_mapping(tmp_path, monkeypatch)
    mapping["mapping_source"]["semantic_mapping"] = (
        "generic performance implies BIOS performance"
    )
    mapping["strict_conditions_met"] = True
    mapping["failed_conditions"] = []
    evidence["bios_thermal_management"]["dell_platform_profile_mapping"] = mapping

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert _bios_gate(verification)["passed"] is False


def test_readable_root_only_bios_value_is_authoritative_over_fallbacks(
    tmp_path, monkeypatch
):
    project, _, evidence = _valid_evidence(tmp_path, monkeypatch, confirmed=True)
    mapping = _strict_dell_mapping(tmp_path, monkeypatch)
    evidence["bios_thermal_management"].update(
        {
            "path": str(reentry.THERMAL_MANAGEMENT_PATH),
            "readable": True,
            "value": "Optimized",
            "error": None,
            "dell_platform_profile_mapping": mapping,
        }
    )

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    gate = _bios_gate(verification)
    assert gate["passed"] is False
    assert "directly readable" in gate["detail"]
    assert "fallback evidence is not accepted" in gate["detail"]


def test_root_only_ultraperformance_remains_preferred_evidence(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(
        tmp_path, monkeypatch, confirmed=False
    )
    evidence["bios_thermal_management"].update(
        {
            "path": str(reentry.THERMAL_MANAGEMENT_PATH),
            "readable": True,
            "value": "UltraPerformance",
            "error": None,
        }
    )

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    gate = _bios_gate(verification)
    assert gate["passed"] is True
    assert "root-only BIOS attribute" in gate["detail"]


def test_tracked_template_is_exact_null_contract_and_execution_paths_are_hashed():
    assert reentry.GPU_IDLE_MAX_AGE_SECONDS == 4 * 60 * 60
    expected = reentry.declaration_template()
    template = json.loads(
        (PROJECT_ROOT / "validation/gpu-reentry-declaration.template.json").read_text()
    )
    live_null_declaration = json.loads(
        (
            PROJECT_ROOT
            / "validation/results/gpu-reentry/operator-declaration.json"
        ).read_text()
    )

    assert template == expected
    assert live_null_declaration == expected
    assert {
        "validation/gpu_reentry_evidence.py",
        "validation/run_caviar.py",
        "validation/run_caviar_batch.py",
        "validation/open_video_review.py",
        "validation/run_loaf.py",
        "validation/run_loaf_batch.py",
        "validation/gpu_guarded_process.py",
    }.issubset(set(reentry.GUARD_PATHS))


def test_require_reentry_evidence_rechecks_ttl_and_guard_hashes(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(tmp_path, monkeypatch)
    report = project / "validation/results/reentry/evidence.json"
    report.write_text(json.dumps(evidence), encoding="utf-8")

    accepted = reentry.require_reentry_evidence(report, project_root=project, now=NOW)
    assert accepted["load_authority_granted_by_gate"] is False

    stale_now = NOW + timedelta(seconds=reentry.GPU_IDLE_MAX_AGE_SECONDS + 1)
    with pytest.raises(reentry.ReentryEvidenceError, match="fresh_idle_gpu_telemetry"):
        reentry.require_reentry_evidence(report, project_root=project, now=stale_now)

    (project / reentry.GUARD_PATHS[0]).write_text("# changed\n", encoding="utf-8")
    with pytest.raises(reentry.ReentryEvidenceError, match="current_guard_code_hashes"):
        reentry.require_reentry_evidence(report, project_root=project, now=NOW)


def test_declaration_file_hash_and_embedded_payload_must_match(tmp_path, monkeypatch):
    project, declaration_path, evidence = _valid_evidence(tmp_path, monkeypatch)
    declaration_path.write_text("{}\n", encoding="utf-8")

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert "operator_declaration_integrity" in verification["failed_gate_ids"]


def test_physical_declaration_older_than_24_hours_blocks(tmp_path, monkeypatch):
    project, declaration_path, evidence = _valid_evidence(tmp_path, monkeypatch)
    old = (NOW - timedelta(hours=25)).isoformat()
    declaration = evidence["operator_declaration"]
    declaration["declared_at_utc"] = old
    declaration["bios_thermal_management"]["observed_at_utc"] = old
    declaration["cooling_and_epsa"]["checked_at_utc"] = old
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    evidence["operator_declaration_file"]["sha256"] = _hash(declaration_path)

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert {
        "bios_thermal_management",
        "original_240w_adapter_declaration",
        "cooling_and_epsa_acknowledgement",
    }.issubset(verification["failed_gate_ids"])


def test_idle_power_limit_drop_or_slowdown_blocks(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(tmp_path, monkeypatch)
    sample = evidence["idle_gpu_telemetry"]["samples"][0]
    sample["snapshot"].update(
        {
            "power_current_limit_w": "55.00",
            "clock_event_hw_thermal_slowdown": "Active",
        }
    )
    sample["safety_assessment"] = reentry.assess_gpu_safety(
        sample["snapshot"],
        power_limit_drop_tolerance_w=reentry.POWER_LIMIT_DROP_TOLERANCE_W,
    )

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    gate = next(
        item
        for item in verification["gates"]
        if item["id"] == "idle_gpu_quality_diagnostics"
    )
    assert gate["passed"] is False
    assert gate["required"] is True
    assert "below default" in gate["detail"]
    assert "slowdown flag" in gate["detail"]


def test_workstation_managed_null_physical_and_quality_diagnostics_do_not_block(
    tmp_path, monkeypatch
):
    project, declaration_path, evidence = _valid_evidence(
        tmp_path,
        monkeypatch,
        confirmed=False,
        operating_policy_id=reentry.WORKSTATION_MANAGED_POLICY_ID,
    )
    # Prove that even an absent declaration file is informational in this mode.
    declaration_path.unlink()
    sample = evidence["idle_gpu_telemetry"]["samples"][0]
    sample["snapshot"].update(
        {
            "temperature_c": "92",
            "power_current_limit_w": "55.00",
            "clock_event_hw_thermal_slowdown": "Active",
        }
    )
    sample["safety_assessment"] = reentry.assess_gpu_safety(
        sample["snapshot"],
        power_limit_drop_tolerance_w=reentry.POWER_LIMIT_DROP_TOLERANCE_W,
    )
    evidence["platform_thermal"] = {
        "available": False,
        "values": {},
        "errors": ["test fixture intentionally unavailable"],
    }

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "ready_for_operator_review"
    assert verification["failed_gate_ids"] == []
    assert verification["operating_policy"]["id"] == "workstation_managed"
    by_id = {gate["id"]: gate for gate in verification["gates"]}
    for gate_id in reentry.WORKSTATION_INFORMATIONAL_GATE_IDS:
        assert by_id[gate_id]["required"] is False
    assert by_id["operator_declaration_integrity"]["passed"] is False
    assert by_id["bios_thermal_management"]["passed"] is False
    assert by_id["original_240w_adapter_declaration"]["passed"] is False
    assert by_id["cooling_and_epsa_acknowledgement"]["passed"] is False
    assert by_id["idle_gpu_quality_diagnostics"]["passed"] is False
    assert by_id["platform_fan_acpi_snapshot"]["passed"] is False

    report = project / "validation/results/reentry/workstation.json"
    report.write_text(json.dumps(evidence), encoding="utf-8")
    receipt = reentry.require_reentry_evidence(
        report, project_root=project, now=NOW
    )
    assert receipt["load_authority_granted_by_gate"] is False
    assert receipt["execution_authority"] == {
        "source": "explicit_user_instruction",
        "granted_by_this_evidence": False,
        "cryptographic_identity_authentication": False,
    }
    assert "does not cryptographically authenticate identity" in receipt[
        "verification"
    ]["authorization_policy"]


def test_workstation_managed_xorg_only_utilization_is_informational(
    tmp_path, monkeypatch
):
    project, _, evidence = _valid_evidence(
        tmp_path,
        monkeypatch,
        confirmed=False,
        operating_policy_id=reentry.WORKSTATION_MANAGED_POLICY_ID,
    )
    for sample in evidence["idle_gpu_telemetry"]["samples"]:
        sample["snapshot"]["gpu_utilization_percent"] = "36"
        sample["safety_assessment"] = reentry.assess_gpu_safety(
            sample["snapshot"],
            power_limit_drop_tolerance_w=reentry.POWER_LIMIT_DROP_TOLERANCE_W,
        )

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)
    by_id = {gate["id"]: gate for gate in verification["gates"]}

    assert verification["status"] == "ready_for_operator_review"
    assert by_id["fresh_idle_gpu_telemetry"]["passed"] is True
    assert by_id["idle_gpu_quality_diagnostics"]["passed"] is False
    assert by_id["idle_gpu_quality_diagnostics"]["required"] is False
    assert "compute-client query is empty" in by_id[
        "idle_gpu_quality_diagnostics"
    ]["detail"]


@pytest.mark.parametrize("compute_state", ("missing", "unreadable", "present"))
def test_workstation_managed_high_utilization_requires_readable_empty_compute_query(
    tmp_path, monkeypatch, compute_state
):
    project, _, evidence = _valid_evidence(
        tmp_path,
        monkeypatch,
        confirmed=False,
        operating_policy_id=reentry.WORKSTATION_MANAGED_POLICY_ID,
    )
    for sample in evidence["idle_gpu_telemetry"]["samples"]:
        sample["snapshot"]["gpu_utilization_percent"] = "24"
        sample["safety_assessment"] = reentry.assess_gpu_safety(
            sample["snapshot"],
            power_limit_drop_tolerance_w=reentry.POWER_LIMIT_DROP_TOLERANCE_W,
        )
    if compute_state == "missing":
        evidence.pop("compute_processes")
    elif compute_state == "unreadable":
        evidence["compute_processes"].update(
            {"available": False, "count": None, "error": "query failed"}
        )
    else:
        evidence["compute_processes"].update(
            {
                "processes": [
                    {
                        "pid": "1234",
                        "process_name": "python3",
                        "gpu_uuid": "GPU-test",
                    }
                ],
                "count": 1,
            }
        )

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "blocked"
    assert "fresh_idle_gpu_telemetry" in verification["failed_gate_ids"]


def test_legacy_strict_high_utilization_still_blocks_with_empty_compute_query(
    tmp_path, monkeypatch
):
    project, _, evidence = _valid_evidence(tmp_path, monkeypatch)
    for sample in evidence["idle_gpu_telemetry"]["samples"]:
        sample["snapshot"]["gpu_utilization_percent"] = "24"
        sample["safety_assessment"] = reentry.assess_gpu_safety(
            sample["snapshot"],
            power_limit_drop_tolerance_w=reentry.POWER_LIMIT_DROP_TOLERANCE_W,
        )

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "blocked"
    assert "fresh_idle_gpu_telemetry" in verification["failed_gate_ids"]


def test_collection_defaults_to_exact_workstation_managed_policy_without_hardware(
    tmp_path, monkeypatch
):
    project, declaration_path, fixture = _valid_evidence(
        tmp_path,
        monkeypatch,
        confirmed=False,
        operating_policy_id=reentry.WORKSTATION_MANAGED_POLICY_ID,
    )
    collection_now = datetime.now(timezone.utc)
    monkeypatch.setattr(reentry, "utc_now", lambda: collection_now.isoformat())
    monkeypatch.setattr(
        reentry, "collect_gpu_identity", lambda *args, **kwargs: deepcopy(fixture["gpu_identity"])
    )
    monkeypatch.setattr(
        reentry,
        "collect_compute_processes",
        lambda *args, **kwargs: deepcopy(fixture["compute_processes"]),
    )
    monkeypatch.setattr(
        reentry,
        "collect_idle_gpu_telemetry",
        lambda *args, **kwargs: deepcopy(fixture["idle_gpu_telemetry"]),
    )
    monkeypatch.setattr(
        reentry,
        "collect_linux_power_profiles",
        lambda *args, **kwargs: deepcopy(fixture["linux_power_profiles"]),
    )
    monkeypatch.setattr(
        reentry, "collect_ac_online", lambda *args, **kwargs: deepcopy(fixture["ac_power"])
    )
    monkeypatch.setattr(
        reentry,
        "collect_bios_thermal_management",
        lambda *args, **kwargs: deepcopy(fixture["bios_thermal_management"]),
    )
    monkeypatch.setattr(
        reentry,
        "collect_platform_thermal",
        lambda *args, **kwargs: deepcopy(fixture["platform_thermal"]),
    )
    monkeypatch.setattr(
        reentry,
        "collect_xid_log",
        lambda *args, **kwargs: deepcopy(fixture["xid_current_boot"]),
    )

    collected = reentry.collect_evidence(
        project_root=project,
        declaration_path=declaration_path,
        sleeper=lambda _: None,
    )

    assert collected["operating_policy"] == reentry.operating_policy_contract(
        reentry.WORKSTATION_MANAGED_POLICY_ID
    )
    assert collected["collection_policy"]["operating_policy_id"] == (
        reentry.WORKSTATION_MANAGED_POLICY_ID
    )
    assert collected["verification"]["status"] == "ready_for_operator_review"


@pytest.mark.parametrize(
    ("case", "failed_gate"),
    (
        ("stale", "fresh_idle_gpu_telemetry"),
        ("missing_identity", "fresh_idle_gpu_telemetry"),
        ("missing_xid", "no_xid_current_boot"),
        ("xid_present", "no_xid_current_boot"),
        ("ac_offline", "ac_mains_online"),
        ("profile_balanced", "linux_power_profiles_performance"),
        ("code_tamper", "current_guard_code_hashes"),
    ),
)
def test_workstation_managed_still_rejects_missing_or_stale_technical_evidence(
    tmp_path, monkeypatch, case, failed_gate
):
    project, _, evidence = _valid_evidence(
        tmp_path,
        monkeypatch,
        confirmed=False,
        operating_policy_id=reentry.WORKSTATION_MANAGED_POLICY_ID,
    )
    verify_now = NOW
    if case == "stale":
        verify_now = NOW + timedelta(seconds=reentry.GPU_IDLE_MAX_AGE_SECONDS + 1)
    elif case == "missing_identity":
        evidence["gpu_identity"] = {"available": False, "fields": None}
    elif case == "missing_xid":
        evidence["xid_current_boot"] = {"available": False, "count": 0}
    elif case == "xid_present":
        evidence["xid_current_boot"].update(
            {"count": 1, "lines": ["NVRM: Xid (PCI:0000:01:00): 79"]}
        )
    elif case == "ac_offline":
        evidence["ac_power"].update({"available": True, "mains_online": False})
    elif case == "profile_balanced":
        evidence["linux_power_profiles"]["powerprofilesctl"]["value"] = "balanced"
    elif case == "code_tamper":
        (project / reentry.GUARD_PATHS[0]).write_text(
            "# modified after evidence collection\n", encoding="utf-8"
        )

    verification = reentry.verify_evidence(
        evidence, project_root=project, now=verify_now
    )

    assert verification["status"] == "blocked"
    assert failed_gate in verification["failed_gate_ids"]


@pytest.mark.parametrize(
    "bad_policy",
    (
        None,
        "workstation_managed",
        {"id": "unknown_policy"},
        {
            **reentry.operating_policy_contract(
                reentry.WORKSTATION_MANAGED_POLICY_ID
            ),
            "hardware_protection_owner": "caller_overrides_all_checks",
        },
    ),
)
def test_malformed_or_unknown_operating_policy_fails_closed(
    tmp_path, monkeypatch, bad_policy
):
    project, _, evidence = _valid_evidence(
        tmp_path,
        monkeypatch,
        confirmed=False,
        operating_policy_id=reentry.WORKSTATION_MANAGED_POLICY_ID,
    )
    evidence["operating_policy"] = bad_policy

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "blocked"
    assert "operating_policy_contract" in verification["failed_gate_ids"]


def test_explicit_policy_collection_binding_tamper_fails_closed(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(
        tmp_path,
        monkeypatch,
        confirmed=False,
        operating_policy_id=reentry.WORKSTATION_MANAGED_POLICY_ID,
    )
    evidence["collection_policy"]["operating_policy_id"] = (
        reentry.LEGACY_STRICT_PHYSICAL_POLICY_ID
    )

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "blocked"
    assert "operating_policy_contract" in verification["failed_gate_ids"]


def test_missing_or_unsupported_evidence_schema_fails_closed(tmp_path, monkeypatch):
    project, _, evidence = _valid_evidence(
        tmp_path,
        monkeypatch,
        confirmed=False,
        operating_policy_id=reentry.WORKSTATION_MANAGED_POLICY_ID,
    )
    evidence["schema_version"] = "deepsafe.gpu-reentry-evidence/unknown"

    verification = reentry.verify_evidence(evidence, project_root=project, now=NOW)

    assert verification["status"] == "blocked"
    assert "operating_policy_contract" in verification["failed_gate_ids"]


def test_xid_collector_accepts_quiet_journalctl_no_match_as_empty_boot_log():
    calls = []

    def runner(command, _timeout):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 1, "", "")

    value = reentry.collect_xid_log(runner)

    assert value == {
        "available": True,
        "source": "journalctl",
        "count": 0,
        "lines": [],
        "read_only_command": [
            "journalctl",
            "--quiet",
            "-k",
            "-b",
            "--no-pager",
            "--grep",
            "NVRM: Xid",
        ],
        "errors": [],
    }
    assert calls == [value["read_only_command"]]


@pytest.mark.parametrize(
    "command",
    (
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=name",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        ["powerprofilesctl", "get"],
        [
            "journalctl",
            "--quiet",
            "-k",
            "-b",
            "--no-pager",
            "--grep",
            "NVRM: Xid",
        ],
    ),
)
def test_read_only_probe_allowlist_accepts_only_query_commands(command):
    assert reentry._is_read_only_command(command) is True


@pytest.mark.parametrize(
    "command",
    (
        ["nvidia-smi", "-pl", "115"],
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=name",
            "--format=csv,noheader,nounits",
            "--power-limit=55",
        ],
        ["nvidia-smi", "--gpu-reset"],
        ["powerprofilesctl", "set", "performance"],
        ["sudo", "cat", str(reentry.THERMAL_MANAGEMENT_PATH)],
        ["docker", "run", "anything"],
    ),
)
def test_read_only_probe_allowlist_rejects_mutation_and_load_commands(command):
    assert reentry._is_read_only_command(command) is False
