from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path

import jsonschema
import pytest

from validation import gpu_lease_v6_independent_review as review


@pytest.fixture(scope="session")
def audit() -> dict[str, object]:
    value = review.audit_current_state()
    assert value["decision"] == review.DECISION
    return value


@pytest.fixture(scope="session")
def receipt(audit: dict[str, object]) -> dict[str, object]:
    if review.DEFAULT_RECEIPT.exists():
        return review.load_receipt()
    return review.build_receipt(audit, independent_tests=25)


def _resign(value: dict[str, object]) -> dict[str, object]:
    value["review_fingerprint_sha256"] = review.fingerprint(
        value, "review_fingerprint_sha256"
    )
    return value


def test_independent_schema_is_valid_and_closed() -> None:
    schema = review.strict_json((review.ROOT / review.SCHEMA_REL).read_bytes(), "schema")
    jsonschema.Draft202012Validator.check_schema(schema)
    report = review.schema_closure(schema)
    assert report["all_property_objects_closed"] is True


@pytest.mark.parametrize(
    "raw,match",
    [
        (b'{"a":1,"a":2}', "duplicate"),
        (b'{"a":NaN}', "non-finite"),
        (b"\xef\xbb\xbf{}", "BOM"),
    ],
)
def test_strict_json_rejects_ambiguous_inputs(raw: bytes, match: str) -> None:
    with pytest.raises(review.GPULeaseV6IndependentReviewError, match=match):
        review.strict_json(raw, "adversarial")


@pytest.mark.parametrize(
    "path",
    ["/absolute", "../escape", "a/../b", "a//b", "a\\b", "a\x00b"],
)
def test_repository_path_grammar_fails_closed(path: str) -> None:
    with pytest.raises(review.GPULeaseV6IndependentReviewError, match="unsafe path"):
        review._parts(path)


def test_exact_subject_and_predecessor_pins_replay(audit: dict[str, object]) -> None:
    subject = audit["subject"]
    assert subject["artifacts"] == review.SUBJECT_PINS
    assert subject["frozen_predecessors"] == review.FROZEN_PREDECESSOR_PINS


def test_handoff_schema_and_canonical_fingerprint_replay(audit: dict[str, object]) -> None:
    verification = audit["subject"]["handoff"]["verification"]
    assert verification["schema_valid"] is True
    assert verification["canonical_fingerprint_valid"] is True
    assert audit["subject"]["handoff"]["fingerprint_sha256"] == review.HANDOFF_FINGERPRINT


def test_contract_is_locked_and_canonically_fingerprinted(audit: dict[str, object]) -> None:
    contract = audit["verification"]["contract"]
    assert contract["canonical_fingerprint_valid"] is True
    assert contract["status_locked_no_live_plan"] is True
    assert contract["default_or_published_plan"] is False


def test_frozen_v4_wrapping_and_v5_ineligibility_are_exact(audit: dict[str, object]) -> None:
    contract = audit["verification"]["contract"]
    assert contract["v4_frozen_execution_base_exact"] is True
    assert contract["v5_frozen_predecessor_exact_and_ineligible"] is True
    assert audit["verification"]["host"]["v5_activation_eligible"] is False


def test_current_docker_and_nvidia_smi_are_hash_only_exact(audit: dict[str, object]) -> None:
    host = audit["verification"]["host"]
    assert host["v6_requalified_tools"] == review.V6_TOOL_PINS
    assert host["tools_hashed_not_executed"] is True
    assert review.RESOURCE_BOUNDARY["docker_called"] is False
    assert review.RESOURCE_BOUNDARY["nvidia_smi_called"] is False


def test_frozen_v5_tool_pins_remain_honestly_drifted(audit: dict[str, object]) -> None:
    host = audit["verification"]["host"]
    assert host["v5_expected_tools"] == review.V5_TOOL_PINS
    assert all(
        host["v6_requalified_tools"][name] != review.V5_TOOL_PINS[name]
        for name in review.V5_TOOL_PINS
    )


def test_frozen_v5_nine_failures_are_not_reclassified(audit: dict[str, object]) -> None:
    evidence = audit["verification"]["tests"]["frozen_v5_regression"]
    assert evidence["result"] == "EXPECTED_FAIL_CLOSED"
    assert (evidence["passed"], evidence["failed"]) == (17, 9)
    observation = audit["external_gate_observations"][0]
    assert observation["would_be_execution_readiness_severity"] == "P1"
    assert observation["subject_severity"] is None


def test_r7_receipt_and_current_boot_replay(audit: dict[str, object]) -> None:
    r7 = audit["verification"]["driver_r7"]
    assert r7["canonical_fingerprint_valid"] is True
    assert r7["boot_id"] == review.R7_BOOT_ID
    assert r7["boot_matches_now"] is True
    assert r7["gpu_workload_authorized"] is False


def test_schema_replay_requires_outer_user_and_workload_binding(audit: dict[str, object]) -> None:
    schemas = audit["verification"]["schemas"]
    assert schemas["outer_user_notification_required"] is True
    assert schemas["outer_plan_id_gpu_owner_argv_image_binding_required"] is True
    assert schemas["plan_exact_gpu_owner_argv_image_and_artifact_projection"] is True


def test_schema_replay_requires_activation_lineage(audit: dict[str, object]) -> None:
    schemas = audit["verification"]["schemas"]
    assert schemas["activation_plan_outer_v4_terminal_binding_required"] is True
    assert schemas["v5_activation_eligible_const_false"] is True
    assert schemas["current_boot_driver_binding_required"] is True


def test_ast_audit_keeps_sensitive_calls_in_three_narrow_sites(audit: dict[str, object]) -> None:
    surface = audit["verification"]["author_source_ast"]
    assert {(row["call"], row["function"]) for row in surface["sensitive_call_sites"]} == {
        ("subprocess.Popen", "run_plan"),
        ("subprocess.run", "_probe_gpu_identity_from_held_fd"),
        ("v4.main", "_held_activate"),
    }
    assert surface["forbidden_calls"] == []
    assert surface["network_or_model_imports"] == []


def test_ast_audit_keeps_gate_transition_after_last_boot_replay(audit: dict[str, object]) -> None:
    surface = audit["verification"]["author_source_ast"]
    assert surface["module_closed_literals"] is True
    assert surface["true_gate_assignments_held_child_only"] is True
    assert surface["last_boot_before_gate_transition"] is True
    assert surface["v5_true_assignments"] == 0


def test_ast_audit_keeps_gpu_probe_execute_only(audit: dict[str, object]) -> None:
    surface = audit["verification"]["author_source_ast"]
    assert surface["gpu_probe_execute_mode_only"] is True
    assert surface["held_bootstrap_reconstructs_frozen_modules"] is True


def test_delivery_root_is_owner_only_and_empty(audit: dict[str, object]) -> None:
    delivery = audit["verification"]["delivery"]
    assert delivery["v6_result_root"]["mode"] == "0700"
    assert delivery["v6_result_root"]["entries"] == []
    assert delivery["default_workload_plan"] is None
    assert delivery["published_live_plan"] is None


def test_no_user_outer_plan_or_activation_receipt_is_claimed(audit: dict[str, object]) -> None:
    delivery = audit["verification"]["delivery"]
    assert delivery["user_notification_present"] is False
    assert delivery["outer_acceptance_present"] is False
    assert delivery["activation_receipt_present"] is False
    assert review.AUTHORITY["workload_plan_present"] is False


def test_focused_and_v1_v4_test_evidence_is_exact(audit: dict[str, object]) -> None:
    evidence = audit["verification"]["tests"]
    assert evidence["focused_v6_run_1"]["passed"] == 39
    assert evidence["focused_v6_run_2"]["passed"] == 39
    assert evidence["v1_v4_regression"]["passed"] == 138


def test_static_acceptance_has_zero_subject_findings(audit: dict[str, object]) -> None:
    assert audit["decision"] == "ACCEPT_STATIC_NON_EXECUTION_ONLY"
    assert audit["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}
    assert audit["findings"] == []


def test_all_execution_and_production_authority_is_false() -> None:
    assert review.AUTHORITY["independent_static_bundle_acceptance_present"] is True
    assert all(
        value is False
        for key, value in review.AUTHORITY.items()
        if key != "independent_static_bundle_acceptance_present"
    )
    assert all(value is False for value in review.PERMISSIONS.values())


def test_reviewer_never_imports_subject_or_subprocess() -> None:
    source = inspect.getsource(review)
    assert "from validation import gpu_lease_v6" not in source
    assert "import gpu_lease_v6" not in source
    tree = ast.parse(source)
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "subprocess" not in imports


def test_candidate_or_published_receipt_verifies(receipt: dict[str, object]) -> None:
    result = review.verify_receipt(receipt, replay_subject=True)
    assert result["decision"] == review.DECISION
    assert result["execution_authorized"] is False
    assert result["production_authorized"] is False


@pytest.mark.parametrize(
    "field",
    [
        "gpu_lease_api_ready",
        "live_plan_authorized",
        "user_notification_accepted",
        "outer_workload_accepted",
        "activation_receipt_present",
        "production_authorized",
    ],
)
def test_authority_escalation_rejected_after_refingerprint(
    receipt: dict[str, object], field: str
) -> None:
    changed = copy.deepcopy(receipt)
    changed["authority"][field] = True
    with pytest.raises(review.GPULeaseV6IndependentReviewError):
        review.verify_receipt(_resign(changed))


def test_subject_pin_tamper_rejected_after_refingerprint(receipt: dict[str, object]) -> None:
    changed = copy.deepcopy(receipt)
    changed["subject"]["artifacts"]["implementation"]["sha256"] = "0" * 64
    with pytest.raises(review.GPULeaseV6IndependentReviewError, match="subject pins"):
        review.verify_receipt(_resign(changed))


def test_v5_eligibility_tamper_rejected_after_refingerprint(receipt: dict[str, object]) -> None:
    changed = copy.deepcopy(receipt)
    changed["verification"]["host"]["v5_activation_eligible"] = True
    with pytest.raises(review.GPULeaseV6IndependentReviewError, match="V5 eligibility"):
        review.verify_receipt(_resign(changed))


def test_receipt_schema_rejects_unknown_top_level_field(receipt: dict[str, object]) -> None:
    changed = copy.deepcopy(receipt)
    changed["unexpected"] = True
    schema = json.loads((review.ROOT / review.SCHEMA_REL).read_text(encoding="utf-8"))
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(changed))
    assert any("Additional properties" in error.message for error in errors)


def test_required_next_preserves_all_live_gates() -> None:
    assert review.REQUIRED_NEXT == [
        "explicit_user_notification_after_this_static_review",
        "immutable_outer_acceptance_for_one_exact_plan_id_gpu_owner_argv_and_image",
        "fresh_externally_pinned_v6_workload_plan_with_no_defaults",
        "separate_live_launch_authorization",
        "current_boot_r7_and_current_tool_replay_at_future_launch",
        "separate_v6_activation_receipt_after_a_real_terminal_run",
    ]


def test_review_control_paths_are_additive_only(audit: dict[str, object]) -> None:
    paths = {pin["path"] for pin in audit["review_control_pins"]}
    assert paths == {review.REVIEWER_REL, review.SCHEMA_REL, review.TEST_REL, review.DOC_REL}
    assert not paths & {pin["path"] for pin in review.SUBJECT_PINS.values()}
