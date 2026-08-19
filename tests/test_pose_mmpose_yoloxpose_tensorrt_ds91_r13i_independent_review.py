from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_tensorrt_ds91_r13i_independent_review as review


@pytest.fixture(scope="session")
def subject_documents() -> tuple[dict, dict, dict, dict, dict]:
    controller = json.loads((review.ROOT / review.CONTROLLER_REL).read_text(encoding="utf-8"))
    template = json.loads((review.ROOT / review.TEMPLATE_REL).read_text(encoding="utf-8"))
    controller_schema = json.loads(
        (review.ROOT / review.AUTHOR_PINS["controller_schema"]["path"]).read_text(encoding="utf-8")
    )
    template_schema = json.loads(
        (review.ROOT / review.AUTHOR_PINS["workload_template_schema"]["path"]).read_text(encoding="utf-8")
    )
    handoff = json.loads((review.ROOT / review.HANDOFF_REL).read_text(encoding="utf-8"))
    return controller, template, controller_schema, template_schema, handoff


@pytest.fixture(scope="session")
def full_audit() -> dict:
    return review.audit_current_state()


@pytest.fixture(scope="session")
def receipt(full_audit: dict) -> dict:
    return review.build_receipt(full_audit, independent_tests=70)


def test_canonical_bytes_are_stable() -> None:
    assert review.canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_canonical_bytes_reject_nonfinite() -> None:
    with pytest.raises(ValueError):
        review.canonical_bytes({"bad": float("nan")})


def test_fingerprint_removes_only_named_field() -> None:
    value = {"a": 1, "self": "wrong"}
    assert review.fingerprint(value, "self") == hashlib.sha256(b'{"a":1}').hexdigest()


def test_command_argv_hash_has_typed_envelope() -> None:
    one = review.command_argv_sha256(["a", "bc"])
    two = review.command_argv_sha256(["ab", "c"])
    assert one != two


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'\xef\xbb\xbf{"a":1}',
        b'\xff',
        b'[]',
        b'',
    ],
)
def test_strict_json_rejects_malformed_or_ambiguous(raw: bytes) -> None:
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.strict_json(raw, "adversarial")


def test_strict_json_accepts_closed_object() -> None:
    assert review.strict_json(b'{"a":[true,null,1]}', "ok") == {"a": [True, None, 1]}


@pytest.mark.parametrize(
    "path",
    ["", "/absolute", "../escape", "a/../b", "a//b", "./a", "a/./b", "a\\b", "a\x00b"],
)
def test_relative_path_policy_rejects_traversal(path: str) -> None:
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review._parts(path)


def test_relative_path_policy_accepts_repository_path() -> None:
    assert review._parts("a/b/c.json") == ("a", "b", "c.json")


def _make_file(root: Path, relative: str, data: bytes = b"payload", mode: int = 0o440) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mode": f"{mode:04o}",
    }


def test_workspace_reader_exact_good(tmp_path: Path) -> None:
    pin = _make_file(tmp_path, "a/b.bin")
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.read_exact(pin, label="good") == b"payload"
        reader.replay_root()


@pytest.mark.parametrize("field,value", [("bytes", 1), ("sha256", "0" * 64), ("mode", "0400")])
def test_workspace_reader_rejects_wrong_pin(tmp_path: Path, field: str, value: object) -> None:
    pin = _make_file(tmp_path, "a.bin")
    pin[field] = value
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.PoseR13IIndependentReviewError):
            reader.read_exact(pin, label="wrong")


def test_workspace_reader_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    target.chmod(0o440)
    (tmp_path / "link").symlink_to(target)
    pin = {"path": "link", "bytes": 7, "sha256": hashlib.sha256(b"payload").hexdigest(), "mode": "0440"}
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.read_exact(pin, label="symlink")


def test_workspace_reader_rejects_symlink_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    _make_file(real, "file")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    pin = {"path": "alias/file", "bytes": 7, "sha256": hashlib.sha256(b"payload").hexdigest(), "mode": "0440"}
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.read_exact(pin, label="component")


def test_workspace_reader_rejects_hardlink(tmp_path: Path) -> None:
    pin = _make_file(tmp_path, "one")
    os.link(tmp_path / "one", tmp_path / "two")
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.PoseR13IIndependentReviewError):
            reader.read_exact(pin, label="hardlink")


def test_workspace_reader_pin_regular(tmp_path: Path) -> None:
    pin = _make_file(tmp_path, "pin")
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.pin_regular("pin") == pin


def test_workspace_reader_output_absence_and_presence(tmp_path: Path) -> None:
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.exists_no_follow("missing/deep/file") is False
    _make_file(tmp_path, "present/file")
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.exists_no_follow("present/file") is True


def test_workspace_reader_rejects_root_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        with review.WorkspaceReader(alias):
            pass


def test_current_author_source_is_ast_only_safe() -> None:
    raw = (review.ROOT / review.AUTHOR_PINS["controller_source"]["path"]).read_bytes()
    report = review.audit_author_source(raw)
    assert report["parsed_only_not_imported_or_executed"] is True
    assert all(value is False for value in report["execution_flag_literals"].values())


def _mutated_source(replacement: tuple[bytes, bytes] | None = None, suffix: bytes = b"") -> bytes:
    raw = (review.ROOT / review.AUTHOR_PINS["controller_source"]["path"]).read_bytes()
    if replacement:
        old, new = replacement
        assert old in raw
        raw = raw.replace(old, new, 1)
    return raw + suffix


def test_author_ast_rejects_true_gate() -> None:
    raw = _mutated_source((b"EXECUTION_AUTHORIZED = False", b"EXECUTION_AUTHORIZED = True"))
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.audit_author_source(raw)


@pytest.mark.parametrize("suffix", [b"\nimport subprocess\n", b"\neval('1')\n", b"\nos.unlink('x')\n"])
def test_author_ast_rejects_forbidden_surfaces(suffix: bytes) -> None:
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.audit_author_source(_mutated_source(suffix=suffix))


def test_author_ast_rejects_writable_open() -> None:
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.audit_author_source(_mutated_source(suffix=b"\nos.open('x', os.O_WRONLY)\n"))


def test_author_ast_rejects_missing_main_guard() -> None:
    raw = _mutated_source((b'if __name__ == "__main__":', b'if __name__ == "not-main":'))
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.audit_author_source(raw)


def test_independent_schema_is_recursively_closed() -> None:
    schema = json.loads((review.ROOT / review.SCHEMA_REL).read_text(encoding="utf-8"))
    assert review.schemas_recursively_closed(schema)["object_nodes"] >= 10


def test_recursive_schema_closure_rejects_open_object() -> None:
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.schemas_recursively_closed({"type": "object", "properties": {}})


def test_subject_schemas_are_recursively_closed(subject_documents: tuple[dict, dict, dict, dict, dict]) -> None:
    _, _, controller_schema, template_schema, _ = subject_documents
    assert review.schemas_recursively_closed(controller_schema)["all_closed"] == 1
    assert review.schemas_recursively_closed(template_schema)["all_closed"] == 1


def test_handoff_semantics_pass(subject_documents: tuple[dict, dict, dict, dict, dict]) -> None:
    *_, handoff = subject_documents
    assert review.verify_handoff(handoff)["execution_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"execution_authorized": True}),
        lambda value: value["permissions"].update({"accept": True}),
        lambda value: value["author_tests"].update({"passed": 82}),
        lambda value: value["remaining_gates"].pop(),
    ],
)
def test_handoff_semantics_reject_authority_or_evidence_tamper(
    subject_documents: tuple[dict, dict, dict, dict, dict], mutation
) -> None:
    *_, handoff = subject_documents
    changed = copy.deepcopy(handoff)
    mutation(changed)
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.verify_handoff(changed)


@pytest.mark.parametrize("profile", [640, 960])
def test_expected_argv_exact_profile(profile: int) -> None:
    argv = review.expected_argv(profile)
    assert argv[13] == review.IMAGE_REFERENCE
    assert "--fp16" in argv and "--noTF32" in argv and "--skipInference" in argv
    assert not any(item in argv for item in ("--rm", "--detach", "--cidfile"))


def test_expected_argv_rejects_unknown_profile() -> None:
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.expected_argv(1280)


def test_controller_and_template_schema_validation(subject_documents: tuple[dict, dict, dict, dict, dict]) -> None:
    controller, template, controller_schema, template_schema, _ = subject_documents
    report = review.verify_controller(controller, template, controller_schema, template_schema)
    assert report["execution_authorized"] is False
    assert report["parser_ds91_compatibility_accepted"] is False


def test_controller_rejects_satisfied_gate(subject_documents: tuple[dict, dict, dict, dict, dict]) -> None:
    controller, template, controller_schema, template_schema, _ = subject_documents
    changed = copy.deepcopy(controller)
    changed["execution_gate"]["required_acceptances"]["driver595_r7_host_qualification"]["satisfied"] = True
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.verify_controller(changed, template, controller_schema, template_schema)


def test_controller_rejects_parser_abi_escalation(subject_documents: tuple[dict, dict, dict, dict, dict]) -> None:
    controller, template, controller_schema, template_schema, _ = subject_documents
    changed = copy.deepcopy(controller)
    changed["config_contract"]["ds91_abi_compatibility_accepted"] = True
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.verify_controller(changed, template, controller_schema, template_schema)


class _AbsentReader:
    def __init__(self, collision: str | None = None) -> None:
        self.collision = collision

    def exists_no_follow(self, relative: str) -> bool:
        return relative == self.collision


def test_template_semantics_and_output_absence(subject_documents: tuple[dict, dict, dict, dict, dict]) -> None:
    controller, template, *_ = subject_documents
    report = review.verify_template(_AbsentReader(), template, controller)  # type: ignore[arg-type]
    assert set(report) == {"640", "960"}
    assert sum(len(row["outputs_absent"]) for row in report.values()) == 20


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update({"execution_authorized": True}),
        lambda value: value["profiles"]["640"]["precision"].update({"fp16": False}),
        lambda value: value["profiles"]["960"]["optimization_profile"]["input"].update({"opt": [1, 3, 960, 960]}),
        lambda value: value["profiles"]["640"]["docker"].update({"environment": {"X": "1"}}),
        lambda value: value["profiles"]["960"]["docker"]["argv"].append("--rm"),
        lambda value: value["profiles"]["640"]["outputs"]["final_engine"].update({"published_pin": {"x": 1}}),
        lambda value: value["lease_v5"].update({"live_plan_in_template": {"authorized": True}}),
    ],
)
def test_template_rejects_authority_shape_precision_or_output_tamper(
    subject_documents: tuple[dict, dict, dict, dict, dict], mutator
) -> None:
    controller, template, *_ = subject_documents
    changed = copy.deepcopy(template)
    mutator(changed)
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.verify_template(_AbsentReader(), changed, controller)  # type: ignore[arg-type]


def test_template_rejects_existing_future_output(subject_documents: tuple[dict, dict, dict, dict, dict]) -> None:
    controller, template, *_ = subject_documents
    collision = template["profiles"]["640"]["outputs"]["final_engine"]["path"]
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.verify_template(_AbsentReader(collision), template, controller)  # type: ignore[arg-type]


@pytest.mark.parametrize("tool", ["/usr/bin/python3.12", "/usr/bin/docker", "/usr/bin/nvidia-smi"])
def test_external_tools_are_hashed_not_executed(tool: str) -> None:
    row = review._hash_external_regular(tool)
    assert row["executed"] is False
    assert row["mode"] == "0755"
    assert len(row["sha256"]) == 64


def test_external_tool_reader_rejects_unapproved_path() -> None:
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review._hash_external_regular("/bin/sh")


def test_current_boot_matches_r7_constant() -> None:
    assert Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip() == review.R7_BOOT_ID


def test_full_audit_accepts_only_static_subject(full_audit: dict) -> None:
    assert full_audit["decision"] == "ACCEPT"
    assert full_audit["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}
    assert full_audit["permissions"] == review.PERMISSIONS
    assert full_audit["controller"]["execution_authorized"] is False


def test_full_audit_exposes_v5_host_gate_blocker(full_audit: dict) -> None:
    lease = full_audit["foundations"]["gpu_lease_v5"]
    assert lease["host_tool_exact_matches"] == {
        "python_interpreter": True,
        "docker_cli": False,
        "nvidia_smi": False,
    }
    assert lease["current_host_realization_eligible"] is False
    assert lease["activation_would_fail_closed"] is True
    observation = full_audit["external_gate_observations"][0]
    assert observation["would_be_execution_readiness_severity"] == "P1"
    assert observation["subject_severity"] is None


def test_full_audit_replays_r7_current_boot_without_absorbing_it(full_audit: dict) -> None:
    r7 = full_audit["foundations"]["driver595_r7_current_boot"]
    assert r7["decision"] == "ACCEPT" and r7["current_boot_match"] is True
    assert full_audit["accepted_scope"]["driver595_r7_absorbed_into_author_controller"] is False


def test_full_audit_has_exact_profiles(full_audit: dict) -> None:
    assert full_audit["profiles"]["640"]["shape_min_opt_max"]["opt"] == [12, 3, 640, 640]
    assert full_audit["profiles"]["960"]["shape_min_opt_max"]["max"] == [12, 3, 960, 960]
    assert full_audit["model_inputs"]["keypoint_count"] == 17
    assert full_audit["model_inputs"]["same_index_association_required"] is True


def test_receipt_schema_and_fingerprint(receipt: dict) -> None:
    schema = json.loads((review.ROOT / review.SCHEMA_REL).read_text(encoding="utf-8"))
    review._validate_schema(receipt, schema, "test receipt")
    assert receipt["review_fingerprint_sha256"] == review.fingerprint(receipt, "review_fingerprint_sha256")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["authority_boundary"].update({"runtime_or_build_authority_granted": True}),
        lambda value: value["permissions"].update({"call_gpu": True}),
        lambda value: value["accepted_scope"].update({"gpu_lease_v5_current_host_realization_eligible": True}),
        lambda value: value["verification"]["foundations"]["gpu_lease_v5"].update({"activation_would_fail_closed": False}),
        lambda value: value.update({"external_gate_observations": []}),
        lambda value: value["severity_counts"].update({"P1": 1}),
        lambda value: value.update({"unexpected": True}),
    ],
)
def test_receipt_schema_or_verifier_rejects_authority_tamper(receipt: dict, mutator) -> None:
    changed = copy.deepcopy(receipt)
    mutator(changed)
    changed["review_fingerprint_sha256"] = review.fingerprint(changed, "review_fingerprint_sha256")
    with pytest.raises((review.PoseR13IIndependentReviewError, json.JSONDecodeError)):
        review.verify_receipt(changed)


def test_verify_receipt_without_subject_replay(receipt: dict) -> None:
    report = review.verify_receipt(receipt)
    assert report["decision"] == "ACCEPT"
    assert report["external_execution_gate_blocker"] == "P1"
    assert report["execution_authorized"] is False


def test_verify_receipt_with_fresh_subject_replay(receipt: dict) -> None:
    report = review.verify_receipt(receipt, replay_subject=True)
    assert report["replayed_subject"] is True
    assert report["v5_host_realization_eligible"] is False


def test_build_receipt_rejects_small_test_count(full_audit: dict) -> None:
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.build_receipt(full_audit, independent_tests=34)


def test_publish_receipt_is_atomic_noreplace(tmp_path: Path, receipt: dict) -> None:
    container = tmp_path / "container"
    container.mkdir()
    output = container / "run" / "receipt.json"
    size, digest = review.publish_receipt(output, receipt, expected_output=output)
    assert size == output.stat().st_size
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o550
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.publish_receipt(output, receipt, expected_output=output)


def test_publish_receipt_rejects_wrong_expected_path(tmp_path: Path, receipt: dict) -> None:
    output = tmp_path / "one" / "receipt.json"
    expected = tmp_path / "two" / "receipt.json"
    with pytest.raises(review.PoseR13IIndependentReviewError):
        review.publish_receipt(output, receipt, expected_output=expected)
