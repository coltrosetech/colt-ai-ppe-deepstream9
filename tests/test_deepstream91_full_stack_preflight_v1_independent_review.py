from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from validation import deepstream91_full_stack_preflight_v1 as subject
from validation import deepstream91_full_stack_preflight_v1_independent_review as review
from validation import full_stack_benchmark_contract_v1 as legacy


ROOT = Path(__file__).resolve().parents[1]


def _successor_plan() -> dict:
    return subject.load_plan()


def _resign_successor(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("self_fingerprint", None)
    result["self_fingerprint"] = hashlib.sha256(subject.canonical_bytes(result)).hexdigest()
    return result


def _resign_review(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("review_fingerprint_sha256", None)
    result["review_fingerprint_sha256"] = review.review_fingerprint(result)
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _subject_pin(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _legacy_pin(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _copy_control_tree(destination: Path) -> None:
    paths = (
        "validation/plans/deepstream-full-stack-benchmark-v1.json",
        "validation/open_video_review/scenes.json",
        "validation/accepted-roots/ds91-engine-builder-r1c3-terminal/terminal-root-r1c3.json",
        "validation/contracts/gpu-lease-v5.json",
    )
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def test_checked_in_review_receipt_verifies_a_rejection() -> None:
    result = review.verify(review.load_review())
    assert result["status"] == "verified_immutable_rejection"
    assert result["decision"] == "REJECT"
    assert result["severity_counts"] == {"P0": 0, "P1": 4, "P2": 1}
    assert result["execution_ready"] is False


def test_independent_baseline_without_media_replay() -> None:
    result = review.independent_baseline(replay_media=False)
    assert result["source_count"] == 12
    assert result["distinct_video_types"] == 12
    assert result["authorization_all_false"] is True
    assert result["blockers_exact"] is True


def test_independent_baseline_replays_all_exact_media() -> None:
    result = review.independent_baseline(replay_media=True)
    assert result["media_replayed"] is True
    assert result["profiles"] == [640, 960]
    assert result["simulated_streams"] == 12
    assert result["measurement_seconds_per_profile"] == 300


def test_current_legacy_projection_is_structurally_a_full_runtime_plan() -> None:
    predecessor = review.load_workspace_json(
        "validation/plans/deepstream-full-stack-benchmark-v1.json"
    )
    assert predecessor["runtime"]["deepstream_version"] == "9.0.0"
    assert predecessor["runtime"]["fusion"]["fusion_plugin_pin"] is not None
    assert predecessor["profiles"][0]["modules"][0]["engine_pin"] is not None
    assert "authorization_plan_pin" in predecessor["profiles"][0]


def test_checked_in_subject_stays_blocked_and_non_authorizing() -> None:
    report = subject.validate_plan(_successor_plan(), replay_media=False)
    assert report["status"] == "pass_blocked_as_designed"
    assert report["execution_ready"] is False
    assert report["legacy_ds90_runtime_reused"] is False
    assert report["gpu_or_docker_called"] is False


def test_resigned_replacement_builder_can_carry_unchecked_runtime_scope(
    tmp_path: Path,
) -> None:
    _copy_control_tree(tmp_path)
    plan = _successor_plan()
    builder_path = tmp_path / plan["runtime_target"]["prepared_closed_engine_builder_root"]["path"]
    replacement = {
        "schema_version": "deepsafe.ds91-engine-builder-terminal-accepted-root/v1",
        "status": "accepted_terminal_prepared_closed_image_identity_and_evidence_only",
        "accepted_scope": {
            "terminal_accepted": True,
            "gpu_runtime": False,
            "tensorrt_engine_build": False,
            "production": False,
            "deepstream_runtime": True,
            "inference": True,
            "nested_unverified_claim": {"execution_ready": True},
        },
    }
    _write_json(builder_path, replacement)
    plan["runtime_target"]["prepared_closed_engine_builder_root"] = _subject_pin(
        builder_path, tmp_path
    )
    result = subject.validate_plan(
        _resign_successor(plan), root=tmp_path, replay_media=False
    )
    assert result["status"] == "pass_blocked_as_designed"


def test_resigned_replacement_lease_can_carry_unchecked_plan_semantics(
    tmp_path: Path,
) -> None:
    _copy_control_tree(tmp_path)
    plan = _successor_plan()
    lease_path = tmp_path / plan["gpu_lease"]["contract"]["path"]
    replacement = {
        "schema_version": "deepsafe.gpu-lease-contract/v5",
        "status": "locked-no-live-plan",
        "authorized_owner_kinds": ["capacity_5min"],
        "activation_policy": {
            "published_live_plan": False,
            "default_plan": True,
            "caller_argument_overrides": True,
        },
        "receipt_policy": {"published_receipt_in_this_delivery": True},
        "nested_unverified_claim": {"live_plan_available": True},
    }
    _write_json(lease_path, replacement)
    plan["gpu_lease"]["contract"] = _subject_pin(lease_path, tmp_path)
    result = subject.validate_plan(
        _resign_successor(plan), root=tmp_path, replay_media=False
    )
    assert result["status"] == "pass_blocked_as_designed"


def test_resigned_control_projection_accepts_media_substitution_with_matching_scene() -> None:
    work_parent = ROOT / "validation/work"
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ds91-preflight-review-", dir=work_parent) as name:
        work = Path(name)
        replacement_media = work / "replacement-media.bin"
        replacement_media.write_bytes(b"not-an-mp4-independent-lineage-fixture\n")

        predecessor = legacy.load_json(
            ROOT / "validation/plans/deepstream-full-stack-benchmark-v1.json"
        )
        relative_media = replacement_media.relative_to(ROOT).as_posix()
        predecessor["sources"][0]["video_type"] = "replacement_exact_bytes"
        predecessor["sources"][0]["view_types"] = ["high_oblique", "adverse_weather"]
        predecessor["sources"][0]["plan_uri"] = f"file:///workspace/{relative_media}"
        predecessor["sources"][0]["media_pin"] = _legacy_pin(replacement_media, ROOT)
        predecessor["fingerprint_sha256"] = legacy.fingerprint(predecessor)
        predecessor_path = work / "legacy-source-replacement.json"
        _write_json(predecessor_path, predecessor)

        scenes = review.load_workspace_json("validation/open_video_review/scenes.json")
        scenes["scenes"][0]["video_path"] = relative_media
        scenes_path = work / "scene-replacement.json"
        _write_json(scenes_path, scenes)

        plan = _successor_plan()
        plan["source_matrix"]["legacy_plan_source_projection"] = _subject_pin(
            predecessor_path, ROOT
        )
        plan["source_matrix"]["qualitative_scene_contract"] = _subject_pin(
            scenes_path, ROOT
        )
        result = subject.validate_plan(
            _resign_successor(plan), root=ROOT, replay_media=True
        )
        assert result["status"] == "pass_blocked_as_designed"
        assert result["source_count"] == 12
        assert result["distinct_video_types"] == 12


def test_subject_accepts_parent_directory_symlink_inside_selected_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    real = root / "real"
    real.mkdir(parents=True)
    artifact = real / "artifact.bin"
    artifact.write_bytes(b"exact-content")
    (root / "alias").symlink_to(real, target_is_directory=True)
    pin = {
        "path": "alias/artifact.bin",
        "bytes": len(b"exact-content"),
        "sha256": hashlib.sha256(b"exact-content").hexdigest(),
    }
    assert subject.replay_pin(pin, "parent-symlink", root=root) == b"exact-content"


def test_subject_rejects_final_component_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"content")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(subject.DS91PreflightError, match="cannot open exact pin"):
        subject.read_stable_regular(link)


def test_subject_rejects_hard_link(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"content")
    os.link(first, second)
    with pytest.raises(subject.DS91PreflightError, match="hard-linked"):
        subject.read_stable_regular(first)


def test_subject_rejects_static_root_escape_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    artifact = outside / "artifact.bin"
    artifact.write_bytes(b"outside")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    pin = {
        "path": "escape/artifact.bin",
        "bytes": len(b"outside"),
        "sha256": hashlib.sha256(b"outside").hexdigest(),
    }
    with pytest.raises(subject.DS91PreflightError, match="path escapes root"):
        subject.replay_pin(pin, "root-escape", root=root)


def test_subject_resolve_then_open_is_not_anchored_to_selected_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    slot = root / "slot"
    outside = tmp_path / "outside"
    slot.mkdir(parents=True)
    outside.mkdir()
    (slot / "artifact.bin").write_bytes(b"inside-version")
    expected = b"outside-version"
    (outside / "artifact.bin").write_bytes(expected)
    pin = {
        "path": "slot/artifact.bin",
        "bytes": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
    }

    real_open = os.open
    selected = root / "slot" / "artifact.bin"
    changed = False

    def open_after_identity_change(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal changed
        if not changed and Path(path) == selected:
            (root / "slot").rename(root / "slot-before-open")
            (root / "slot").symlink_to(outside, target_is_directory=True)
            changed = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(subject.os, "open", open_after_identity_change)
    assert subject.replay_pin(pin, "identity-change", root=root) == expected
    assert changed is True


def test_review_reader_rejects_parent_component_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    real = root / "real"
    real.mkdir(parents=True)
    (real / "artifact.bin").write_bytes(b"content")
    (root / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        with review.held_regular("alias/artifact.bin", maximum=100, root=root):
            pass


def test_subject_duplicate_json_is_rejected() -> None:
    with pytest.raises(subject.DS91PreflightError, match="duplicate JSON key"):
        subject.strict_object(b'{"x":1,"x":2}', "fixture")


def test_review_duplicate_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        review.strict_json(b'{"x":1,"x":2}', "fixture")


def test_review_nonfinite_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite JSON token"):
        review.strict_json(b'{"x":NaN}', "fixture")


def test_direct_subject_script_entrypoint_currently_fails() -> None:
    process = subprocess.run(
        [sys.executable, "validation/deepstream91_full_stack_preflight_v1.py", "--skip-media-replay"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "ModuleNotFoundError" in process.stderr


def test_module_subject_entrypoint_succeeds_and_stays_blocked() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.deepstream91_full_stack_preflight_v1",
            "--skip-media-replay",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["status"] == "pass_blocked_as_designed"


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        ("decision", lambda value: value.__setitem__("decision", "ACCEPT")),
        (
            "authority",
            lambda value: value["authority"].__setitem__("gpu_workload_authorized", True),
        ),
        ("findings", lambda value: value["findings"].pop()),
        (
            "severity",
            lambda value: value["findings"][0].__setitem__("severity", "P2"),
        ),
        (
            "criteria",
            lambda value: value["criterion_assessment"][5].__setitem__("result", "PASS"),
        ),
        (
            "subject_pin",
            lambda value: value["subject_pins"][0].__setitem__("sha256", "0" * 64),
        ),
        (
            "review_source_pin",
            lambda value: value["review_source_pins"][0].__setitem__("sha256", "0" * 64),
        ),
        (
            "baseline",
            lambda value: value["baseline_replay"].__setitem__("blockers_exact", False),
        ),
        (
            "adversarial",
            lambda value: value["adversarial_replay"].__setitem__(
                "media_substitution_accepted", False
            ),
        ),
    ],
)
def test_semantically_resigned_review_mutations_fail_closed(
    field: str, mutator: object
) -> None:
    value = review.load_review()
    mutator(value)  # type: ignore[operator]
    with pytest.raises(ValueError):
        review.verify(_resign_review(value))


def test_corrupt_review_fingerprint_fails_closed() -> None:
    value = review.load_review()
    value["review_fingerprint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="review fingerprint mismatch"):
        review.verify(value)


def test_resigned_extra_nested_review_field_fails_schema() -> None:
    value = review.load_review()
    value["scope"]["hidden_authority"] = False
    with pytest.raises(ValueError, match="schema"):
        review.verify(_resign_review(value))
