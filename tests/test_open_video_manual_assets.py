import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from validation import open_video_manual_assets as assets


class _FakeFrame:
    shape = (3, 4, 3)

    def __init__(self, value=0, detections=None):
        self.value = value
        self.detections = detections

    def copy(self):
        return _FakeFrame(self.value, self.detections)


@pytest.fixture(autouse=True)
def _cpu_visual_stubs(monkeypatch):
    """Exercise the materializer without requiring optional OpenCV/NumPy wheels."""

    monkeypatch.setattr(
        assets,
        "_draw_overlay",
        lambda frame, detections, threshold: _FakeFrame(
            frame.value, (len(detections), threshold)
        ),
    )
    monkeypatch.setattr(
        assets,
        "_png_bytes",
        lambda frame: (
            b"\x89PNG\r\n\x1a\nfixture:"
            + str(frame.value).encode("ascii")
            + b":"
            + repr(frame.detections).encode("ascii")
        ),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source(index: int, *, sensitive: bool = False):
    scene_id = "closed-scene" if sensitive else "ordinary-scene"
    return {
        "schema_version": "deepsafe.open-video-source-frame-review/v1",
        "record_id": f"closed.f{index:06d}" if sensitive else f"ordinary.f{index:06d}",
        "scene_id": scene_id,
        "sensitive": sensitive,
        "frame": {
            "index": index,
            "timestamp_seconds": index / 5,
        },
        "observation": {
            "person_expected": "not_reviewed" if sensitive else "yes",
            "visible_person_count_range": {
                "min": None if sensitive else 1,
                "max": None if sensitive else 1,
            },
            "scorable_person_count_range": {
                "min": None if sensitive else 1,
                "max": None if sensitive else 1,
            },
            "scale_classes": [] if sensitive else ["medium"],
            "dominant_scale": None if sensitive else "medium",
            "occlusion": "unknown" if sensitive else "none",
            "top_view": False,
            "high_oblique": False,
            "medium_close": not sensitive,
            "partial_body_only": False,
            "ambiguity": [],
            "notes": [],
        },
    }


def _prediction(scene_id: str, frame_index: int):
    confidence = 0.2 if frame_index == 0 else 0.9
    return {
        "schema_version": "deepsafe.person-detections/v1",
        "sequence_id": scene_id,
        "frame_index": frame_index,
        "image_width": 4,
        "image_height": 3,
        "timestamp_ns": frame_index * 200_000_000,
        "source_uri": "file:///workspace/fixture.mp4",
        "model_id": "fixture-yolo",
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": confidence,
                "bbox_norm_xywh": [0.0, 0.0, 0.5, 2 / 3],
            }
        ],
    }


def _valid_source(*_args, **_kwargs):
    records = _args[0]
    return {
        "valid": True,
        "record_count": len(records),
        "closed_review_records": sum(row["sensitive"] for row in records),
    }


def _complete_inspection(*_args, **_kwargs):
    return {
        "state": "complete",
        "reasons": [],
        "checks": {"contiguous_zero_based": True},
    }


def _decoder(_path: Path, indices: set[int]):
    return {
        index: _FakeFrame(index % 255)
        for index in indices
    }


def _fixture(tmp_path: Path):
    workspace = tmp_path / "workspace"
    results = workspace / "validation/results"
    output = results / "open-video-review/manual-assets"
    source_path = workspace / "validation/open_video_review/source.jsonl"
    plan_path = results / "open-video-review/campaign-plan.json"
    video_path = workspace / "data/ordinary.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"cpu-only-video-fixture")

    source_rows = [_source(index) for index in range(21)] + [
        _source(0, sensitive=True)
    ]
    _jsonl(source_path, source_rows)

    scenes = [
        {
            "id": "ordinary-scene",
            "sensitive": False,
            "video": {
                "path": "data/ordinary.mp4",
                "size_bytes": video_path.stat().st_size,
                "sha256": _sha(video_path),
            },
            "video_metadata": {
                "width": 4,
                "height": 3,
                "exact_decoded_frame_count": 21,
            },
        },
        {
            "id": "closed-scene",
            "sensitive": True,
            "video": {
                "path": "data/closed.mp4",
                "size_bytes": 1,
                "sha256": "f" * 64,
            },
            "video_metadata": {
                "width": 4,
                "height": 3,
                "exact_decoded_frame_count": 1,
            },
        },
    ]
    jobs = []
    for profile in assets.PROFILES:
        run_relative = f"validation/results/open-video-review/ordinary-scene/{profile}"
        run_root = workspace / run_relative
        guard_path = run_root / "safety/gpu-guard-report.json"
        guard_path.parent.mkdir(parents=True)
        guard_path.write_text(
            json.dumps(
                {
                    "schema_version": "deepsafe.gpu-guarded-process/v1",
                    "status": "complete",
                }
            ),
            encoding="utf-8",
        )
        (run_root / "run-manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "sequence_id": "ordinary-scene",
                    "model_input": profile,
                    "gpu_safety": {
                        "status": "complete",
                        "report": f"{run_relative}/safety/gpu-guard-report.json",
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_root / "conversion.json").write_text(
            json.dumps(
                {
                    "decoded_frame_files": 21,
                    "exported_frame_records": 21,
                    "skipped_unannotated_frames": 0,
                    "kitti_coordinate_dimensions": [4, 3],
                    "json_image_dimensions": [4, 3],
                }
            ),
            encoding="utf-8",
        )
        _jsonl(
            run_root / "predictions.jsonl",
            [_prediction("ordinary-scene", index) for index in range(21)],
        )
        jobs.append(
            {
                "job_id": f"ordinary-scene:{profile}",
                "scene_id": "ordinary-scene",
                "model_input": profile,
                "sensitive": False,
                "run_root": run_relative,
                "expected_frame_count": 21,
                "coordinate_contract": {
                    "source_dimensions": [4, 3],
                    "streammux_dimensions": [4, 3],
                },
            }
        )
    plan = {
        "schema_version": "deepsafe.open-video-review-plan/v1",
        "campaign": {
            "model_input_sizes": [640, 960],
            "review_confidence_threshold": 0.25,
        },
        "scenes": scenes,
        "jobs": jobs,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return {
        "workspace": workspace,
        "results": results,
        "output": output,
        "source": source_path,
        "plan": plan_path,
        "source_rows": source_rows,
        "plan_object": plan,
        "video": video_path,
    }


def _materialize(fixture, **overrides):
    arguments = {
        "source_records_path": fixture["source"],
        "campaign_plan_path": fixture["plan"],
        "results_root": fixture["results"],
        "output_root": fixture["output"],
        "workspace_root": fixture["workspace"],
        "source_validator": _valid_source,
        "inspector": _complete_inspection,
        "decoder": _decoder,
    }
    arguments.update(overrides)
    return assets.materialize_from_paths(**arguments)


@pytest.mark.parametrize(
    "surface",
    ["source", "plan", "prediction", "guard", "manifest", "conversion"],
)
def test_materializer_rejects_duplicate_keys_on_every_json_input_surface(
    tmp_path, surface
):
    fixture = _fixture(tmp_path)
    if surface == "source":
        path = fixture["source"]
        before = path.read_text(encoding="utf-8")
        after = before.replace(
            '"person_expected":"yes"',
            '"person_expected":"no","person_expected":"yes"',
            1,
        )
    elif surface == "plan":
        path = fixture["plan"]
        before = path.read_text(encoding="utf-8")
        after = before.replace(
            '"review_confidence_threshold": 0.25',
            '"review_confidence_threshold": 0.5, '
            '"review_confidence_threshold": 0.25',
            1,
        )
    else:
        run_root = (
            fixture["results"] / "open-video-review/ordinary-scene/640"
        )
        path = {
            "prediction": run_root / "predictions.jsonl",
            "guard": run_root / "safety/gpu-guard-report.json",
            "manifest": run_root / "run-manifest.json",
            "conversion": run_root / "conversion.json",
        }[surface]
        before = path.read_text(encoding="utf-8")
        replacements = {
            "prediction": (
                '"class_name":"person"',
                '"class_name":"vehicle","class_name":"person"',
            ),
            "guard": (
                '"status": "complete"',
                '"status": "failed", "status": "complete"',
            ),
            "manifest": (
                '"status": "complete"',
                '"status": "failed", "status": "complete"',
            ),
            "conversion": (
                '"decoded_frame_files": 21',
                '"decoded_frame_files": 0, "decoded_frame_files": 21',
            ),
        }
        old, new = replacements[surface]
        after = before.replace(old, new, 1)
    assert after != before
    path.write_text(after, encoding="utf-8")

    with pytest.raises(assets.MaterializationError):
        _materialize(fixture)
    assert not (fixture["output"] / "index.json").exists()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_materializer_rejects_nonfinite_prediction_numbers(tmp_path, token):
    fixture = _fixture(tmp_path)
    path = (
        fixture["results"]
        / "open-video-review/ordinary-scene/640/predictions.jsonl"
    )
    before = path.read_text(encoding="utf-8")
    after = before.replace('"confidence":0.2', f'"confidence":{token}', 1)
    assert after != before
    path.write_text(after, encoding="utf-8")

    with pytest.raises(assets.MaterializationError):
        _materialize(fixture)
    assert not (fixture["output"] / "index.json").exists()


def test_default_dry_plan_builds_exact_matrix_without_inspection_or_decode(tmp_path):
    fixture = _fixture(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry plan attempted to inspect/decode")

    # Core dry planning accepts no inspector/decoder and therefore cannot call
    # either GPU-job inspection or video decoding.
    result = assets.build_dry_run_plan(
        fixture["source_rows"],
        fixture["plan_object"],
        source_validator=_valid_source,
        scene_manifest=fixture["workspace"] / "unused-scenes.json",
        normalization_report=fixture["workspace"] / "unused-normalization.tsv",
        workspace_root=fixture["workspace"],
    )
    forbidden  # make the no-call intent explicit to static readers
    assert result["mode"] == "dry-run"
    assert result["expected_decision_count"] == 42
    assert result["gpu_execution_requested"] is False
    assert result["network_requested"] is False
    assert len(result["ordinary_jobs_requiring_complete_gpu_guard"]) == 2
    assert not any(row["scene_id"] == "closed-scene" for row in result["decisions"])


def test_materializes_exact_42_by_four_matrix_and_filters_at_plan_threshold(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    inspected = []

    def inspecting(job, scene):
        inspected.append((scene["id"], job["model_input"]))
        return _complete_inspection(job, scene)

    def forbidden_subprocess(*_args, **_kwargs):
        raise AssertionError("materializer attempted a subprocess/GPU/Docker action")

    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    index = _materialize(fixture, inspector=inspecting)

    assert index["schema_version"] == assets.INDEX_SCHEMA_VERSION
    assert index["decision_count"] == 42
    assert index["asset_count"] == 168
    assert index["sensitive_media_included"] is False
    assert len(index["decisions"]) == 42
    assert len(index["assets"]) == 168
    assert not any(row["scene_id"] == "closed-scene" for row in index["decisions"])
    assert inspected == [("ordinary-scene", 640), ("ordinary-scene", 960)]
    assert {
        (row["source_review_id"], row["model_input"])
        for row in index["decisions"]
    } == {
        (f"ordinary.f{frame:06d}", profile)
        for frame in range(21)
        for profile in (640, 960)
    }
    for decision in index["decisions"]:
        assert set(decision["evidence"]) == set(assets.ASSET_KINDS)
        assert decision["source_observation"] == fixture["source_rows"][
            decision["frame_index"]
        ]["observation"]
        for kind, asset_id in decision["evidence"].items():
            assert asset_id.startswith("a_")
            entry = index["assets"][asset_id]
            assert list(entry) == list(assets.ASSET_FIELDS)
            assert entry["kind"] == kind
            assert not Path(entry["relative_path"]).is_absolute()
            assert ".." not in Path(entry["relative_path"]).parts
            path = fixture["results"] / entry["relative_path"]
            assert path.is_file()
            assert path.stat().st_size == entry["size_bytes"]
            assert _sha(path) == entry["sha256"]

    low = next(
        row
        for row in index["decisions"]
        if row["frame_index"] == 0 and row["model_input"] == 640
    )
    prediction_asset = index["assets"][low["evidence"]["predictions"]]
    prediction_row = json.loads(
        (fixture["results"] / prediction_asset["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert prediction_row["frame_index"] == 0
    assert prediction_row["detections"] == []


def test_ids_and_index_are_deterministic_and_schema_valid(tmp_path):
    fixture = _fixture(tmp_path)
    first = _materialize(fixture)
    first_bytes = (fixture["output"] / "index.json").read_bytes()
    second = _materialize(fixture)
    assert second == first
    assert (fixture["output"] / "index.json").read_bytes() == first_bytes

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        Path("validation/schemas/open-video-manual-assets-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(first)


@pytest.mark.parametrize("failure", ["inspector", "guard", "missing_prediction", "video"])
def test_tamper_missing_and_guard_fail_closed_before_index(tmp_path, failure):
    fixture = _fixture(tmp_path)
    kwargs = {}
    if failure == "inspector":
        kwargs["inspector"] = lambda *_args, **_kwargs: {
            "state": "incomplete",
            "reasons": ["GPU guard failed"],
        }
    elif failure == "guard":
        guard = next(
            fixture["results"].glob(
                "open-video-review/ordinary-scene/*/safety/gpu-guard-report.json"
            )
        )
        guard.write_text(
            json.dumps(
                {
                    "schema_version": "deepsafe.gpu-guarded-process/v1",
                    "status": "failed",
                }
            ),
            encoding="utf-8",
        )
    elif failure == "missing_prediction":
        predictions = (
            fixture["results"]
            / "open-video-review/ordinary-scene/640/predictions.jsonl"
        )
        rows = predictions.read_text(encoding="utf-8").splitlines()
        predictions.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    else:
        fixture["video"].write_bytes(b"tampered-video")

    with pytest.raises(assets.MaterializationError):
        _materialize(fixture, **kwargs)
    assert not (fixture["output"] / "index.json").exists()


def test_rejects_symlinked_prediction_and_wrong_decode_identity_or_dimensions(tmp_path):
    fixture = _fixture(tmp_path)
    predictions = (
        fixture["results"]
        / "open-video-review/ordinary-scene/640/predictions.jsonl"
    )
    copy_path = predictions.with_name("predictions-copy.jsonl")
    predictions.replace(copy_path)
    predictions.symlink_to(copy_path.name)
    with pytest.raises(assets.MaterializationError, match="symlinks"):
        _materialize(fixture)

    fixture = _fixture(tmp_path / "identity")
    with pytest.raises(assets.MaterializationError, match="wrong exact-frame identity"):
        _materialize(fixture, decoder=lambda _path, indices: {})

    fixture = _fixture(tmp_path / "dimensions")
    with pytest.raises(assets.MaterializationError, match="dimensions/channels"):
        _materialize(
            fixture,
            decoder=lambda _path, indices: {
                index: type("WrongFrame", (), {"shape": (2, 4, 3)})()
                for index in indices
            },
        )


def test_atomic_failure_preserves_previous_valid_index(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    _materialize(fixture)
    index_path = fixture["output"] / "index.json"
    original = index_path.read_bytes()
    real_atomic_write = assets._atomic_write

    def fail_index(path, content):
        if path.name == "index.json":
            raise OSError("injected index publication failure")
        return real_atomic_write(path, content)

    monkeypatch.setattr(assets, "_atomic_write", fail_index)
    with pytest.raises(OSError, match="injected"):
        _materialize(fixture)
    assert index_path.read_bytes() == original
    assert json.loads(original)["status"] == "complete"


@pytest.mark.parametrize("run_root", ["../../outside", "/tmp/outside"])
def test_rejects_unsafe_run_root_before_calling_inspector(tmp_path, run_root):
    fixture = _fixture(tmp_path)
    for job in fixture["plan_object"]["jobs"]:
        if job["model_input"] == 640:
            job["run_root"] = run_root
    fixture["plan"].write_text(
        json.dumps(fixture["plan_object"]), encoding="utf-8"
    )
    inspected = []

    def spy_inspector(*_args, **_kwargs):
        inspected.append(True)
        return _complete_inspection()

    with pytest.raises(assets.MaterializationError):
        _materialize(fixture, inspector=spy_inspector)
    assert inspected == []
    assert not (fixture["output"] / "index.json").exists()


@pytest.mark.parametrize("filename", ["run-manifest.json", "conversion.json"])
def test_manifest_or_conversion_change_before_publication_is_rejected(
    tmp_path, filename
):
    fixture = _fixture(tmp_path)
    target = (
        fixture["results"]
        / "open-video-review/ordinary-scene/640"
        / filename
    )

    def mutating_decoder(_path: Path, indices: set[int]):
        target.write_bytes(target.read_bytes() + b"\n")
        return _decoder(_path, indices)

    with pytest.raises(assets.MaterializationError, match="changed before publication"):
        _materialize(fixture, decoder=mutating_decoder)
    assert not (fixture["output"] / "index.json").exists()


def test_asset_entry_contract_has_no_embedded_asset_id():
    decision = {
        "decision_id": "source:640",
        "source_review_id": "source",
        "scene_id": "scene",
        "frame_index": 1,
        "model_input": 640,
    }
    entry = assets._asset_entry(
        decision=decision,
        kind="predictions",
        relative_path=(
            "open-video-review/manual-assets/objects/"
            + "b_"
            + "0" * 64
            + "/a_"
            + "1" * 64
            + ".jsonl"
        ),
        content=b"{}\n",
    )
    assert tuple(entry) == assets.ASSET_FIELDS
    assert "asset_id" not in entry
