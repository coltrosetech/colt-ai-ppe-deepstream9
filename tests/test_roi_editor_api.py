import json
from pathlib import Path

from fastapi.testclient import TestClient

from admin.app import app, pipeline


def _valid_plan(video_id: str = "01") -> dict:
    return {
        "video_id": video_id,
        "start_seconds": 12.5,
        "end_seconds": 18.25,
        "rois": [
            {
                "name": "Kısıtlı Alan",
                "points": [
                    {"x": 0.2, "y": 0.2},
                    {"x": 0.8, "y": 0.2},
                    {"x": 0.75, "y": 0.8},
                    {"x": 0.25, "y": 0.8},
                ],
            }
        ],
    }


def _prepare_roots(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    validation_root = tmp_path / "validation-results"
    data_root = tmp_path / "data"
    (validation_root / "content-editor/clean-sources").mkdir(parents=True)
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(validation_root))
    monkeypatch.setenv("DEEPSAFE_DATA", str(data_root))
    monkeypatch.setenv("DEEPSAFE_ADMIN_TOKEN", "change-me")
    return validation_root, data_root


def test_lists_only_closed_approved_registry_with_editor_metadata(
    monkeypatch,
    tmp_path,
):
    validation_root, _ = _prepare_roots(monkeypatch, tmp_path)
    (validation_root / "content-editor/clean-sources/01.mp4").write_bytes(
        b"clean-preview"
    )

    with TestClient(app) as client:
        response = client.get("/api/roi-editor/videos")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    videos = response.json()["videos"]
    assert [item["video_id"] for item in videos] == [
        "01",
        "02",
        "03",
        "04",
        "06",
        "07",
        "08",
        "14",
        "18",
    ]
    first = videos[0]
    assert {
        "video_id",
        "display_name",
        "stream_url",
        "duration_seconds",
        "width",
        "height",
        "fps",
        "frame_count",
    } <= first.keys()
    assert first["stream_url"] == "/api/roi-editor/videos/01/stream"
    assert first["playable"] is True
    assert videos[1]["playable"] is False
    assert "unavailable_reason" in videos[1]
    assert all("source_path" not in item for item in videos)


def test_stream_supports_range_and_rejects_unknown_or_missing_preview(
    monkeypatch,
    tmp_path,
):
    validation_root, _ = _prepare_roots(monkeypatch, tmp_path)
    preview = validation_root / "content-editor/clean-sources/01.mp4"
    preview.write_bytes(b"0123456789")

    with TestClient(app) as client:
        response = client.get(
            "/api/roi-editor/videos/01/stream",
            headers={"Range": "bytes=2-5"},
        )
        invalid_range = client.get(
            "/api/roi-editor/videos/01/stream",
            headers={"Range": "bytes=999-1000"},
        )
        unknown = client.get("/api/roi-editor/videos/not-approved/stream")
        missing = client.get("/api/roi-editor/videos/02/stream")

    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.headers["content-disposition"].startswith("inline;")
    assert invalid_range.status_code == 416
    assert invalid_range.headers["content-range"] == "*/10"
    assert unknown.status_code == 404
    assert missing.status_code == 409


def test_stream_rejects_symlink_escape(monkeypatch, tmp_path):
    validation_root, _ = _prepare_roots(monkeypatch, tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not-approved")
    (
        validation_root / "content-editor/clean-sources/01.mp4"
    ).symlink_to(outside)

    with TestClient(app) as client:
        listing = client.get("/api/roi-editor/videos").json()
        response = client.get("/api/roi-editor/videos/01/stream")

    assert listing["videos"][0]["playable"] is False
    assert response.status_code == 409


def test_create_plan_is_atomic_immutable_and_does_not_start_pipeline(
    monkeypatch,
    tmp_path,
):
    _, data_root = _prepare_roots(monkeypatch, tmp_path)

    def forbidden_start(*_args, **_kwargs):
        raise AssertionError("plan save must not start pipeline")

    monkeypatch.setattr(pipeline, "start", forbidden_start)
    with TestClient(app) as client:
        response = client.post("/api/roi-editor/plans", json=_valid_plan())
        assert response.status_code == 201
        receipt = response.json()
        detail = client.get(receipt["plan_url"])

    assert receipt["status"] == "ready"
    assert receipt["gpu_or_model_execution"] is False
    assert len(receipt["plan_id"]) == 32
    assert detail.status_code == 200
    assert detail.headers["etag"].startswith('"')
    plan = detail.json()
    assert plan["plan_id"] == receipt["plan_id"]
    assert plan["status"] == "ready"
    assert plan["clip"] == {
        "semantics": "half_open",
        "start_seconds": 12.5,
        "end_seconds": 18.25,
        "start_frame": 375,
        "end_frame_exclusive": 548,
        "output_frame_count": 173,
    }
    assert plan["rois"][0]["coordinate_space"] == "source_video_normalized"
    assert plan["rois"][0]["polygon_norm"][0] == [0.2, 0.2]
    assert plan["execution"]["requested"] is False
    assert plan["execution"]["gpu_or_model_execution"] is False

    saved = data_root / "roi-editor/plans" / f"{receipt['plan_id']}.json"
    assert saved.is_file()
    assert saved.stat().st_mode & 0o077 == 0
    assert json.loads(saved.read_text(encoding="utf-8")) == plan


def test_plan_mutation_requires_configured_admin_token(monkeypatch, tmp_path):
    _prepare_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("DEEPSAFE_ADMIN_TOKEN", "editor-secret")

    with TestClient(app) as client:
        public_listing = client.get("/api/roi-editor/videos")
        rejected = client.post("/api/roi-editor/plans", json=_valid_plan())
        accepted = client.post(
            "/api/roi-editor/plans",
            json=_valid_plan(),
            headers={"Authorization": "Bearer editor-secret"},
        )

    assert public_listing.status_code == 200
    assert rejected.status_code == 401
    assert accepted.status_code == 201


def test_local_panel_mode_accepts_plan_without_admin_token(monkeypatch, tmp_path):
    _prepare_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("DEEPSAFE_ADMIN_TOKEN", "editor-secret")
    monkeypatch.setenv("DEEPSAFE_ADMIN_AUTH_DISABLED", "true")

    with TestClient(app) as client:
        response = client.post("/api/roi-editor/plans", json=_valid_plan())

    assert response.status_code == 201
    assert response.json()["gpu_or_model_execution"] is False


def test_plan_validation_rejects_unsafe_geometry_bounds_and_extra_fields(
    monkeypatch,
    tmp_path,
):
    _prepare_roots(monkeypatch, tmp_path)
    invalid_payloads = []

    duplicate = _valid_plan()
    duplicate["rois"][0]["points"][3] = {"x": 0.2, "y": 0.2}
    invalid_payloads.append(duplicate)

    self_intersecting = _valid_plan()
    self_intersecting["rois"][0]["points"] = [
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.9},
        {"x": 0.9, "y": 0.1},
        {"x": 0.1, "y": 0.9},
    ]
    invalid_payloads.append(self_intersecting)

    outside = _valid_plan()
    outside["rois"][0]["points"][0]["x"] = 1.01
    invalid_payloads.append(outside)

    too_long = _valid_plan()
    too_long["end_seconds"] = 301.0
    invalid_payloads.append(too_long)

    no_complete_frame = _valid_plan()
    no_complete_frame["start_seconds"] = 1.001
    no_complete_frame["end_seconds"] = 1.002
    invalid_payloads.append(no_complete_frame)

    extra = _valid_plan()
    extra["execute"] = True
    invalid_payloads.append(extra)

    with TestClient(app) as client:
        responses = [
            client.post("/api/roi-editor/plans", json=payload)
            for payload in invalid_payloads
        ]
        unknown = client.post(
            "/api/roi-editor/plans",
            json=_valid_plan("99"),
        )
        traversal = client.get("/api/roi-editor/plans/not-a-plan-id")

    assert all(response.status_code == 422 for response in responses)
    assert unknown.status_code == 404
    assert traversal.status_code == 404
