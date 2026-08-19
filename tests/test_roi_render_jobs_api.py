import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from admin.roi_editor import (
    APPROVED_VIDEOS,
    ProcessingPlanIn,
    RoiEditorService,
)
from admin.roi_render_jobs import (
    RoiRenderJobService,
    create_roi_render_jobs_router,
)


VIDEO = APPROVED_VIDEOS[0]


def _plan_value(*, roi_count: int = 1) -> ProcessingPlanIn:
    roi = {
        "name": "Kısıtlı Alan",
        "points": [
            {"x": 0.2, "y": 0.2},
            {"x": 0.8, "y": 0.2},
            {"x": 0.75, "y": 0.8},
            {"x": 0.25, "y": 0.8},
        ],
    }
    return ProcessingPlanIn.model_validate(
        {
            "video_id": VIDEO.video_id,
            "start_seconds": 12.5,
            "end_seconds": 18.25,
            "rois": [
                {**roi, "name": f"Kısıtlı Alan {index + 1}"}
                for index in range(roi_count)
            ],
        }
    )


def _prepare_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    source = repository / VIDEO.source_path
    detections = repository / VIDEO.detections_path
    template = (
        repository
        / f"content/configs/colt-collbrai-person-full-{VIDEO.video_id}.json"
    )
    source.parent.mkdir(parents=True)
    detections.parent.mkdir(parents=True)
    template.parent.mkdir(parents=True)
    source.write_bytes(b"approved-video")
    detections.write_text("{}\n", encoding="utf-8")
    template.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.content-roi-demo/v1",
                "demo_id": "template-only",
                "title": "İNSAN TESPİTİ",
                "camera_label": "CAM-01",
                "disclosure": {
                    "mode": "production_inference",
                    "label": "ÜRETİM ÇIKTISI",
                },
                "source": {
                    "asset_id": "approved-01",
                    "video_path": VIDEO.source_path,
                    "source_url": "https://example.test/source",
                    "license_id": "CC-BY-4.0",
                    "license_url": "https://example.test/license",
                    "attribution": "Test source",
                    "modification_notice": "Test rendering",
                },
                "detections": {
                    "kind": "predictions_jsonl",
                    "path": VIDEO.detections_path,
                    "sequence_id": VIDEO.sequence_id,
                    "confidence_threshold": 0.25,
                    "expected_model_id": VIDEO.expected_model_id,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return repository


def _services(
    tmp_path: Path,
    process_factory,
) -> tuple[RoiEditorService, RoiRenderJobService, Path, Path]:
    repository = _prepare_repository(tmp_path)
    data_root = tmp_path / "state"
    render_root = repository / "runtime-results/roi-editor/jobs"
    editor = RoiEditorService(
        data_root=data_root,
        approved_videos=(VIDEO,),
    )
    jobs = RoiRenderJobService(
        editor,
        approved_videos=(VIDEO,),
        repository_root=repository,
        data_root=data_root,
        render_root=render_root,
        process_factory=process_factory,
    )
    return editor, jobs, repository, data_root


def _app(jobs: RoiRenderJobService) -> FastAPI:
    def auth(authorization: str | None = Header(default=None)):
        if authorization != "Bearer editor-secret":
            raise HTTPException(401, "unauthorized")

    @asynccontextmanager
    async def lifespan(_app):
        jobs.start()
        try:
            yield
        finally:
            jobs.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.include_router(create_roi_render_jobs_router(jobs, auth))
    return app


def _wait_for_status(
    client: TestClient,
    job_url: str,
    expected: str,
) -> dict:
    for _attempt in range(200):
        response = client.get(job_url)
        assert response.status_code == 200
        value = response.json()
        if value["status"] == expected:
            return value
        time.sleep(0.005)
    raise AssertionError(f"job did not reach {expected}")


def test_execute_is_authenticated_cpu_only_and_serves_completed_mp4(
    tmp_path: Path,
):
    calls = []
    repository_holder: dict[str, Path] = {}

    class CompleteProcess:
        pid = None

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_process(command, **kwargs):
        calls.append((command, kwargs))
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        repository = repository_holder["path"]
        output = repository / config["output"]["directory"]
        output.mkdir(parents=True)
        (output / "demo.mp4").write_bytes(b"0123456789")
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "rendered",
                    "demo_id": config["demo_id"],
                    "gpu_or_model_execution": False,
                }
            ),
            encoding="utf-8",
        )
        return CompleteProcess()

    editor, jobs, repository, data_root = _services(
        tmp_path,
        fake_process,
    )
    repository_holder["path"] = repository
    receipt = editor.create_plan(_plan_value())

    with TestClient(_app(jobs)) as client:
        rejected = client.post(
            f"/api/roi-editor/plans/{receipt['plan_id']}/execute"
        )
        accepted = client.post(
            f"/api/roi-editor/plans/{receipt['plan_id']}/execute",
            headers={"Authorization": "Bearer editor-secret"},
        )
        assert rejected.status_code == 401
        assert accepted.status_code == 202
        queued = accepted.json()
        assert queued["gpu_or_model_execution"] is False
        completed = _wait_for_status(
            client,
            queued["job_url"],
            "complete",
        )
        output = client.get(
            completed["output_url"],
            headers={"Range": "bytes=2-5"},
        )

    assert completed["progress"] == 100
    assert completed["progress_percent"] == 100
    assert completed["gpu_or_model_execution"] is False
    assert output.status_code == 206
    assert output.content == b"2345"
    assert output.headers["content-range"] == "bytes 2-5/10"
    assert len(calls) == 1
    command, process_options = calls[0]
    assert command[:3] == [
        __import__("sys").executable,
        "-m",
        "content.roi_demo",
    ]
    assert command[-2:] == ["--render", "--allow-missing-preview-states"]
    assert process_options["shell"] is False
    assert process_options["start_new_session"] is True
    assert process_options["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert process_options["env"]["NVIDIA_VISIBLE_DEVICES"] == "none"

    config_path = Path(command[command.index("--config") + 1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["source"]["video_path"] == VIDEO.source_path
    assert config["detections"]["path"] == VIDEO.detections_path
    assert config["clip"] == {
        "start_seconds": 12.5,
        "end_seconds": 18.25,
    }
    assert config["roi"]["polygon_norm"][0] == [0.2, 0.2]
    assert config["output"]["directory"].startswith(
        "runtime-results/roi-editor/jobs/"
    )
    state_files = list((data_root / "roi-editor/jobs").glob("*.json"))
    assert len(state_files) == 1
    assert not list((data_root / "roi-editor/jobs").glob(".*.tmp"))


def test_execute_revalidates_saved_server_bindings_and_requires_one_roi(
    tmp_path: Path,
):
    calls = []

    def forbidden_process(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid plan must not start a process")

    editor, jobs, _repository, data_root = _services(
        tmp_path,
        forbidden_process,
    )
    tampered_receipt = editor.create_plan(_plan_value())
    tampered_path = (
        data_root
        / "roi-editor/plans"
        / f"{tampered_receipt['plan_id']}.json"
    )
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["analytics"]["detections_path"] = "other/predictions.jsonl"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    multi_receipt = editor.create_plan(_plan_value(roi_count=2))

    with TestClient(_app(jobs)) as client:
        tampered_response = client.post(
            f"/api/roi-editor/plans/{tampered_receipt['plan_id']}/execute",
            headers={"Authorization": "Bearer editor-secret"},
        )
        multi_response = client.post(
            f"/api/roi-editor/plans/{multi_receipt['plan_id']}/execute",
            headers={"Authorization": "Bearer editor-secret"},
        )

    assert tampered_response.status_code == 409
    assert multi_response.status_code == 409
    assert calls == []


def test_shutdown_terminates_active_process_and_marks_job_error(
    tmp_path: Path,
):
    started = threading.Event()

    class BlockingProcess:
        pid = None

        def __init__(self):
            self.return_code = None
            self.terminated = False

        def poll(self):
            return self.return_code

        def terminate(self):
            self.terminated = True
            self.return_code = -15

        def kill(self):
            self.return_code = -9

        def wait(self, timeout=None):
            return self.return_code

    process = BlockingProcess()

    def blocking_process(*_args, **_kwargs):
        started.set()
        return process

    editor, jobs, _repository, _data_root = _services(
        tmp_path,
        blocking_process,
    )
    receipt = editor.create_plan(_plan_value())
    jobs.start()
    queued = jobs.enqueue(receipt["plan_id"])
    assert started.wait(timeout=2)

    jobs.shutdown()
    state = jobs.get_job(queued["job_id"])

    assert process.terminated is True
    assert state["status"] == "error"
    assert state["output_url"] is None
    assert state["gpu_or_model_execution"] is False
