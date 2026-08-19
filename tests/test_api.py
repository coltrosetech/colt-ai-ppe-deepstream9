import os
os.environ["DEEPSAFE_DATA"] = "/tmp/deepsafe-test"
from fastapi.testclient import TestClient
from admin.app import app, store


def test_health():
    with TestClient(app) as client:
        result = client.get("/api/health").json()
        assert result["ok"] is True
        assert result["deepstream"]["person_profile"]["max_batch_size"] == 12


def test_select_person_inference_profile():
    original = store.load()
    try:
        with TestClient(app) as client:
            response = client.put("/api/inference", json={"person_profile": "yolo11s-960"})
            assert response.status_code == 200
            selected = response.json()["selected"]
            assert selected["input_width"] == 960
            assert selected["config_file"] == "/models/person/960/config_infer_primary.txt"
            status = client.get("/api/status").json()
            assert status["source_plan"]["benchmark_simulated_sources"] == 12
            assert status["source_plan"]["benchmark_duration_seconds"] == 300
    finally:
        store.save(original)


def test_direct_api_cannot_enable_unconfigured_pose_or_ppe():
    original = store.load()
    try:
        with TestClient(app) as client:
            for module in ("pose", "ppe"):
                analytics = client.get("/api/analytics").json()
                analytics[module]["enabled"] = True
                response = client.put("/api/analytics", json=analytics)
                assert response.status_code == 409
                assert module in response.json()["detail"]
                assert client.get("/api/analytics").json()[module]["enabled"] is False
    finally:
        store.save(original)
