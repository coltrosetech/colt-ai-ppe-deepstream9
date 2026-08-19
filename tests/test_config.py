from pathlib import Path

import pytest

from deepstream.config import render_config, resolve_person_profile
from admin.store import _normalise


def test_person_graph_omits_unready_placeholder_modules(tmp_path):
    state = {"sources": [{"id": "1", "name": "cam", "uri": "rtsp://example/cam"}],
             "analytics": {"person": {"enabled": True, "confidence": .3, "interval": 0},
                           "pose": {"enabled": False, "confidence": .3, "interval": 0},
                           "ppe": {"enabled": False, "confidence": .4, "interval": 1}}}
    text = render_config(state, tmp_path / "app.ini").read_text()
    assert text.count("[source0]") == 1
    assert "[secondary-gie" not in text
    assert "[primary-gie]\nenable=1" in text
    assert "batch-size=1" in text
    assert "[source0]\nenable=1\ntype=4\nuri=rtsp://example/cam" in text
    assert "latency=200" in text
    assert "rtsp-reconnect-interval-sec=10" in text


def test_inference_resolution(tmp_path):
    state = {"sources": [], "inference": {"person_profile": "yolo11s-960"},
             "analytics": {"person": {"enabled": True, "confidence": .3, "interval": 0},
                           "pose": {"enabled": False, "confidence": .3, "interval": 0},
                           "ppe": {"enabled": False, "confidence": .4, "interval": 1}}}
    text = render_config(state, tmp_path / "app.ini").read_text()
    assert "width=1920\nheight=1080" in text
    assert "config-file=/models/person/960/config_infer_primary.txt" in text
    assert resolve_person_profile(state)["input_width"] == 960


@pytest.mark.parametrize("module", ["pose", "ppe"])
def test_unconfigured_modules_fail_closed(module, tmp_path):
    analytics = {
        "person": {"enabled": True, "confidence": .3, "interval": 0},
        "pose": {"enabled": False, "confidence": .3, "interval": 0},
        "ppe": {"enabled": False, "confidence": .4, "interval": 1},
    }
    analytics[module]["enabled"] = True
    state = {
        "sources": [{"id": "1", "name": "cam", "uri": "file:///video.mp4"}],
        "analytics": analytics,
    }
    with pytest.raises(ValueError, match=f"hazir olmayan moduller acilamaz: {module}"):
        render_config(state, tmp_path / "app.ini")


def test_person_enable_flag_is_rendered(tmp_path):
    state = {"sources": [{"id": "1", "name": "cam", "uri": "file:///video.mp4"}],
             "analytics": {"person": {"enabled": False, "confidence": .3, "interval": 0},
                           "pose": {"enabled": False, "confidence": .3, "interval": 0},
                           "ppe": {"enabled": False, "confidence": .4, "interval": 1}}}
    text = render_config(state, tmp_path / "app.ini").read_text()
    assert "[primary-gie]\nenable=0" in text


def test_person_engine_batch_limit(tmp_path):
    sources = [{"id": str(i), "name": f"cam-{i}", "uri": f"rtsp://example/{i}"} for i in range(13)]
    state = {"sources": sources, "inference": {"person_profile": "yolo11s-640"},
             "analytics": {"person": {"enabled": True, "confidence": .3, "interval": 0},
                           "pose": {"enabled": False, "confidence": .3, "interval": 0},
                           "ppe": {"enabled": False, "confidence": .4, "interval": 1}}}
    with pytest.raises(ValueError, match="en fazla 12 kaynak"):
        render_config(state, tmp_path / "app.ini")


def test_legacy_resolution_state_migrates_to_explicit_profile():
    state = _normalise({"sources": [], "inference": {"width": 960}, "analytics": {}})
    assert state["inference"]["person_profile"] == "yolo11s-960"
    assert state["inference"]["streammux_width"] == 1920


def test_explicit_source_types_and_rtsp_only_properties(tmp_path):
    sources = [
        {"id": "0", "name": "file", "uri": "file:///workspace/data/video.mp4"},
        {"id": "1", "name": "http", "uri": "https://example.test/video.mp4"},
        {"id": "2", "name": "rtsp", "uri": "rtsp://example.test/live"},
        {"id": "3", "name": "rtsps", "uri": "rtsps://example.test/secure"},
    ]
    state = {"sources": sources, "inference": {"person_profile": "yolo11s-640"},
             "analytics": {"person": {"enabled": True, "confidence": .3, "interval": 0},
                           "pose": {"enabled": False, "confidence": .3, "interval": 0},
                           "ppe": {"enabled": False, "confidence": .4, "interval": 1}}}
    text = render_config(state, tmp_path / "app.ini").read_text()
    blocks = {block.splitlines()[0]: block for block in text.split("\n\n") if block.startswith("[source")}

    assert "type=2" in blocks["[source0]"]
    assert "type=2" in blocks["[source1]"]
    assert "type=4" in blocks["[source2]"]
    assert "type=4" in blocks["[source3]"]
    assert "latency=" not in blocks["[source0]"]
    assert "rtsp-reconnect-interval-sec=" not in blocks["[source1]"]
    assert "latency=200" in blocks["[source2]"]
    assert "rtsp-reconnect-interval-sec=10" in blocks["[source3]"]


def test_deploy_configs_emit_only_coco_person_metadata():
    expected = list(range(1, 80))
    root = Path(__file__).resolve().parents[1]
    for size in (640, 960):
        text = (root / f"models/person/{size}/config_infer_primary.txt").read_text()
        line = next(line for line in text.splitlines() if line.startswith("filter-out-class-ids="))
        filtered = [int(value) for value in line.partition("=")[2].split(";")]
        assert filtered == expected
