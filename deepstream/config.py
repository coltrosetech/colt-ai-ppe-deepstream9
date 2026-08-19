from pathlib import Path


DEFAULT_PERSON_PROFILE = "yolo11s-640"
MODULE_ARTIFACTS_CONFIGURED = {
    "person": True,
    "pose": False,
    "ppe": False,
}
PERSON_PROFILES = {
    "yolo11s-640": {
        "id": "yolo11s-640",
        "label": "YOLO11s 640x640 FP16",
        "model": "YOLO11s",
        "input_width": 640,
        "input_height": 640,
        "precision": "FP16",
        "max_batch_size": 12,
        "benchmark_sources": 12,
        "benchmark_duration_seconds": 300,
        "config_file": "/models/person/640/config_infer_primary.txt",
        "onnx_file": "/models/person/640/yolo11s.onnx",
        "engine_file": "/models/person/640/yolo11s_b12_gpu0_fp16.engine",
    },
    "yolo11s-960": {
        "id": "yolo11s-960",
        "label": "YOLO11s 960x960 FP16",
        "model": "YOLO11s",
        "input_width": 960,
        "input_height": 960,
        "precision": "FP16",
        "max_batch_size": 12,
        "benchmark_sources": 12,
        "benchmark_duration_seconds": 300,
        "config_file": "/models/person/960/config_infer_primary.txt",
        "onnx_file": "/models/person/960/yolo11s.onnx",
        "engine_file": "/models/person/960/yolo11s_b12_gpu0_fp16.engine",
    },
}


def resolve_person_profile(state: dict) -> dict:
    """Return the selected, deployable person model profile.

    ``width`` is accepted only as a migration path for state files created by the
    first scaffold. New state stores an explicit profile id so model resolution
    cannot silently diverge from the selected TensorRT artifacts.
    """
    inference = state.get("inference", {})
    profile_id = inference.get("person_profile") or inference.get("profile")
    if profile_id is None:
        profile_id = "yolo11s-960" if int(inference.get("width", 640)) == 960 else DEFAULT_PERSON_PROFILE
    try:
        return PERSON_PROFILES[profile_id].copy()
    except KeyError as exc:
        choices = ", ".join(PERSON_PROFILES)
        raise ValueError(f"Bilinmeyen insan algilama profili: {profile_id}. Secenekler: {choices}") from exc


def list_person_profiles() -> list[dict]:
    return [profile.copy() for profile in PERSON_PROFILES.values()]


def module_readiness() -> dict[str, dict[str, bool]]:
    """Return the server-side module gate used by both config and admin APIs.

    Pose and PPE deliberately remain closed until their model, TensorRT,
    DeepStream and evidence contracts are present.  A disabled UI control is not
    a security or correctness boundary, so direct API/config callers are checked
    here as well.
    """

    return {
        name: {"artifacts_configured": configured}
        for name, configured in MODULE_ARTIFACTS_CONFIGURED.items()
    }


def validate_module_selection(state: dict) -> None:
    analytics = state.get("analytics", {})
    unavailable = sorted(
        name
        for name, configured in MODULE_ARTIFACTS_CONFIGURED.items()
        if not configured and bool(analytics.get(name, {}).get("enabled"))
    )
    if unavailable:
        raise ValueError(
            "Model/engine/DeepStream kaniti hazir olmayan moduller acilamaz: "
            + ", ".join(unavailable)
        )


def render_config(state: dict, output: Path) -> Path:
    """Render the currently deployable person-only DeepStream graph.

    The final three-model graph uses independent full-frame inference branches
    and ``nvdsmetamux``.  Until that runtime and the pose/PPE artifacts exist, it
    is incorrect to emit placeholder SGIE groups that could be mistaken for a
    deployable implementation.
    """
    validate_module_selection(state)
    sources = state["sources"]
    a = state["analytics"]
    inference = state.get("inference", {})
    profile = resolve_person_profile(state)
    if len(sources) > profile["max_batch_size"]:
        raise ValueError(
            f"{profile['id']} en fazla {profile['max_batch_size']} kaynak icin derlendi; "
            f"{len(sources)} kaynak verildi"
        )
    mux_width = int(inference.get("streammux_width", 1920))
    mux_height = int(inference.get("streammux_height", 1080))
    lines = [
        "[application]", "enable-perf-measurement=1", "perf-measurement-interval-sec=5", "",
        "[tiled-display]", "enable=0", "",
        "[streammux]", f"batch-size={max(1, len(sources))}", f"width={mux_width}", f"height={mux_height}",
        "batched-push-timeout=40000", "live-source=1", "sync-inputs=0", "",
        "[primary-gie]", f"enable={int(a['person']['enabled'])}", "gie-unique-id=1", f"batch-size={max(1, len(sources))}",
        f"config-file={profile['config_file']}",
        f"interval={int(a['person']['interval'])}", "",
        "[tracker]", "enable=1", "tracker-width=640", "tracker-height=384",
        "ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml", "",
        "[sink0]", "enable=1", "type=1", "sync=0", "qos=0", "",
    ]
    for i, source in enumerate(sources):
        uri = source["uri"]
        is_rtsp = uri.lower().startswith(("rtsp://", "rtsps://"))
        lines += [f"[source{i}]", "enable=1", f"type={4 if is_rtsp else 2}", f"uri={uri}"]
        if is_rtsp:
            lines += ["latency=200", "rtsp-reconnect-interval-sec=10"]
        lines += ["drop-frame-interval=0", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return output
