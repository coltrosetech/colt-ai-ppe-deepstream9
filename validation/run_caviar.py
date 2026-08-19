#!/usr/bin/env python3
"""Run one CAVIAR clip through DeepStream 9 and evaluate person detections."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from validation.ds9_runtime_compatibility import (
    DEFAULT_RECEIPT as DEFAULT_DS9_COMPATIBILITY_RECEIPT,
    write_pending_report as write_ds9_pending_report,
)
from validation.kitti_to_jsonl import caviar_frame_indices, convert_kitti_directory
from validation.gpu_guarded_process import (
    DEFAULT_KILL_GRACE_SECONDS,
    DEFAULT_MAX_TEMPERATURE_C,
    DEFAULT_POWER_LIMIT_DROP_TOLERANCE_W,
    DEFAULT_SLOWDOWN_CONSECUTIVE_SAMPLES,
    GpuGuardError,
    run_guarded_docker,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REENTRY_EVIDENCE = Path(
    "validation/results/gpu-reentry/current/evidence.json"
)


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not result:
        raise ValueError("sequence id has no path-safe characters")
    return result


def _inside_repo(path: Path) -> Path:
    candidate = REPO_ROOT / path if not path.is_absolute() else path
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must be inside workspace {REPO_ROOT}: {candidate}") from exc
    cursor = REPO_ROOT
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"symlink path components are forbidden: {cursor}")
        if not cursor.exists():
            break
    resolved = candidate.resolve()
    if resolved != candidate:
        raise ValueError(f"path must be canonical and symlink-free: {candidate}")
    return resolved


def _container_path(path: Path) -> str:
    return "/workspace/" + _inside_repo(path).relative_to(REPO_ROOT).as_posix()


def probe_video(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected exactly one selected video stream")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    raw_frames = stream.get("nb_frames")
    frames = int(raw_frames) if raw_frames not in (None, "N/A") else None
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"{path}: invalid video metadata")
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def calculate_streammux_dimensions(
    source_width: int,
    source_height: int,
    model_size: int,
    *,
    max_nvinfer_upscale: float = 1.0,
    policy: str = "no-nvinfer-upscale",
) -> tuple[int, int, float]:
    """Return even streammux dimensions and source-to-mux scale.

    ``model-active-area`` scales both low- and high-resolution sources so the
    longest edge is the model input and nvinfer only adds symmetric padding.
    The legacy/default policy preserves high-resolution source dimensions but
    pre-upscales low-resolution inputs enough to avoid the inaccurate >1x
    nvinfer upscale path found during CAVIAR validation.
    """

    if min(source_width, source_height) <= 0:
        raise ValueError("source dimensions must be positive")
    if model_size <= 0:
        raise ValueError("model_size must be positive")
    if not math.isfinite(max_nvinfer_upscale) or max_nvinfer_upscale <= 0:
        raise ValueError("max_nvinfer_upscale must be positive")
    if policy == "model-active-area":
        mux_scale = model_size / max(source_width, source_height)
        # Pin the long edge exactly.  Computing it again through floating-point
        # multiplication can turn an exact 960 into 960.0000000001 and then an
        # even-ceiling into 962 for some aspect ratios (for example 2720x1530).
        if source_width >= source_height:
            mux_width = model_size
            mux_height = math.ceil(source_height * mux_scale / 2) * 2
        else:
            mux_width = math.ceil(source_width * mux_scale / 2) * 2
            mux_height = model_size
        return mux_width, mux_height, mux_scale
    elif policy == "no-nvinfer-upscale":
        mux_scale = max(
            1.0,
            model_size
            / (max_nvinfer_upscale * max(source_width, source_height)),
        )
    else:
        raise ValueError(f"unsupported streammux policy: {policy}")
    # NV12 surfaces are safest with even dimensions. Rounding upward guarantees
    # the configured nvinfer upscale ceiling and avoids cropping one source row.
    mux_width = math.ceil(source_width * mux_scale / 2) * 2
    mux_height = math.ceil(source_height * mux_scale / 2) * 2
    return mux_width, mux_height, mux_scale


def render_infer_config(
    model_size: int,
    export_threshold: float,
    *,
    parser: str = "cuda",
    included_class_ids: tuple[int, ...] = (0,),
) -> str:
    """Render the YOLO PGIE config for an explicit COCO class subset.

    Person-only callers keep the historical default.  Industrial-scene
    callers may additionally expose COCO class 7 (``truck``) as low-cost
    vehicle evidence without running a second network or rebuilding the
    TensorRT engine.
    """

    if (
        not included_class_ids
        or len(included_class_ids) != len(set(included_class_ids))
        or any(
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < 80
            for class_id in included_class_ids
        )
    ):
        raise ValueError("included_class_ids must be unique COCO class IDs")
    included = set(included_class_ids)
    filtered = ";".join(
        str(class_id) for class_id in range(80) if class_id not in included
    )
    parser_function = (
        "NvDsInferParseYoloCuda" if parser == "cuda" else "NvDsInferParseYolo"
    )
    return f"""[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=/workspace/models/person/{model_size}/yolo11s.onnx
model-engine-file=/workspace/models/person/{model_size}/yolo11s_b12_gpu0_fp16.engine
labelfile-path=/workspace/models/person/{model_size}/labels.txt
batch-size=1
network-mode=2
num-detected-classes=80
filter-out-class-ids={filtered}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name={parser_function}
custom-lib-path=/opt/deepsafe/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold={export_threshold:.8g}
topk=300
"""


def render_deepstream_config(
    *, video: Path, kitti_dir: Path, infer_config: Path, width: int, height: int
) -> str:
    return render_deepstream_config_paths(
        video_container_path=_container_path(video),
        kitti_container_path=_container_path(kitti_dir),
        infer_config_container_path=_container_path(infer_config),
        width=width,
        height=height,
    )


def render_deepstream_config_paths(
    *,
    video_container_path: str,
    kitti_container_path: str,
    infer_config_container_path: str,
    width: int,
    height: int,
) -> str:
    """Render config text from already canonical container paths."""

    return f"""[application]
enable-perf-measurement=1
perf-measurement-interval-sec=1
gie-kitti-output-dir={kitti_container_path}

[tiled-display]
enable=0

[source0]
enable=1
type=2
uri=file://{video_container_path}
gpu-id=0
cudadec-memtype=0

[sink0]
enable=1
type=1
sync=0
qos=0

[osd]
enable=0

[streammux]
gpu-id=0
live-source=0
batch-size=1
batched-push-timeout=40000
width={width}
height={height}
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
batch-size=1
interval=0
gie-unique-id=1
nvbuf-memory-type=0
config-file={infer_config_container_path}

[tests]
file-loop=0
"""


def build_docker_command(
    *,
    container_name: str,
    image: str,
    gpu: int | str,
    app_config: Path,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    app_config = app_config.resolve()
    repo_root = repo_root.resolve()
    relative = app_config.relative_to(repo_root).as_posix()
    run_root = app_config.parent.parent.resolve()
    run_relative = run_root.relative_to(repo_root).as_posix()
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--gpus",
        f"device={gpu}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility",
        "-v",
        f"{repo_root}:/workspace:ro",
        "-v",
        f"{run_root}:/workspace/{run_relative}:rw",
        "-w",
        "/workspace",
        image,
        "deepstream-app",
        "-c",
        "/workspace/" + relative,
    ]


def _file_pin(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    if not content:
        raise ValueError(f"cannot pin empty artifact: {path}")
    return {
        "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _load_batch_binding(
    *,
    batch_manifest: Path | None,
    job_id: str | None,
    args: argparse.Namespace,
    run_root: Path,
) -> dict[str, Any] | None:
    if batch_manifest is None and job_id is None:
        return None
    if batch_manifest is None or not job_id:
        raise ValueError("--batch-manifest and --job-id must be supplied together")
    manifest_path = _inside_repo(batch_manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "deepsafe.caviar-batch-plan/v1":
        raise ValueError("batch manifest schema is unsupported")
    campaign = payload.get("campaign")
    jobs = payload.get("jobs")
    if not isinstance(campaign, dict) or not isinstance(jobs, list):
        raise ValueError("batch manifest campaign/jobs are invalid")
    selected = [item for item in jobs if isinstance(item, dict) and item.get("job_id") == job_id]
    if len(selected) != 1:
        raise ValueError(f"batch manifest must contain job {job_id!r} exactly once")
    job = selected[0]
    expected = {
        "sequence_id": args.sequence,
        "model_input": args.model_size,
        "run_root": run_root.relative_to(REPO_ROOT).as_posix(),
    }
    for key, value in expected.items():
        if job.get(key) != value:
            raise ValueError(f"batch job {key} differs from invocation")
    quality_binding = campaign.get("quality_policy")
    runtime_contract = campaign.get("model_runtime_contract")
    if not isinstance(quality_binding, dict) or not isinstance(runtime_contract, dict):
        raise ValueError("batch manifest has no quality/runtime contract")
    from validation.person_quality_policy import load_policy_execution_binding

    policy_artifact = quality_binding.get("artifact")
    if not isinstance(policy_artifact, dict) or not isinstance(policy_artifact.get("path"), str):
        raise ValueError("batch quality-policy artifact pin is invalid")
    live = load_policy_execution_binding(
        Path(policy_artifact["path"]),
        project_root=REPO_ROOT,
        require_approved=not args.dry_run,
    )
    if live["quality_policy"] != quality_binding:
        raise ValueError("batch quality-policy snapshot differs from live policy")
    if live["model_runtime_contract"] != runtime_contract:
        raise ValueError("batch runtime snapshot differs from live policy")
    if live["dataset_catalog"] != campaign.get("dataset_catalog"):
        raise ValueError("batch dataset snapshot differs from live policy")
    authorization = quality_binding.get("campaign_authorization")
    if not args.dry_run:
        if not isinstance(authorization, dict):
            raise ValueError("batch execution has no campaign authorization")
        authorized_root = _inside_repo(
            Path(authorization["authorized_results_root"])
        )
        try:
            run_root.relative_to(authorized_root)
        except ValueError as exc:
            raise ValueError(
                "batch run root is outside the nonce-authorized session"
            ) from exc
        if campaign.get("campaign_nonce") != authorization.get("campaign_nonce"):
            raise ValueError("batch campaign nonce differs from approval")
        session_claim = campaign.get("session_claim_artifact")
        if not isinstance(session_claim, dict) or not isinstance(
            session_claim.get("path"), str
        ):
            raise ValueError("batch session claim pin is missing")
        if _file_pin(_inside_repo(Path(session_claim["path"]))) != session_claim:
            raise ValueError("batch session claim changed before job start")
    else:
        session_claim = campaign.get("session_claim_artifact")
    dataset_item = next(
        (
            item
            for item in live["dataset_catalog"]
            if item["sequence_id"] == args.sequence
        ),
        None,
    )
    if dataset_item is None:
        raise ValueError("batch sequence is absent from approved native dataset catalog")
    sequence_rows = [
        item
        for item in payload.get("sequences", [])
        if isinstance(item, dict) and item.get("sequence_id") == args.sequence
    ]
    if len(sequence_rows) != 1:
        raise ValueError("batch sequence record is missing/duplicated")
    if (
        sequence_rows[0].get("video_artifact") != dataset_item["video"]
        or sequence_rows[0].get("ground_truth_artifact")
        != dataset_item["ground_truth"]
        or sequence_rows[0].get("video_metadata")
        != dataset_item["video_metadata"]
        or sequence_rows[0].get("frame_mapping") != dataset_item["frame_mapping"]
    ):
        raise ValueError(
            "batch native video/GT/ffprobe/frame-map differs from approved catalog"
        )
    if campaign.get("container_image") != runtime_contract.get("requested_container_image"):
        raise ValueError("batch requested image differs from runtime policy")
    if args.container_image != runtime_contract.get("requested_container_image"):
        raise ValueError("invocation requested image differs from runtime policy")
    if args.streammux_policy != campaign.get("streammux_policy"):
        raise ValueError("invocation streammux policy differs from batch plan")
    execution_contract = runtime_contract["execution_contract"]
    invocation_contract = {
        "bbox_parser": args.parser,
        "export_threshold": args.export_threshold,
        "evaluation_confidence": args.evaluation_confidence,
        "iou": args.iou,
        "streammux_policy": args.streammux_policy,
        "max_nvinfer_upscale": args.max_nvinfer_upscale,
        "max_temperature_c": args.max_temperature_c,
        "power_limit_drop_tolerance_w": args.power_limit_drop_tolerance_w,
        "slowdown_consecutive_samples": args.slowdown_consecutive_samples,
        "kill_grace_seconds": args.kill_grace,
    }
    for key, value in invocation_contract.items():
        if execution_contract.get(key) != value:
            raise ValueError(f"invocation {key} differs from approved execution contract")
    if args.gpu != int(runtime_contract["gpu_contract"]["index"]):
        raise ValueError("invocation GPU index differs from approved GPU contract")
    reentry_path = _inside_repo(args.reentry_evidence).relative_to(REPO_ROOT).as_posix()
    if reentry_path != execution_contract["reentry_evidence"]["path"]:
        raise ValueError("invocation re-entry evidence differs from approved contract")
    compatibility_path = _inside_repo(args.ds9_compatibility_receipt).relative_to(
        REPO_ROOT
    ).as_posix()
    if compatibility_path != execution_contract["ds9_compatibility_receipt"]["path"]:
        raise ValueError(
            "invocation DS9 compatibility receipt differs from approved contract"
        )
    campaign_compatibility = campaign.get("ds9_runtime_compatibility")
    if not isinstance(campaign_compatibility, dict) or campaign_compatibility.get(
        "receipt"
    ) != compatibility_path:
        raise ValueError("batch DS9 compatibility receipt path differs")
    if not args.dry_run and campaign_compatibility.get("status") != "production_ready":
        raise ValueError("batch DS9 compatibility snapshot is not production-ready")
    profile_contract = runtime_contract["profiles"].get(str(args.model_size))
    if job.get("model_contract") != profile_contract:
        raise ValueError("batch job model contract differs from campaign contract")
    planned_command = job.get("command")
    if not isinstance(planned_command, list) or not all(
        isinstance(value, str) and value for value in planned_command
    ):
        raise ValueError("batch job command is invalid")
    return {
        "job_id": job_id,
        "planned_command": copy.deepcopy(planned_command),
        "quality_policy": copy.deepcopy(quality_binding),
        "model_contract": copy.deepcopy(profile_contract),
        "runtime_contract": copy.deepcopy(runtime_contract),
        "dataset_item": copy.deepcopy(dataset_item),
        "campaign_authorization": copy.deepcopy(authorization),
        "session_claim": copy.deepcopy(session_claim),
    }


def _live_pin_map(expected: dict[str, Any], context: str) -> dict[str, Any]:
    actual: dict[str, Any] = {}
    for name, pin in expected.items():
        if not isinstance(pin, dict) or not isinstance(pin.get("path"), str):
            raise ValueError(f"{context} {name} pin is invalid")
        current = _file_pin(_inside_repo(Path(pin["path"])))
        if current != pin:
            raise ValueError(f"{context} {name} live size/hash differs")
        actual[name] = current
    return actual


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    content = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("short write while persisting immutable job receipt")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_and_tee(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def parse_deepstream_fps(log_path: Path) -> dict[str, object]:
    """Extract single-source current/average FPS samples from deepstream-app."""

    samples: list[dict[str, float]] = []
    pattern = re.compile(r"\*\*PERF:\s+([0-9]+(?:\.[0-9]+)?)\s+\(([0-9]+(?:\.[0-9]+)?)\)")
    for match in pattern.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        samples.append({"current": float(match.group(1)), "average": float(match.group(2))})
    return {
        "samples": samples,
        "last_reported_average": samples[-1]["average"] if samples else None,
    }


def attest_engine_load(log_path: Path, model_size: int) -> dict[str, Any]:
    """Require DeepStream to deserialize the policy-pinned engine, never rebuild."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    engine_path = (
        f"/workspace/models/person/{model_size}/"
        "yolo11s_b12_gpu0_fp16.engine"
    )
    deserialize = re.findall(
        r"deserialized trt engine from\s*:(\S+)", text, flags=re.IGNORECASE
    )
    selected = re.findall(
        r"Use deserialized engine model:\s*(\S+)", text, flags=re.IGNORECASE
    )
    forbidden_patterns = {
        "engine_rebuild": len(
            re.findall(
                r"(?:\b(?:building|creating|generating)\b[^\n]{0,80}\bengine\b|\bserialize(?:d|ing)?\s+(?:cuda\s+)?engine\s+to)",
                text,
                flags=re.IGNORECASE,
            )
        ),
        "onnx_fallback": len(
            re.findall(
                r"(?:parsing|building from|fallback to).*onnx",
                text,
                flags=re.IGNORECASE,
            )
        ),
    }
    passed = (
        deserialize == [engine_path]
        and selected == [engine_path]
        and not any(forbidden_patterns.values())
    )
    attestation = {
        "schema_version": "deepsafe.caviar-engine-load-attestation/v1",
        "model_input": model_size,
        "expected_engine_container_path": engine_path,
        "deserialize_paths": deserialize,
        "selected_engine_paths": selected,
        "forbidden_fallback_patterns": forbidden_patterns,
        "status": "pass" if passed else "fail",
    }
    if not passed:
        raise ValueError("DeepStream engine load attestation failed or fallback/rebuild occurred")
    return attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="prediction/GT sequence id")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-size", type=int, choices=(640, 960), required=True)
    parser.add_argument("--ground-truth", type=Path, help="single CAVIAR CVML XML")
    parser.add_argument(
        "--export-threshold",
        type=float,
        default=0.001,
        help="PGIE serialization threshold; keep low so AP is not truncated",
    )
    parser.add_argument("--evaluation-confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--parser",
        choices=("cuda", "cpu"),
        default="cuda",
        help="DeepStream-Yolo bbox parser implementation (default: cuda)",
    )
    parser.add_argument(
        "--max-nvinfer-upscale",
        type=float,
        default=1.0,
        help=(
            "pre-upscale low-resolution frames in streammux so nvinfer never "
            "upscales by more than this factor; 1.0 makes nvinfer pad only "
            "and is the accuracy-validation default"
        ),
    )
    parser.add_argument(
        "--streammux-policy",
        choices=("no-nvinfer-upscale", "model-active-area"),
        default="no-nvinfer-upscale",
        help=(
            "model-active-area scales the source longest edge directly to the "
            "model input; default preserves high-resolution decoded dimensions"
        ),
    )
    parser.add_argument("--container-image", default="deepsafe-deepstream:9.0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--reentry-evidence",
        type=Path,
        default=DEFAULT_REENTRY_EVIDENCE,
        help="current fail-closed GPU re-entry evidence bundle",
    )
    parser.add_argument(
        "--ds9-compatibility-receipt",
        type=Path,
        default=DEFAULT_DS9_COMPATIBILITY_RECEIPT,
        help="current production-ready DeepStream 9 compatibility receipt",
    )
    parser.add_argument(
        "--max-temperature-c", type=float, default=DEFAULT_MAX_TEMPERATURE_C
    )
    parser.add_argument(
        "--power-limit-drop-tolerance-w",
        type=float,
        default=DEFAULT_POWER_LIMIT_DROP_TOLERANCE_W,
    )
    parser.add_argument(
        "--slowdown-consecutive-samples",
        type=int,
        default=DEFAULT_SLOWDOWN_CONSECUTIVE_SAMPLES,
    )
    parser.add_argument(
        "--kill-grace", type=int, default=DEFAULT_KILL_GRACE_SECONDS
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        help="batch plan carrying the owner-approved immutable runtime contract",
    )
    parser.add_argument("--job-id", help="exact job identity in --batch-manifest")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    try:
        if not 0 <= args.export_threshold <= 1:
            raise ValueError("export threshold must be in [0, 1]")
        if not 0 <= args.evaluation_confidence <= 1:
            raise ValueError("evaluation confidence must be in [0, 1]")
        if not 0 < args.iou <= 1:
            raise ValueError("IoU must be in (0, 1]")
        if not math.isfinite(args.max_nvinfer_upscale) or args.max_nvinfer_upscale <= 0:
            raise ValueError("max nvinfer upscale must be positive")
        if not math.isfinite(args.max_temperature_c) or args.max_temperature_c <= 0:
            raise ValueError("max temperature must be positive")
        if (
            not math.isfinite(args.power_limit_drop_tolerance_w)
            or args.power_limit_drop_tolerance_w <= 0
        ):
            raise ValueError("power-limit drop tolerance must be positive")
        if args.slowdown_consecutive_samples <= 0:
            raise ValueError("slowdown consecutive samples must be positive")
        if args.kill_grace <= 0:
            raise ValueError("kill grace must be positive")
        video = _inside_repo(args.video)
        if not video.is_file():
            raise ValueError(f"video not found: {video}")
        ground_truth = _inside_repo(args.ground_truth) if args.ground_truth else None
        if ground_truth is not None and not ground_truth.is_file():
            raise ValueError(f"ground truth not found: {ground_truth}")
        metadata = probe_video(video)
        source_width = int(metadata["width"])
        source_height = int(metadata["height"])
        mux_width, mux_height, mux_scale = calculate_streammux_dimensions(
            source_width,
            source_height,
            args.model_size,
            max_nvinfer_upscale=args.max_nvinfer_upscale,
            policy=args.streammux_policy,
        )
        name = _safe_name(args.sequence)
        run_root = _inside_repo(
            args.run_root
            or Path("validation/results/caviar") / name / str(args.model_size)
        )
        batch_binding = _load_batch_binding(
            batch_manifest=args.batch_manifest,
            job_id=args.job_id,
            args=args,
            run_root=run_root,
        )
        if run_root.exists() and any(run_root.iterdir()):
            if not args.force:
                raise ValueError(f"run directory is not empty (use --force): {run_root}")
            shutil.rmtree(run_root)
        kitti_dir = run_root / "kitti"
        kitti_dir.mkdir(parents=True, exist_ok=True)

        generated = run_root / "generated"
        generated.mkdir(parents=True, exist_ok=True)
        infer_config = generated / "config-infer-primary.txt"
        app_config = generated / "deepstream-app.txt"
        # DeepStream 9 deepstream-app only dispatches KeyFile parsing for .txt.
        assert infer_config.suffix == ".txt" and app_config.suffix == ".txt"
        infer_config.write_text(
            render_infer_config(
                args.model_size, args.export_threshold, parser=args.parser
            ),
            encoding="utf-8",
        )
        app_config.write_text(
            render_deepstream_config(
                video=video,
                kitti_dir=kitti_dir,
                infer_config=infer_config,
                width=mux_width,
                height=mux_height,
            ),
            encoding="utf-8",
        )

        container_name = f"deepsafe-caviar-{name.lower()}-{args.model_size}-{os.getpid()}"
        docker_gpu_device: int | str = args.gpu
        if batch_binding is not None:
            docker_gpu_device = batch_binding["runtime_contract"]["gpu_contract"][
                "uuid"
            ]
        requested_docker_command = build_docker_command(
            container_name=container_name,
            image=args.container_image,
            gpu=docker_gpu_device,
            app_config=app_config,
        )
        reentry_evidence = _inside_repo(args.reentry_evidence)
        ds9_compatibility_receipt = _inside_repo(
            args.ds9_compatibility_receipt
        )
        runtime_binding: dict[str, Any] | None = None
        if batch_binding is not None:
            runtime_contract = batch_binding.pop("runtime_contract")
            profile_contract = batch_binding["model_contract"]
            dataset_item = batch_binding["dataset_item"]
            runtime_binding = {
                "model_id": profile_contract["model_id"],
                "model_artifacts_preflight": _live_pin_map(
                    profile_contract["model_artifacts"], "model preflight"
                ),
                "model_artifacts_postflight": None,
                "control_artifacts_preflight": _live_pin_map(
                    runtime_contract["control_artifacts"], "control preflight"
                ),
                "control_artifacts_postflight": None,
                "generated_configs": {
                    "deepstream_app": _file_pin(app_config),
                    "primary_inference": _file_pin(infer_config),
                },
                "container": {
                    "requested_image": args.container_image,
                    "resolved_image_id": None,
                    "container_name": container_name,
                    "requested_command": requested_docker_command,
                    "command": None,
                },
                "input_artifacts_preflight": _live_pin_map(
                    {
                        "video": dataset_item["video"],
                        "ground_truth": dataset_item["ground_truth"],
                    },
                    "input preflight",
                ),
                "input_artifacts_postflight": None,
            }
        manifest = {
            "sequence_id": args.sequence,
            "video": str(video.relative_to(REPO_ROOT)),
            "ground_truth": (
                str(ground_truth.relative_to(REPO_ROOT)) if ground_truth else None
            ),
            "model": f"yolo11s-{args.model_size}-fp16",
            "model_input": args.model_size,
            "bbox_parser": args.parser,
            "export_threshold": args.export_threshold,
            "evaluation_confidence": args.evaluation_confidence,
            "iou": args.iou,
            "video_metadata": metadata,
            "streammux": {
                "width": mux_width,
                "height": mux_height,
                "source_to_mux_scale": mux_scale,
                "max_nvinfer_upscale": args.max_nvinfer_upscale,
                "policy": args.streammux_policy,
            },
            "deepstream_config": str(app_config.relative_to(REPO_ROOT)),
            "infer_config": str(infer_config.relative_to(REPO_ROOT)),
            "deepstream_config_suffix_requirement": ".txt",
            "docker_requested_command": requested_docker_command,
            "docker_command": None,
            "batch_binding": batch_binding,
            "runtime_binding": runtime_binding,
            "gpu_safety": {
                "execution_boundary": "validation.gpu_guarded_process/v1",
                "reentry_evidence": str(reentry_evidence.relative_to(REPO_ROOT)),
                "active_monitoring": True,
                "sample_interval_seconds": 1.0,
                "max_temperature_c": args.max_temperature_c,
                "power_limit_drop_tolerance_w": args.power_limit_drop_tolerance_w,
                "slowdown_consecutive_samples": args.slowdown_consecutive_samples,
                "kill_grace_seconds": args.kill_grace,
                "report": str((run_root / "safety/gpu-guard-report.json").relative_to(REPO_ROOT)),
            },
            "ds9_runtime_compatibility": {
                "receipt": str(ds9_compatibility_receipt.relative_to(REPO_ROOT)),
                "status": "pending_static_probe" if args.dry_run else "required",
                "pending_report": str(
                    (run_root / "ds9-runtime-compatibility-pending.json").relative_to(
                        REPO_ROOT
                    )
                ),
            },
            "status": "dry-run" if args.dry_run else "running",
        }
        manifest_path = run_root / "run-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.dry_run:
            pending = write_ds9_pending_report(
                run_root / "ds9-runtime-compatibility-pending.json",
                requested_image=args.container_image,
                project_root=REPO_ROOT,
                launch_scope=f"caviar:{args.sequence}:{args.model_size}",
            )
            manifest["ds9_runtime_compatibility"]["pending"] = pending
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
            return 0

        deepstream_log = run_root / "deepstream.log"
        started = time.monotonic()
        guard_report = run_guarded_docker(
            requested_docker_command,
            project_root=REPO_ROOT,
            artifact_root=run_root / "safety",
            log_path=deepstream_log,
            container_name=container_name,
            image=args.container_image,
            gpu_index=args.gpu,
            reentry_evidence_path=reentry_evidence,
            ds9_compatibility_receipt_path=ds9_compatibility_receipt,
            max_temperature_c=args.max_temperature_c,
            power_limit_drop_tolerance_w=args.power_limit_drop_tolerance_w,
            slowdown_consecutive_samples=args.slowdown_consecutive_samples,
            kill_grace_seconds=args.kill_grace,
        )
        manifest["gpu_safety"]["status"] = guard_report["status"]
        executed_docker_command = guard_report.get("command")
        if not isinstance(executed_docker_command, list):
            raise ValueError("GPU guard did not return the resolved Docker command")
        manifest["docker_command"] = executed_docker_command
        manifest["ds9_runtime_compatibility"]["status"] = guard_report[
            "ds9_runtime_compatibility"
        ]["status"]
        manifest["ds9_runtime_compatibility"]["binding"] = guard_report[
            "ds9_runtime_compatibility"
        ]
        if runtime_binding is not None:
            resolved_image_id = guard_report.get("preflight", {}).get(
                "resolved_image_id"
            )
            if not isinstance(resolved_image_id, str) or not resolved_image_id:
                raise ValueError("GPU guard did not return a resolved image identity")
            runtime_binding["container"]["resolved_image_id"] = resolved_image_id
            runtime_binding["container"]["command"] = executed_docker_command
            runtime_binding["model_artifacts_postflight"] = _live_pin_map(
                batch_binding["model_contract"]["model_artifacts"],
                "model postflight",
            )
            runtime_binding["control_artifacts_postflight"] = _live_pin_map(
                runtime_binding["control_artifacts_preflight"],
                "control postflight",
            )
            runtime_binding["input_artifacts_postflight"] = _live_pin_map(
                runtime_binding["input_artifacts_preflight"],
                "input postflight",
            )
            manifest["gpu_safety"]["preflight"] = _file_pin(
                run_root / "safety/gpu-preflight.json"
            )
            manifest["gpu_safety"]["guard_report"] = _file_pin(
                run_root / "safety/gpu-guard-report.json"
            )
            manifest["gpu_safety"]["guard_receipt"] = _file_pin(
                run_root / "safety/gpu-guard-artifact-receipt.json"
            )
        manifest["deepstream_wall_seconds"] = round(time.monotonic() - started, 6)
        manifest["deepstream_fps"] = parse_deepstream_fps(deepstream_log)
        manifest["engine_load_attestation"] = attest_engine_load(
            deepstream_log, args.model_size
        )
        predictions = run_root / "predictions.jsonl"
        include_frames = caviar_frame_indices(ground_truth) if ground_truth else None
        conversion = convert_kitti_directory(
            kitti_dir,
            predictions,
            sequence_id=args.sequence,
            image_width=int(metadata["width"]),
            image_height=int(metadata["height"]),
            labels_path=REPO_ROOT
            / "models"
            / "person"
            / str(args.model_size)
            / "labels.txt",
            coordinate_width=mux_width,
            coordinate_height=mux_height,
            expected_frames=(
                int(metadata["frames"]) if metadata["frames"] is not None else None
            ),
            include_frames=include_frames,
            fps=float(metadata["fps"]),
            source_uri="file://" + _container_path(video),
            model_id=f"yolo11s-{args.model_size}-fp16",
        )
        (run_root / "conversion.json").write_text(
            json.dumps(conversion, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if ground_truth:
            evaluation_command = [
                sys.executable,
                "-m",
                "evaluation",
                "--predictions",
                str(predictions),
                "--ground-truth",
                str(ground_truth),
                "--ground-truth-format",
                "caviar",
                "--sequence-id",
                args.sequence,
                "--image-width",
                str(metadata["width"]),
                "--image-height",
                str(metadata["height"]),
                "--iou",
                str(args.iou),
                "--confidence",
                str(args.evaluation_confidence),
                "--output",
                str(run_root / "evaluation.json"),
            ]
            completed = subprocess.run(
                evaluation_command,
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            manifest["evaluation_command"] = evaluation_command

        manifest["status"] = "complete"
        manifest["conversion"] = conversion
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if batch_binding is not None:
            receipt_artifact_paths = {
                "run_manifest": manifest_path,
                "conversion": run_root / "conversion.json",
                "predictions": run_root / "predictions.jsonl",
                "evaluation": run_root / "evaluation.json",
                "deepstream_log": deepstream_log,
                "deepstream_config": app_config,
                "infer_config": infer_config,
                "gpu_guard_report": run_root / "safety/gpu-guard-report.json",
                "gpu_guard_receipt": (
                    run_root / "safety/gpu-guard-artifact-receipt.json"
                ),
                "gpu_preflight": run_root / "safety/gpu-preflight.json",
                "ds9_runtime_compatibility_receipt": ds9_compatibility_receipt,
            }
            job_receipt = {
                "schema_version": "deepsafe.caviar-job-receipt/v1",
                "created_at_utc": _utc_now(),
                "job_id": batch_binding["job_id"],
                "quality_policy": batch_binding["quality_policy"],
                "model_contract": batch_binding["model_contract"],
                "dataset_item": batch_binding["dataset_item"],
                "campaign_authorization": batch_binding[
                    "campaign_authorization"
                ],
                "session_claim": batch_binding["session_claim"],
                "artifacts": {
                    name: _file_pin(path)
                    for name, path in receipt_artifact_paths.items()
                },
            }
            receipt_path = run_root / "job-receipt.json"
            _write_immutable_json(receipt_path, job_receipt)
        print(f"completed: {run_root}")
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        if manifest is not None and manifest_path is not None:
            manifest["status"] = "safety_abort" if isinstance(exc, GpuGuardError) else "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            try:
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
