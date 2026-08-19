import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from validation.site_distance_runner_v2 import (
    EXECUTION_VERSION,
    acquire_endurance_supervisor_lock,
    acquire_gpu_lock,
    assert_endurance_inactive,
    build_execution_receipt,
    execution_guard_session,
    execute_plan,
    finalize_profile_manifest,
    generate_run_plan,
    materialize_docker_command,
    source_uri_from_media_ledger,
    verify_live_gpu_binding,
    verify_profile_config_semantics,
)
from validation.site_distance_readiness_v2 import (
    _calibration,
    _nominal_bin,
    _schema_validate,
    build_calibration,
    fit_homography,
    frame_key_sha256,
    pin_file,
    project_ground,
    validate_preflight,
    validate_profile_pair,
)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _pin(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _phase1_fixture(tmp_path: Path):
    mount = tmp_path / "mount-evidence.md"
    mount.write_text("mount bolts and camera pose verified\n", encoding="utf-8")
    survey = tmp_path / "survey.md"
    survey.write_text("independent total-station control report\n", encoding="utf-8")
    camera = _write_json(
        tmp_path / "camera-configuration.json",
        {
            "schema_version": "deepsafe.site-camera-configuration/v2",
            "status": "locked",
            "site_id": "site-a",
            "camera_id": "camera-a",
            "camera_configuration_id": "camera-a-fixed-lens-stream-v1",
            "configuration_locked_at": "2026-07-16T10:00:00Z",
            "camera": {
                "manufacturer": "Example",
                "model": "CCTV-1",
                "serial_number": "SERIAL-1",
                "firmware_version": "1.2.3",
            },
            "optics": {
                "lens_model": "fixed-4mm",
                "projection_model": "rectilinear",
                "focal_length_mm": 4.0,
                "zoom_mode": "fixed",
                "zoom_value": "1.0x",
                "focus_mode": "locked",
                "focus_value": "position-42",
                "distortion_coefficients": [],
                "dewarp_enabled": False,
            },
            "mount": {
                "fixed": True,
                "camera_ground_position_m": [0.0, 0.0],
                "height_m": 6.0,
                "yaw_deg": 0.0,
                "pitch_deg": 45.0,
                "roll_deg": 0.0,
                "evidence": _pin(mount),
            },
            "source_stream": {
                "width": 1920,
                "height": 1080,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "rotation_degrees_clockwise": 0,
                "pixel_format": "yuv420p",
                "crop_pixel_xywh": [0, 0, 1920, 1080],
            },
            "calibration_coordinate_space": {
                "coordinate_space_id": "source-rgb24-1920x1080-v1",
                "width": 1920,
                "height": 1080,
                "pixel_origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "box_format": "pixel_xywh",
                "transform_chain": [
                    {
                        "order": 0,
                        "operation": "identity",
                        "parameters": "source frame unchanged",
                    }
                ],
            },
        },
    )

    def point(point_id, u, v):
        return {
            "point_id": point_id,
            "image_pixel": [u, v],
            "ground_m": [(u - 960) * 0.02, (1080 - v) * 0.05],
            "ground_measurement_uncertainty_m": 0.02,
            "independently_measured": True,
        }

    controls = _write_json(
        tmp_path / "control-points.json",
        {
            "schema_version": "deepsafe.site-ground-plane-control-points/v2",
            "status": "complete",
            "calibration_id": "calibration-a-v2",
            "site_id": "site-a",
            "camera_id": "camera-a",
            "camera_configuration_id": "camera-a-fixed-lens-stream-v1",
            "camera_configuration": _pin(camera),
            "coordinate_space_id": "source-rgb24-1920x1080-v1",
            "image": {"width": 1920, "height": 1080},
            "ground_coordinate_reference": {
                "name": "site-grid-a",
                "unit": "m",
                "x_axis": "east",
                "y_axis": "north",
                "origin_description": "camera projected onto ground",
            },
            "distance_definition": "horizontal_ground_range_from_camera_ground_position",
            "valid_image_polygon_px": [
                [200, 300],
                [1720, 300],
                [1720, 1000],
                [200, 1000],
            ],
            "fit_points": [
                point("fit-1", 560, 720),
                point("fit-2", 1360, 720),
                point("fit-3", 560, 480),
                point("fit-4", 1360, 480),
                point("fit-5", 960, 640),
            ],
            "holdout_points": [
                point("hold-20", 960, 680),
                point("hold-25", 960, 580),
                point("hold-left", 760, 680),
                point("hold-right", 1160, 580),
            ],
            "quality_policy": {
                "minimum_fit_points": 5,
                "minimum_holdout_points": 4,
                "maximum_fit_error_m": 0.15,
                "maximum_holdout_error_m": 0.20,
                "maximum_holdout_rmse_m": 0.15,
                "maximum_control_point_uncertainty_m": 0.05,
                "required_holdout_distance_coverage_m": [20, 25],
            },
            "verification": {
                "method": "total_station",
                "measured_by": "surveyor-a",
                "measured_at": "2026-07-16T11:00:00Z",
                "document": _pin(survey),
            },
        },
    )
    return camera, controls


def test_normalized_homography_fit_is_deterministic_and_exact():
    points = []
    for index, (u, v) in enumerate(
        ((10, 20), (100, 20), (10, 100), (100, 100), (55, 60), (80, 75))
    ):
        denominator = 1 + 0.0005 * u - 0.0002 * v
        points.append(
            {
                "point_id": f"p{index}",
                "image_pixel": [u, v],
                "ground_m": [
                    (0.04 * u + 0.003 * v + 2) / denominator,
                    (-0.002 * u + 0.05 * v - 1) / denominator,
                ],
            }
        )
    first = fit_homography(points)
    second = fit_homography(points)
    assert first == second
    projected = project_ground(first, (33, 66))
    denominator = 1 + 0.0005 * 33 - 0.0002 * 66
    assert projected == pytest.approx(
        ((0.04 * 33 + 0.003 * 66 + 2) / denominator,
         (-0.002 * 33 + 0.05 * 66 - 1) / denominator),
        abs=1e-9,
    )


def test_calibration_is_recomputable_and_schema_valid(tmp_path):
    camera, controls = _phase1_fixture(tmp_path)
    first = build_calibration(camera, controls)
    second = build_calibration(camera, controls)
    assert first == second
    assert first["status"] == "verified"
    assert first["fit"]["maximum_error_m"] == pytest.approx(0.0, abs=1e-10)
    assert first["holdout"]["maximum_error_m"] == pytest.approx(0.0, abs=1e-10)
    assert first["uncertainty"]["conservative_combined_distance_uncertainty_m"] == 0.02
    _schema_validate(
        first, "site-ground-plane-calibration-v2.schema.json", "test calibration"
    )
    output = _write_json(tmp_path / "calibration-v2.json", first)
    observed, model = _calibration(output, camera)
    assert observed == first
    assert model.camera_position == (0.0, 0.0)
    assert pin_file(output)["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_holdout_error_is_independently_recomputed_and_rejected(tmp_path):
    camera, controls = _phase1_fixture(tmp_path)
    value = json.loads(controls.read_text())
    value["holdout_points"][1]["ground_m"][1] += 1.0
    _write_json(controls, value)
    with pytest.raises(ValueError, match="holdout reprojection error"):
        build_calibration(camera, controls)


def test_fit_and_holdout_points_must_be_independent(tmp_path):
    camera, controls = _phase1_fixture(tmp_path)
    value = json.loads(controls.read_text())
    value["holdout_points"][0]["point_id"] = value["fit_points"][0]["point_id"]
    _write_json(controls, value)
    with pytest.raises(ValueError, match="point_id values must be globally unique"):
        build_calibration(camera, controls)


def test_fisheye_configuration_requires_pinned_dewarp(tmp_path):
    camera, controls = _phase1_fixture(tmp_path)
    value = json.loads(camera.read_text())
    value["optics"]["projection_model"] = "fisheye_equidistant"
    _write_json(camera, value)
    with pytest.raises(ValueError, match="require a pinned dewarp map"):
        build_calibration(camera, controls)


def test_holdout_must_span_both_20_and_25_m_boundaries(tmp_path):
    camera, controls = _phase1_fixture(tmp_path)
    value = json.loads(controls.read_text())
    for point in value["holdout_points"]:
        point["ground_m"] = [0.0, 22.0]
    _write_json(controls, value)
    with pytest.raises(ValueError, match="20 m near boundary"):
        build_calibration(camera, controls)


def _phase2_fixture(tmp_path: Path):
    camera, controls = _phase1_fixture(tmp_path)
    calibration = _write_json(
        tmp_path / "calibration-v2.json", build_calibration(camera, controls)
    )
    source = tmp_path / "site-source.mp4"
    source.write_bytes(b"deterministic-site-source-placeholder")
    frames = [
        {
            "frame_index": index,
            "timestamp_ns": index * 40_000_000,
            "image_width": 1920,
            "image_height": 1080,
            "rgb24_sha256": hashlib.sha256(f"rgb-frame-{index}".encode()).hexdigest(),
            "annotation_selected": True,
        }
        for index in range(6)
    ]
    ledger_value = {
        "schema_version": "deepsafe.site-distance-media-frame-ledger/v2",
        "status": "complete",
        "dataset_id": "site-distance-a-v2",
        "site_id": "site-a",
        "camera_id": "camera-a",
        "camera_configuration_id": "camera-a-fixed-lens-stream-v1",
        "calibration_id": "calibration-a-v2",
        "sequence_id": "site-sequence-a",
        "camera_configuration": _pin(camera),
        "calibration": _pin(calibration),
        "source_asset": _pin(source),
        "source_probe": {
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "fps_numerator": 25,
            "fps_denominator": 1,
            "time_base_numerator": 1,
            "time_base_denominator": 90000,
            "decoded_frame_count": 6,
            "duration_ns": 240_000_000,
            "probe_tool": "ffprobe",
            "probe_tool_version": "7.0",
        },
        "coordinate_space_id": "source-rgb24-1920x1080-v1",
        "frame_hash_policy": "sha256_of_rgb24_row_major_after_declared_coordinate_transform",
        "frame_key_sha256": frame_key_sha256("site-sequence-a", frames),
        "frames": frames,
        "created_by": "frame-ledger-operator",
        "created_at": "2026-07-16T12:00:00Z",
    }
    media = _write_json(tmp_path / "media-frame-ledger.json", ledger_value)

    annotation = tmp_path / "annotation-qa.md"
    annotation.write_text("all visible people and empty frames reviewed\n", encoding="utf-8")
    approval = tmp_path / "acceptance-approval.md"
    approval.write_text("owner approved criteria before inference\n", encoding="utf-8")
    bin_requirements = []
    for minimum in range(20, 25):
        maximum = minimum + 1
        bin_requirements.append(
            {
                "bin_id": f"{minimum}-{maximum}m",
                "minimum_inclusive_m": minimum,
                "maximum_m": maximum,
                "maximum_boundary": "inclusive" if maximum == 25 else "exclusive",
                "minimum_instances": 1,
                "minimum_independent_events": 1,
                "minimum_unambiguous_independent_events": 1,
            }
        )
    acceptance_value = {
        "schema_version": "deepsafe.site-distance-acceptance/v2",
        "status": "approved",
        "criterion_id": "site-owner-distance-v2",
        "task": "person_detection",
        "evidence_kind": "deployment_site_calibrated_ground_plane",
        "profiles": [640, 960],
        "distance_scope": {
            "unit": "m",
            "minimum_inclusive": 20,
            "maximum_inclusive": 25,
            "boundary_policy": "lower_inclusive_upper_exclusive_except_24_25m_upper_inclusive",
        },
        "uncertainty_policy": {
            "maximum_total_distance_uncertainty_m": 0.25,
            "quota_eligibility_rule": "distance_uncertainty_interval_must_be_fully_contained_in_one_bin",
            "exact_25m_policy": "nominal_25m_is_in_24_25m; boundary_ambiguous_samples_are_reported_but_not_unambiguous_quota",
        },
        "distance_bins": bin_requirements,
        "endpoint_requirement": {
            "minimum_nominal_distance_m": 24.5,
            "maximum_nominal_distance_inclusive_m": 25,
            "minimum_independent_events": 1,
        },
        "evaluation_config": {
            "iou_threshold": 0.5,
            "confidence_threshold": 0.25,
            "serialization_confidence_floor": 0.0,
            "distance_point": "bbox_bottom_center_ground_contact",
            "metric_geometry": "axis_aligned_bbox_iou",
            "ap_definition": "101_point_interpolated_ap_at_configured_iou",
        },
        "rules": [
            {"metric": "precision", "operator": "gte", "threshold": 0.85, "applies_to": "each_profile"},
            {"metric": "recall", "operator": "gte", "threshold": 0.80, "applies_to": "each_profile"},
            {"metric": "f1", "operator": "gte", "threshold": 0.82, "applies_to": "each_profile"},
            {"metric": "ap_101_point", "operator": "gte", "threshold": 0.70, "applies_to": "each_profile"},
        ],
        "approval": {
            "approved_by": "project-owner",
            "approved_at": "2026-07-16T13:00:00Z",
            "document": _pin(approval),
        },
    }
    acceptance = _write_json(tmp_path / "acceptance-v2.json", acceptance_value)

    distances = [20.5, 21.5, 22.5, 23.5, 24.8]
    gt_frames = []
    for index, frame in enumerate(frames):
        persons = []
        if index < len(distances):
            bottom_y = 1080 - distances[index] / 0.05
            persons.append(
                {
                    "object_id": f"person-{index}",
                    "independent_event_id": f"event-{index}",
                    "bbox_pixel_xywh": [940, bottom_y - 100, 40, 100],
                    "ground_contact_uncertainty_px": 0.0,
                    "additional_distance_uncertainty_m": 0.0,
                    "occlusion_fraction": 0.0,
                    "truncated": False,
                    "ignored": False,
                }
            )
        gt_frames.append(
            {
                "frame_index": frame["frame_index"],
                "rgb24_sha256": frame["rgb24_sha256"],
                "persons": persons,
            }
        )
    gt_value = {
        "schema_version": "deepsafe.site-distance-ground-truth/v2",
        "status": "complete",
        "evidence_kind": "deployment_site_calibrated_person_ground_truth",
        "dataset_id": "site-distance-a-v2",
        "site_id": "site-a",
        "camera_id": "camera-a",
        "camera_configuration_id": "camera-a-fixed-lens-stream-v1",
        "calibration_id": "calibration-a-v2",
        "sequence_id": "site-sequence-a",
        "camera_configuration": _pin(camera),
        "calibration": _pin(calibration),
        "media_frame_ledger": _pin(media),
        "source_asset_sha256": _pin(source)["sha256"],
        "coordinate_space_id": "source-rgb24-1920x1080-v1",
        "image": {"width": 1920, "height": 1080},
        "annotation": {
            "status": "verified",
            "all_visible_people_in_calibrated_roi_annotated": True,
            "empty_frames_retained_for_false_positive_measurement": True,
            "box_format": "pixel_xywh",
            "distance_point": "bbox_bottom_center_ground_contact",
            "independent_event_definition": "one physical person passage counts once per bin",
            "reviewed_by": "annotation-qa",
            "reviewed_at": "2026-07-16T12:30:00Z",
            "document": _pin(annotation),
        },
        "frames": gt_frames,
    }
    ground_truth = _write_json(tmp_path / "ground-truth-v2.json", gt_value)
    return {
        "camera": camera,
        "controls": controls,
        "calibration": calibration,
        "source": source,
        "media": media,
        "ground_truth": ground_truth,
        "acceptance": acceptance,
    }


def _preflight(paths):
    return validate_preflight(
        camera_configuration_path=paths["camera"],
        calibration_path=paths["calibration"],
        media_frame_ledger_path=paths["media"],
        ground_truth_path=paths["ground_truth"],
        acceptance_path=paths["acceptance"],
    )


def test_preflight_proves_five_bins_endpoint_and_exact_frame_set(tmp_path):
    paths = _phase2_fixture(tmp_path)
    result = _preflight(paths)
    assert result["status"] == "ready"
    assert result["inference_executed_by_tool"] is False
    assert [row["bin_id"] for row in result["distance_bin_coverage"]] == [
        "20-21m", "21-22m", "22-23m", "23-24m", "24-25m"
    ]
    assert all(row["status"] == "pass" for row in result["distance_bin_coverage"])
    assert result["endpoint_coverage"]["independent_events"] == 1
    assert result["media_frame_contract"]["selected_frames"] == 6
    _schema_validate(
        result, "site-distance-preflight-receipt-v2.schema.json", "test preflight"
    )


def test_exact_25m_is_final_bin_nominal_but_over_25_is_outside():
    assert _nominal_bin(25.0) == ("24-25m", 24.0, 25.0, True)
    assert _nominal_bin(25.0000001) is None


def test_preflight_fails_when_one_distance_bin_has_no_independent_event(tmp_path):
    paths = _phase2_fixture(tmp_path)
    value = json.loads(paths["ground_truth"].read_text())
    value["frames"][2]["persons"] = []
    _write_json(paths["ground_truth"], value)
    with pytest.raises(ValueError, match="22-23m coverage quota"):
        _preflight(paths)


def test_preflight_rejects_frame_rgb_hash_drift(tmp_path):
    paths = _phase2_fixture(tmp_path)
    value = json.loads(paths["ground_truth"].read_text())
    value["frames"][0]["rgb24_sha256"] = "0" * 64
    _write_json(paths["ground_truth"], value)
    with pytest.raises(ValueError, match="RGB hash does not match ledger"):
        _preflight(paths)


def test_ap_acceptance_requires_zero_serialization_floor(tmp_path):
    paths = _phase2_fixture(tmp_path)
    value = json.loads(paths["acceptance"].read_text())
    value["evaluation_config"]["serialization_confidence_floor"] = 0.1
    _write_json(paths["acceptance"], value)
    with pytest.raises(ValueError, match="serialization_confidence_floor=0.0"):
        _preflight(paths)


def _write_predictions(path: Path, profile: int, media: dict) -> Path:
    records = []
    for frame in media["frames"]:
        records.append(
            {
                "schema_version": "deepsafe.person-detections/v1",
                "sequence_id": "site-sequence-a",
                "frame_index": frame["frame_index"],
                "image_width": 1920,
                "image_height": 1080,
                "timestamp_ns": frame["timestamp_ns"],
                "source_uri": "file:///validation/inputs/distance-25m/site-source.mp4",
                "model_id": "site-person-yolo-v2",
                "detections": [],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    return path


def _profile_pair_fixture(tmp_path: Path):
    paths = _phase2_fixture(tmp_path)
    media_value = json.loads(paths["media"].read_text())
    artifacts = {}
    for name, content in {
        "model.onnx": b"same-model-weights",
        "parser.so": b"same-parser-library",
        "labels.txt": b"person\n",
        "deepstream-app": b"deepstream-app-9.0",
    }.items():
        artifact = tmp_path / name
        artifact.write_bytes(content)
        artifacts[name] = artifact
    invariants = {
        "binding_id": "site-pair-binding-v2",
        "base_model_id": "site-person-yolo-v2",
        "model_weights": _pin(artifacts["model.onnx"]),
        "parser_library": _pin(artifacts["parser.so"]),
        "label_file": _pin(artifacts["labels.txt"]),
        "runtime": {
            "deepstream_version": "9.0",
            "container_image_reference": "nvcr.io/nvidia/deepstream:9.0-triton-multiarch",
            "container_image_digest": "sha256:" + "a" * 64,
            "deepstream_app": _pin(artifacts["deepstream-app"]),
            "cuda_version": "13.0",
            "tensorrt_version": "10.13",
            "driver_version": "590.48.01",
            "gpu_uuid": "GPU-8cbaba1c-2629-a732-f528-66f459089ef6",
        },
        "preprocessing": {
            "color_format": "RGB",
            "maintain_aspect_ratio": True,
            "symmetric_padding": True,
            "normalization_factor": 1 / 255,
            "offsets": [0, 0, 0],
        },
        "postprocessing": {
            "person_class_name": "person",
            "person_class_id": 0,
            "cluster_mode": "NMS",
            "nms_iou_threshold": 0.45,
            "serialization_confidence_floor": 0.0,
            "bbox_coordinate_space": "declared_calibration_coordinate_space",
        },
        "permitted_profile_differences": [
            "profile", "network_input", "engine", "inference_config",
            "predictions", "runtime_log_and_safety_receipt", "run_timestamps"
        ],
    }
    manifests = {}
    for profile in (640, 960):
        directory = tmp_path / f"profile-{profile}"
        directory.mkdir()
        engine = directory / "model.engine"
        engine.write_bytes(f"engine-{profile}".encode())
        config = directory / "infer.txt"
        config.write_text(f"network-input={profile}\n", encoding="utf-8")
        log = directory / "deepstream.log"
        log.write_text("DeepStream 9 run completed\n", encoding="utf-8")
        safety = directory / "safety.json"
        safety.write_text('{"status":"pass"}\n', encoding="utf-8")
        predictions = _write_predictions(directory / "predictions.jsonl", profile, media_value)
        manifest_value = {
            "schema_version": "deepsafe.site-distance-profile-run/v2",
            "status": "complete",
            "evidence_kind": "deployment_site_deepstream9_profile_inference",
            "profile": profile,
            "dataset_id": "site-distance-a-v2",
            "site_id": "site-a",
            "camera_id": "camera-a",
            "camera_configuration_id": "camera-a-fixed-lens-stream-v1",
            "calibration_id": "calibration-a-v2",
            "sequence_id": "site-sequence-a",
            "coordinate_space_id": "source-rgb24-1920x1080-v1",
            "source_asset_sha256": _pin(paths["source"])["sha256"],
            "camera_configuration": _pin(paths["camera"]),
            "calibration": _pin(paths["calibration"]),
            "media_frame_ledger": _pin(paths["media"]),
            "ground_truth": _pin(paths["ground_truth"]),
            "acceptance": _pin(paths["acceptance"]),
            "predictions": _pin(predictions),
            "frame_contract": {
                "status": "exact",
                "expected_frames": 6,
                "serialized_frames": 6,
                "frame_key_sha256": media_value["frame_key_sha256"],
                "all_selected_frames_serialized_including_empty_detections": True,
            },
            "cross_profile_invariants": invariants,
            "profile_runtime": {
                "network_input": {"width": profile, "height": profile},
                "precision": "FP16",
                "engine": _pin(engine),
                "inference_config": _pin(config),
                "runtime_log": _pin(log),
                "safety_receipt": _pin(safety),
                "exit_code": 0,
                "safety_guard_status": "pass",
                "started_at": f"2026-07-16T1{4 if profile == 640 else 5}:00:00Z",
                "completed_at": f"2026-07-16T1{4 if profile == 640 else 5}:05:00Z",
            },
        }
        manifests[profile] = _write_json(directory / "run-manifest.json", manifest_value)
    paths["manifests"] = manifests
    return paths


def _pair(paths):
    return validate_profile_pair(
        camera_configuration_path=paths["camera"],
        calibration_path=paths["calibration"],
        media_frame_ledger_path=paths["media"],
        ground_truth_path=paths["ground_truth"],
        acceptance_path=paths["acceptance"],
        profile_640_manifest_path=paths["manifests"][640],
        profile_960_manifest_path=paths["manifests"][960],
    )


def test_profile_pair_binds_same_model_parser_and_deepstream9_runtime(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    result = _pair(paths)
    assert result["status"] == "verified"
    assert result["profiles"] == [640, 960]
    assert result["deepstream_version"] == "9.0"
    assert result["same_model_parser_runtime"] is True
    assert result["inference_executed_by_tool"] is False


def test_profile_pair_rejects_parser_drift(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    manifest = json.loads(paths["manifests"][960].read_text())
    parser = tmp_path / "different-parser.so"
    parser.write_bytes(b"different-parser")
    manifest["cross_profile_invariants"]["parser_library"] = _pin(parser)
    _write_json(paths["manifests"][960], manifest)
    with pytest.raises(ValueError, match="model, parser, preprocessing"):
        _pair(paths)


def test_profile_pair_rejects_acceptance_approved_after_run(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    acceptance = json.loads(paths["acceptance"].read_text())
    acceptance["approval"]["approved_at"] = "2026-07-16T16:00:00Z"
    _write_json(paths["acceptance"], acceptance)
    for manifest_path in paths["manifests"].values():
        manifest = json.loads(manifest_path.read_text())
        manifest["acceptance"] = _pin(paths["acceptance"])
        _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not approved before inference"):
        _pair(paths)


def _runner_fixture(tmp_path: Path, *, host_gpu_index: int = 0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = _phase2_fixture(tmp_path)
    preflight = _write_json(tmp_path / "preflight-receipt.json", _preflight(paths))
    artifacts = {}
    for name, content in {
        "model.onnx": b"one-dynamic-shape-person-model",
        "parser.so": b"one-deepstream9-parser",
        "labels.txt": b"person\n",
        "deepstream-app": b"deepstream-app-9.0-identity",
    }.items():
        path = tmp_path / "runtime" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts[name] = path
    status = _write_json(
        tmp_path / "validation/results/endurance/current/status.json",
        {"schema_version": "fixture/v1", "state": "complete"},
    )
    live = _write_json(
        tmp_path / "validation/results/endurance/current/live.json",
        {"schema_version": "fixture/v1", "state": "stopped"},
    )
    binding = _write_json(
        tmp_path / "runtime-binding.json",
        {
            "schema_version": "deepsafe.site-distance-runtime-binding/v2",
            "status": "locked",
            "binding_id": "site-person-dynamic-binding-v2",
            "base_model_id": "site-person-yolo-v2",
            "model_weights": _pin(artifacts["model.onnx"]),
            "parser_library": _pin(artifacts["parser.so"]),
            "label_file": _pin(artifacts["labels.txt"]),
            "runtime": {
                "deepstream_version": "9.0",
                "container_image_reference": "nvcr.io/nvidia/deepstream:9.0-triton-multiarch",
                "container_image_digest": "sha256:" + "a" * 64,
                "deepstream_app": _pin(artifacts["deepstream-app"]),
                "cuda_version": "13.0",
                "tensorrt_version": "10.13",
                "driver_version": "590.48.01",
                "gpu_uuid": "GPU-8cbaba1c-2629-a732-f528-66f459089ef6",
            },
            "preprocessing": {
                "color_format": "RGB",
                "maintain_aspect_ratio": True,
                "symmetric_padding": True,
                "normalization_factor": 1 / 255,
                "offsets": [0, 0, 0],
            },
            "postprocessing": {
                "person_class_name": "person",
                "person_class_id": 0,
                "cluster_mode": "NMS",
                "nms_iou_threshold": 0.45,
                "serialization_confidence_floor": 0.0,
                "bbox_coordinate_space": "declared_calibration_coordinate_space",
            },
            "inference_backend": {
                "parse_bbox_func_name": "NvDsInferParseYoloCuda",
                "engine_create_func_name": "NvDsInferYoloCudaEngineGet",
                "num_detected_classes": 1,
                "topk": 300,
            },
            "execution": {
                "container_runtime": "docker",
                "host_gpu_index": host_gpu_index,
                "container_gpu_ordinal": 0,
                "workspace_container_root": "/workspace",
                "validation_gpu_lock_path_template": "/tmp/deepsafe-caviar-gpu{host_gpu_index}.lock",
                "active_endurance_status_files": [
                    "validation/results/endurance/current/status.json",
                    "validation/results/endurance/current/live.json",
                ],
                "endurance_supervisor_lock": "validation/results/endurance/current/supervisor.lock",
                "execution_ready": False,
                "readiness_blockers": [
                    "onnx_dynamic_640_960_profile_attestation_missing",
                    "deepstream9_parser_abi_attestation_missing",
                    "tensorrt_engine_load_attestation_missing",
                    "endurance_global_gpu_lock_contract_missing",
                ],
            },
            "permitted_profile_differences": [
                "profile", "network_input", "engine", "inference_config",
                "predictions", "runtime_log_and_safety_receipt", "run_timestamps",
            ],
        },
    )
    paths.update(
        {
            "preflight": preflight,
            "binding": binding,
            "output_root": tmp_path / "generated/site-distance-v2",
            "status": status,
            "live": live,
        }
    )
    return paths


def _generate_runner(paths, tmp_path):
    return generate_run_plan(
        preflight_receipt_path=paths["preflight"],
        camera_configuration_path=paths["camera"],
        calibration_path=paths["calibration"],
        media_frame_ledger_path=paths["media"],
        ground_truth_path=paths["ground_truth"],
        acceptance_path=paths["acceptance"],
        runtime_binding_path=paths["binding"],
        output_root=paths["output_root"],
        workspace_root=tmp_path,
    )


def test_runner_generate_only_is_deterministic_and_semantically_bound(tmp_path):
    paths = _runner_fixture(tmp_path)
    first, plan_path = _generate_runner(paths, tmp_path)
    first_bytes = plan_path.read_bytes()
    second, second_path = _generate_runner(paths, tmp_path)
    assert first == second
    assert first_bytes == second_path.read_bytes()
    assert first["generate_only"] is True
    assert first["execution_requested"] is False
    assert first["inference_executed_by_generator"] is False
    assert first["execution_ready"] is False
    assert first["execution_blockers"] == [
        "onnx_dynamic_640_960_profile_attestation_missing",
        "deepstream9_parser_abi_attestation_missing",
        "tensorrt_engine_load_attestation_missing",
        "endurance_global_gpu_lock_contract_missing",
    ]
    assert first["semantic_equivalence"]["same_model_parser_runtime"] is True
    assert first["jobs"][0]["network_input"] == {"width": 640, "height": 640}
    assert first["jobs"][1]["network_input"] == {"width": 960, "height": 960}
    model_pin = first["cross_profile_invariants"]["model_weights"]
    assert model_pin["path"] == "runtime/model.onnx"
    assert model_pin["sha256"] == json.loads(paths["binding"].read_text())[
        "model_weights"
    ]["sha256"]
    assert first["jobs"][0]["command"][0:4] == [
        "docker", "run", "--rm", "--pull=never"
    ]
    assert "deepstream:9.0-triton-multiarch@sha256:" in first["jobs"][0]["command"][-4]
    serialized = json.dumps(first, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert ".:/workspace:ro" in first["jobs"][0]["command"]
    assert all(not pin["path"].startswith("/") for pin in first["inputs"].values())
    infer_text = (paths["output_root"] / "640/infer.txt").read_text()
    assert "infer-dims=3;640;640\n" in infer_text
    assert "infer-dims=3;640;640;0" not in infer_text
    assert "network-input-order=0\n" in infer_text
    _schema_validate(first, "site-distance-run-plan-v2.schema.json", "test run plan")


def test_portable_command_is_cross_checkout_stable_and_materialized_only_at_launch(tmp_path):
    first_root = tmp_path / "checkout-a"
    second_root = tmp_path / "checkout-b"
    first_paths = _runner_fixture(first_root)
    second_paths = _runner_fixture(second_root)
    first, _ = _generate_runner(first_paths, first_root)
    second, _ = _generate_runner(second_paths, second_root)
    assert first["jobs"][0]["command"] == second["jobs"][0]["command"]
    assert (first_paths["output_root"] / "640/infer.txt").read_bytes() == (
        second_paths["output_root"] / "640/infer.txt"
    ).read_bytes()
    portable = first["jobs"][0]["command"]
    assert str(first_root) not in json.dumps(portable)
    materialized = materialize_docker_command(portable, first_root)
    assert str(first_root.resolve()) in json.dumps(materialized)
    assert portable != materialized


def test_runner_jsonl_source_uri_comes_from_pinned_media_not_app_config(tmp_path):
    paths = _runner_fixture(tmp_path)
    plan, _ = _generate_runner(paths, tmp_path)
    uri = source_uri_from_media_ledger(paths["media"], tmp_path)
    assert uri == "file:///workspace/site-source.mp4"
    assert uri != "file://" + plan["jobs"][0]["command"][-1]
    app_text = (paths["output_root"] / "640/deepstream.txt").read_text()
    assert f"uri={uri}" in app_text


def test_runner_rejects_stored_preflight_drift(tmp_path):
    paths = _runner_fixture(tmp_path)
    value = json.loads(paths["preflight"].read_text())
    value["identity"]["dataset_id"] = "tampered-dataset"
    _write_json(paths["preflight"], value)
    with pytest.raises(ValueError, match="stored preflight receipt differs"):
        _generate_runner(paths, tmp_path)


def test_profile_config_semantics_rejects_nms_drift(tmp_path):
    paths = _runner_fixture(tmp_path)
    _, _ = _generate_runner(paths, tmp_path)
    root = paths["output_root"]
    infer_960 = root / "960/infer.txt"
    infer_960.write_text(
        infer_960.read_text().replace("nms-iou-threshold=0.45000000000000001", "nms-iou-threshold=0.4"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="semantics differ outside input/engine"):
        verify_profile_config_semantics(
            app_640=root / "640/deepstream.txt", infer_640=root / "640/infer.txt",
            app_960=root / "960/deepstream.txt", infer_960=infer_960,
        )


def test_endurance_guard_requires_both_terminal_states(tmp_path):
    paths = _runner_fixture(tmp_path)
    observed = assert_endurance_inactive(tmp_path)
    assert [item["state"] for item in observed] == ["complete", "stopped"]
    assert [item["path"] for item in observed] == [
        "validation/results/endurance/current/status.json",
        "validation/results/endurance/current/live.json",
    ]
    assert str(tmp_path) not in json.dumps(observed)
    _write_json(paths["live"], {"schema_version": "fixture/v1", "state": "running"})
    with pytest.raises(RuntimeError, match="active endurance guard rejected"):
        assert_endurance_inactive(tmp_path)


def test_runtime_binding_rejects_arbitrary_or_duplicate_endurance_paths(tmp_path):
    paths = _runner_fixture(tmp_path)
    value = json.loads(paths["binding"].read_text())
    value["execution"]["active_endurance_status_files"] = [
        "validation/results/endurance/current/status.json",
        "validation/results/endurance/current/status.json",
    ]
    _write_json(paths["binding"], value)
    with pytest.raises(ValueError, match="runtime binding"):
        _generate_runner(paths, tmp_path)


def test_dual_lock_order_is_held_across_guard_session(tmp_path):
    _runner_fixture(tmp_path)
    gpu_lock = tmp_path / "validation-gpu.lock"

    def no_containers(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with execution_guard_session(
        workspace_root=tmp_path,
        validation_gpu_lock=gpu_lock,
        container_probe_runner=no_containers,
    ) as evidence:
        assert evidence["post_lock_container_check"]["status"] == "pass"
        with pytest.raises(RuntimeError, match="supervisor lock is held"):
            with acquire_endurance_supervisor_lock(tmp_path):
                pass
        with pytest.raises(RuntimeError, match="already held"):
            with acquire_gpu_lock(gpu_lock):
                pass


def test_post_lock_container_recheck_closes_launch_race(tmp_path):
    _runner_fixture(tmp_path)
    calls = 0

    def racing_container(*args, **kwargs):
        nonlocal calls
        calls += 1
        stdout = "deepsafe-endurance-999-1\n" if calls == 3 else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match="active endurance container"):
        with execution_guard_session(
            workspace_root=tmp_path,
            validation_gpu_lock=tmp_path / "validation-gpu.lock",
            container_probe_runner=racing_container,
        ):
            pass
    assert calls == 3


def test_gpu_host_index_uuid_and_container_ordinal_are_exact(tmp_path):
    paths = _runner_fixture(tmp_path, host_gpu_index=7)
    plan, _ = _generate_runner(paths, tmp_path)
    binding = json.loads(paths["binding"].read_text())
    uuid = binding["runtime"]["gpu_uuid"]

    def gpu_map(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=f"7, {uuid}\n", stderr="")

    observed = verify_live_gpu_binding(binding, command_runner=gpu_map)
    assert observed == {
        "status": "verified",
        "host_gpu_index": 7,
        "gpu_uuid": uuid,
        "docker_gpu_request": f"device={uuid}",
        "container_gpu_ordinal": 0,
    }
    command = plan["jobs"][0]["command"]
    assert command[command.index("--gpus") + 1] == f"device={uuid}"
    assert plan["execution_guards"]["validation_gpu_lock"].endswith("gpu7.lock")
    config = (paths["output_root"] / "640/infer.txt").read_text()
    assert "gpu-id=0\n" in config
    assert "gpu-id=7\n" not in config

    def wrong_map(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=f"0, {uuid}\n", stderr="")

    with pytest.raises(RuntimeError, match="does not map exactly"):
        verify_live_gpu_binding(binding, command_runner=wrong_map)


def test_execute_requires_exact_confirmation_before_any_process(tmp_path):
    paths = _runner_fixture(tmp_path)
    _, plan_path = _generate_runner(paths, tmp_path)
    called = False

    def forbidden_process(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("process runner must not be reached")

    with pytest.raises(RuntimeError, match="requires --confirm"):
        execute_plan(
            plan_path=plan_path, confirmation="NO",
            workspace_root=tmp_path, process_runner=forbidden_process,
        )
    assert called is False
    with pytest.raises(RuntimeError, match="not execution-ready"):
        execute_plan(
            plan_path=plan_path, confirmation="RUN_SITE_DISTANCE_V2",
            workspace_root=tmp_path, process_runner=forbidden_process,
            control_runner=forbidden_process,
        )
    assert called is False


def test_gpu_lock_is_nonblocking_and_exclusive(tmp_path):
    lock = tmp_path / "gpu0.lock"
    with acquire_gpu_lock(lock):
        with pytest.raises(RuntimeError, match="already held"):
            with acquire_gpu_lock(lock):
                pass
    with acquire_gpu_lock(lock):
        assert lock.is_file()


def _runner_profile_artifacts(paths, profile=640):
    directory = paths["output_root"] / str(profile)
    engine = directory / "model.engine"
    engine.write_bytes(f"profile-{profile}-engine".encode())
    media = json.loads(paths["media"].read_text())
    predictions = _write_predictions(directory / "predictions.jsonl", profile, media)
    runtime_log = directory / "deepstream.log"
    runtime_log.write_text("portable_command_sha256=fixture\nexit_code=0\n", encoding="utf-8")
    return directory, engine, predictions, runtime_log, media


def _conversion_fixture(media):
    return {
        "schema_version": "deepsafe.person-detections/v1",
        "sequence_id": media["sequence_id"],
        "decoded_frame_files": media["source_probe"]["decoded_frame_count"],
        "exported_frame_records": len(media["frames"]),
        "skipped_unannotated_frames": 0,
        "recognized_kitti_files_all_sources": media["source_probe"]["decoded_frame_count"],
        "json_image_dimensions": [1920, 1080],
        "kitti_coordinate_dimensions": [1920, 1080],
        "person_detections": 0,
        "ignored_non_person_detections": 0,
        "clipped_person_boxes": 0,
        "dropped_degenerate_person_boxes": 0,
    }


def test_finalize_manifest_rejects_fixture_v1_receipt_bypass(tmp_path):
    paths = _runner_fixture(tmp_path)
    _, plan_path = _generate_runner(paths, tmp_path)
    directory, engine, predictions, runtime_log, _ = _runner_profile_artifacts(paths)
    safety = _write_json(
        directory / "execution-receipt.json",
        {"schema_version": "fixture/v1", "status": "pass", "profile": 640},
    )
    with pytest.raises(ValueError, match="site-distance-execution-receipt-v2"):
        finalize_profile_manifest(
            plan_path=plan_path, profile=640, engine_path=engine,
            predictions_path=predictions, runtime_log_path=runtime_log,
            safety_receipt_path=safety, started_at="2026-07-16T14:00:00Z",
            completed_at="2026-07-16T14:05:00Z",
            output_path=directory / "run-manifest.json", workspace_root=tmp_path,
        )


def test_strict_receipt_binds_plan_runtime_configs_predictions_and_guards(tmp_path):
    paths = _runner_fixture(tmp_path)
    plan, plan_path = _generate_runner(paths, tmp_path)
    directory, engine, predictions, runtime_log, media = _runner_profile_artifacts(paths)
    binding = json.loads(paths["binding"].read_text())
    uuid = binding["runtime"]["gpu_uuid"]
    states = [
        {
            "path": "validation/results/endurance/current/status.json",
            "state": "complete",
        },
        {
            "path": "validation/results/endurance/current/live.json",
            "state": "stopped",
        },
    ]
    container_check = {
        "status": "pass",
        "container_prefix": "deepsafe-endurance-",
        "matching_running_containers": 0,
    }
    safety = directory / "execution-receipt.json"
    receipt = build_execution_receipt(
        plan_path=plan_path, profile=640, engine_path=engine,
        predictions_path=predictions, runtime_log_path=runtime_log,
        gpu_binding={
            "status": "verified", "host_gpu_index": 0, "gpu_uuid": uuid,
            "docker_gpu_request": f"device={uuid}", "container_gpu_ordinal": 0,
        },
        pre_lock_states=states, post_lock_states=states,
        pre_lock_container_check=container_check,
        post_lock_container_check=container_check,
        conversion=_conversion_fixture(media),
        started_at="2026-07-16T14:00:00Z",
        completed_at="2026-07-16T14:05:00Z", output_path=safety,
        workspace_root=tmp_path,
    )
    assert receipt["schema_version"] == EXECUTION_VERSION
    assert receipt["plan_fingerprint_sha256"] == plan["plan_fingerprint_sha256"]
    assert receipt["predictions"]["path"].endswith("640/predictions.jsonl")
    assert str(tmp_path) not in json.dumps(receipt, sort_keys=True)
    _schema_validate(
        receipt, "site-distance-execution-receipt-v2.schema.json", "test receipt"
    )

    tampered = json.loads(safety.read_text())
    tampered["predictions"]["sha256"] = "0" * 64
    _write_json(safety, tampered)
    with pytest.raises(ValueError, match="predictions pin differs"):
        finalize_profile_manifest(
            plan_path=plan_path, profile=640, engine_path=engine,
            predictions_path=predictions, runtime_log_path=runtime_log,
            safety_receipt_path=safety, started_at="2026-07-16T14:00:00Z",
            completed_at="2026-07-16T14:05:00Z",
            output_path=directory / "run-manifest.json", workspace_root=tmp_path,
        )

    _write_json(safety, receipt)
    with pytest.raises(RuntimeError, match="not execution-ready"):
        finalize_profile_manifest(
            plan_path=plan_path, profile=640, engine_path=engine,
            predictions_path=predictions, runtime_log_path=runtime_log,
            safety_receipt_path=safety, started_at="2026-07-16T14:00:00Z",
            completed_at="2026-07-16T14:05:00Z",
            output_path=directory / "run-manifest.json", workspace_root=tmp_path,
        )
    assert not (directory / "run-manifest.json").exists()
