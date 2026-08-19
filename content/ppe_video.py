#!/usr/bin/env python3
"""Render DeepStream PPE predictions with the shared COLT AI video theme."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np

from content.theme import BRAND_NAME, THEME, THEME_ID
from content.person_zone_rules import (
    EVENT_SCHEMA_VERSION as SAFETY_EVENT_SCHEMA_VERSION,
    PersonZoneRuleEngine,
    PersonZoneRuleError,
    ZoneArea,
    ZoneRuleConfig,
)


class PpeVideoError(RuntimeError):
    """The source, prediction stream, or rendered artifact is invalid."""


LABELS = {
    "helmet": "KASK",
    "hi_vis": "REFLEKTIF YELEK",
    "no_helmet": "KASK YOK",
    "no_hi_vis": "YELEK YOK",
}
PERSON_EVENT_SCHEMA_VERSION = "colt-ai.person-ppe-events/v1"
TRANSITION_EVENT_SCHEMA_VERSION = "colt-ai.person-ppe-transitions/v1"
EVIDENCE_LABELS = {
    "present": "VAR",
    "absent": "YOK",
    "unknown": "BELIRSIZ",
}
WALKWAY_ALERT_LABEL = "YURUYUS YOLU DISI"
DEFAULT_WALKWAY_ALERT_POLICY = {
    "tracking_identity": "nvdcf_track_id",
    "person_anchor": "bbox_bottom_center",
    "zone_rule": "outside_all_safe_walkways",
    "enter_debounce_frames": 6,
    "exit_debounce_frames": 4,
    "track_ttl_frames": 15,
    "ppe_scope": "all_tracked_persons",
}
_TURKISH_ASCII = str.maketrans(
    {
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
)


def _opencv_text(value: object) -> str:
    """Return stable ASCII text for OpenCV's built-in Hershey fonts."""

    translated = str(value).translate(_TURKISH_ASCII)
    return " ".join(
        "".join(
            character if 32 <= ord(character) <= 126 else " "
            for character in translated
        ).split()
    )


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_rois(
    rois: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Normalize optional ROI polygons; ``None`` deliberately means full frame."""

    if rois is None:
        return None
    if not isinstance(rois, Sequence) or isinstance(rois, (str, bytes)) or not rois:
        raise PpeVideoError("ROIs must be a non-empty sequence when supplied")
    result: list[dict[str, Any]] = []
    for index, roi in enumerate(rois, 1):
        if not isinstance(roi, Mapping):
            raise PpeVideoError(f"ROI {index} must be an object")
        roi_id = roi.get("roi_id")
        name = roi.get("name")
        points_value = roi.get("points")
        if not isinstance(roi_id, str) or not roi_id.strip():
            raise PpeVideoError(f"ROI {index} id is invalid")
        if not isinstance(name, str) or not name.strip():
            raise PpeVideoError(f"ROI {index} name is invalid")
        if (
            not isinstance(points_value, Sequence)
            or isinstance(points_value, (str, bytes))
            or not 3 <= len(points_value) <= 16
        ):
            raise PpeVideoError(f"ROI {index} needs 3..16 points")
        points: list[dict[str, float]] = []
        for point in points_value:
            if not isinstance(point, Mapping):
                raise PpeVideoError(f"ROI {index} point is invalid")
            x = point.get("x")
            y = point.get("y")
            if (
                not _finite(x)
                or not _finite(y)
                or not 0.0 <= float(x) <= 1.0
                or not 0.0 <= float(y) <= 1.0
            ):
                raise PpeVideoError(
                    f"ROI {index} points must be finite and normalized"
                )
            points.append({"x": float(x), "y": float(y)})
        area = abs(
            sum(
                points[point_index]["x"]
                * points[(point_index + 1) % len(points)]["y"]
                - points[(point_index + 1) % len(points)]["x"]
                * points[point_index]["y"]
                for point_index in range(len(points))
            )
            / 2.0
        )
        if area < 1e-5:
            raise PpeVideoError(f"ROI {index} area is too small")
        result.append(
            {
                "roi_id": roi_id.strip(),
                "name": name.strip(),
                "points": points,
            }
        )
    return result


def _validate_safe_walkways(
    safe_walkways: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize v2 allowed polygons without turning them into a filter."""

    if safe_walkways is None:
        return []
    if (
        not isinstance(safe_walkways, Sequence)
        or isinstance(safe_walkways, (str, bytes))
        or len(safe_walkways) > 8
    ):
        raise PpeVideoError("safe_walkways must contain zero to eight areas")
    if not safe_walkways:
        return []
    legacy_shape: list[dict[str, Any]] = []
    for index, raw in enumerate(safe_walkways, 1):
        if not isinstance(raw, Mapping):
            raise PpeVideoError(f"safe walkway {index} must be an object")
        area_type = raw.get(
            "area_type",
            raw.get("roi_type", raw.get("type")),
        )
        if area_type != "safe_walkway":
            raise PpeVideoError(
                f"safe walkway {index} must have type safe_walkway"
            )
        area_id = raw.get("area_id", raw.get("roi_id"))
        legacy_shape.append(
            {
                "roi_id": area_id,
                "name": raw.get("name"),
                "points": raw.get("points", raw.get("polygon_norm")),
            }
        )
    normalized = _validate_rois(legacy_shape)
    return [
        {
            "area_id": area["roi_id"],
            "roi_id": area["roi_id"],
            "area_type": "safe_walkway",
            "name": area["name"],
            "points": area["points"],
        }
        for area in normalized
    ]


def _validate_alert_policy(
    value: Mapping[str, Any] | None,
    *,
    walkway_enabled: bool,
) -> dict[str, Any]:
    policy = dict(DEFAULT_WALKWAY_ALERT_POLICY)
    if value is not None:
        if not isinstance(value, Mapping):
            raise PpeVideoError("alert_policy must be an object")
        policy.update(value)
    expected_rule = (
        "outside_all_safe_walkways" if walkway_enabled else "disabled"
    )
    if value is None and not walkway_enabled:
        policy["zone_rule"] = "disabled"
    if policy.get("zone_rule") != expected_rule:
        raise PpeVideoError(
            f"alert_policy.zone_rule must be {expected_rule}"
        )
    if policy.get("tracking_identity") != "nvdcf_track_id":
        raise PpeVideoError(
            "alert_policy.tracking_identity must be nvdcf_track_id"
        )
    if policy.get("person_anchor") != "bbox_bottom_center":
        raise PpeVideoError(
            "alert_policy.person_anchor must be bbox_bottom_center"
        )
    allowed_ppe_scopes = (
        {"all_tracked_persons"}
        if walkway_enabled
        else {"all_tracked_persons", "legacy_roi_visual_filter"}
    )
    if policy.get("ppe_scope") not in allowed_ppe_scopes:
        raise PpeVideoError(
            "alert_policy.ppe_scope is invalid"
        )
    for field in (
        "enter_debounce_frames",
        "exit_debounce_frames",
        "track_ttl_frames",
    ):
        number = policy.get(field)
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number <= 0
        ):
            raise PpeVideoError(f"alert_policy.{field} must be positive")
    return policy


def _inside_any_roi(
    box: tuple[float, float, float, float],
    rois: Sequence[Mapping[str, Any]] | None,
) -> bool:
    if rois is None:
        return True
    x, y, width, height = box
    center = (x + width / 2.0, y + height / 2.0)
    for roi in rois:
        contour = np.asarray(
            [
                [float(point["x"]), float(point["y"])]
                for point in roi["points"]
            ],
            dtype=np.float32,
        )
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            return True
    return False


def _draw_rois(
    frame: np.ndarray,
    rois: Sequence[Mapping[str, Any]] | None,
) -> None:
    if rois is None:
        return
    height, width = frame.shape[:2]
    polygons = [
        np.asarray(
            [
                [
                    max(0, min(width - 1, int(round(float(point["x"]) * width)))),
                    max(
                        0,
                        min(
                            height - 1,
                            int(round(float(point["y"]) * height)),
                        ),
                    ),
                ]
                for point in roi["points"]
            ],
            dtype=np.int32,
        )
        for roi in rois
    ]
    overlay = frame.copy()
    cv2.fillPoly(overlay, polygons, THEME.bgr("warning"), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0.0, frame)
    line_width = max(2, int(round(height / 540)))
    for roi, polygon in zip(rois, polygons):
        cv2.polylines(
            frame,
            [polygon],
            True,
            THEME.bgr("panel"),
            line_width + 3,
            cv2.LINE_AA,
        )
        cv2.polylines(
            frame,
            [polygon],
            True,
            THEME.bgr("warning"),
            line_width,
            cv2.LINE_AA,
        )
        anchor_x = int(polygon[0][0])
        anchor_y = int(polygon[0][1])
        label = _opencv_text(roi["name"])
        font_scale = max(0.42, min(0.68, width / 2600.0))
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1
        )
        label_top = max(0, anchor_y - label_height - baseline - 10)
        label_right = min(width - 1, anchor_x + label_width + 14)
        cv2.rectangle(
            frame,
            (anchor_x, label_top),
            (label_right, anchor_y),
            THEME.bgr("background"),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (anchor_x + 7, max(label_height + 1, anchor_y - baseline - 4)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            THEME.bgr("warning"),
            1,
            cv2.LINE_AA,
        )


def _draw_safe_walkways(
    frame: np.ndarray,
    safe_walkways: Sequence[Mapping[str, Any]],
) -> None:
    """Draw allowed walking polygons in cyan without filtering people."""

    if not safe_walkways:
        return
    height, width = frame.shape[:2]
    polygons = [
        np.asarray(
            [
                [
                    max(0, min(width - 1, int(round(float(point["x"]) * width)))),
                    max(
                        0,
                        min(
                            height - 1,
                            int(round(float(point["y"]) * height)),
                        ),
                    ),
                ]
                for point in walkway["points"]
            ],
            dtype=np.int32,
        )
        for walkway in safe_walkways
    ]
    overlay = frame.copy()
    cv2.fillPoly(overlay, polygons, THEME.bgr("safe"), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0.0, frame)
    line_width = max(2, int(round(height / 540)))
    for walkway, polygon in zip(safe_walkways, polygons):
        cv2.polylines(
            frame,
            [polygon],
            True,
            THEME.bgr("panel"),
            line_width + 3,
            cv2.LINE_AA,
        )
        cv2.polylines(
            frame,
            [polygon],
            True,
            THEME.bgr("safe"),
            line_width,
            cv2.LINE_AA,
        )
        anchor_x = int(polygon[0][0])
        anchor_y = int(polygon[0][1])
        label = _opencv_text(
            f"GUVENLI YURUYUS YOLU - {walkway['name']}"
        )
        font_scale = max(0.42, min(0.68, width / 2600.0))
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1
        )
        label_top = max(0, anchor_y - label_height - baseline - 10)
        label_right = min(width - 1, anchor_x + label_width + 14)
        cv2.rectangle(
            frame,
            (anchor_x, label_top),
            (label_right, anchor_y),
            THEME.bgr("background"),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (anchor_x + 7, max(label_height + 1, anchor_y - baseline - 4)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            THEME.bgr("safe"),
            1,
            cv2.LINE_AA,
        )


def _prediction_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PpeVideoError(
                    f"predictions line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise PpeVideoError(
                    f"predictions line {line_number} is not an object"
                )
            yield row


def _validate_detection(
    detection: Mapping[str, Any],
) -> tuple[str, str, float, tuple[float, float, float, float]]:
    canonical = str(detection.get("canonical_class", ""))
    if canonical not in LABELS:
        raise PpeVideoError(f"unsupported PPE class: {canonical!r}")
    compliance = str(detection.get("compliance", ""))
    if compliance not in {"compliant", "noncompliant", "violation"}:
        raise PpeVideoError(f"invalid PPE compliance: {compliance!r}")
    confidence = detection.get("confidence")
    if not _finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
        raise PpeVideoError("PPE confidence is outside [0,1]")
    box = detection.get("bbox_norm_xywh")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(_finite(value) for value in box)
    ):
        raise PpeVideoError("PPE bbox_norm_xywh must contain four finite values")
    x, y, width, height = (float(value) for value in box)
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1.000001
        or y + height > 1.000001
    ):
        raise PpeVideoError("PPE bbox_norm_xywh is outside the frame")
    return canonical, compliance, float(confidence), (x, y, width, height)


def _validate_identity(value: object, *, label: str) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PpeVideoError(f"{label} is invalid")
    if isinstance(value, int):
        if value < 0:
            raise PpeVideoError(f"{label} is invalid")
        return value
    if isinstance(value, str) and value.strip() and len(value.strip()) <= 64:
        return value.strip()
    raise PpeVideoError(f"{label} is invalid")


def _validate_equipment(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PpeVideoError(f"{label} must be an object")
    evidence = value.get("evidence")
    if evidence not in EVIDENCE_LABELS:
        raise PpeVideoError(
            f"{label}.evidence must be present, absent, or unknown"
        )
    state = value.get("state", evidence)
    if state not in EVIDENCE_LABELS:
        raise PpeVideoError(
            f"{label}.state must be present, absent, or unknown"
        )
    confidence = value.get("confidence", 0.0)
    if not _finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
        raise PpeVideoError(f"{label}.confidence is outside [0,1]")
    state_confidence = value.get("state_confidence", confidence)
    if not _finite(state_confidence) or not 0.0 <= float(state_confidence) <= 1.0:
        raise PpeVideoError(f"{label}.state_confidence is outside [0,1]")
    default_link = "matched" if evidence in {"present", "absent"} else (
        "unknown_no_observation"
    )
    link_status = value.get("link_status", default_link)
    if not isinstance(link_status, str) or not link_status.strip():
        raise PpeVideoError(f"{label}.link_status is invalid")
    link_status = link_status.strip()
    evidence_source = value.get("evidence_source")
    if evidence_source is None:
        if link_status == "matched":
            evidence_source = "detector_observation"
        elif link_status == "inferred_visible_missing":
            evidence_source = "visible_zone_no_positive_observation"
        elif link_status == "unknown_conflicting":
            evidence_source = "conflicting_observations"
        elif link_status == "unknown_no_observation":
            evidence_source = "insufficient_visibility_or_no_observation"
    valid_contracts = {
        "detector_observation": (
            frozenset({"present", "absent"}),
            "matched",
        ),
        "visible_zone_no_positive_observation": (
            frozenset({"absent"}),
            "inferred_visible_missing",
        ),
        "conflicting_observations": (
            frozenset({"unknown"}),
            "unknown_conflicting",
        ),
        "insufficient_visibility_or_no_observation": (
            frozenset({"unknown"}),
            "unknown_no_observation",
        ),
        "verified_present_grace_no_observation": (
            frozenset({"unknown"}),
            "unknown_no_observation",
        ),
    }
    contract = valid_contracts.get(str(evidence_source))
    if (
        contract is None
        or evidence not in contract[0]
        or link_status != contract[1]
    ):
        raise PpeVideoError(
            f"{label} evidence/source/link_status contract is inconsistent"
        )
    if "observations" in value:
        raw_observations = value.get("observations")
        if not isinstance(raw_observations, list):
            raise PpeVideoError(f"{label}.observations must be a list")
        if evidence_source == "detector_observation" and not raw_observations:
            raise PpeVideoError(
                f"{label} matched detector evidence needs an observation"
            )
        if (
            evidence_source
            in {
                "visible_zone_no_positive_observation",
                "insufficient_visibility_or_no_observation",
                "verified_present_grace_no_observation",
            }
            and raw_observations
        ):
            raise PpeVideoError(
                f"{label} no-observation evidence cannot carry observations"
            )
        if (
            evidence_source == "conflicting_observations"
            and len(raw_observations) < 2
        ):
            raise PpeVideoError(
                f"{label} conflicting evidence needs two observations"
            )
    observation_id = _validate_identity(
        value.get("observation_id"), label=f"{label}.observation_id"
    )
    return {
        "evidence": str(evidence),
        "state": str(state),
        "state_source": (
            "temporal_state" if "state" in value else "evidence_fallback"
        ),
        "confidence": float(confidence),
        "state_confidence": float(state_confidence),
        "evidence_source": str(evidence_source),
        "link_status": link_status,
        "observation_id": observation_id,
    }


def _validate_zone_safety(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PpeVideoError("zone_safety must be an object")
    footpoint = value.get("footpoint_norm_xy")
    if (
        not isinstance(footpoint, Sequence)
        or isinstance(footpoint, (str, bytes))
        or len(footpoint) != 2
        or not all(_finite(item) for item in footpoint)
        or not all(0.0 <= float(item) <= 1.0 for item in footpoint)
    ):
        raise PpeVideoError(
            "zone_safety.footpoint_norm_xy must be normalized"
        )
    violation_active = value.get("violation_active")
    if not isinstance(violation_active, bool):
        raise PpeVideoError("zone_safety.violation_active must be boolean")
    active_types = value.get("active_violation_types")
    if (
        not isinstance(active_types, Sequence)
        or isinstance(active_types, (str, bytes))
        or any(
            item not in {
                "restricted_area_intrusion",
                "walkway_violation",
            }
            for item in active_types
        )
    ):
        raise PpeVideoError(
            "zone_safety.active_violation_types is invalid"
        )
    area_ids = value.get("contributing_area_ids")
    if (
        not isinstance(area_ids, Sequence)
        or isinstance(area_ids, (str, bytes))
        or any(
            not isinstance(item, str) or not item.strip()
            for item in area_ids
        )
    ):
        raise PpeVideoError(
            "zone_safety.contributing_area_ids is invalid"
        )
    if violation_active != bool(active_types):
        raise PpeVideoError("zone_safety active state is inconsistent")
    return {
        **dict(value),
        "footpoint_norm_xy": [float(item) for item in footpoint],
        "violation_active": violation_active,
        "active_violation_types": list(active_types),
        "contributing_area_ids": [str(item) for item in area_ids],
    }


def _validate_vehicle_context(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PpeVideoError("vehicle_context must be an object")
    for field in (
        "raw_match",
        "suppression_active",
        "ppe_alert_eligible",
        "walkway_alert_eligible",
    ):
        if not isinstance(value.get(field), bool):
            raise PpeVideoError(f"vehicle_context.{field} must be boolean")
    role = value.get("role")
    if role not in {None, "forklift_operator"}:
        raise PpeVideoError("vehicle_context.role is invalid")
    if bool(value["suppression_active"]) != (role == "forklift_operator"):
        raise PpeVideoError("vehicle_context role/active state is inconsistent")
    vehicle_track_id = value.get("vehicle_track_id")
    if vehicle_track_id is not None and (
        not isinstance(vehicle_track_id, int)
        or isinstance(vehicle_track_id, bool)
        or vehicle_track_id < 0
    ):
        raise PpeVideoError("vehicle_context.vehicle_track_id is invalid")
    person_ioa = value.get("person_ioa")
    if (
        not _finite(person_ioa)
        or not 0.0 <= float(person_ioa) <= 1.0
    ):
        raise PpeVideoError("vehicle_context.person_ioa is outside [0,1]")
    scopes = value.get("suppression_scopes", [])
    if (
        not isinstance(scopes, Sequence)
        or isinstance(scopes, (str, bytes))
        or any(scope not in {"ppe", "walkway"} for scope in scopes)
    ):
        raise PpeVideoError("vehicle_context.suppression_scopes is invalid")
    return {
        **dict(value),
        "vehicle_track_id": vehicle_track_id,
        "person_ioa": float(person_ioa),
        "suppression_scopes": list(scopes),
    }


def _validate_person(person: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(person, Mapping):
        raise PpeVideoError("person must be an object")
    person_id = _validate_identity(person.get("person_id"), label="person_id")
    track_id = _validate_identity(person.get("track_id"), label="track_id")
    if person_id is None and track_id is None:
        raise PpeVideoError("person needs person_id or track_id")
    box = person.get("bbox_norm_xywh")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(_finite(value) for value in box)
    ):
        raise PpeVideoError(
            "person bbox_norm_xywh must contain four finite values"
        )
    x, y, width, height = (float(value) for value in box)
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1.000001
        or y + height > 1.000001
    ):
        raise PpeVideoError("person bbox_norm_xywh is outside the frame")
    confidence = person.get("confidence")
    if confidence is not None and (
        not _finite(confidence) or not 0.0 <= float(confidence) <= 1.0
    ):
        raise PpeVideoError("person confidence is outside [0,1]")
    occluded = person.get("occluded", False)
    if not isinstance(occluded, bool):
        raise PpeVideoError("person occluded must be boolean")
    vehicle_context = _validate_vehicle_context(
        person.get("vehicle_context")
    )
    zone_alert_eligible = person.get("zone_alert_eligible", True)
    if not isinstance(zone_alert_eligible, bool):
        raise PpeVideoError("person zone_alert_eligible must be boolean")
    return {
        "person_id": person_id,
        "track_id": track_id,
        "bbox_norm_xywh": [x, y, width, height],
        "confidence": None if confidence is None else float(confidence),
        "occluded": occluded,
        "helmet": _validate_equipment(person.get("helmet"), label="helmet"),
        "hi_vis": _validate_equipment(person.get("hi_vis"), label="hi_vis"),
        "zone_safety": _validate_zone_safety(person.get("zone_safety")),
        "vehicle_context": vehicle_context,
        "zone_alert_eligible": zone_alert_eligible,
    }


def _validated_persons(
    persons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(persons, Sequence) or isinstance(persons, (str, bytes)):
        raise PpeVideoError("persons must be a list")
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for raw_person in persons:
        person = _validate_person(raw_person)
        for field in ("track_id", "person_id"):
            value = person[field]
            if value is None:
                continue
            identity = (field, str(value))
            if identity in identities:
                raise PpeVideoError(f"duplicate {field} inside frame: {value}")
            identities.add(identity)
        result.append(person)
    return result


def _person_display_id(person: Mapping[str, Any]) -> str:
    identity = person.get("track_id")
    if identity is None:
        identity = person.get("person_id")
    return str(identity)


def _walkway_violation(person: Mapping[str, Any]) -> bool:
    vehicle_context = person.get("vehicle_context")
    if (
        isinstance(vehicle_context, Mapping)
        and vehicle_context.get("walkway_alert_eligible") is False
    ):
        return False
    zone_safety = person.get("zone_safety")
    return bool(
        isinstance(zone_safety, Mapping)
        and zone_safety.get("violation_active") is True
        and "walkway_violation"
        in zone_safety.get("active_violation_types", ())
    )


def person_ppe_events(
    *,
    frame_index: int,
    persons: Sequence[Mapping[str, Any]],
    camera_label: str,
    timestamp_ns: int | None = None,
) -> list[dict[str, Any]]:
    """Return legacy per-frame snapshots for each stable PPE violation.

    Display and alert decisions use temporal ``state`` when supplied. The
    instantaneous ``evidence`` remains diagnostic, and is used as a backward-
    compatible state fallback only for older prediction rows. The video
    renderer does not use these snapshots when the fusion stream supplies
    transition events.
    """

    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise PpeVideoError("frame_index must be a non-negative integer")
    if frame_index < 0:
        raise PpeVideoError("frame_index must be a non-negative integer")
    if timestamp_ns is not None and (
        isinstance(timestamp_ns, bool)
        or not isinstance(timestamp_ns, int)
        or timestamp_ns < 0
    ):
        raise PpeVideoError("timestamp_ns must be a non-negative integer")
    normalized = _validated_persons(persons)
    events: list[dict[str, Any]] = []
    for person in normalized:
        vehicle_context = person.get("vehicle_context")
        if (
            isinstance(vehicle_context, Mapping)
            and vehicle_context.get("ppe_alert_eligible") is False
        ):
            continue
        violations = [
            equipment
            for equipment in ("helmet", "hi_vis")
            if person[equipment]["state"] == "absent"
        ]
        if not violations:
            continue
        events.append(
            {
                "schema_version": PERSON_EVENT_SCHEMA_VERSION,
                "event_type": "ppe_noncompliance",
                "severity": "critical" if "helmet" in violations else "warning",
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "camera_label": camera_label,
                "person_id": person["person_id"],
                "track_id": person["track_id"],
                "bbox_norm_xywh": list(person["bbox_norm_xywh"]),
                "violations": [
                    f"{equipment}_missing" for equipment in violations
                ],
                "equipment": {
                    equipment: {
                        "state": person[equipment]["state"],
                        "state_source": person[equipment]["state_source"],
                        "evidence": person[equipment]["evidence"],
                        "confidence": person[equipment]["confidence"],
                        "link_status": person[equipment]["link_status"],
                    }
                    for equipment in ("helmet", "hi_vis")
                },
                "operator_attention": True,
            }
        )
    return events


def transition_event_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    person_mode: bool,
) -> dict[str, Any]:
    """Validate and summarize fusion transitions without per-frame alert spam."""

    flags = ["alarm_events" in row for row in rows]
    if any(flags) and not all(flags):
        raise PpeVideoError(
            "alarm_events must be present on every frame in transition mode"
        )
    if not all(flags):
        return {
            "schema_version": None,
            "mode": (
                "legacy_person_rows_without_transitions"
                if person_mode
                else "legacy_equipment_boxes_without_transitions"
            ),
            "source": None,
            "scope": "no_transition_stream_available",
            "per_frame_violation_events_emitted": False,
            "unknown_state_emits_event": False,
            "observations": {},
            "open_alarms_at_end": [],
        }
    if not person_mode:
        raise PpeVideoError(
            "alarm_events require person-centric prediction rows"
        )

    counts: Counter[str] = Counter()
    open_alarms: set[tuple[int, str]] = set()
    for row_index, row in enumerate(rows):
        raw_events = row.get("alarm_events")
        if not isinstance(raw_events, list):
            raise PpeVideoError(
                f"alarm_events must be a list at frame {row_index}"
            )
        seen: set[tuple[str, str, int]] = set()
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                raise PpeVideoError(
                    f"alarm event must be an object at frame {row_index}"
                )
            kind = raw_event.get("kind")
            event_type = raw_event.get("type")
            track_id = raw_event.get("track_id")
            frame_index = raw_event.get("frame_index")
            if kind not in {"started", "ended"}:
                raise PpeVideoError(
                    f"alarm event kind is invalid at frame {row_index}"
                )
            if event_type not in {"no_helmet", "no_hi_vis"}:
                raise PpeVideoError(
                    f"alarm event type is invalid at frame {row_index}"
                )
            if (
                not isinstance(track_id, int)
                or isinstance(track_id, bool)
                or track_id < 0
            ):
                raise PpeVideoError(
                    f"alarm event track_id is invalid at frame {row_index}"
                )
            if frame_index != row_index:
                raise PpeVideoError(
                    f"alarm event frame_index differs at frame {row_index}"
                )
            identity = (str(kind), str(event_type), track_id)
            if identity in seen:
                raise PpeVideoError(
                    f"duplicate alarm transition at frame {row_index}"
                )
            seen.add(identity)
            alarm_key = (track_id, str(event_type))
            if kind == "started":
                if alarm_key in open_alarms:
                    raise PpeVideoError(
                        f"alarm started twice without ending at frame {row_index}"
                    )
                open_alarms.add(alarm_key)
            else:
                if alarm_key not in open_alarms:
                    raise PpeVideoError(
                        f"alarm ended without a start at frame {row_index}"
                    )
                open_alarms.remove(alarm_key)
            counts["total"] += 1
            counts[str(kind)] += 1
            counts[f"{kind}:{event_type}"] += 1
    return {
        "schema_version": TRANSITION_EVENT_SCHEMA_VERSION,
        "mode": "fusion_state_transitions",
        "source": "alarm_events",
        "scope": "full_frame_person_track_state_changes",
        "per_frame_violation_events_emitted": False,
        "unknown_state_emits_event": False,
        "observations": dict(sorted(counts.items())),
        "open_alarms_at_end": [
            {"track_id": track_id, "type": event_type}
            for track_id, event_type in sorted(open_alarms)
        ],
    }


def draw_ppe_frame(
    frame: np.ndarray,
    *,
    detections: Sequence[Mapping[str, Any]],
    camera_label: str,
    rois: Sequence[Mapping[str, Any]] | None = None,
    safe_walkways: Sequence[Mapping[str, Any]] | None = None,
    persons: Sequence[Mapping[str, Any]] | None = None,
    vehicles: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, Counter[str]]:
    """Draw a model-name-free PPE frame using the shared navy theme.

    ``persons`` selects the preferred person-centric contract. Each human gets
    one tracked bbox carrying helmet and hi-vis status. Omitting ``persons``
    preserves the legacy equipment-box renderer for existing prediction files.
    """

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise PpeVideoError("frame must be a BGR HxWx3 image")
    normalized_rois = _validate_rois(rois)
    normalized_walkways = _validate_safe_walkways(safe_walkways)
    if normalized_rois is not None and normalized_walkways:
        raise PpeVideoError(
            "legacy rois and safe_walkways cannot be combined"
        )
    return _draw_ppe_frame_validated(
        frame,
        detections=detections,
        camera_label=camera_label,
        rois=normalized_rois,
        safe_walkways=normalized_walkways,
        persons=persons,
        vehicles=vehicles,
    )


def _box_pixels(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x, y, box_width, box_height = box
    left = max(0, min(width - 1, int(round(x * width))))
    top = max(0, min(height - 1, int(round(y * height))))
    right = max(0, min(width - 1, int(round((x + box_width) * width))))
    bottom = max(0, min(height - 1, int(round((y + box_height) * height))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _draw_vehicle_candidates(
    frame: np.ndarray,
    vehicles: Sequence[Mapping[str, Any]] | None,
) -> int:
    if vehicles is None:
        return 0
    if not isinstance(vehicles, Sequence) or isinstance(
        vehicles, (str, bytes)
    ):
        raise PpeVideoError("vehicles must be a list")
    height, width = frame.shape[:2]
    rendered = 0
    for index, vehicle in enumerate(vehicles):
        if not isinstance(vehicle, Mapping):
            raise PpeVideoError(f"vehicles[{index}] must be an object")
        if vehicle.get("class_name") != "forklift_candidate":
            raise PpeVideoError(
                f"vehicles[{index}] class_name must be forklift_candidate"
            )
        box = vehicle.get("bbox_norm_xywh")
        if (
            not isinstance(box, Sequence)
            or isinstance(box, (str, bytes))
            or len(box) != 4
            or not all(_finite(value) for value in box)
        ):
            raise PpeVideoError(f"vehicles[{index}] bbox is invalid")
        normalized_box = tuple(float(value) for value in box)
        pixels = _box_pixels(normalized_box, width=width, height=height)
        if pixels is None:
            raise PpeVideoError(f"vehicles[{index}] bbox is degenerate")
        confidence = vehicle.get("confidence")
        if not _finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise PpeVideoError(
                f"vehicles[{index}] confidence is outside [0,1]"
            )
        track_id = vehicle.get("track_id")
        if (
            not isinstance(track_id, int)
            or isinstance(track_id, bool)
            or track_id < 0
        ):
            raise PpeVideoError(f"vehicles[{index}] track_id is invalid")
        left, top, right, bottom = pixels
        color = THEME.bgr("warning")
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            THEME.bgr("panel"),
            7,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            3,
            cv2.LINE_AA,
        )
        label = f"FORKLIFT ADAYI #{track_id}"
        font_scale = max(0.48, min(0.78, width / 2350.0))
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            1,
        )
        label_top = max(0, top - text_height - baseline - 12)
        cv2.rectangle(
            frame,
            (left, label_top),
            (min(width - 1, left + text_width + 16), top),
            THEME.bgr("background"),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (left + 7, max(text_height + 2, top - baseline - 5)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            THEME.bgr("text"),
            1,
            cv2.LINE_AA,
        )
        rendered += 1
    return rendered


def _draw_person_status(
    frame: np.ndarray,
    *,
    person: Mapping[str, Any],
) -> None:
    height, width = frame.shape[:2]
    pixels = _box_pixels(
        person["bbox_norm_xywh"], width=width, height=height
    )
    if pixels is None:
        return
    left, top, right, bottom = pixels
    helmet_state = person["helmet"]["state"]
    hi_vis_state = person["hi_vis"]["state"]
    vehicle_context = person.get("vehicle_context")
    forklift_operator = bool(
        isinstance(vehicle_context, Mapping)
        and vehicle_context.get("suppression_active") is True
    )
    ppe_eligible = not (
        isinstance(vehicle_context, Mapping)
        and vehicle_context.get("ppe_alert_eligible") is False
    )
    walkway_violation = _walkway_violation(person)
    explicit_violation = (
        (ppe_eligible and "absent" in {helmet_state, hi_vis_state})
        or walkway_violation
    )
    uncertain = ppe_eligible and "unknown" in {helmet_state, hi_vis_state}
    token = "violation" if explicit_violation else "warning" if uncertain else "safe"
    color = THEME.bgr(token)

    cv2.rectangle(
        frame, (left, top), (right, bottom), THEME.bgr("panel"), 6, cv2.LINE_AA
    )
    cv2.rectangle(frame, (left, top), (right, bottom), color, 3, cv2.LINE_AA)
    if (ppe_eligible and helmet_state == "absent") or walkway_violation:
        alert_overlay = frame.copy()
        cv2.rectangle(
            alert_overlay, (left, top), (right, bottom), color, -1, cv2.LINE_AA
        )
        cv2.addWeighted(alert_overlay, 0.06, frame, 0.94, 0.0, frame)
        corner = max(12, min(34, int(round(min(width, height) * 0.026))))
        line_width = max(3, int(round(height / 300)))
        for start, horizontal, vertical in (
            ((left, top), (left + corner, top), (left, top + corner)),
            ((right, top), (right - corner, top), (right, top + corner)),
            ((left, bottom), (left + corner, bottom), (left, bottom - corner)),
            ((right, bottom), (right - corner, bottom), (right, bottom - corner)),
        ):
            cv2.line(frame, start, horizontal, color, line_width, cv2.LINE_AA)
            cv2.line(frame, start, vertical, color, line_width, cv2.LINE_AA)

    identity = _person_display_id(person)
    title = (
        f"FORKLIFT SURUCUSU #{identity}"
        if forklift_operator
        else f"INSAN #{identity}"
    )
    status = (
        "PPE / YAYA ALARMI BASTIRILDI"
        if forklift_operator and not ppe_eligible
        else (
            f"KASK {EVIDENCE_LABELS[helmet_state]}  |  "
            f"YELEK {EVIDENCE_LABELS[hi_vis_state]}"
        )
    )
    lines = [title, status]
    if walkway_violation:
        lines.append(WALKWAY_ALERT_LABEL)
    font_scale = max(0.42, min(0.76, width / 2450.0))
    title_scale = font_scale * 1.04
    metrics = [
        cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_DUPLEX,
            title_scale if index == 0 else font_scale,
            1,
        )
        for index, line in enumerate(lines)
    ]
    padding = 7
    block_height = int(
        sum(size[1] + baseline for size, baseline in metrics)
        + padding * (len(lines) + 1)
    )
    block_width = max(size[0] for size, _ in metrics) + padding * 2
    label_top = top - block_height
    if label_top < 0:
        label_top = top
    label_bottom = min(height - 1, label_top + block_height)
    label_right = min(width - 1, left + block_width)
    cv2.rectangle(
        frame,
        (left, label_top),
        (label_right, label_bottom),
        THEME.bgr("background"),
        -1,
    )
    cv2.rectangle(
        frame,
        (left, label_top),
        (label_right, label_bottom),
        color,
        1,
        cv2.LINE_AA,
    )
    cursor_y = label_top + padding
    for index, (line, (size, baseline)) in enumerate(zip(lines, metrics)):
        cursor_y += size[1]
        cv2.putText(
            frame,
            line,
            (left + padding, min(label_bottom - baseline - 1, cursor_y)),
            cv2.FONT_HERSHEY_DUPLEX,
            title_scale if index == 0 else font_scale,
            THEME.bgr("text") if index == 0 else color,
            1,
            cv2.LINE_AA,
        )
        cursor_y += baseline + padding


def _draw_alert_panel(
    frame: np.ndarray,
    *,
    camera_label: str,
    helmet_person_ids: Sequence[str],
    walkway_person_ids: Sequence[str],
) -> None:
    rows: list[tuple[str, Sequence[str]]] = []
    if helmet_person_ids:
        rows.append(("KRITIK KASK UYARISI", helmet_person_ids))
    if walkway_person_ids:
        rows.append((WALKWAY_ALERT_LABEL, walkway_person_ids))
    if not rows:
        return
    height, width = frame.shape[:2]
    row_height = max(42, min(68, int(round(height * 0.064))))
    panel_height = row_height * len(rows)
    panel_top = height - panel_height
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, panel_top),
        (width, height),
        THEME.bgr("violation"),
        -1,
    )
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0.0, frame)
    font_scale = max(0.58, min(0.94, width / 1950.0))
    for index, (label, person_ids) in enumerate(rows):
        row_top = panel_top + index * row_height
        if index:
            cv2.line(
                frame,
                (24, row_top),
                (width - 24, row_top),
                THEME.bgr("background"),
                1,
                cv2.LINE_AA,
            )
        shown_ids = ", ".join(f"#{identity}" for identity in person_ids[:4])
        if len(person_ids) > 4:
            shown_ids += f" +{len(person_ids) - 4}"
        text = f"{label}   INSAN {shown_ids}   {camera_label}"
        cv2.putText(
            frame,
            text,
            (24, row_top + int(row_height * 0.66)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            THEME.bgr("text"),
            2,
            cv2.LINE_AA,
        )


def _draw_ppe_frame_validated(
    frame: np.ndarray,
    *,
    detections: Sequence[Mapping[str, Any]],
    camera_label: str,
    rois: Sequence[Mapping[str, Any]] | None,
    safe_walkways: Sequence[Mapping[str, Any]],
    persons: Sequence[Mapping[str, Any]] | None = None,
    vehicles: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, Counter[str]]:
    height, width = frame.shape[:2]
    _draw_rois(frame, rois)
    _draw_safe_walkways(frame, safe_walkways)
    counts: Counter[str] = Counter()
    vehicle_count = _draw_vehicle_candidates(frame, vehicles)
    if vehicle_count:
        counts["forklift_candidate"] += vehicle_count
    person_mode = persons is not None
    helmet_alert_ids: list[str] = []
    walkway_alert_ids: list[str] = []
    if person_mode:
        normalized_persons = _validated_persons(persons)
        for person in normalized_persons:
            box = person["bbox_norm_xywh"]
            # ``rois`` is the v1 legacy visual filter. Safe walkways are an
            # alert rule and deliberately never remove a person from view.
            if not _inside_any_roi(box, rois):
                continue
            counts["person"] += 1
            vehicle_context = person.get("vehicle_context")
            ppe_eligible = not (
                isinstance(vehicle_context, Mapping)
                and vehicle_context.get("ppe_alert_eligible") is False
            )
            if (
                isinstance(vehicle_context, Mapping)
                and vehicle_context.get("suppression_active") is True
            ):
                counts["forklift_operator"] += 1
            for equipment, present_key, absent_key in (
                ("helmet", "helmet", "no_helmet"),
                ("hi_vis", "hi_vis", "no_hi_vis"),
            ):
                state = person[equipment]["state"]
                if state == "present":
                    counts[present_key] += 1
                elif state == "absent" and ppe_eligible:
                    counts[absent_key] += 1
                elif state == "absent":
                    counts[f"suppressed_{absent_key}"] += 1
                else:
                    counts[f"{equipment}_unknown"] += 1
            if ppe_eligible and person["helmet"]["state"] == "absent":
                helmet_alert_ids.append(_person_display_id(person))
            if _walkway_violation(person):
                counts["walkway_violation"] += 1
                walkway_alert_ids.append(_person_display_id(person))
            _draw_person_status(frame, person=person)
    else:
        for detection in detections:
            canonical, compliance, confidence, box = _validate_detection(detection)
            if not _inside_any_roi(box, rois):
                continue
            pixels = _box_pixels(box, width=width, height=height)
            if pixels is None:
                continue
            left, top, right, bottom = pixels
            counts[canonical] += 1
            color = THEME.bgr(
                "safe" if compliance == "compliant" else "violation"
            )
            outline = THEME.bgr("panel")
            cv2.rectangle(
                frame, (left, top), (right, bottom), outline, 5, cv2.LINE_AA
            )
            cv2.rectangle(
                frame, (left, top), (right, bottom), color, 2, cv2.LINE_AA
            )
            label = f"{LABELS[canonical]}  %{confidence * 100:.0f}"
            font_scale = max(0.48, min(0.78, width / 2400.0))
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1
            )
            label_top = max(0, top - label_height - baseline - 12)
            cv2.rectangle(
                frame,
                (left, label_top),
                (min(width - 1, left + label_width + 16), top),
                THEME.bgr("background"),
                -1,
            )
            cv2.putText(
                frame,
                label,
                (left + 8, max(label_height + 2, top - baseline - 5)),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

    panel_height = max(58, min(82, int(round(height * 0.064))))
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (width, panel_height),
        THEME.bgr("background"),
        -1,
    )
    cv2.addWeighted(overlay, 0.91, frame, 0.09, 0.0, frame)
    font_scale = max(0.68, min(1.05, width / 1900.0))
    cv2.putText(
        frame,
        BRAND_NAME,
        (24, int(panel_height * 0.62)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        THEME.bgr("text"),
        2,
        cv2.LINE_AA,
    )
    violation_count = (
        counts["no_helmet"]
        + counts["no_hi_vis"]
        + counts["walkway_violation"]
    )
    analysis_label = "PPE + ALAN ANALIZI" if safe_walkways else "PPE ANALIZI"
    status = f"{camera_label}   {analysis_label}   "
    if person_mode:
        status += f"{counts['person']:02d} INSAN"
    else:
        status += f"{sum(counts.values()):02d} BULGU"
    if violation_count:
        status += f"   {violation_count:02d} UYARI"
    status_scale = font_scale * 0.70
    (status_width, _), _ = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_DUPLEX, status_scale, 1
    )
    cv2.putText(
        frame,
        status,
        (max(24, width - status_width - 24), int(panel_height * 0.62)),
        cv2.FONT_HERSHEY_DUPLEX,
        status_scale,
        THEME.bgr("violation" if violation_count else "safe"),
        1,
        cv2.LINE_AA,
    )
    _draw_alert_panel(
        frame,
        camera_label=camera_label,
        helmet_person_ids=helmet_alert_ids,
        walkway_person_ids=walkway_alert_ids,
    )
    return frame, counts


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frames": int(stream["nb_read_frames"]),
    }


def _safety_event_contract(
    events: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    alert_policy: Mapping[str, Any],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    open_violations: set[tuple[int, str]] = set()
    for event in events:
        kind = str(event["kind"])
        event_type = str(event["type"])
        track_id = int(event["track_id"])
        key = (track_id, event_type)
        if kind == "started":
            open_violations.add(key)
        else:
            open_violations.discard(key)
        counts["total"] += 1
        counts[kind] += 1
        counts[f"{kind}:{event_type}"] += 1
    return {
        "schema_version": SAFETY_EVENT_SCHEMA_VERSION if enabled else None,
        "enabled": enabled,
        "mode": (
            "track_state_transitions" if enabled else "disabled"
        ),
        "scope": (
            "full_frame_alert_eligible_tracked_persons"
            if alert_policy.get("forklift_driver_suppression")
            else "full_frame_all_tracked_persons"
        ),
        "rule": alert_policy["zone_rule"],
        "person_anchor": alert_policy["person_anchor"],
        "enter_debounce_frames": alert_policy["enter_debounce_frames"],
        "exit_debounce_frames": alert_policy["exit_debounce_frames"],
        "track_ttl_frames": alert_policy["track_ttl_frames"],
        "per_frame_violation_events_emitted": False,
        "observations": dict(sorted(counts.items())),
        "open_violations_at_end": [
            {"track_id": track_id, "type": event_type}
            for track_id, event_type in sorted(open_violations)
        ],
    }


def render(
    *,
    source: Path,
    predictions: Path,
    output: Path,
    camera_label: str,
    rois: Sequence[Mapping[str, Any]] | None = None,
    safe_walkways: Sequence[Mapping[str, Any]] | None = None,
    alert_policy: Mapping[str, Any] | None = None,
    keep_intermediate: bool = False,
) -> dict[str, Any]:
    if not source.is_file():
        raise PpeVideoError(f"source is not a file: {source}")
    if not predictions.is_file():
        raise PpeVideoError(f"predictions are not a file: {predictions}")
    if output.exists():
        raise PpeVideoError(f"output already exists: {output}")

    normalized_rois = _validate_rois(rois)
    normalized_walkways = _validate_safe_walkways(safe_walkways)
    if normalized_rois is not None and normalized_walkways:
        raise PpeVideoError(
            "legacy rois and safe_walkways cannot be combined"
        )
    normalized_alert_policy = _validate_alert_policy(
        alert_policy,
        walkway_enabled=bool(normalized_walkways),
    )
    rows = list(_prediction_rows(predictions))
    if not rows:
        raise PpeVideoError("prediction stream is empty")
    expected_indices = list(range(len(rows)))
    if [int(row.get("frame_index", -1)) for row in rows] != expected_indices:
        raise PpeVideoError("prediction frame indices are not contiguous from zero")
    person_mode_flags = ["persons" in row for row in rows]
    if any(person_mode_flags) and not all(person_mode_flags):
        raise PpeVideoError(
            "persons must be present on every frame in person-centric mode"
        )
    person_mode = all(person_mode_flags)
    forklift_mode = any(
        isinstance(row.get("vehicles"), list) and bool(row["vehicles"])
        for row in rows
    )
    if normalized_walkways and not person_mode:
        raise PpeVideoError(
            "safe walkway analysis requires person-centric prediction rows"
        )
    event_contract = transition_event_contract(rows, person_mode=person_mode)
    zone_engine: PersonZoneRuleEngine | None = None
    if normalized_walkways:
        try:
            zone_engine = PersonZoneRuleEngine(
                [
                    ZoneArea.from_mapping(walkway)
                    for walkway in normalized_walkways
                ],
                ZoneRuleConfig(
                    walkway_enter_frames=int(
                        normalized_alert_policy["enter_debounce_frames"]
                    ),
                    walkway_exit_frames=int(
                        normalized_alert_policy["exit_debounce_frames"]
                    ),
                    track_ttl_frames=int(
                        normalized_alert_policy["track_ttl_frames"]
                    ),
                ),
            )
        except PersonZoneRuleError as exc:
            raise PpeVideoError(f"safe walkway contract is invalid: {exc}") from exc

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise PpeVideoError("could not open source video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise PpeVideoError("source video metadata is invalid")

    output.parent.mkdir(parents=True, exist_ok=True)
    total_counts: Counter[str] = Counter()
    safety_events: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=".ppe-render-", dir=output.parent
    ) as temporary:
        intermediate = Path(temporary) / "intermediate.mp4"
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise PpeVideoError("could not create intermediate video")
        rendered = 0
        try:
            for row in rows:
                ok, frame = capture.read()
                if not ok:
                    raise PpeVideoError(
                        f"source ended before prediction frame {rendered}"
                    )
                if (
                    int(row.get("image_width", -1)) != width
                    or int(row.get("image_height", -1)) != height
                ):
                    raise PpeVideoError(
                        f"prediction geometry differs at frame {rendered}"
                    )
                detections = row.get("detections", [] if person_mode else None)
                if not isinstance(detections, list):
                    raise PpeVideoError(
                        f"detections must be a list at frame {rendered}"
                    )
                persons = row.get("persons") if person_mode else None
                if person_mode and not isinstance(persons, list):
                    raise PpeVideoError(
                        f"persons must be a list at frame {rendered}"
                    )
                vehicles = row.get("vehicles")
                if vehicles is not None and not isinstance(vehicles, list):
                    raise PpeVideoError(
                        f"vehicles must be a list at frame {rendered}"
                    )
                if zone_engine is not None:
                    try:
                        zone_frame = zone_engine.process_frame(
                            frame_index=rendered,
                            persons=persons,
                        )
                    except PersonZoneRuleError as exc:
                        raise PpeVideoError(
                            f"safe walkway frame {rendered} is invalid: {exc}"
                        ) from exc
                    persons = zone_frame["persons"]
                    safety_events.extend(zone_frame["safety_events"])
                frame, frame_counts = _draw_ppe_frame_validated(
                    frame,
                    detections=detections,
                    camera_label=camera_label,
                    rois=normalized_rois,
                    safe_walkways=normalized_walkways,
                    persons=persons,
                    vehicles=vehicles,
                )
                total_counts.update(frame_counts)
                writer.write(frame)
                rendered += 1
        finally:
            capture.release()
            writer.release()

        transcode = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(intermediate),
                "-an",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "19",
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if transcode.returncode:
            raise PpeVideoError(
                f"NVENC transcode failed: {transcode.stderr[-500:]}"
            )
        if keep_intermediate:
            retained = output.with_suffix(".intermediate.mp4")
            retained.write_bytes(intermediate.read_bytes())

    probe = _probe(output)
    if (
        probe["codec"] != "h264"
        or probe["width"] != width
        or probe["height"] != height
        or probe["frames"] != len(rows)
    ):
        output.unlink(missing_ok=True)
        raise PpeVideoError(f"rendered video contract differs: {probe}")
    return {
        "schema_version": "colt-ai.ppe-video-result/v1",
        "status": "rendered",
        "brand_name": BRAND_NAME,
        "theme_id": THEME_ID,
        "model_name_visible": False,
        "source": str(source),
        "predictions": str(predictions),
        "output": str(output),
        "camera_label": camera_label,
        "render_mode": (
            "person_ppe_forklift_association_safe_walkway"
            if person_mode and forklift_mode and normalized_walkways
            else (
                "person_ppe_forklift_association"
                if person_mode and forklift_mode
                else (
                    "person_ppe_association_safe_walkway"
                    if person_mode and normalized_walkways
                    else (
                        "person_ppe_association"
                        if person_mode
                        else "legacy_equipment_boxes"
                    )
                )
            )
        ),
        "analysis_scope": (
            "full_frame_person_ppe_forklift_with_safe_walkway_rules"
            if normalized_walkways and forklift_mode
            else (
                "full_frame_person_ppe_with_safe_walkway_rules"
                if normalized_walkways
                else (
                    "legacy_roi_visual_filter"
                    if normalized_rois is not None
                    else (
                        "full_frame_person_ppe_forklift"
                        if forklift_mode
                        else "full_frame_person_ppe"
                    )
                )
            )
        ),
        "person_overlay": {
            "enabled": person_mode,
            "bbox_anchor": "person_fusion_track",
            "identity_fields": ["track_id", "person_id"],
            "status_fields": (
                [
                    "helmet",
                    "hi_vis",
                    "zone_safety",
                    *(["vehicle_context"] if forklift_mode else []),
                ]
                if normalized_walkways
                else [
                    "helmet",
                    "hi_vis",
                    *(["vehicle_context"] if forklift_mode else []),
                ]
            ),
            "decision_field": "state",
            "instantaneous_diagnostic_field": "evidence",
            "missing_state_fallback": "evidence",
            "unknown_is_violation": False,
            "helmet_absent_alert_panel": True,
            "walkway_violation_alert_panel": bool(normalized_walkways),
            "zone_filter_applied": False if normalized_walkways else None,
        },
        "forklift_driver_suppression": {
            "enabled": forklift_mode,
            "vehicle_rendered": forklift_mode,
            "vehicle_class": (
                "forklift_candidate" if forklift_mode else None
            ),
            "person_bbox_preserved": True if forklift_mode else None,
            "ppe_alerts_suppressed_only_after_temporal_association": (
                True if forklift_mode else None
            ),
            "walkway_alerts_suppressed_only_after_temporal_association": (
                True if forklift_mode else None
            ),
        },
        "event_contract": {
            **event_contract,
            "decision_field": "state",
            "instantaneous_diagnostic_field": "evidence",
            "missing_state_fallback": "evidence",
            "roi_scope": (
                "full_frame_person_ppe_safe_walkway"
                if normalized_walkways
                else (
                    "visual_filter_only_full_frame_transitions"
                    if normalized_rois is not None
                    else "full_frame"
                )
            ),
        },
        "roi_filter": {
            "enabled": normalized_rois is not None,
            "policy": (
                "object_bbox_center_inside"
                if normalized_rois is not None
                else "full_frame"
            ),
            "combination": "any_roi" if normalized_rois is not None else None,
            "rois": normalized_rois or [],
        },
        "safe_walkways": {
            "enabled": bool(normalized_walkways),
            "person_scope": "full_frame_no_zone_filter",
            "rule": normalized_alert_policy["zone_rule"],
            "person_anchor": normalized_alert_policy["person_anchor"],
            "combination": (
                "allowed_union_outside_all_violation"
                if normalized_walkways
                else None
            ),
            "areas": normalized_walkways,
        },
        "safety_event_contract": _safety_event_contract(
            safety_events,
            enabled=bool(normalized_walkways),
            alert_policy=normalized_alert_policy,
        ),
        "safety_events": safety_events,
        "video": probe,
        "class_observations": dict(sorted(total_counts.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-label", required=True)
    parser.add_argument(
        "--rois-json",
        type=Path,
        help="Optional JSON file containing a list of normalized ROI objects.",
    )
    parser.add_argument(
        "--safe-walkways-json",
        type=Path,
        help=(
            "Optional JSON file containing v2 safe_walkway polygons; these "
            "never filter people."
        ),
    )
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rois = None
    if args.rois_json is not None:
        try:
            rois = json.loads(args.rois_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PpeVideoError(f"could not load ROI JSON: {exc}") from exc
    safe_walkways = None
    if args.safe_walkways_json is not None:
        try:
            safe_walkways = json.loads(
                args.safe_walkways_json.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PpeVideoError(
                f"could not load safe walkway JSON: {exc}"
            ) from exc
    result = render(
        source=args.input,
        predictions=args.predictions_jsonl,
        output=args.output,
        camera_label=args.camera_label,
        rois=rois,
        safe_walkways=safe_walkways,
        keep_intermediate=args.keep_intermediate,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
