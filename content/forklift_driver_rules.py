"""Track-aware forklift-driver association and alert suppression.

The rule engine consumes frame-aligned person and vehicle tracks.  It does not
remove people from the output: every person record is copied and augmented
with a ``vehicle_context`` object.  Downstream PPE and walkway rules can use
the explicit eligibility flags without losing the person's bbox or identity.

Association is deliberately geometry-only and deterministic.  Candidate pairs
must pass a vehicle-confidence threshold, person intersection-over-area (IoA)
threshold, and optionally a person-center-inside-vehicle check.  Pairs are
greedily selected by descending IoA with stable track-ID tie breakers, so one
vehicle can suppress at most one person and one person can occupy at most one
vehicle.

Activation and clearing use independent frame hysteresis.  A missing person is
an unknown observation: it neither enters nor clears suppression.  Active
state is retained until the person returns, the visible person accumulates
enough unmatched frames, or the track reaches its TTL.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO, TypeAlias


Bbox: TypeAlias = tuple[float, float, float, float]

SCHEMA_VERSION = "colt-ai.forklift-driver-frame/v1"
EVENT_SCHEMA_VERSION = "colt-ai.forklift-driver-transition/v1"
ROLE = "forklift_operator"
EVENT_TYPE = "forklift_operator_suppression"
_BBOX_FIELDS = (
    "bbox_xyxy",
    "bbox_xyxy_px",
    "bbox_norm_xyxy",
    "bbox_norm_xywh",
)
_RAW_KITTI_NAME = re.compile(
    r"^00_000_(?P<frame_index>\d{6,})\.txt$"
)


class ForkliftDriverRuleError(ValueError):
    """The configuration or a frame input violates the rule contract."""


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ForkliftDriverRuleError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ForkliftDriverRuleError(
            f"{label} must be a finite number"
        ) from exc
    if not math.isfinite(number):
        raise ForkliftDriverRuleError(f"{label} must be a finite number")
    return number


def _probability(value: object, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise ForkliftDriverRuleError(f"{label} must be inside [0,1]")
    return number


def _positive_integer(value: object, *, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ForkliftDriverRuleError(f"{label} must be a positive integer")
    return value


def _track_id(record: Mapping[str, Any], *, label: str) -> int:
    value = record.get("track_id")
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ForkliftDriverRuleError(
            f"{label}.track_id must be a non-negative integer"
        )
    return value


def _bbox(record: Mapping[str, Any], *, label: str) -> Bbox:
    present = [field for field in _BBOX_FIELDS if field in record]
    if len(present) != 1:
        choices = ", ".join(_BBOX_FIELDS)
        raise ForkliftDriverRuleError(
            f"{label} must contain exactly one bbox field: {choices}"
        )
    raw = record[present[0]]
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 4
    ):
        raise ForkliftDriverRuleError(
            f"{label}.{present[0]} must be [x1,y1,x2,y2]"
        )
    first, second, third, fourth = (
        _finite_number(value, label=f"{label}.{present[0]}[{index}]")
        for index, value in enumerate(raw)
    )
    if present[0] == "bbox_norm_xywh":
        x1, y1 = first, second
        x2, y2 = first + third, second + fourth
    else:
        x1, y1, x2, y2 = first, second, third, fourth
    if x2 <= x1 or y2 <= y1:
        raise ForkliftDriverRuleError(
            f"{label}.{present[0]} must have positive width and height"
        )
    return x1, y1, x2, y2


def person_intersection_over_area(person_bbox: Bbox, vehicle_bbox: Bbox) -> float:
    """Return intersection area divided by the person's bbox area."""

    px1, py1, px2, py2 = person_bbox
    vx1, vy1, vx2, vy2 = vehicle_bbox
    intersection_width = max(0.0, min(px2, vx2) - max(px1, vx1))
    intersection_height = max(0.0, min(py2, vy2) - max(py1, vy1))
    person_area = (px2 - px1) * (py2 - py1)
    return intersection_width * intersection_height / person_area


def person_center_inside_vehicle(
    person_bbox: Bbox,
    vehicle_bbox: Bbox,
) -> bool:
    """Return whether the person's bbox center is inside the vehicle bbox."""

    px1, py1, px2, py2 = person_bbox
    vx1, vy1, vx2, vy2 = vehicle_bbox
    center_x = (px1 + px2) / 2.0
    center_y = (py1 + py2) / 2.0
    return vx1 <= center_x <= vx2 and vy1 <= center_y <= vy2


def _bbox_iou(left: Bbox, right: Bbox) -> float:
    intersection_width = max(
        0.0,
        min(left[2], right[2]) - max(left[0], right[0]),
    )
    intersection_height = max(
        0.0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class ForkliftDriverConfig:
    """Geometry, hysteresis, and downstream suppression policy."""

    vehicle_confidence: float = 0.35
    person_ioa: float = 0.55
    require_center_inside: bool = True
    enter_frames: int = 4
    exit_frames: int = 8
    ttl_frames: int = 30
    vehicle_confidence_ttl_frames: int = 60
    suppress_ppe: bool = True
    suppress_walkway: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vehicle_confidence",
            _probability(
                self.vehicle_confidence,
                label="vehicle_confidence",
            ),
        )
        object.__setattr__(
            self,
            "person_ioa",
            _probability(self.person_ioa, label="person_ioa"),
        )
        for field_name in (
            "require_center_inside",
            "suppress_ppe",
            "suppress_walkway",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ForkliftDriverRuleError(
                    f"{field_name} must be a boolean"
                )
        for field_name in (
            "enter_frames",
            "exit_frames",
            "ttl_frames",
            "vehicle_confidence_ttl_frames",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(
                    getattr(self, field_name),
                    label=field_name,
                ),
            )

    @property
    def suppression_scopes(self) -> tuple[str, ...]:
        scopes: list[str] = []
        if self.suppress_ppe:
            scopes.append("ppe")
        if self.suppress_walkway:
            scopes.append("walkway")
        return tuple(scopes)


@dataclass(frozen=True)
class _PersonInput:
    index: int
    track_id: int
    bbox: Bbox
    record: Mapping[str, Any]


@dataclass(frozen=True)
class _VehicleInput:
    index: int
    track_id: int
    bbox: Bbox
    confidence: float
    record: Mapping[str, Any]


@dataclass(frozen=True)
class _Match:
    person_index: int
    person_track_id: int
    vehicle_index: int
    vehicle_track_id: int
    vehicle_confidence: float
    ioa: float


@dataclass
class _TrackState:
    track_id: int
    active: bool = False
    enter_streak: int = 0
    exit_streak: int = 0
    last_seen_frame: int = -1
    vehicle_track_id: int | None = None
    last_match_ioa: float = 0.0


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> tuple[list[_PersonInput], list[_VehicleInput]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ForkliftDriverRuleError(f"{kind}s must be a sequence")
    seen_track_ids: set[int] = set()
    people: list[_PersonInput] = []
    vehicles: list[_VehicleInput] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ForkliftDriverRuleError(
                f"{kind}s[{index}] must be an object"
            )
        label = f"{kind}s[{index}]"
        track_id = _track_id(record, label=label)
        if track_id in seen_track_ids:
            raise ForkliftDriverRuleError(
                f"duplicate {kind} track_id inside frame: {track_id}"
            )
        seen_track_ids.add(track_id)
        box = _bbox(record, label=label)
        if kind == "person":
            people.append(_PersonInput(index, track_id, box, record))
        else:
            raw_confidence = record.get(
                "confidence",
                record.get("tracker_confidence"),
            )
            confidence = _probability(
                raw_confidence,
                label=f"{label}.confidence",
            )
            vehicles.append(
                _VehicleInput(
                    index,
                    track_id,
                    box,
                    confidence,
                    record,
                )
            )
    return people, vehicles


def associate_people_to_vehicles(
    persons: Sequence[Mapping[str, Any]],
    vehicles: Sequence[Mapping[str, Any]],
    config: ForkliftDriverConfig | None = None,
) -> dict[int, dict[str, Any]]:
    """Return deterministic one-to-one raw matches keyed by person track ID."""

    policy = config or ForkliftDriverConfig()
    people, _ = _validate_records(persons, kind="person")
    _, vehicle_rows = _validate_records(vehicles, kind="vehicle")
    candidates: list[_Match] = []
    for person in people:
        for vehicle in vehicle_rows:
            if vehicle.confidence < policy.vehicle_confidence:
                continue
            ioa = person_intersection_over_area(person.bbox, vehicle.bbox)
            if ioa < policy.person_ioa:
                continue
            if (
                policy.require_center_inside
                and not person_center_inside_vehicle(
                    person.bbox,
                    vehicle.bbox,
                )
            ):
                continue
            candidates.append(
                _Match(
                    person.index,
                    person.track_id,
                    vehicle.index,
                    vehicle.track_id,
                    vehicle.confidence,
                    ioa,
                )
            )

    candidates.sort(
        key=lambda match: (
            -match.ioa,
            -match.vehicle_confidence,
            match.person_track_id,
            match.vehicle_track_id,
            match.person_index,
            match.vehicle_index,
        )
    )
    occupied_people: set[int] = set()
    occupied_vehicles: set[int] = set()
    selected: dict[int, dict[str, Any]] = {}
    for match in candidates:
        if (
            match.person_track_id in occupied_people
            or match.vehicle_track_id in occupied_vehicles
        ):
            continue
        occupied_people.add(match.person_track_id)
        occupied_vehicles.add(match.vehicle_track_id)
        selected[match.person_track_id] = {
            "person_track_id": match.person_track_id,
            "vehicle_track_id": match.vehicle_track_id,
            "vehicle_confidence": match.vehicle_confidence,
            "person_ioa": match.ioa,
        }
    return selected


class ForkliftDriverRuleEngine:
    """Maintain track-level forklift-operator suppression state."""

    def __init__(self, config: ForkliftDriverConfig | None = None) -> None:
        self.config = config or ForkliftDriverConfig()
        self._tracks: dict[int, _TrackState] = {}
        self._qualified_vehicle_frames: dict[int, int] = {}
        self._last_frame_index: int | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._qualified_vehicle_frames.clear()
        self._last_frame_index = None

    def _qualified_vehicles(
        self,
        frame_index: int,
        vehicles: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep a tracked vehicle qualified across NvDCF score dips.

        NvDCF's exported confidence is a correlation response rather than a
        fresh detector probability.  Once a vehicle track crosses the policy
        threshold, retaining that qualification for a bounded TTL avoids
        turning an otherwise continuous forklift track on and off.
        """

        _, validated = _validate_records(vehicles, kind="vehicle")
        effective: list[dict[str, Any]] = []
        for vehicle in validated:
            if vehicle.confidence >= self.config.vehicle_confidence:
                self._qualified_vehicle_frames[vehicle.track_id] = frame_index
            last_qualified = self._qualified_vehicle_frames.get(
                vehicle.track_id
            )
            if (
                last_qualified is None
                or frame_index - last_qualified
                > self.config.vehicle_confidence_ttl_frames
            ):
                continue
            record = dict(vehicle.record)
            record["confidence"] = max(
                vehicle.confidence,
                self.config.vehicle_confidence,
            )
            effective.append(record)
        expired = [
            track_id
            for track_id, qualified_frame in self._qualified_vehicle_frames.items()
            if frame_index - qualified_frame
            > self.config.vehicle_confidence_ttl_frames
        ]
        for track_id in expired:
            del self._qualified_vehicle_frames[track_id]
        return effective

    def _transition(
        self,
        *,
        frame_index: int,
        kind: str,
        track_id: int,
        vehicle_track_id: int | None,
        person_ioa: float,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "kind": kind,
            "type": EVENT_TYPE,
            "frame_index": frame_index,
            "track_id": track_id,
            "person_track_id": track_id,
            "vehicle_track_id": vehicle_track_id,
            "person_ioa": round(person_ioa, 6),
            "suppression_scopes": list(self.config.suppression_scopes),
            "reason": reason,
        }

    def _expire_tracks(self, frame_index: int) -> list[dict[str, Any]]:
        transitions: list[dict[str, Any]] = []
        expired = sorted(
            track_id
            for track_id, state in self._tracks.items()
            if frame_index - state.last_seen_frame > self.config.ttl_frames
        )
        for track_id in expired:
            state = self._tracks.pop(track_id)
            if state.active:
                transitions.append(
                    self._transition(
                        frame_index=frame_index,
                        kind="ended",
                        track_id=track_id,
                        vehicle_track_id=state.vehicle_track_id,
                        person_ioa=state.last_match_ioa,
                        reason="track_expired",
                    )
                )
        return transitions

    def process_frame(
        self,
        *,
        frame_index: int,
        persons: Sequence[Mapping[str, Any]],
        vehicles: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Copy and augment one frame of person records."""

        if (
            not isinstance(frame_index, int)
            or isinstance(frame_index, bool)
            or frame_index < 0
        ):
            raise ForkliftDriverRuleError(
                "frame_index must be a non-negative integer"
            )
        if (
            self._last_frame_index is not None
            and frame_index <= self._last_frame_index
        ):
            raise ForkliftDriverRuleError(
                "frame_index must increase strictly between calls"
            )
        people, _ = _validate_records(persons, kind="person")
        qualified_vehicles = self._qualified_vehicles(
            frame_index,
            vehicles,
        )
        raw_matches = associate_people_to_vehicles(
            persons,
            qualified_vehicles,
            self.config,
        )
        transitions = self._expire_tracks(frame_index)
        contexts: dict[int, dict[str, Any]] = {}

        for person in sorted(people, key=lambda value: value.track_id):
            state = self._tracks.get(person.track_id)
            if state is None:
                state = _TrackState(track_id=person.track_id)
                self._tracks[person.track_id] = state
            state.last_seen_frame = frame_index
            match = raw_matches.get(person.track_id)
            if match is not None:
                state.exit_streak = 0
                state.enter_streak += 1
                state.vehicle_track_id = int(match["vehicle_track_id"])
                state.last_match_ioa = float(match["person_ioa"])
                if (
                    not state.active
                    and state.enter_streak >= self.config.enter_frames
                ):
                    # Hysteresis may keep the old person track alive during an
                    # NvDCF identity handoff.  Enforce the configured
                    # one-occupant invariant across active state as well as
                    # raw frame association.
                    for other_track_id, other_state in sorted(
                        self._tracks.items()
                    ):
                        if (
                            other_track_id == person.track_id
                            or not other_state.active
                            or other_state.vehicle_track_id
                            != state.vehicle_track_id
                        ):
                            continue
                        transitions.append(
                            self._transition(
                                frame_index=frame_index,
                                kind="ended",
                                track_id=other_track_id,
                                vehicle_track_id=(
                                    other_state.vehicle_track_id
                                ),
                                person_ioa=other_state.last_match_ioa,
                                reason="operator_track_reassigned",
                            )
                        )
                        other_state.active = False
                        other_state.enter_streak = 0
                        other_state.exit_streak = 0
                        other_state.vehicle_track_id = None
                        other_state.last_match_ioa = 0.0
                    state.active = True
                    transitions.append(
                        self._transition(
                            frame_index=frame_index,
                            kind="started",
                            track_id=person.track_id,
                            vehicle_track_id=state.vehicle_track_id,
                            person_ioa=state.last_match_ioa,
                            reason="enter_hysteresis_met",
                        )
                    )
            elif state.active:
                state.enter_streak = 0
                state.exit_streak += 1
                if state.exit_streak >= self.config.exit_frames:
                    ended_vehicle_track_id = state.vehicle_track_id
                    ended_ioa = state.last_match_ioa
                    state.active = False
                    state.exit_streak = 0
                    state.vehicle_track_id = None
                    state.last_match_ioa = 0.0
                    transitions.append(
                        self._transition(
                            frame_index=frame_index,
                            kind="ended",
                            track_id=person.track_id,
                            vehicle_track_id=ended_vehicle_track_id,
                            person_ioa=ended_ioa,
                            reason="exit_hysteresis_met",
                        )
                    )
            else:
                state.enter_streak = 0
                state.exit_streak = 0
                state.vehicle_track_id = None
                state.last_match_ioa = 0.0

            raw_match = match is not None
            current_vehicle_track_id = (
                int(match["vehicle_track_id"])
                if raw_match
                else state.vehicle_track_id if state.active else None
            )
            current_ioa = float(match["person_ioa"]) if raw_match else 0.0
            suppression_active = state.active
            contexts[person.track_id] = {
                "raw_match": raw_match,
                "suppression_active": suppression_active,
                "role": ROLE if suppression_active else None,
                "vehicle_track_id": current_vehicle_track_id,
                "person_ioa": round(current_ioa, 6),
                "ppe_alert_eligible": not (
                    suppression_active and self.config.suppress_ppe
                ),
                "walkway_alert_eligible": not (
                    suppression_active and self.config.suppress_walkway
                ),
                "suppression_scopes": (
                    list(self.config.suppression_scopes)
                    if suppression_active
                    else []
                ),
                "enter_streak": state.enter_streak,
                "exit_streak": state.exit_streak,
            }

        augmented_people: list[dict[str, Any]] = []
        for person in people:
            augmented = dict(person.record)
            augmented["vehicle_context"] = contexts[person.track_id]
            augmented_people.append(augmented)

        self._last_frame_index = frame_index
        transitions.sort(
            key=lambda event: (
                int(event["frame_index"]),
                int(event["track_id"]),
                0 if event["kind"] == "ended" else 1,
            )
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "frame_index": frame_index,
            "persons": augmented_people,
            "raw_associations": [
                {
                    **raw_matches[track_id],
                    "person_ioa": round(
                        float(raw_matches[track_id]["person_ioa"]),
                        6,
                    ),
                }
                for track_id in sorted(raw_matches)
            ],
            "transition_events": transitions,
            "active_operator_track_ids": sorted(
                track_id
                for track_id, state in self._tracks.items()
                if state.active
            ),
        }


def augment_frame_record(
    person_record: Mapping[str, Any],
    vehicle_record: Mapping[str, Any],
    engine: ForkliftDriverRuleEngine,
    *,
    persons_key: str = "persons",
    vehicles_key: str = "vehicles",
) -> dict[str, Any]:
    """Augment one pair of aligned JSON-style frame records."""

    if not isinstance(person_record, Mapping) or not isinstance(
        vehicle_record,
        Mapping,
    ):
        raise ForkliftDriverRuleError("frame records must be objects")
    person_frame = person_record.get("frame_index")
    vehicle_frame = vehicle_record.get("frame_index")
    if person_frame != vehicle_frame:
        raise ForkliftDriverRuleError(
            "person and vehicle frame_index values must match"
        )
    persons = person_record.get(persons_key)
    vehicles = vehicle_record.get(vehicles_key)
    if not isinstance(persons, list):
        raise ForkliftDriverRuleError(
            f"person record needs a {persons_key} list"
        )
    if not isinstance(vehicles, list):
        raise ForkliftDriverRuleError(
            f"vehicle record needs a {vehicles_key} list"
        )
    frame = engine.process_frame(
        frame_index=person_frame,
        persons=persons,
        vehicles=vehicles,
    )
    augmented = dict(person_record)
    augmented[persons_key] = frame["persons"]
    augmented["vehicle_context"] = {
        "schema_version": SCHEMA_VERSION,
        "raw_associations": frame["raw_associations"],
        "transition_events": frame["transition_events"],
        "active_operator_track_ids": frame["active_operator_track_ids"],
    }
    return augmented


def _jsonl_rows(stream: TextIO, *, label: str) -> Iterable[Mapping[str, Any]]:
    for line_number, raw in enumerate(stream, 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ForkliftDriverRuleError(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise ForkliftDriverRuleError(
                f"{label}:{line_number}: frame must be an object"
            )
        yield value


def merge_raw_vehicle_detector_fallback(
    tracked_detections_jsonl: Path,
    raw_kitti_directory: Path,
    output_jsonl: Path,
    *,
    coordinate_width: int,
    coordinate_height: int,
    minimum_detector_confidence: float = 0.05,
    minimum_iou: float = 0.15,
    track_ttl_frames: int = 60,
) -> dict[str, Any]:
    """Fill NvDCF vehicle gaps with the PGIE's raw class-7 detections.

    Person rows and native NvDCF vehicle rows remain untouched.  A raw truck
    box is first linked to a native vehicle track in the same frame, then to a
    bounded local track from earlier raw observations.  Only when neither
    association is possible is a synthetic vehicle-only ID allocated.  This
    preserves current geometry during NvDCF correlation-output gaps without
    changing the canonical person identity stream.
    """

    if output_jsonl.exists():
        raise ForkliftDriverRuleError(f"output already exists: {output_jsonl}")
    if (
        isinstance(coordinate_width, bool)
        or not isinstance(coordinate_width, int)
        or coordinate_width <= 0
        or isinstance(coordinate_height, bool)
        or not isinstance(coordinate_height, int)
        or coordinate_height <= 0
    ):
        raise ForkliftDriverRuleError(
            "raw KITTI coordinate dimensions must be positive integers"
        )
    detector_threshold = _probability(
        minimum_detector_confidence,
        label="minimum_detector_confidence",
    )
    iou_threshold = _probability(minimum_iou, label="minimum_iou")
    ttl = _positive_integer(track_ttl_frames, label="track_ttl_frames")
    if not raw_kitti_directory.is_dir():
        raise ForkliftDriverRuleError(
            f"raw KITTI directory is absent: {raw_kitti_directory}"
        )

    raw_files: dict[int, Path] = {}
    for path in raw_kitti_directory.iterdir():
        match = _RAW_KITTI_NAME.fullmatch(path.name)
        if match is None:
            continue
        frame_index = int(match["frame_index"])
        if frame_index in raw_files:
            raise ForkliftDriverRuleError(
                f"duplicate raw KITTI frame: {frame_index}"
            )
        raw_files[frame_index] = path

    def raw_vehicles(frame_index: int) -> list[dict[str, Any]]:
        path = raw_files.get(frame_index)
        if path is None:
            return []
        candidates: list[dict[str, Any]] = []
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not raw.strip():
                continue
            fields = raw.split()
            if fields[0].casefold() != "truck":
                continue
            if len(fields) < 16:
                raise ForkliftDriverRuleError(
                    f"{path}:{line_number}: incomplete raw KITTI row"
                )
            try:
                values = [float(value) for value in fields[1:16]]
            except ValueError as exc:
                raise ForkliftDriverRuleError(
                    f"{path}:{line_number}: invalid raw KITTI field"
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ForkliftDriverRuleError(
                    f"{path}:{line_number}: non-finite raw KITTI field"
                )
            left, top, right, bottom = values[3:7]
            confidence = values[14]
            if confidence < detector_threshold:
                continue
            left = max(0.0, min(float(coordinate_width), left))
            top = max(0.0, min(float(coordinate_height), top))
            right = max(0.0, min(float(coordinate_width), right))
            bottom = max(0.0, min(float(coordinate_height), bottom))
            if right <= left or bottom <= top:
                continue
            candidates.append(
                {
                    "bbox": (
                        left / coordinate_width,
                        top / coordinate_height,
                        right / coordinate_width,
                        bottom / coordinate_height,
                    ),
                    "confidence": min(1.0, confidence),
                }
            )
        candidates.sort(
            key=lambda candidate: (
                -float(candidate["confidence"]),
                candidate["bbox"],
            )
        )
        return candidates

    # track_id -> (last normalized xyxy bbox, last frame)
    vehicle_tracks: dict[int, tuple[Bbox, int]] = {}
    next_synthetic_id = 1_000_000
    frame_count = 0
    raw_observations = 0
    fallback_observations = 0
    native_enriched_observations = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_jsonl.name}.",
        suffix=".tmp",
        dir=output_jsonl.parent,
    )
    try:
        with (
            tracked_detections_jsonl.open(encoding="utf-8") as tracked_stream,
            os.fdopen(descriptor, "w", encoding="utf-8") as output_stream,
        ):
            for expected_frame, record in enumerate(
                _jsonl_rows(
                    tracked_stream,
                    label=str(tracked_detections_jsonl),
                )
            ):
                frame_index = record.get("frame_index")
                if frame_index != expected_frame:
                    raise ForkliftDriverRuleError(
                        "tracked detection frame indices must be contiguous "
                        "from zero"
                    )
                raw_detections = record.get("detections")
                if not isinstance(raw_detections, list):
                    raise ForkliftDriverRuleError(
                        "tracked frame needs a detections list"
                    )
                detections = [
                    dict(detection)
                    for detection in raw_detections
                    if isinstance(detection, Mapping)
                ]
                if len(detections) != len(raw_detections):
                    raise ForkliftDriverRuleError(
                        "tracked detections must contain only objects"
                    )
                native: list[tuple[int, int, Bbox]] = []
                for detection_index, detection in enumerate(detections):
                    if (
                        str(detection.get("class_name", ""))
                        .strip()
                        .casefold()
                        != "forklift_candidate"
                    ):
                        continue
                    track_id = _track_id(
                        detection,
                        label=f"detections[{detection_index}]",
                    )
                    box = _bbox(
                        detection,
                        label=f"detections[{detection_index}]",
                    )
                    native.append((detection_index, track_id, box))
                    vehicle_tracks[track_id] = (box, frame_index)

                candidates = raw_vehicles(frame_index)
                raw_observations += len(candidates)
                assignments: dict[int, tuple[int, int | None]] = {}
                used_raw: set[int] = set()
                used_tracks: set[int] = set()
                native_pairs = sorted(
                    (
                        _bbox_iou(candidate["bbox"], box),
                        raw_index,
                        detection_index,
                        track_id,
                    )
                    for raw_index, candidate in enumerate(candidates)
                    for detection_index, track_id, box in native
                    if _bbox_iou(candidate["bbox"], box) >= iou_threshold
                )
                for _, raw_index, detection_index, track_id in reversed(
                    native_pairs
                ):
                    if raw_index in used_raw or track_id in used_tracks:
                        continue
                    used_raw.add(raw_index)
                    used_tracks.add(track_id)
                    assignments[raw_index] = (track_id, detection_index)

                active_tracks = {
                    track_id: box
                    for track_id, (box, last_frame) in vehicle_tracks.items()
                    if frame_index - last_frame <= ttl
                    and track_id not in used_tracks
                }
                history_pairs = sorted(
                    (
                        _bbox_iou(candidate["bbox"], box),
                        raw_index,
                        track_id,
                    )
                    for raw_index, candidate in enumerate(candidates)
                    if raw_index not in used_raw
                    for track_id, box in active_tracks.items()
                    if _bbox_iou(candidate["bbox"], box) >= iou_threshold
                )
                for _, raw_index, track_id in reversed(history_pairs):
                    if raw_index in used_raw or track_id in used_tracks:
                        continue
                    used_raw.add(raw_index)
                    used_tracks.add(track_id)
                    assignments[raw_index] = (track_id, None)

                for raw_index in range(len(candidates)):
                    if raw_index in assignments:
                        continue
                    while next_synthetic_id in vehicle_tracks:
                        next_synthetic_id += 1
                    assignments[raw_index] = (next_synthetic_id, None)
                    next_synthetic_id += 1

                for raw_index, candidate in enumerate(candidates):
                    track_id, native_index = assignments[raw_index]
                    box = candidate["bbox"]
                    confidence = float(candidate["confidence"])
                    vehicle_tracks[track_id] = (box, frame_index)
                    if native_index is not None:
                        detections[native_index]["confidence"] = max(
                            float(
                                detections[native_index].get(
                                    "confidence",
                                    0.0,
                                )
                            ),
                            confidence,
                        )
                        native_enriched_observations += 1
                        continue
                    detections.append(
                        {
                            "class_id": 7,
                            "class_name": "forklift_candidate",
                            "detector_class_name": "truck",
                            "confidence": confidence,
                            "track_id": track_id,
                            "bbox_norm_xywh": [
                                round(box[0], 10),
                                round(box[1], 10),
                                round(box[2] - box[0], 10),
                                round(box[3] - box[1], 10),
                            ],
                        }
                    )
                    fallback_observations += 1

                vehicle_tracks = {
                    track_id: state
                    for track_id, state in vehicle_tracks.items()
                    if frame_index - state[1] <= ttl
                }
                augmented = dict(record)
                augmented["detections"] = detections
                output_stream.write(
                    json.dumps(
                        augmented,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                output_stream.write("\n")
                frame_count += 1
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary_name, output_jsonl)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
    return {
        "schema_version": "colt-ai.vehicle-detector-fallback/v1",
        "frames": frame_count,
        "raw_detector_observations": raw_observations,
        "native_enriched_observations": native_enriched_observations,
        "fallback_observations": fallback_observations,
        "output": str(output_jsonl),
    }


def augment_jsonl_streams(
    person_jsonl: Path,
    vehicle_jsonl: Path,
    output_jsonl: Path,
    *,
    config: ForkliftDriverConfig | None = None,
    persons_key: str = "persons",
    vehicles_key: str = "vehicles",
) -> dict[str, Any]:
    """Atomically augment aligned person and vehicle JSONL streams."""

    if output_jsonl.exists():
        raise ForkliftDriverRuleError(f"output already exists: {output_jsonl}")
    engine = ForkliftDriverRuleEngine(config)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_jsonl.name}.",
        suffix=".tmp",
        dir=output_jsonl.parent,
    )
    frame_count = 0
    transition_count = 0
    try:
        with (
            person_jsonl.open(encoding="utf-8") as person_stream,
            vehicle_jsonl.open(encoding="utf-8") as vehicle_stream,
            os.fdopen(descriptor, "w", encoding="utf-8") as output_stream,
        ):
            person_rows = _jsonl_rows(
                person_stream,
                label=str(person_jsonl),
            )
            vehicle_rows = _jsonl_rows(
                vehicle_stream,
                label=str(vehicle_jsonl),
            )
            for pair_index, pair in enumerate(
                itertools.zip_longest(person_rows, vehicle_rows),
                1,
            ):
                person_record, vehicle_record = pair
                if person_record is None or vehicle_record is None:
                    raise ForkliftDriverRuleError(
                        "person and vehicle JSONL streams have different "
                        f"frame counts at pair {pair_index}"
                    )
                augmented = augment_frame_record(
                    person_record,
                    vehicle_record,
                    engine,
                    persons_key=persons_key,
                    vehicles_key=vehicles_key,
                )
                output_stream.write(
                    json.dumps(
                        augmented,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                output_stream.write("\n")
                frame_count += 1
                transition_count += len(
                    augmented["vehicle_context"]["transition_events"]
                )
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary_name, output_jsonl)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "frames": frame_count,
        "transitions": transition_count,
        "output": str(output_jsonl),
    }


def augment_person_ppe_with_tracked_vehicles(
    person_ppe_jsonl: Path,
    tracked_detections_jsonl: Path,
    output_jsonl: Path,
    *,
    config: ForkliftDriverConfig | None = None,
) -> dict[str, Any]:
    """Fuse forklift context into the production person-centric PPE stream.

    ``tracked_detections_jsonl`` is the combined DeepStream/NvDCF stream
    produced by :mod:`validation.run_person_deepstream_direct`.  Person rows
    are ignored here; only ``forklift_candidate`` detections become vehicle
    evidence.  PPE transition events are re-derived from the effective,
    post-suppression person alarms so every started event still has one ended
    event when driver suppression activates.
    """

    if output_jsonl.exists():
        raise ForkliftDriverRuleError(f"output already exists: {output_jsonl}")
    policy = config or ForkliftDriverConfig()
    engine = ForkliftDriverRuleEngine(policy)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_jsonl.name}.",
        suffix=".tmp",
        dir=output_jsonl.parent,
    )
    frame_count = 0
    transition_count = 0
    suppressed_person_frames = 0
    candidate_observations = 0
    effective_open: set[tuple[int, str]] = set()
    try:
        with (
            person_ppe_jsonl.open(encoding="utf-8") as person_stream,
            tracked_detections_jsonl.open(encoding="utf-8") as tracked_stream,
            os.fdopen(descriptor, "w", encoding="utf-8") as output_stream,
        ):
            person_rows = _jsonl_rows(
                person_stream,
                label=str(person_ppe_jsonl),
            )
            tracked_rows = _jsonl_rows(
                tracked_stream,
                label=str(tracked_detections_jsonl),
            )
            for pair_index, pair in enumerate(
                itertools.zip_longest(person_rows, tracked_rows),
                1,
            ):
                person_record, tracked_record = pair
                if person_record is None or tracked_record is None:
                    raise ForkliftDriverRuleError(
                        "person-PPE and tracked detection streams have "
                        f"different frame counts at pair {pair_index}"
                    )
                frame_index = person_record.get("frame_index")
                if frame_index != tracked_record.get("frame_index"):
                    raise ForkliftDriverRuleError(
                        "person-PPE and tracked frame_index values differ"
                    )
                for dimension in ("image_width", "image_height"):
                    if (
                        dimension in person_record
                        and dimension in tracked_record
                        and person_record[dimension] != tracked_record[dimension]
                    ):
                        raise ForkliftDriverRuleError(
                            f"person-PPE and tracked {dimension} values differ"
                        )
                detections = tracked_record.get("detections")
                if not isinstance(detections, list):
                    raise ForkliftDriverRuleError(
                        "tracked frame needs a detections list"
                    )
                vehicles = [
                    dict(detection)
                    for detection in detections
                    if isinstance(detection, Mapping)
                    and str(detection.get("class_name", ""))
                    .strip()
                    .casefold()
                    == "forklift_candidate"
                ]
                candidate_observations += len(vehicles)
                augmented = augment_frame_record(
                    person_record,
                    {
                        "frame_index": frame_index,
                        "vehicles": vehicles,
                    },
                    engine,
                )
                augmented["vehicles"] = vehicles

                # A temporarily missing NvDCF person is an unknown
                # observation, not evidence that an effective PPE alarm
                # ended.  Begin from the previous state, replace only tracks
                # visible in this frame, and honor explicit raw ``ended``
                # events (including the upstream track-TTL expiry event).
                desired_open: set[tuple[int, str]] = set(effective_open)
                effective_people: list[dict[str, Any]] = []
                for raw_person in augmented["persons"]:
                    person = dict(raw_person)
                    context = person["vehicle_context"]
                    ppe_eligible = bool(context["ppe_alert_eligible"])
                    walkway_eligible = bool(
                        context["walkway_alert_eligible"]
                    )
                    person["zone_alert_eligible"] = walkway_eligible
                    raw_alarms = person.get("alarms", [])
                    if not isinstance(raw_alarms, list):
                        raise ForkliftDriverRuleError(
                            "person-PPE alarms must be a list"
                        )
                    alarms = [str(alarm) for alarm in raw_alarms]
                    person["raw_alarms"] = alarms
                    if not ppe_eligible and alarms:
                        suppressed_person_frames += 1
                        person["suppressed_alarms"] = alarms
                        person["alarms"] = []
                        person["alarm"] = False
                        person["overall_state"] = (
                            "suppressed_forklift_operator"
                        )
                        alarms = []
                    track_id = _track_id(person, label="person")
                    desired_open = {
                        alarm_key
                        for alarm_key in desired_open
                        if alarm_key[0] != track_id
                    }
                    desired_open.update((track_id, alarm) for alarm in alarms)
                    effective_people.append(person)
                augmented["persons"] = effective_people

                raw_events = person_record.get("alarm_events", [])
                if not isinstance(raw_events, list):
                    raise ForkliftDriverRuleError(
                        "person-PPE alarm_events must be a list"
                    )
                for raw_event in raw_events:
                    if (
                        not isinstance(raw_event, Mapping)
                        or raw_event.get("kind") != "ended"
                    ):
                        continue
                    raw_track_id = raw_event.get("track_id")
                    raw_event_type = raw_event.get("type")
                    if (
                        isinstance(raw_track_id, int)
                        and not isinstance(raw_track_id, bool)
                        and isinstance(raw_event_type, str)
                    ):
                        desired_open.discard(
                            (raw_track_id, raw_event_type)
                        )

                effective_events: list[dict[str, Any]] = []
                for track_id, event_type in sorted(
                    effective_open - desired_open
                ):
                    effective_events.append(
                        {
                            "kind": "ended",
                            "type": event_type,
                            "track_id": track_id,
                            "frame_index": frame_index,
                            "reason": "effective_alarm_state_changed",
                        }
                    )
                for track_id, event_type in sorted(
                    desired_open - effective_open
                ):
                    effective_events.append(
                        {
                            "kind": "started",
                            "type": event_type,
                            "track_id": track_id,
                            "frame_index": frame_index,
                            "reason": "effective_alarm_state_changed",
                        }
                    )
                augmented["raw_alarm_events"] = list(raw_events)
                augmented["alarm_events"] = effective_events
                effective_open = desired_open

                output_stream.write(
                    json.dumps(
                        augmented,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                output_stream.write("\n")
                frame_count += 1
                transition_count += len(
                    augmented["vehicle_context"]["transition_events"]
                )
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary_name, output_jsonl)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "frames": frame_count,
        "transitions": transition_count,
        "forklift_candidate_observations": candidate_observations,
        "suppressed_person_frames": suppressed_person_frames,
        "effective_open_alarms_at_end": [
            {"track_id": track_id, "type": event_type}
            for track_id, event_type in sorted(effective_open)
        ],
        "output": str(output_jsonl),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-jsonl", type=Path, required=True)
    parser.add_argument("--vehicle-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--vehicle-confidence", type=float, default=0.35)
    parser.add_argument("--person-ioa", type=float, default=0.55)
    parser.add_argument("--enter-frames", type=int, default=4)
    parser.add_argument("--exit-frames", type=int, default=8)
    parser.add_argument("--ttl-frames", type=int, default=30)
    parser.add_argument("--no-center-inside", action="store_true")
    parser.add_argument("--keep-ppe-alerts", action="store_true")
    parser.add_argument("--keep-walkway-alerts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ForkliftDriverConfig(
        vehicle_confidence=args.vehicle_confidence,
        person_ioa=args.person_ioa,
        require_center_inside=not args.no_center_inside,
        enter_frames=args.enter_frames,
        exit_frames=args.exit_frames,
        ttl_frames=args.ttl_frames,
        suppress_ppe=not args.keep_ppe_alerts,
        suppress_walkway=not args.keep_walkway_alerts,
    )
    result = augment_jsonl_streams(
        args.person_jsonl,
        args.vehicle_jsonl,
        args.output_jsonl,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EVENT_TYPE",
    "ForkliftDriverConfig",
    "ForkliftDriverRuleEngine",
    "ForkliftDriverRuleError",
    "ROLE",
    "SCHEMA_VERSION",
    "associate_people_to_vehicles",
    "augment_frame_record",
    "augment_jsonl_streams",
    "person_center_inside_vehicle",
    "person_intersection_over_area",
]


if __name__ == "__main__":
    raise SystemExit(main())
