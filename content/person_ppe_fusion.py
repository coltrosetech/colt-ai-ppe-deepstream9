"""Person-centric PPE association, tracking and temporal alarm state.

This module is deliberately independent from the inference backend.  A frame
contains person detections plus helmet/hi-vis observations.  The fusion layer:

* assigns a stable, ephemeral track ID to each visible person;
* associates helmet observations with the person's head zone and vest
  observations with the torso zone;
* turns explicit ``no_*`` observations (or sustained visible-zone absence,
  when enabled) into a per-person violation;
* applies hysteresis before opening or closing an alarm.

The track ID is only a within-camera runtime identifier.  It is not biometric
identity and is never matched across cameras.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class PersonPpeFusionError(ValueError):
    """A frame or configuration violates the person/PPE fusion contract."""


PERSON_CLASS = "person"
EQUIPMENT_CLASSES = frozenset(
    {"helmet", "no_helmet", "hi_vis", "no_hi_vis"}
)
_EQUIPMENT = ("helmet", "hi_vis")


@dataclass(frozen=True)
class PersonPpeFusionConfig:
    """Runtime policy for one camera's person-centric PPE state."""

    track_iou_threshold: float = 0.25
    track_center_distance_threshold: float = 0.55
    track_ttl_frames: int = 15
    alarm_enter_frames: int = 3
    inferred_absence_enter_frames: int = 8
    verified_present_missing_grace_frames: int = 90
    alarm_clear_frames: int = 3
    unknown_after_frames: int = 8
    minimum_person_confidence: float = 0.25
    minimum_person_width: float = 0.015
    minimum_person_height: float = 0.04
    minimum_zone_visible_fraction: float = 0.65
    minimum_absence_zone_width: float = 0.02
    minimum_absence_zone_height: float = 0.025
    minimum_equipment_confidence: float = 0.10
    minimum_observation_coverage: float = 0.35
    association_ambiguity_margin: float = 0.05
    contradictory_confidence_margin: float = 0.10
    # Explicit NO-Hardhat / NO-Safety Vest observations are always accepted.
    # Inferring a violation merely because a positive detector missed an item
    # is an opt-in content policy; production callers should qualify it per
    # camera because occlusion and small people otherwise create false alarms.
    infer_no_helmet_from_missing: bool = False
    infer_no_hi_vis_from_missing: bool = False
    head_zone: tuple[float, float, float, float] = (0.05, 0.00, 0.95, 0.38)
    torso_zone: tuple[float, float, float, float] = (0.05, 0.18, 0.95, 0.75)

    def validate(self) -> None:
        unit_values = {
            "track_iou_threshold": self.track_iou_threshold,
            "track_center_distance_threshold": self.track_center_distance_threshold,
            "minimum_person_confidence": self.minimum_person_confidence,
            "minimum_person_width": self.minimum_person_width,
            "minimum_person_height": self.minimum_person_height,
            "minimum_zone_visible_fraction": self.minimum_zone_visible_fraction,
            "minimum_absence_zone_width": self.minimum_absence_zone_width,
            "minimum_absence_zone_height": self.minimum_absence_zone_height,
            "minimum_equipment_confidence": self.minimum_equipment_confidence,
            "minimum_observation_coverage": self.minimum_observation_coverage,
            "association_ambiguity_margin": self.association_ambiguity_margin,
            "contradictory_confidence_margin": self.contradictory_confidence_margin,
        }
        for name, value in unit_values.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PersonPpeFusionError(f"{name} must be inside [0,1]")
        for name, value in {
            "track_ttl_frames": self.track_ttl_frames,
            "alarm_enter_frames": self.alarm_enter_frames,
            "inferred_absence_enter_frames": self.inferred_absence_enter_frames,
            "verified_present_missing_grace_frames": (
                self.verified_present_missing_grace_frames
            ),
            "alarm_clear_frames": self.alarm_clear_frames,
            "unknown_after_frames": self.unknown_after_frames,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PersonPpeFusionError(f"{name} must be a positive integer")
        for name, zone in {"head_zone": self.head_zone, "torso_zone": self.torso_zone}.items():
            if (
                len(zone) != 4
                or not all(math.isfinite(value) for value in zone)
                or not all(0.0 <= value <= 1.0 for value in zone)
                or zone[2] <= zone[0]
                or zone[3] <= zone[1]
            ):
                raise PersonPpeFusionError(
                    f"{name} must be normalized xyxy with positive area"
                )


@dataclass
class _ChannelState:
    state: str = "unknown"
    confidence: float = 0.0
    latest_evidence: str = "unknown"
    latest_source: str = "none"
    streak: int = 0


@dataclass
class _Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    missed_frames: int = 0
    helmet: _ChannelState = field(default_factory=_ChannelState)
    hi_vis: _ChannelState = field(default_factory=_ChannelState)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _bbox(
    detection: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    raw = detection.get("bbox_norm_xywh")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 4
        or not all(_finite(value) for value in raw)
    ):
        raise PersonPpeFusionError(
            "bbox_norm_xywh must contain four finite values"
        )
    x, y, width, height = (float(value) for value in raw)
    if (
        x < 0.0
        or y < 0.0
        or width <= 0.0
        or height <= 0.0
        or x + width > 1.000001
        or y + height > 1.000001
    ):
        raise PersonPpeFusionError("bbox_norm_xywh is outside the frame")
    return (x, y, width, height)


def _confidence(detection: Mapping[str, Any]) -> float:
    value = detection.get("confidence")
    if not _finite(value) or not 0.0 <= float(value) <= 1.0:
        raise PersonPpeFusionError("confidence must be inside [0,1]")
    return float(value)


def _xyxy(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return (x, y, x + width, y + height)


def _intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_xyxy = _xyxy(first)
    second_xyxy = _xyxy(second)
    return max(
        0.0, min(first_xyxy[2], second_xyxy[2]) - max(first_xyxy[0], second_xyxy[0])
    ) * max(
        0.0, min(first_xyxy[3], second_xyxy[3]) - max(first_xyxy[1], second_xyxy[1])
    )


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = _intersection(first, second)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x = first[0] + first[2] * 0.5
    first_y = first[1] + first[3] * 0.5
    second_x = second[0] + second[2] * 0.5
    second_y = second[1] + second[3] * 0.5
    normalizer = max(first[2], first[3], second[2], second[3], 1.0e-9)
    return math.hypot(first_x - second_x, first_y - second_y) / normalizer


def _zone(
    person: tuple[float, float, float, float],
    normalized_zone: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, width, height = person
    left, top, right, bottom = normalized_zone
    return (
        x + left * width,
        y + top * height,
        (right - left) * width,
        (bottom - top) * height,
    )


def _zone_visible_fraction(
    zone: tuple[float, float, float, float],
) -> float:
    full_area = zone[2] * zone[3]
    visible = _intersection(zone, (0.0, 0.0, 1.0, 1.0))
    return visible / full_area if full_area > 0.0 else 0.0


def _association_score(
    observation: tuple[float, float, float, float],
    zone: tuple[float, float, float, float],
    minimum_coverage: float,
) -> float | None:
    observation_center = (
        observation[0] + observation[2] * 0.5,
        observation[1] + observation[3] * 0.5,
    )
    zone_xyxy = _xyxy(zone)
    if not (
        zone_xyxy[0] <= observation_center[0] <= zone_xyxy[2]
        and zone_xyxy[1] <= observation_center[1] <= zone_xyxy[3]
    ):
        return None
    intersection = _intersection(observation, zone)
    coverage = intersection / (observation[2] * observation[3])
    if coverage < minimum_coverage:
        return None
    zone_center = (zone[0] + zone[2] * 0.5, zone[1] + zone[3] * 0.5)
    distance = math.hypot(
        (observation_center[0] - zone_center[0]) / max(zone[2], 1.0e-9),
        (observation_center[1] - zone_center[1]) / max(zone[3], 1.0e-9),
    )
    proximity = max(0.0, 1.0 - distance)
    return 0.72 * coverage + 0.28 * proximity


def _equipment_and_evidence(canonical_class: str) -> tuple[str, str]:
    if canonical_class in {"helmet", "no_helmet"}:
        return "helmet", "present" if canonical_class == "helmet" else "absent"
    if canonical_class in {"hi_vis", "no_hi_vis"}:
        return "hi_vis", "present" if canonical_class == "hi_vis" else "absent"
    raise PersonPpeFusionError(f"unsupported equipment class {canonical_class!r}")


class PersonPpeFusion:
    """Fuse one ordered camera stream into per-person PPE decisions."""

    def __init__(self, config: PersonPpeFusionConfig | None = None) -> None:
        self.config = config or PersonPpeFusionConfig()
        self.config.validate()
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_frame_index: int | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_frame_index = None

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def _allocate_track_id(self) -> int:
        while self._next_track_id in self._tracks:
            self._next_track_id += 1
        value = self._next_track_id
        self._next_track_id += 1
        return value

    def _associate_people(
        self, people: Sequence[tuple[Mapping[str, Any], tuple[float, float, float, float], float]]
    ) -> tuple[list[_Track], list[_Track]]:
        assignments: dict[int, int] = {}
        occupied_tracks: set[int] = set()

        # Respect a canonical tracker ID when DeepStream already supplied one.
        for detection_index, (detection, box, confidence) in enumerate(people):
            external = detection.get("track_id")
            if external is None:
                continue
            if (
                not isinstance(external, int)
                or isinstance(external, bool)
                or external < 0
                or external in occupied_tracks
            ):
                raise PersonPpeFusionError(
                    "person track_id must be a unique non-negative integer"
                )
            assignments[detection_index] = external
            occupied_tracks.add(external)
            track = self._tracks.get(external)
            if track is None:
                self._tracks[external] = _Track(external, box, confidence)
            self._next_track_id = max(self._next_track_id, external + 1)

        proposals: list[tuple[float, float, int, int]] = []
        for detection_index, (_, box, _) in enumerate(people):
            if detection_index in assignments:
                continue
            for track_id, track in self._tracks.items():
                if track_id in occupied_tracks:
                    continue
                iou = _iou(box, track.bbox)
                distance = _center_distance(box, track.bbox)
                if (
                    iou < self.config.track_iou_threshold
                    and distance > self.config.track_center_distance_threshold
                ):
                    continue
                proposals.append((iou, -distance, detection_index, track_id))
        proposals.sort(key=lambda value: (-value[0], -value[1], value[2], value[3]))
        occupied_detections = set(assignments)
        for _, _, detection_index, track_id in proposals:
            if detection_index in occupied_detections or track_id in occupied_tracks:
                continue
            assignments[detection_index] = track_id
            occupied_detections.add(detection_index)
            occupied_tracks.add(track_id)

        for detection_index in range(len(people)):
            if detection_index not in assignments:
                track_id = self._allocate_track_id()
                assignments[detection_index] = track_id
                occupied_tracks.add(track_id)
                _, box, confidence = people[detection_index]
                self._tracks[track_id] = _Track(track_id, box, confidence)

        for track in self._tracks.values():
            track.missed_frames += 1
        ordered: list[_Track] = []
        for detection_index, (_, box, confidence) in enumerate(people):
            track = self._tracks[assignments[detection_index]]
            track.bbox = box
            track.confidence = confidence
            track.missed_frames = 0
            ordered.append(track)
        expired = [
            track
            for track in self._tracks.values()
            if track.missed_frames > self.config.track_ttl_frames
        ]
        for track in expired:
            del self._tracks[track.track_id]
        return ordered, sorted(expired, key=lambda value: value.track_id)

    def _associate_equipment(
        self,
        tracks: Sequence[_Track],
        detections: Sequence[
            tuple[Mapping[str, Any], tuple[float, float, float, float], float]
        ],
    ) -> dict[tuple[int, str], list[dict[str, Any]]]:
        associated: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for detection, box, confidence in detections:
            if confidence < self.config.minimum_equipment_confidence:
                continue
            canonical = str(detection["canonical_class"])
            equipment, evidence = _equipment_and_evidence(canonical)
            normalized_zone = (
                self.config.head_zone
                if equipment == "helmet"
                else self.config.torso_zone
            )
            candidates: list[tuple[float, int]] = []
            for track in tracks:
                score = _association_score(
                    box,
                    _zone(track.bbox, normalized_zone),
                    self.config.minimum_observation_coverage,
                )
                if score is not None:
                    candidates.append((score, track.track_id))
            candidates.sort(key=lambda value: (-value[0], value[1]))
            if not candidates:
                continue
            if (
                len(candidates) > 1
                and candidates[0][0] - candidates[1][0]
                <= self.config.association_ambiguity_margin
            ):
                continue
            track_id = candidates[0][1]
            associated.setdefault((track_id, equipment), []).append(
                {
                    "evidence": evidence,
                    "canonical_class": canonical,
                    "confidence": confidence,
                    "bbox_norm_xywh": list(box),
                    "association_score": candidates[0][0],
                }
            )
        return associated

    def _frame_evidence(
        self,
        track: _Track,
        equipment: str,
        observations: Sequence[Mapping[str, Any]],
    ) -> tuple[str, float, str, list[dict[str, Any]]]:
        if observations:
            ordered = sorted(
                (dict(observation) for observation in observations),
                key=lambda value: (-float(value["confidence"]), value["canonical_class"]),
            )
            best_by_evidence: dict[str, dict[str, Any]] = {}
            for observation in ordered:
                best_by_evidence.setdefault(str(observation["evidence"]), observation)
            if {"present", "absent"} <= set(best_by_evidence):
                present = best_by_evidence["present"]
                absent = best_by_evidence["absent"]
                difference = abs(
                    float(present["confidence"]) - float(absent["confidence"])
                )
                if difference < self.config.contradictory_confidence_margin:
                    return "unknown", 0.0, "conflicting_observations", ordered
                winner = (
                    present
                    if float(present["confidence"]) > float(absent["confidence"])
                    else absent
                )
            else:
                winner = ordered[0]
            return (
                str(winner["evidence"]),
                float(winner["confidence"]),
                "detector_observation",
                ordered,
            )

        normalized_zone = (
            self.config.head_zone if equipment == "helmet" else self.config.torso_zone
        )
        zone = _zone(track.bbox, normalized_zone)
        sufficiently_visible = (
            track.bbox[2] >= self.config.minimum_person_width
            and track.bbox[3] >= self.config.minimum_person_height
            and zone[2] >= self.config.minimum_absence_zone_width
            and zone[3] >= self.config.minimum_absence_zone_height
            and _zone_visible_fraction(zone)
            >= self.config.minimum_zone_visible_fraction
            # A detector box clipped to the top frame edge cannot prove that
            # the person's head is visible.  Never infer missing PPE there.
            and (equipment != "helmet" or track.bbox[1] > 1.0e-6)
        )
        infer_missing = (
            self.config.infer_no_helmet_from_missing
            if equipment == "helmet"
            else self.config.infer_no_hi_vis_from_missing
        )
        if sufficiently_visible and infer_missing:
            channel = (
                track.helmet if equipment == "helmet" else track.hi_vis
            )
            if channel.state == "present":
                grace_age = (
                    channel.streak
                    if channel.latest_source
                    == "verified_present_grace_no_observation"
                    else 0
                )
                if (
                    grace_age
                    < self.config.verified_present_missing_grace_frames
                ):
                    return (
                        "unknown",
                        0.0,
                        "verified_present_grace_no_observation",
                        [],
                    )
            # Confidence is deliberately bounded: this is negative evidence
            # inferred from a visible body zone, not an explicit no-PPE box.
            return "absent", 0.55, "visible_zone_no_positive_observation", []
        return "unknown", 0.0, "insufficient_visibility_or_no_observation", []

    def _update_channel(
        self,
        channel: _ChannelState,
        *,
        evidence: str,
        confidence: float,
        source: str,
    ) -> tuple[str, str]:
        previous = channel.state
        if channel.latest_evidence == evidence and channel.latest_source == source:
            channel.streak += 1
        else:
            channel.latest_evidence = evidence
            channel.streak = 1
        channel.latest_source = source

        absence_threshold = (
            self.config.inferred_absence_enter_frames
            if source == "visible_zone_no_positive_observation"
            else self.config.alarm_enter_frames
        )
        unknown_threshold = (
            self.config.verified_present_missing_grace_frames
            if source == "verified_present_grace_no_observation"
            else self.config.unknown_after_frames
        )
        if evidence == "absent" and channel.streak >= absence_threshold:
            channel.state = "absent"
            channel.confidence = confidence
        elif evidence == "present" and channel.streak >= self.config.alarm_clear_frames:
            channel.state = "present"
            channel.confidence = confidence
        elif evidence == "unknown" and channel.streak >= unknown_threshold:
            channel.state = "unknown"
            channel.confidence = 0.0
        elif channel.state != "unknown" and evidence == channel.state:
            channel.confidence = confidence
        return previous, channel.state

    def process_frame(
        self,
        *,
        frame_index: int,
        detections: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Process one frame and return person boxes, PPE states and events."""

        if (
            not isinstance(frame_index, int)
            or isinstance(frame_index, bool)
            or frame_index < 0
        ):
            raise PersonPpeFusionError("frame_index must be non-negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise PersonPpeFusionError("frame_index must be strictly increasing")
        if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
            raise PersonPpeFusionError("detections must be a sequence")

        people: list[
            tuple[Mapping[str, Any], tuple[float, float, float, float], float]
        ] = []
        equipment: list[
            tuple[Mapping[str, Any], tuple[float, float, float, float], float]
        ] = []
        filtered_person_detections = 0
        for detection in detections:
            if not isinstance(detection, Mapping):
                raise PersonPpeFusionError("each detection must be an object")
            canonical = str(detection.get("canonical_class", ""))
            box = _bbox(detection)
            confidence = _confidence(detection)
            if canonical == PERSON_CLASS:
                if (
                    confidence < self.config.minimum_person_confidence
                    or box[2] < self.config.minimum_person_width
                    or box[3] < self.config.minimum_person_height
                ):
                    filtered_person_detections += 1
                    continue
                people.append((detection, box, confidence))
            elif canonical in EQUIPMENT_CLASSES:
                equipment.append((detection, box, confidence))
            else:
                raise PersonPpeFusionError(
                    f"unsupported canonical_class {canonical!r}"
                )

        tracks, expired_tracks = self._associate_people(people)
        observations = self._associate_equipment(tracks, equipment)
        persons: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for track in expired_tracks:
            for equipment_name, channel in (
                ("helmet", track.helmet),
                ("hi_vis", track.hi_vis),
            ):
                if channel.state != "absent":
                    continue
                events.append(
                    {
                        "kind": "ended",
                        "type": (
                            "no_helmet"
                            if equipment_name == "helmet"
                            else "no_hi_vis"
                        ),
                        "track_id": track.track_id,
                        "frame_index": frame_index,
                    }
                )
        for track in sorted(tracks, key=lambda value: value.track_id):
            equipment_outputs: dict[str, dict[str, Any]] = {}
            for equipment_name in _EQUIPMENT:
                evidence, confidence, source, matched = self._frame_evidence(
                    track,
                    equipment_name,
                    observations.get((track.track_id, equipment_name), []),
                )
                channel = (
                    track.helmet if equipment_name == "helmet" else track.hi_vis
                )
                previous, current = self._update_channel(
                    channel,
                    evidence=evidence,
                    confidence=confidence,
                    source=source,
                )
                if previous != "absent" and current == "absent":
                    events.append(
                        {
                            "kind": "started",
                            "type": (
                                "no_helmet"
                                if equipment_name == "helmet"
                                else "no_hi_vis"
                            ),
                            "track_id": track.track_id,
                            "frame_index": frame_index,
                        }
                    )
                elif previous == "absent" and current in {"present", "unknown"}:
                    events.append(
                        {
                            "kind": "ended",
                            "type": (
                                "no_helmet"
                                if equipment_name == "helmet"
                                else "no_hi_vis"
                            ),
                            "track_id": track.track_id,
                            "frame_index": frame_index,
                        }
                    )
                if source == "detector_observation":
                    link_status = "matched"
                elif source == "visible_zone_no_positive_observation":
                    link_status = "inferred_visible_missing"
                elif source == "conflicting_observations":
                    link_status = "unknown_conflicting"
                else:
                    link_status = "unknown_no_observation"
                equipment_outputs[equipment_name] = {
                    "state": current,
                    # Evidence is always the current frame's observation.
                    # State is the independently qualified temporal decision.
                    "evidence": evidence,
                    "frame_evidence": evidence,
                    "evidence_source": source,
                    "link_status": link_status,
                    "confidence": round(confidence, 6),
                    "state_confidence": round(channel.confidence, 6),
                    "streak": channel.streak,
                    "observations": matched,
                }
            alarms = [
                name
                for name, channel in (
                    ("no_helmet", track.helmet),
                    ("no_hi_vis", track.hi_vis),
                )
                if channel.state == "absent"
            ]
            overall_state = (
                "noncompliant"
                if alarms
                else (
                    "compliant"
                    if track.helmet.state == "present"
                    and track.hi_vis.state == "present"
                    else "unknown"
                )
            )
            persons.append(
                {
                    "track_id": track.track_id,
                    "person_id": f"KISI-{track.track_id:03d}",
                    "bbox_norm_xywh": [round(value, 8) for value in track.bbox],
                    "confidence": round(track.confidence, 6),
                    "helmet": equipment_outputs["helmet"],
                    "hi_vis": equipment_outputs["hi_vis"],
                    "overall_state": overall_state,
                    "alarm": bool(alarms),
                    "alarms": alarms,
                }
            )
        self._last_frame_index = frame_index
        return {
            "schema_version": "colt-ai.person-ppe-frame/v1",
            "frame_index": frame_index,
            "persons": persons,
            "alarm_events": events,
            "filtered_person_detection_count": filtered_person_detections,
            "unassociated_equipment_count": max(
                0,
                len(equipment)
                - sum(
                    len(values)
                    for values in observations.values()
                ),
            ),
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise PersonPpeFusionError(
                        f"{path}:{line_number}: blank JSONL record"
                    )
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise PersonPpeFusionError(
                        f"{path}:{line_number}: record must be an object"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersonPpeFusionError(f"cannot read prediction stream {path}: {exc}") from exc
    if not rows:
        raise PersonPpeFusionError(f"prediction stream is empty: {path}")
    return rows


def fuse_prediction_streams(
    *,
    ppe_predictions: Path,
    output: Path,
    person_predictions: Path | None = None,
    config: PersonPpeFusionConfig | None = None,
) -> dict[str, Any]:
    """Fuse person and PPE prediction JSONL streams into per-person rows.

    ``person_predictions`` accepts ``deepsafe.person-detections/v1`` rows.
    When omitted, ``ppe_predictions`` must itself contain ``person`` records,
    as produced by the updated SafetyVision diagnostic runner.
    """

    ppe_rows = _read_jsonl(ppe_predictions)
    person_rows = _read_jsonl(person_predictions) if person_predictions else None
    if person_rows is not None and len(person_rows) != len(ppe_rows):
        raise PersonPpeFusionError("person and PPE streams have different lengths")
    if output.exists():
        raise PersonPpeFusionError(f"output already exists: {output}")

    fusion = PersonPpeFusion(config)
    encoded_rows: list[bytes] = []
    event_count = 0
    person_observations = 0
    alarm_person_frames = 0
    for expected_index, ppe_row in enumerate(ppe_rows):
        if int(ppe_row.get("frame_index", -1)) != expected_index:
            raise PersonPpeFusionError(
                "PPE frame indices must be contiguous from zero"
            )
        width = ppe_row.get("image_width")
        height = ppe_row.get("image_height")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise PersonPpeFusionError("PPE frame dimensions are invalid")
        raw_ppe = ppe_row.get("detections")
        if not isinstance(raw_ppe, list):
            raise PersonPpeFusionError("PPE detections must be a list")
        detections = [dict(item) for item in raw_ppe]

        if person_rows is not None:
            person_row = person_rows[expected_index]
            if (
                int(person_row.get("frame_index", -1)) != expected_index
                or person_row.get("image_width") != width
                or person_row.get("image_height") != height
            ):
                raise PersonPpeFusionError(
                    "person/PPE frame index or geometry differs"
                )
            raw_people = person_row.get("detections")
            if not isinstance(raw_people, list):
                raise PersonPpeFusionError("person detections must be a list")
            detections = [
                {
                    **dict(item),
                    "canonical_class": PERSON_CLASS,
                }
                for item in raw_people
                if (
                    str(item.get("class_name", "person"))
                    .strip()
                    .casefold()
                    == PERSON_CLASS.casefold()
                    and item.get("class_id", 0) == 0
                )
            ] + [
                item
                for item in detections
                if item.get("canonical_class") != PERSON_CLASS
            ]
        elif not any(
            item.get("canonical_class") == PERSON_CLASS for item in detections
        ):
            raise PersonPpeFusionError(
                f"PPE frame {expected_index} has no person detections and "
                "--person-predictions was not supplied"
            )

        fused = fusion.process_frame(
            frame_index=expected_index,
            detections=detections,
        )
        fused.update(
            {
                "sequence_id": ppe_row.get("sequence_id"),
                "image_width": width,
                "image_height": height,
            }
        )
        event_count += len(fused["alarm_events"])
        person_observations += len(fused["persons"])
        alarm_person_frames += sum(
            1 for person in fused["persons"] if person["alarm"]
        )
        encoded_rows.append(
            (
                json.dumps(
                    fused,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"".join(encoded_rows))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "schema_version": "colt-ai.person-ppe-fusion-result/v1",
        "status": "complete",
        "frames": len(ppe_rows),
        "person_observations": person_observations,
        "alarm_person_frames": alarm_person_frames,
        "alarm_events": event_count,
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse person tracks with helmet/hi-vis observations"
    )
    parser.add_argument("--ppe-predictions", type=Path, required=True)
    parser.add_argument("--person-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alarm-enter-frames", type=int, default=3)
    parser.add_argument("--inferred-absence-enter-frames", type=int, default=8)
    parser.add_argument(
        "--verified-present-missing-grace-frames",
        type=int,
        default=90,
        help=(
            "Keep a previously verified present state through this many "
            "visible-zone detector misses before inferred absence can start."
        ),
    )
    parser.add_argument("--alarm-clear-frames", type=int, default=3)
    parser.add_argument(
        "--unknown-after-frames",
        type=int,
        default=8,
        help=(
            "Clear non-grace temporal state after this many unknown frames."
        ),
    )
    parser.add_argument("--minimum-person-confidence", type=float, default=0.25)
    parser.add_argument("--minimum-person-width", type=float, default=0.015)
    parser.add_argument("--minimum-person-height", type=float, default=0.04)
    parser.add_argument(
        "--infer-missing-helmet-alarm",
        action="store_true",
        help="Treat sustained visible-head absence as no_helmet evidence.",
    )
    parser.add_argument(
        "--infer-missing-vest-alarm",
        action="store_true",
        help="Treat sustained visible-torso absence as no_hi_vis evidence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PersonPpeFusionConfig(
        alarm_enter_frames=args.alarm_enter_frames,
        inferred_absence_enter_frames=args.inferred_absence_enter_frames,
        verified_present_missing_grace_frames=(
            args.verified_present_missing_grace_frames
        ),
        alarm_clear_frames=args.alarm_clear_frames,
        unknown_after_frames=args.unknown_after_frames,
        minimum_person_confidence=args.minimum_person_confidence,
        minimum_person_width=args.minimum_person_width,
        minimum_person_height=args.minimum_person_height,
        infer_no_helmet_from_missing=args.infer_missing_helmet_alarm,
        infer_no_hi_vis_from_missing=args.infer_missing_vest_alarm,
    )
    result = fuse_prediction_streams(
        ppe_predictions=args.ppe_predictions,
        person_predictions=args.person_predictions,
        output=args.output,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
