"""Backend-independent safety-zone rules for tracked people.

The engine consumes frame-aligned person records carrying a stable ``track_id``
and either ``bbox_norm_xywh`` or ``footpoint_norm_xy``.  It deliberately does
not filter people: input person records are copied to the output and augmented
with ``zone_safety`` so PPE, pose, and other person-centric metadata remains
available over the full frame.

Two area types are supported:

``restricted_zone``
    By default, a person violates the rule when their footpoint is inside any
    configured restricted polygon.  An optional COCO17 pose rule can instead
    require a configured percentage of selected, confidence-qualified
    keypoints to be inside the restricted-polygon union.

``safe_walkway``
    Configured walkway polygons form one allowed union.  A person violates the
    rule when their footpoint is outside every configured walkway polygon.

Polygon boundaries are inclusive.  Consequently, a point on a restricted
boundary is an intrusion while a point on a walkway boundary is safe.

Pose observations are three-state.  A valid observation is either violating or
safe.  A missing, ambiguous, or insufficiently visible observation is unknown:
it neither advances nor clears a temporal streak, and an already active
violation remains active until a valid exit observation or track expiry.

For fences, ``FenceBoundaryRuleConfig`` provides a stricter, track-aware mode.
The oriented two-point boundary replaces projected keypoint/polygon overlap as
the critical decision: only a confidence-qualified hip core on the forbidden
side can open ``restricted_area_intrusion``.  Wrist contact, hip lift, and knee
lift are retained as non-critical approach/climb evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence, TypeAlias


AreaType: TypeAlias = Literal["restricted_zone", "safe_walkway"]
Point: TypeAlias = tuple[float, float]
Bbox: TypeAlias = tuple[float, float, float, float]
RawViolation: TypeAlias = bool | None
FenceState: TypeAlias = Literal["clear", "approach", "climbing", "breach"]

SCHEMA_VERSION = "colt-ai.person-zone-frame/v1"
EVENT_SCHEMA_VERSION = "colt-ai.person-safety-transition/v1"
FENCE_EVENT_SCHEMA_VERSION = "colt-ai.fence-state-transition/v1"
RESTRICTED_ZONE: AreaType = "restricted_zone"
SAFE_WALKWAY: AreaType = "safe_walkway"
_AREA_TYPES = frozenset({RESTRICTED_ZONE, SAFE_WALKWAY})
_EVENT_TYPES: dict[AreaType, str] = {
    RESTRICTED_ZONE: "restricted_area_intrusion",
    SAFE_WALKWAY: "walkway_violation",
}
COCO17_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_COCO17_KEYPOINT_NAMES = frozenset(COCO17_KEYPOINT_NAMES)
_FENCE_STATES = frozenset({"clear", "approach", "climbing", "breach"})


class PersonZoneRuleError(ValueError):
    """The area configuration or frame input violates the runtime contract."""


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise PersonZoneRuleError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PersonZoneRuleError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise PersonZoneRuleError(f"{label} must be a finite number")
    return number


def _normalized_point(value: object, *, label: str) -> Point:
    if isinstance(value, Mapping):
        if set(value) != {"x", "y"}:
            raise PersonZoneRuleError(f"{label} must contain only x and y")
        raw_x, raw_y = value["x"], value["y"]
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        raw_x, raw_y = value
    else:
        raise PersonZoneRuleError(f"{label} must be a normalized [x,y] point")
    point = (
        _finite_number(raw_x, label=f"{label}.x"),
        _finite_number(raw_y, label=f"{label}.y"),
    )
    if not all(0.0 <= coordinate <= 1.0 for coordinate in point):
        raise PersonZoneRuleError(f"{label} must be inside the normalized frame")
    return point


def _normalized_bbox(value: object, *, label: str) -> Bbox:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise PersonZoneRuleError(f"{label} must be normalized [x,y,w,h]")
    x, y, width, height = (
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if (
        x < 0.0
        or y < 0.0
        or width <= 0.0
        or height <= 0.0
        or x + width > 1.000001
        or y + height > 1.000001
    ):
        raise PersonZoneRuleError(f"{label} is outside the normalized frame")
    return x, y, width, height


def bbox_bottom_center(bbox_norm_xywh: Sequence[object]) -> Point:
    """Return the normalized ground-contact proxy for a person bbox."""

    x, y, width, height = _normalized_bbox(
        bbox_norm_xywh,
        label="bbox_norm_xywh",
    )
    return min(1.0, x + width / 2.0), min(1.0, y + height)


def _point_on_segment(
    point: Point,
    start: Point,
    end: Point,
    *,
    epsilon: float = 1.0e-9,
) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > epsilon:
        return False
    return (
        min(ax, bx) - epsilon <= px <= max(ax, bx) + epsilon
        and min(ay, by) - epsilon <= py <= max(ay, by) + epsilon
    )


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return ``True`` for polygon interiors and boundaries."""

    if len(polygon) < 3:
        raise PersonZoneRuleError("polygon needs at least three points")
    inside = False
    px, py = point
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        ax, ay = start
        bx, by = end
        if (ay > py) != (by > py):
            crossing_x = (bx - ax) * (py - ay) / (by - ay) + ax
            if px < crossing_x:
                inside = not inside
    return inside


def _segment_length(start: Point, end: Point) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _signed_distance_to_oriented_line(
    point: Point,
    start: Point,
    end: Point,
) -> float:
    """Signed perpendicular distance from ``point`` to ``start -> end``.

    Positive is the mathematical left side of the oriented line and negative
    is the right side.  Coordinates are normalized image coordinates, so the
    visual meaning of left/right depends on how the operator draws the line.
    """

    length = _segment_length(start, end)
    if length <= 1.0e-12:
        raise PersonZoneRuleError("fence boundary points must be distinct")
    return (
        (end[0] - start[0]) * (point[1] - start[1])
        - (end[1] - start[1]) * (point[0] - start[0])
    ) / length


def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
    """Return Euclidean distance to the finite segment, endpoints included."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    squared_length = dx * dx + dy * dy
    if squared_length <= 1.0e-24:
        raise PersonZoneRuleError("fence boundary points must be distinct")
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / squared_length
    projection = min(1.0, max(0.0, projection))
    nearest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def _polygon_area(points: Sequence[Point]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


@dataclass(frozen=True)
class ZoneArea:
    """One normalized safety polygon."""

    area_id: str
    area_type: AreaType
    points: tuple[Point, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.area_id, str)
            or not self.area_id.strip()
            or len(self.area_id.strip()) > 64
        ):
            raise PersonZoneRuleError("area_id must be a non-empty string")
        object.__setattr__(self, "area_id", self.area_id.strip())
        if self.area_type not in _AREA_TYPES:
            raise PersonZoneRuleError(
                "area_type must be restricted_zone or safe_walkway"
            )
        if not isinstance(self.name, str) or len(self.name.strip()) > 80:
            raise PersonZoneRuleError("area name is invalid")
        object.__setattr__(self, "name", self.name.strip())
        if not 3 <= len(self.points) <= 16:
            raise PersonZoneRuleError("area polygon needs 3..16 points")
        normalized = tuple(
            _normalized_point(point, label=f"{self.area_id}.points[{index}]")
            for index, point in enumerate(self.points)
        )
        if len(set(normalized)) != len(normalized):
            raise PersonZoneRuleError("area polygon has duplicate points")
        if _polygon_area(normalized) < 1.0e-5:
            raise PersonZoneRuleError("area polygon must have non-zero area")
        object.__setattr__(self, "points", normalized)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ZoneArea":
        """Normalize the queue/UI mapping form into a validated area."""

        if not isinstance(value, Mapping):
            raise PersonZoneRuleError("area must be an object")
        area_type = value.get(
            "area_type",
            value.get("roi_type", value.get("type")),
        )
        raw_points = value.get("points", value.get("polygon_norm"))
        if (
            not isinstance(raw_points, Sequence)
            or isinstance(raw_points, (str, bytes))
        ):
            raise PersonZoneRuleError("area points must be a sequence")
        return cls(
            area_id=str(value.get("area_id", value.get("roi_id", ""))),
            area_type=str(area_type),  # type: ignore[arg-type]
            points=tuple(
                _normalized_point(point, label=f"area.points[{index}]")
                for index, point in enumerate(raw_points)
            ),
            name=str(value.get("name", "")),
        )


@dataclass(frozen=True)
class PoseKeypointRuleConfig:
    """Restricted-zone rule over a selected subset of COCO17 keypoints.

    ``minimum_inside_ratio`` always uses the configured keypoint count as its
    denominator.  Missing, invisible, or low-confidence keypoints therefore do
    not become an artificially small denominator.  ``minimum_visible`` is a
    separate observation-validity gate.
    """

    keypoint_names: tuple[str, ...]
    minimum_confidence: float = 0.30
    minimum_visible: int = 1
    minimum_inside_ratio: float = 0.50

    def __post_init__(self) -> None:
        names = self.keypoint_names
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or not names
        ):
            raise PersonZoneRuleError(
                "pose keypoint_names must be a non-empty sequence"
            )
        normalized = tuple(names)
        if any(not isinstance(name, str) for name in normalized):
            raise PersonZoneRuleError(
                "pose keypoint_names must contain only COCO17 names"
            )
        if len(normalized) != len(set(normalized)):
            raise PersonZoneRuleError("pose keypoint_names must be unique")
        unsupported = [
            name for name in normalized if name not in _COCO17_KEYPOINT_NAMES
        ]
        if unsupported:
            raise PersonZoneRuleError(
                f"unsupported COCO17 keypoint name: {unsupported[0]}"
            )
        object.__setattr__(self, "keypoint_names", normalized)
        self.validate()

    def validate(self) -> None:
        confidence = _finite_number(
            self.minimum_confidence,
            label="minimum_confidence",
        )
        if not 0.0 <= confidence <= 1.0:
            raise PersonZoneRuleError(
                "minimum_confidence must be inside [0,1]"
            )
        ratio = _finite_number(
            self.minimum_inside_ratio,
            label="minimum_inside_ratio",
        )
        if not 0.0 < ratio <= 1.0:
            raise PersonZoneRuleError(
                "minimum_inside_ratio must be inside (0,1]"
            )
        if (
            not isinstance(self.minimum_visible, int)
            or isinstance(self.minimum_visible, bool)
            or not 1 <= self.minimum_visible <= len(self.keypoint_names)
        ):
            raise PersonZoneRuleError(
                "minimum_visible must be inside [1,keypoint_count]"
            )


@dataclass(frozen=True)
class FenceBoundaryRuleConfig:
    """Track-aware fence policy over one oriented normalized boundary.

    ``forbidden_side`` is relative to the operator-drawn
    ``boundary_start -> boundary_end`` direction.  The line itself belongs to
    both closed half-planes, so it is forbidden inclusively.  This deterministic
    convention avoids boundary flicker and lets the renderer show the selected
    side without deriving it from a polygon winding order.

    A critical breach is based only on the midpoint of confidence-qualified
    hip keypoints.  Partial overlap by hands, shoulders, knees, or ankles can
    produce approach/climb evidence but can never open a restricted-area
    safety event.
    """

    boundary_start: Point
    boundary_end: Point
    forbidden_side: Literal["left", "right"]
    contact_band: float = 0.03
    minimum_confidence: float = 0.30
    minimum_core_visible: int = 1
    breach_enter_frames: int = 4
    breach_exit_frames: int = 4
    approach_keypoint_names: tuple[str, ...] = (
        "left_wrist",
        "right_wrist",
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )
    approach_minimum_count: int = 1
    wrist_contact_required: int = 1
    hip_rise_ratio: float = 0.08
    raised_knee_ratio: float = 0.10
    climb_enter_frames: int = 2
    climb_exit_frames: int = 2
    history_window_frames: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "boundary_start",
            _normalized_point(
                self.boundary_start,
                label="fence.boundary_start",
            ),
        )
        object.__setattr__(
            self,
            "boundary_end",
            _normalized_point(
                self.boundary_end,
                label="fence.boundary_end",
            ),
        )
        if _segment_length(self.boundary_start, self.boundary_end) <= 1.0e-6:
            raise PersonZoneRuleError(
                "fence boundary points must be distinct"
            )
        if self.forbidden_side not in {"left", "right"}:
            raise PersonZoneRuleError(
                "fence forbidden_side must be left or right"
            )
        names = self.approach_keypoint_names
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or not names
            or any(not isinstance(name, str) for name in names)
        ):
            raise PersonZoneRuleError(
                "approach_keypoint_names must be a non-empty COCO17 sequence"
            )
        normalized_names = tuple(names)
        if len(normalized_names) != len(set(normalized_names)):
            raise PersonZoneRuleError(
                "approach_keypoint_names must be unique"
            )
        unsupported = [
            name
            for name in normalized_names
            if name not in _COCO17_KEYPOINT_NAMES
        ]
        if unsupported:
            raise PersonZoneRuleError(
                f"unsupported COCO17 keypoint name: {unsupported[0]}"
            )
        object.__setattr__(
            self,
            "approach_keypoint_names",
            normalized_names,
        )
        self.validate()

    @property
    def boundary_points(self) -> tuple[Point, Point]:
        return self.boundary_start, self.boundary_end

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FenceBoundaryRuleConfig":
        """Build the strict policy from its queue/content mapping form."""

        if not isinstance(value, Mapping):
            raise PersonZoneRuleError("fence boundary rule must be an object")
        raw_boundary = value.get("boundary")
        if raw_boundary is not None:
            if (
                not isinstance(raw_boundary, Sequence)
                or isinstance(raw_boundary, (str, bytes))
                or len(raw_boundary) != 2
            ):
                raise PersonZoneRuleError(
                    "fence boundary must contain exactly two points"
                )
            boundary_start, boundary_end = raw_boundary
        else:
            boundary_start = value.get("boundary_start")
            boundary_end = value.get("boundary_end")
        names = value.get(
            "approach_keypoint_names",
            cls.__dataclass_fields__["approach_keypoint_names"].default,
        )
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
        ):
            raise PersonZoneRuleError(
                "approach_keypoint_names must be a sequence"
            )
        return cls(
            boundary_start=boundary_start,  # type: ignore[arg-type]
            boundary_end=boundary_end,  # type: ignore[arg-type]
            forbidden_side=str(value.get("forbidden_side", "")),  # type: ignore[arg-type]
            contact_band=value.get("contact_band", 0.03),  # type: ignore[arg-type]
            minimum_confidence=value.get("minimum_confidence", 0.30),  # type: ignore[arg-type]
            minimum_core_visible=value.get("minimum_core_visible", 1),  # type: ignore[arg-type]
            breach_enter_frames=value.get("breach_enter_frames", 4),  # type: ignore[arg-type]
            breach_exit_frames=value.get("breach_exit_frames", 4),  # type: ignore[arg-type]
            approach_keypoint_names=tuple(names),
            approach_minimum_count=value.get("approach_minimum_count", 1),  # type: ignore[arg-type]
            wrist_contact_required=value.get("wrist_contact_required", 1),  # type: ignore[arg-type]
            hip_rise_ratio=value.get("hip_rise_ratio", 0.08),  # type: ignore[arg-type]
            raised_knee_ratio=value.get("raised_knee_ratio", 0.10),  # type: ignore[arg-type]
            climb_enter_frames=value.get("climb_enter_frames", 2),  # type: ignore[arg-type]
            climb_exit_frames=value.get("climb_exit_frames", 2),  # type: ignore[arg-type]
            history_window_frames=value.get("history_window_frames", 30),  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        for name, value in (
            ("contact_band", self.contact_band),
            ("minimum_confidence", self.minimum_confidence),
            ("hip_rise_ratio", self.hip_rise_ratio),
            ("raised_knee_ratio", self.raised_knee_ratio),
        ):
            number = _finite_number(value, label=name)
            if not 0.0 <= number <= 1.0:
                raise PersonZoneRuleError(f"{name} must be inside [0,1]")
        for name, value in (
            ("minimum_core_visible", self.minimum_core_visible),
            ("breach_enter_frames", self.breach_enter_frames),
            ("breach_exit_frames", self.breach_exit_frames),
            ("approach_minimum_count", self.approach_minimum_count),
            ("wrist_contact_required", self.wrist_contact_required),
            ("climb_enter_frames", self.climb_enter_frames),
            ("climb_exit_frames", self.climb_exit_frames),
            ("history_window_frames", self.history_window_frames),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise PersonZoneRuleError(f"{name} must be a positive integer")
        if self.minimum_core_visible > 2:
            raise PersonZoneRuleError(
                "minimum_core_visible must be 1 or 2"
            )
        if self.approach_minimum_count > len(
            self.approach_keypoint_names
        ):
            raise PersonZoneRuleError(
                "approach_minimum_count exceeds configured keypoints"
            )
        if self.wrist_contact_required > 2:
            raise PersonZoneRuleError(
                "wrist_contact_required must be 1 or 2"
            )
        if self.history_window_frames < 2:
            raise PersonZoneRuleError(
                "history_window_frames must be at least 2"
            )


@dataclass(frozen=True)
class ZoneRuleConfig:
    """Temporal qualification and track-retention policy."""

    restricted_enter_frames: int = 3
    restricted_exit_frames: int = 6
    walkway_enter_frames: int = 6
    walkway_exit_frames: int = 4
    track_ttl_frames: int = 15
    restricted_pose_rule: PoseKeypointRuleConfig | None = None
    restricted_fence_rule: FenceBoundaryRuleConfig | None = None

    def validate(self) -> None:
        for name, value in (
            ("restricted_enter_frames", self.restricted_enter_frames),
            ("restricted_exit_frames", self.restricted_exit_frames),
            ("walkway_enter_frames", self.walkway_enter_frames),
            ("walkway_exit_frames", self.walkway_exit_frames),
            ("track_ttl_frames", self.track_ttl_frames),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PersonZoneRuleError(f"{name} must be a positive integer")
        if self.restricted_pose_rule is not None:
            if not isinstance(
                self.restricted_pose_rule,
                PoseKeypointRuleConfig,
            ):
                raise PersonZoneRuleError(
                    "restricted_pose_rule must be PoseKeypointRuleConfig"
                )
            self.restricted_pose_rule.validate()
        if self.restricted_fence_rule is not None:
            if not isinstance(
                self.restricted_fence_rule,
                FenceBoundaryRuleConfig,
            ):
                raise PersonZoneRuleError(
                    "restricted_fence_rule must be FenceBoundaryRuleConfig"
                )
            self.restricted_fence_rule.validate()

    def debounce(self, area_type: AreaType) -> tuple[int, int]:
        if area_type == RESTRICTED_ZONE:
            return self.restricted_enter_frames, self.restricted_exit_frames
        return self.walkway_enter_frames, self.walkway_exit_frames


@dataclass
class _RuleState:
    active: bool = False
    observed_violation: bool | None = None
    streak: int = 0
    active_area_ids: tuple[str, ...] = ()

    def interrupt_streak(self) -> None:
        self.observed_violation = None
        self.streak = 0


@dataclass(frozen=True)
class _FenceHistorySample:
    frame_index: int
    hip_y: float
    person_height: float
    knee_y: tuple[tuple[str, float], ...]


@dataclass
class _FenceTrackState:
    state: FenceState = "clear"
    climb_active: bool = False
    climb_observed: bool | None = None
    climb_streak: int = 0
    history: list[_FenceHistorySample] = field(default_factory=list)

    def update_climb(
        self,
        observation: bool | None,
        *,
        enter_frames: int,
        exit_frames: int,
    ) -> None:
        if observation is None:
            return
        if self.climb_observed == observation:
            self.climb_streak += 1
        else:
            self.climb_observed = observation
            self.climb_streak = 1
        if not self.climb_active:
            if observation and self.climb_streak >= enter_frames:
                self.climb_active = True
            return
        if not observation and self.climb_streak >= exit_frames:
            self.climb_active = False


@dataclass
class _TrackState:
    track_id: int
    footpoint: Point
    missed_frames: int = 0
    rules: dict[AreaType, _RuleState] = field(
        default_factory=lambda: {
            RESTRICTED_ZONE: _RuleState(),
            SAFE_WALKWAY: _RuleState(),
        }
    )
    fence: _FenceTrackState = field(default_factory=_FenceTrackState)


class PersonZoneRuleEngine:
    """Evaluate tracked people against restricted and allowed-area unions."""

    def __init__(
        self,
        areas: Sequence[ZoneArea | Mapping[str, Any]],
        config: ZoneRuleConfig | None = None,
    ) -> None:
        if not isinstance(areas, Sequence) or isinstance(areas, (str, bytes)):
            raise PersonZoneRuleError("areas must be a sequence")
        normalized = tuple(
            area if isinstance(area, ZoneArea) else ZoneArea.from_mapping(area)
            for area in areas
        )
        if not normalized:
            raise PersonZoneRuleError("at least one safety area is required")
        area_ids = [area.area_id for area in normalized]
        if len(area_ids) != len(set(area_ids)):
            raise PersonZoneRuleError("area_id values must be unique")
        self.areas = normalized
        self.config = config or ZoneRuleConfig()
        self.config.validate()
        self._areas_by_type: dict[AreaType, tuple[ZoneArea, ...]] = {
            area_type: tuple(
                area for area in normalized if area.area_type == area_type
            )
            for area_type in (RESTRICTED_ZONE, SAFE_WALKWAY)
        }
        self._tracks: dict[int, _TrackState] = {}
        self._last_frame_index: int | None = None

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._last_frame_index = None

    @staticmethod
    def _track_id(person: Mapping[str, Any]) -> int:
        value = person.get("track_id")
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise PersonZoneRuleError(
                "each person needs a non-negative integer track_id"
            )
        return value

    @staticmethod
    def _footpoint(person: Mapping[str, Any]) -> Point:
        if "footpoint_norm_xy" in person:
            return _normalized_point(
                person["footpoint_norm_xy"],
                label="footpoint_norm_xy",
            )
        if "bbox_norm_xywh" not in person:
            raise PersonZoneRuleError(
                "each person needs bbox_norm_xywh or footpoint_norm_xy"
            )
        return bbox_bottom_center(person["bbox_norm_xywh"])

    def _raw_rule(
        self,
        area_type: AreaType,
        footpoint: Point,
    ) -> tuple[RawViolation, tuple[str, ...]]:
        areas = self._areas_by_type[area_type]
        if not areas:
            return False, ()
        containing = tuple(
            area.area_id
            for area in areas
            if point_in_polygon(footpoint, area.points)
        )
        if area_type == RESTRICTED_ZONE:
            return bool(containing), containing
        # Every walkway polygon contributes to the allowed union.  When the
        # person is outside that union, all configured walkway IDs explain the
        # violated rule.
        return not containing, (() if containing else tuple(a.area_id for a in areas))

    @staticmethod
    def _unknown_pose_evidence(
        config: PoseKeypointRuleConfig,
        *,
        reason: str,
        pose_score: float | None = None,
        visible_count: int = 0,
        inside_count: int = 0,
        inside_keypoint_names: Sequence[str] = (),
        contributing_area_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        selected_count = len(config.keypoint_names)
        return {
            "mode": "pose_keypoint_ratio",
            "observation_valid": False,
            "reason": reason,
            "selected_keypoint_names": list(config.keypoint_names),
            "selected_count": selected_count,
            "visible_count": visible_count,
            "inside_count": inside_count,
            "inside_ratio": round(inside_count / selected_count, 8),
            "inside_keypoint_names": list(inside_keypoint_names),
            "contributing_area_ids": list(contributing_area_ids),
            "minimum_confidence": float(config.minimum_confidence),
            "minimum_visible": config.minimum_visible,
            "minimum_inside_ratio": float(config.minimum_inside_ratio),
            "ratio_denominator": "configured_keypoints",
            "pose_score": (
                round(pose_score, 8) if pose_score is not None else None
            ),
        }

    def _restricted_pose_rule(
        self,
        person: Mapping[str, Any],
        config: PoseKeypointRuleConfig,
    ) -> tuple[RawViolation, tuple[str, ...], dict[str, Any]]:
        """Evaluate selected pose joints against the restricted-area union."""

        pose = person.get("pose")
        if not isinstance(pose, Mapping):
            evidence = self._unknown_pose_evidence(
                config,
                reason="pose_missing_or_ambiguous",
            )
            return None, (), evidence

        association_status = pose.get(
            "association_status",
            pose.get("link_status"),
        )
        if association_status is not None:
            matched = (
                association_status == 0
                and not isinstance(association_status, bool)
            ) or (
                isinstance(association_status, str)
                and association_status.strip().casefold() == "matched"
            )
            if not matched:
                evidence = self._unknown_pose_evidence(
                    config,
                    reason="pose_missing_or_ambiguous",
                )
                return None, (), evidence
        if pose.get("ambiguous") is True:
            evidence = self._unknown_pose_evidence(
                config,
                reason="pose_missing_or_ambiguous",
            )
            return None, (), evidence

        try:
            pose_score = _finite_number(
                pose.get("score"),
                label="pose.score",
            )
            if not 0.0 <= pose_score <= 1.0:
                raise PersonZoneRuleError("pose.score must be inside [0,1]")
            _normalized_bbox(
                pose.get("bbox_norm_xywh"),
                label="pose.bbox_norm_xywh",
            )
        except PersonZoneRuleError:
            evidence = self._unknown_pose_evidence(
                config,
                reason="pose_missing_or_ambiguous",
            )
            return None, (), evidence

        raw_keypoints = pose.get("keypoints")
        if (
            not isinstance(raw_keypoints, Sequence)
            or isinstance(raw_keypoints, (str, bytes))
        ):
            evidence = self._unknown_pose_evidence(
                config,
                reason="pose_missing_or_ambiguous",
                pose_score=pose_score,
            )
            return None, (), evidence

        selected = frozenset(config.keypoint_names)
        qualified: dict[str, Point] = {}
        seen_selected: set[str] = set()
        for raw_keypoint in raw_keypoints:
            if not isinstance(raw_keypoint, Mapping):
                continue
            name = raw_keypoint.get("name")
            if name not in selected:
                continue
            if name in seen_selected:
                evidence = self._unknown_pose_evidence(
                    config,
                    reason="pose_missing_or_ambiguous",
                    pose_score=pose_score,
                )
                return None, (), evidence
            seen_selected.add(name)
            if raw_keypoint.get("visible") is not True:
                continue
            try:
                confidence = _finite_number(
                    raw_keypoint.get("confidence"),
                    label=f"pose.keypoints.{name}.confidence",
                )
                if (
                    not 0.0 <= confidence <= 1.0
                    or confidence < config.minimum_confidence
                ):
                    continue
                point = _normalized_point(
                    [
                        raw_keypoint.get("x_norm"),
                        raw_keypoint.get("y_norm"),
                    ],
                    label=f"pose.keypoints.{name}",
                )
            except PersonZoneRuleError:
                continue
            qualified[str(name)] = point

        visible_count = len(qualified)
        inside_names: list[str] = []
        touched_area_ids: set[str] = set()
        areas = self._areas_by_type[RESTRICTED_ZONE]
        for name in config.keypoint_names:
            point = qualified.get(name)
            if point is None:
                continue
            containing = tuple(
                area.area_id
                for area in areas
                if point_in_polygon(point, area.points)
            )
            if not containing:
                continue
            inside_names.append(name)
            touched_area_ids.update(containing)

        inside_count = len(inside_names)
        ordered_area_ids = tuple(
            area.area_id
            for area in areas
            if area.area_id in touched_area_ids
        )
        if visible_count < config.minimum_visible:
            evidence = self._unknown_pose_evidence(
                config,
                reason="insufficient_visible_keypoints",
                pose_score=pose_score,
                visible_count=visible_count,
                inside_count=inside_count,
                inside_keypoint_names=inside_names,
                contributing_area_ids=ordered_area_ids,
            )
            return None, (), evidence

        inside_ratio = inside_count / len(config.keypoint_names)
        violation = inside_ratio >= config.minimum_inside_ratio
        evidence = {
            "mode": "pose_keypoint_ratio",
            "observation_valid": True,
            "reason": "ratio_evaluated",
            "selected_keypoint_names": list(config.keypoint_names),
            "selected_count": len(config.keypoint_names),
            "visible_count": visible_count,
            "inside_count": inside_count,
            "inside_ratio": round(inside_ratio, 8),
            "inside_keypoint_names": inside_names,
            "contributing_area_ids": list(ordered_area_ids),
            "minimum_confidence": float(config.minimum_confidence),
            "minimum_visible": config.minimum_visible,
            "minimum_inside_ratio": float(config.minimum_inside_ratio),
            "ratio_denominator": "configured_keypoints",
            "pose_score": round(pose_score, 8),
        }
        return (
            violation,
            ordered_area_ids if violation else (),
            evidence,
        )

    @staticmethod
    def _unknown_fence_evidence(
        config: FenceBoundaryRuleConfig,
        *,
        reason: str,
        pose_score: float | None = None,
        visible_keypoint_names: Sequence[str] = (),
    ) -> dict[str, Any]:
        return {
            "mode": "fence_boundary_state",
            "observation_valid": False,
            "reason": reason,
            "boundary_start": list(config.boundary_start),
            "boundary_end": list(config.boundary_end),
            "forbidden_side": config.forbidden_side,
            "contact_band": float(config.contact_band),
            "minimum_confidence": float(config.minimum_confidence),
            "visible_keypoint_names": list(visible_keypoint_names),
            "core_keypoint_names": [],
            "core_point_norm_xy": None,
            "signed_core_distance": None,
            "core_side": "unknown",
            "core_on_forbidden_side": None,
            "breach_entry_eligible": None,
            "breach_requires_climb": True,
            "contact_keypoint_names": [],
            "contact_count": 0,
            "approach_candidate": None,
            "wrist_visible_count": 0,
            "wrist_contact_names": [],
            "wrist_contact_count": 0,
            "tracked_person_height": None,
            "hip_baseline_y": None,
            "hip_upward_displacement_ratio": None,
            "hip_rise_threshold": float(config.hip_rise_ratio),
            "knee_visible_names": [],
            "knee_upward_displacement_ratio": None,
            "raised_knee": None,
            "raised_knee_threshold": float(config.raised_knee_ratio),
            "climb_observation_valid": False,
            "climb_candidate": None,
            "pose_score": (
                round(pose_score, 8) if pose_score is not None else None
            ),
        }

    def _fence_pose_points(
        self,
        person: Mapping[str, Any],
        config: FenceBoundaryRuleConfig,
    ) -> tuple[float, Bbox, dict[str, Point]] | tuple[None, None, dict[str, Point]]:
        """Return strict confidence-qualified pose points or an unknown marker."""

        pose = person.get("pose")
        if not isinstance(pose, Mapping):
            return None, None, {}
        association_status = pose.get(
            "association_status",
            pose.get("link_status"),
        )
        if association_status is not None:
            matched = (
                association_status == 0
                and not isinstance(association_status, bool)
            ) or (
                isinstance(association_status, str)
                and association_status.strip().casefold() == "matched"
            )
            if not matched:
                return None, None, {}
        if pose.get("ambiguous") is True:
            return None, None, {}
        try:
            pose_score = _finite_number(
                pose.get("score"),
                label="pose.score",
            )
            if not 0.0 <= pose_score <= 1.0:
                return None, None, {}
            pose_bbox = _normalized_bbox(
                pose.get("bbox_norm_xywh"),
                label="pose.bbox_norm_xywh",
            )
        except PersonZoneRuleError:
            return None, None, {}
        raw_keypoints = pose.get("keypoints")
        if (
            not isinstance(raw_keypoints, Sequence)
            or isinstance(raw_keypoints, (str, bytes))
        ):
            return None, None, {}

        needed_names = frozenset(
            (
                *config.approach_keypoint_names,
                "left_hip",
                "right_hip",
                "left_wrist",
                "right_wrist",
                "left_knee",
                "right_knee",
            )
        )
        points: dict[str, Point] = {}
        seen: set[str] = set()
        for raw_keypoint in raw_keypoints:
            if not isinstance(raw_keypoint, Mapping):
                continue
            name = raw_keypoint.get("name")
            if name not in needed_names:
                continue
            name = str(name)
            if name in seen:
                return None, None, {}
            seen.add(name)
            if raw_keypoint.get("visible") is not True:
                continue
            try:
                confidence = _finite_number(
                    raw_keypoint.get("confidence"),
                    label=f"pose.keypoints.{name}.confidence",
                )
                if (
                    not 0.0 <= confidence <= 1.0
                    or confidence < config.minimum_confidence
                ):
                    continue
                points[name] = _normalized_point(
                    (
                        raw_keypoint.get("x_norm"),
                        raw_keypoint.get("y_norm"),
                    ),
                    label=f"pose.keypoints.{name}",
                )
            except PersonZoneRuleError:
                continue
        return pose_score, pose_bbox, points

    @staticmethod
    def _tracked_person_height(
        person: Mapping[str, Any],
        pose_bbox: Bbox,
    ) -> float:
        if "bbox_norm_xywh" in person:
            try:
                return _normalized_bbox(
                    person["bbox_norm_xywh"],
                    label="bbox_norm_xywh",
                )[3]
            except PersonZoneRuleError:
                pass
        return pose_bbox[3]

    def _restricted_fence_rule(
        self,
        person: Mapping[str, Any],
        track: _TrackState,
        config: FenceBoundaryRuleConfig,
        *,
        frame_index: int,
    ) -> tuple[RawViolation, tuple[str, ...], dict[str, Any]]:
        """Evaluate one tracked pose against the strict fence boundary."""

        pose_score, pose_bbox, qualified = self._fence_pose_points(
            person,
            config,
        )
        if pose_score is None or pose_bbox is None:
            return (
                None,
                (),
                self._unknown_fence_evidence(
                    config,
                    reason="pose_missing_or_ambiguous",
                ),
            )

        hip_names = tuple(
            name
            for name in ("left_hip", "right_hip")
            if name in qualified
        )
        if len(hip_names) < config.minimum_core_visible:
            return (
                None,
                (),
                self._unknown_fence_evidence(
                    config,
                    reason="insufficient_visible_core_keypoints",
                    pose_score=pose_score,
                    visible_keypoint_names=tuple(qualified),
                ),
            )
        core = (
            sum(qualified[name][0] for name in hip_names) / len(hip_names),
            sum(qualified[name][1] for name in hip_names) / len(hip_names),
        )
        signed_distance = _signed_distance_to_oriented_line(
            core,
            config.boundary_start,
            config.boundary_end,
        )
        epsilon = 1.0e-9
        core_side = (
            "boundary"
            if abs(signed_distance) <= epsilon
            else ("left" if signed_distance > 0.0 else "right")
        )
        if config.forbidden_side == "left":
            core_forbidden = signed_distance >= -epsilon
        else:
            core_forbidden = signed_distance <= epsilon

        contact_names = tuple(
            name
            for name in config.approach_keypoint_names
            if name in qualified
            and _distance_to_segment(
                qualified[name],
                config.boundary_start,
                config.boundary_end,
            )
            <= float(config.contact_band) + epsilon
        )
        approach_candidate = (
            len(contact_names) >= config.approach_minimum_count
        )
        wrist_names = ("left_wrist", "right_wrist")
        visible_wrists = tuple(
            name for name in wrist_names if name in qualified
        )
        wrist_contacts = tuple(
            name
            for name in visible_wrists
            if _distance_to_segment(
                qualified[name],
                config.boundary_start,
                config.boundary_end,
            )
            <= float(config.contact_band) + epsilon
        )

        person_height = self._tracked_person_height(person, pose_bbox)
        minimum_history_frame = frame_index - config.history_window_frames
        track.fence.history[:] = [
            sample
            for sample in track.fence.history
            if sample.frame_index >= minimum_history_frame
        ]
        hip_baseline_y = (
            max(sample.hip_y for sample in track.fence.history)
            if track.fence.history
            else None
        )
        hip_rise_ratio = (
            max(0.0, (hip_baseline_y - core[1]) / person_height)
            if hip_baseline_y is not None
            else 0.0
        )

        knee_names = tuple(
            name
            for name in ("left_knee", "right_knee")
            if name in qualified
        )
        knee_rise_by_name: dict[str, float] = {}
        for name in knee_names:
            prior_y = [
                dict(sample.knee_y)[name]
                for sample in track.fence.history
                if name in dict(sample.knee_y)
            ]
            if prior_y:
                knee_rise_by_name[name] = max(
                    0.0,
                    (max(prior_y) - qualified[name][1]) / person_height,
                )
        knee_rise_ratio = (
            max(knee_rise_by_name.values())
            if knee_rise_by_name
            else 0.0
        )
        # Knees are a confirming cue when visible.  If the crop/pose does not
        # expose a knee, wrists plus tracked hip lift remain sufficient.
        raised_knee: bool | None = (
            knee_rise_ratio >= config.raised_knee_ratio
            if knee_names
            else None
        )
        knee_gate = raised_knee is not False
        climb_observation_valid = (
            len(visible_wrists) >= config.wrist_contact_required
        )
        climb_candidate: bool | None
        if not climb_observation_valid:
            climb_candidate = None
        else:
            climb_candidate = (
                len(wrist_contacts) >= config.wrist_contact_required
                and hip_rise_ratio >= config.hip_rise_ratio
                and knee_gate
            )
        track.fence.update_climb(
            climb_candidate,
            enter_frames=config.climb_enter_frames,
            exit_frames=config.climb_exit_frames,
        )
        track.fence.history.append(
            _FenceHistorySample(
                frame_index=frame_index,
                hip_y=core[1],
                person_height=person_height,
                knee_y=tuple(
                    (name, qualified[name][1]) for name in knee_names
                ),
            )
        )

        restricted_state = track.rules[RESTRICTED_ZONE]
        breach_entry_eligible = (
            track.fence.climb_active or restricted_state.active
        )
        # Perspective can project the hip core across a top rail while a
        # person is still standing safely in front of it.  A new breach may
        # therefore only enter from a climb-qualified state.  Once opened, the
        # active rule itself keeps the breach held while the core remains on
        # the forbidden side, even though climbing is no longer the displayed
        # state.
        raw_breach = core_forbidden and breach_entry_eligible
        restricted_area_ids = tuple(
            area.area_id
            for area in self._areas_by_type[RESTRICTED_ZONE]
        )
        evidence = {
            "mode": "fence_boundary_state",
            "observation_valid": True,
            "reason": "boundary_core_evaluated",
            "boundary_start": list(config.boundary_start),
            "boundary_end": list(config.boundary_end),
            "forbidden_side": config.forbidden_side,
            "contact_band": float(config.contact_band),
            "minimum_confidence": float(config.minimum_confidence),
            "visible_keypoint_names": list(qualified),
            "core_keypoint_names": list(hip_names),
            "core_point_norm_xy": [
                round(core[0], 8),
                round(core[1], 8),
            ],
            "signed_core_distance": round(signed_distance, 8),
            "core_side": core_side,
            "core_on_forbidden_side": core_forbidden,
            "breach_entry_eligible": breach_entry_eligible,
            "breach_requires_climb": True,
            "contact_keypoint_names": list(contact_names),
            "contact_count": len(contact_names),
            "approach_candidate": approach_candidate,
            "wrist_visible_count": len(visible_wrists),
            "wrist_contact_names": list(wrist_contacts),
            "wrist_contact_count": len(wrist_contacts),
            "tracked_person_height": round(person_height, 8),
            "hip_baseline_y": (
                round(hip_baseline_y, 8)
                if hip_baseline_y is not None
                else None
            ),
            "hip_upward_displacement_ratio": round(hip_rise_ratio, 8),
            "hip_rise_threshold": float(config.hip_rise_ratio),
            "knee_visible_names": list(knee_names),
            "knee_upward_displacement_ratio": round(
                knee_rise_ratio,
                8,
            ),
            "raised_knee": raised_knee,
            "raised_knee_threshold": float(config.raised_knee_ratio),
            "climb_observation_valid": climb_observation_valid,
            "climb_candidate": climb_candidate,
            "pose_score": round(pose_score, 8),
        }
        return (
            raw_breach,
            restricted_area_ids if raw_breach else (),
            evidence,
        )

    def _event(
        self,
        *,
        kind: Literal["started", "ended"],
        area_type: AreaType,
        track: _TrackState,
        frame_index: int,
        area_ids: Sequence[str],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "kind": kind,
            "type": _EVENT_TYPES[area_type],
            "track_id": track.track_id,
            "frame_index": frame_index,
            "footpoint_norm_xy": [
                round(track.footpoint[0], 8),
                round(track.footpoint[1], 8),
            ],
            "contributing_area_ids": list(area_ids),
            "reason": reason,
        }

    @staticmethod
    def _fence_event(
        *,
        kind: Literal["started", "ended"],
        event_type: Literal["fence_approach", "fence_climb_attempt"],
        track_id: int,
        frame_index: int,
        from_state: FenceState,
        to_state: FenceState,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": FENCE_EVENT_SCHEMA_VERSION,
            "kind": kind,
            "type": event_type,
            "track_id": track_id,
            "frame_index": frame_index,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
        }

    def _set_fence_state(
        self,
        *,
        track: _TrackState,
        new_state: FenceState,
        frame_index: int,
        reason: str,
    ) -> list[dict[str, Any]]:
        if new_state not in _FENCE_STATES:
            raise PersonZoneRuleError("invalid internal fence state")
        previous = track.fence.state
        if previous == new_state:
            return []
        events: list[dict[str, Any]] = []
        if previous == "approach":
            events.append(
                self._fence_event(
                    kind="ended",
                    event_type="fence_approach",
                    track_id=track.track_id,
                    frame_index=frame_index,
                    from_state=previous,
                    to_state=new_state,
                    reason=reason,
                )
            )
        elif previous == "climbing":
            events.append(
                self._fence_event(
                    kind="ended",
                    event_type="fence_climb_attempt",
                    track_id=track.track_id,
                    frame_index=frame_index,
                    from_state=previous,
                    to_state=new_state,
                    reason=reason,
                )
            )
        if new_state == "approach":
            events.append(
                self._fence_event(
                    kind="started",
                    event_type="fence_approach",
                    track_id=track.track_id,
                    frame_index=frame_index,
                    from_state=previous,
                    to_state=new_state,
                    reason=reason,
                )
            )
        elif new_state == "climbing":
            events.append(
                self._fence_event(
                    kind="started",
                    event_type="fence_climb_attempt",
                    track_id=track.track_id,
                    frame_index=frame_index,
                    from_state=previous,
                    to_state=new_state,
                    reason=reason,
                )
            )
        track.fence.state = new_state
        return events

    def _update_rule(
        self,
        *,
        area_type: AreaType,
        track: _TrackState,
        raw_violation: RawViolation,
        raw_area_ids: tuple[str, ...],
        frame_index: int,
        enter_frames: int | None = None,
        exit_frames: int | None = None,
        enter_reason: str = "debounce_qualified",
        exit_reason: str = "debounce_cleared",
    ) -> dict[str, Any] | None:
        state = track.rules[area_type]
        if raw_violation is None:
            # Unknown pose evidence pauses both entry and exit qualification.
            # The previous observation/streak and any active state are retained.
            return None
        if state.observed_violation == raw_violation:
            state.streak += 1
        else:
            state.observed_violation = raw_violation
            state.streak = 1
        default_enter, default_exit = self.config.debounce(area_type)
        enter_frames = enter_frames or default_enter
        exit_frames = exit_frames or default_exit

        if not state.active:
            if raw_violation and state.streak >= enter_frames:
                state.active = True
                state.active_area_ids = raw_area_ids
                return self._event(
                    kind="started",
                    area_type=area_type,
                    track=track,
                    frame_index=frame_index,
                    area_ids=state.active_area_ids,
                    reason=enter_reason,
                )
            return None

        if raw_violation:
            state.active_area_ids = raw_area_ids
            return None
        if state.streak < exit_frames:
            return None
        previous_area_ids = state.active_area_ids
        state.active = False
        state.active_area_ids = ()
        return self._event(
            kind="ended",
            area_type=area_type,
            track=track,
            frame_index=frame_index,
            area_ids=previous_area_ids,
            reason=exit_reason,
        )

    def _expire_tracks(
        self,
        *,
        frame_index: int,
        seen_track_ids: set[int],
        elapsed_frames: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        fence_events: list[dict[str, Any]] = []
        expired: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in seen_track_ids:
                continue
            track.missed_frames += elapsed_frames
            for state in track.rules.values():
                state.interrupt_streak()
            if track.missed_frames <= self.config.track_ttl_frames:
                continue
            for area_type in (RESTRICTED_ZONE, SAFE_WALKWAY):
                state = track.rules[area_type]
                if not state.active:
                    continue
                events.append(
                    self._event(
                        kind="ended",
                        area_type=area_type,
                        track=track,
                        frame_index=frame_index,
                        area_ids=state.active_area_ids,
                        reason="track_expired",
                    )
                )
            if track.fence.state in {"approach", "climbing"}:
                event_type = (
                    "fence_approach"
                    if track.fence.state == "approach"
                    else "fence_climb_attempt"
                )
                fence_events.append(
                    self._fence_event(
                        kind="ended",
                        event_type=event_type,
                        track_id=track.track_id,
                        frame_index=frame_index,
                        from_state=track.fence.state,
                        to_state="clear",
                        reason="track_expired",
                    )
                )
            expired.append(track_id)
        for track_id in expired:
            del self._tracks[track_id]
        return events, fence_events

    def process_frame(
        self,
        *,
        frame_index: int,
        persons: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return full-frame people augmented with zone state and transitions."""

        if (
            not isinstance(frame_index, int)
            or isinstance(frame_index, bool)
            or frame_index < 0
        ):
            raise PersonZoneRuleError("frame_index must be non-negative")
        if (
            self._last_frame_index is not None
            and frame_index <= self._last_frame_index
        ):
            raise PersonZoneRuleError("frame_index must be strictly increasing")
        if not isinstance(persons, Sequence) or isinstance(persons, (str, bytes)):
            raise PersonZoneRuleError("persons must be a sequence")

        elapsed_frames = (
            1
            if self._last_frame_index is None
            else frame_index - self._last_frame_index
        )
        normalized_people: list[tuple[Mapping[str, Any], int, Point]] = []
        seen_track_ids: set[int] = set()
        for person in persons:
            if not isinstance(person, Mapping):
                raise PersonZoneRuleError("each person must be an object")
            track_id = self._track_id(person)
            if track_id in seen_track_ids:
                raise PersonZoneRuleError(
                    f"duplicate track_id inside frame: {track_id}"
                )
            seen_track_ids.add(track_id)
            normalized_people.append((person, track_id, self._footpoint(person)))

        events, fence_events = self._expire_tracks(
            frame_index=frame_index,
            seen_track_ids=seen_track_ids,
            elapsed_frames=elapsed_frames,
        )
        output_people: list[dict[str, Any]] = []
        for person, track_id, footpoint in normalized_people:
            zone_alert_eligible = person.get("zone_alert_eligible", True)
            if not isinstance(zone_alert_eligible, bool):
                raise PersonZoneRuleError(
                    "zone_alert_eligible must be boolean when provided"
                )
            track = self._tracks.get(track_id)
            if track is None:
                track = _TrackState(track_id=track_id, footpoint=footpoint)
                self._tracks[track_id] = track
            track.footpoint = footpoint
            track.missed_frames = 0

            rule_outputs: dict[str, dict[str, Any]] = {}
            anchor_evidence: dict[str, Any] | None = None
            fence_evidence: dict[str, Any] | None = None
            for area_type in (RESTRICTED_ZONE, SAFE_WALKWAY):
                if not zone_alert_eligible:
                    # Keep the tracked person in the full-frame output while
                    # treating an externally qualified vehicle operator as
                    # alert-ineligible. A valid non-violation observation
                    # deliberately advances normal exit debounce, so an alarm
                    # opened before driver qualification closes cleanly.
                    raw_violation, raw_area_ids = False, ()
                    if (
                        area_type == RESTRICTED_ZONE
                        and (
                            self.config.restricted_pose_rule is not None
                            or self.config.restricted_fence_rule is not None
                        )
                    ):
                        anchor_evidence = {
                            "mode": "external_alert_eligibility",
                            "observation_valid": True,
                            "reason": "zone_alert_suppressed",
                        }
                        if self.config.restricted_fence_rule is not None:
                            fence_evidence = dict(anchor_evidence)
                elif (
                    area_type == RESTRICTED_ZONE
                    and self._areas_by_type[RESTRICTED_ZONE]
                    and self.config.restricted_fence_rule is not None
                ):
                    (
                        raw_violation,
                        raw_area_ids,
                        fence_evidence,
                    ) = self._restricted_fence_rule(
                        person,
                        track,
                        self.config.restricted_fence_rule,
                        frame_index=frame_index,
                    )
                    anchor_evidence = fence_evidence
                elif (
                    area_type == RESTRICTED_ZONE
                    and self._areas_by_type[RESTRICTED_ZONE]
                    and self.config.restricted_pose_rule is not None
                ):
                    (
                        raw_violation,
                        raw_area_ids,
                        anchor_evidence,
                    ) = self._restricted_pose_rule(
                        person,
                        self.config.restricted_pose_rule,
                    )
                else:
                    raw_violation, raw_area_ids = self._raw_rule(
                        area_type,
                        footpoint,
                    )
                state = track.rules[area_type]
                if not zone_alert_eligible and state.active:
                    previous_area_ids = state.active_area_ids
                    state.active = False
                    state.active_area_ids = ()
                    state.observed_violation = False
                    state.streak = 1
                    event = self._event(
                        kind="ended",
                        area_type=area_type,
                        track=track,
                        frame_index=frame_index,
                        area_ids=previous_area_ids,
                        reason="external_alert_suppressed",
                    )
                else:
                    update_options: dict[str, Any] = {}
                    if (
                        area_type == RESTRICTED_ZONE
                        and self.config.restricted_fence_rule is not None
                    ):
                        update_options = {
                            "enter_frames": (
                                self.config.restricted_fence_rule
                                .breach_enter_frames
                            ),
                            "exit_frames": (
                                self.config.restricted_fence_rule
                                .breach_exit_frames
                            ),
                            "enter_reason": "core_crossing_qualified",
                            "exit_reason": "core_crossing_cleared",
                        }
                    event = self._update_rule(
                        area_type=area_type,
                        track=track,
                        raw_violation=raw_violation,
                        raw_area_ids=raw_area_ids,
                        frame_index=frame_index,
                        **update_options,
                    )
                if event is not None:
                    events.append(event)
                state = track.rules[area_type]
                rule_outputs[area_type] = {
                    "configured": bool(self._areas_by_type[area_type]),
                    "raw_violation": raw_violation,
                    "raw_contributing_area_ids": list(raw_area_ids),
                    "alert_eligible": zone_alert_eligible,
                    "violation_active": state.active,
                    "contributing_area_ids": list(state.active_area_ids),
                    "streak": state.streak,
                }

            fence_config = self.config.restricted_fence_rule
            if fence_config is not None and self._areas_by_type[RESTRICTED_ZONE]:
                if not zone_alert_eligible:
                    track.fence.climb_active = False
                    track.fence.climb_observed = False
                    track.fence.climb_streak = 1
                    fence_events.extend(
                        self._set_fence_state(
                            track=track,
                            new_state="clear",
                            frame_index=frame_index,
                            reason="external_alert_suppressed",
                        )
                    )
                elif (
                    fence_evidence is not None
                    and fence_evidence.get("observation_valid") is True
                ):
                    restricted_active = track.rules[RESTRICTED_ZONE].active
                    if restricted_active:
                        track.fence.climb_active = False
                        track.fence.climb_observed = None
                        track.fence.climb_streak = 0
                        next_fence_state: FenceState = "breach"
                    elif track.fence.climb_active:
                        next_fence_state = "climbing"
                    elif fence_evidence.get("approach_candidate") is True:
                        next_fence_state = "approach"
                    else:
                        next_fence_state = "clear"
                    fence_events.extend(
                        self._set_fence_state(
                            track=track,
                            new_state=next_fence_state,
                            frame_index=frame_index,
                            reason="fence_evidence_updated",
                        )
                    )
                # Unknown pose observations deliberately retain stage, climb
                # debounce, and critical entry/exit streaks.
                if fence_evidence is not None:
                    fence_evidence["fence_state"] = track.fence.state
                    fence_evidence["climb_active"] = (
                        track.fence.climb_active
                    )
                    fence_evidence["climb_streak"] = track.fence.climb_streak
                    restricted_state = track.rules[RESTRICTED_ZONE]
                    fence_evidence["breach_active"] = restricted_state.active
                    fence_evidence["breach_observation"] = (
                        restricted_state.observed_violation
                    )
                    fence_evidence["breach_streak"] = restricted_state.streak
                    fence_evidence["breach_enter_frames"] = (
                        fence_config.breach_enter_frames
                    )
                    fence_evidence["breach_exit_frames"] = (
                        fence_config.breach_exit_frames
                    )

            active_types = [
                _EVENT_TYPES[area_type]
                for area_type in (RESTRICTED_ZONE, SAFE_WALKWAY)
                if track.rules[area_type].active
            ]
            contributing_area_ids = list(
                dict.fromkeys(
                    area_id
                    for area_type in (RESTRICTED_ZONE, SAFE_WALKWAY)
                    for area_id in track.rules[area_type].active_area_ids
                )
            )
            output_person = dict(person)
            zone_safety = {
                "footpoint_norm_xy": [
                    round(footpoint[0], 8),
                    round(footpoint[1], 8),
                ],
                "violation_active": bool(active_types),
                "active_violation_types": active_types,
                "contributing_area_ids": contributing_area_ids,
                "rules": rule_outputs,
            }
            if anchor_evidence is not None:
                zone_safety["anchor_evidence"] = anchor_evidence
            if fence_evidence is not None:
                zone_safety["fence_state"] = track.fence.state
                zone_safety["fence_evidence"] = fence_evidence
            output_person["zone_safety"] = zone_safety
            output_people.append(output_person)

        self._last_frame_index = frame_index
        result = {
            "schema_version": SCHEMA_VERSION,
            "frame_index": frame_index,
            "persons": output_people,
            "safety_events": events,
            "person_visibility_policy": "full_frame_no_zone_filter",
        }
        if self.config.restricted_fence_rule is not None:
            result["fence_events"] = fence_events
        return result


__all__ = [
    "AreaType",
    "COCO17_KEYPOINT_NAMES",
    "EVENT_SCHEMA_VERSION",
    "FENCE_EVENT_SCHEMA_VERSION",
    "FenceBoundaryRuleConfig",
    "FenceState",
    "PersonZoneRuleEngine",
    "PersonZoneRuleError",
    "PoseKeypointRuleConfig",
    "RESTRICTED_ZONE",
    "SAFE_WALKWAY",
    "SCHEMA_VERSION",
    "ZoneArea",
    "ZoneRuleConfig",
    "bbox_bottom_center",
    "point_in_polygon",
]
