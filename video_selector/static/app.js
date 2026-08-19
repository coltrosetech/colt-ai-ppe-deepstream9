(() => {
  "use strict";

  const STORAGE_KEY = "colt-ai-video-roi-draft-v2";
  const LEGACY_STORAGE_KEY = "colt-ai-video-roi-draft-v1";
  const COCO17_KEYPOINTS = Object.freeze([
    { id: "nose", label: "Burun", group: "Yüz" },
    { id: "left_eye", label: "Sol göz", group: "Yüz" },
    { id: "right_eye", label: "Sağ göz", group: "Yüz" },
    { id: "left_ear", label: "Sol kulak", group: "Yüz" },
    { id: "right_ear", label: "Sağ kulak", group: "Yüz" },
    { id: "left_shoulder", label: "Sol omuz", group: "Gövde" },
    { id: "right_shoulder", label: "Sağ omuz", group: "Gövde" },
    { id: "left_elbow", label: "Sol dirsek", group: "Kollar" },
    { id: "right_elbow", label: "Sağ dirsek", group: "Kollar" },
    { id: "left_wrist", label: "Sol bilek", group: "Kollar" },
    { id: "right_wrist", label: "Sağ bilek", group: "Kollar" },
    { id: "left_hip", label: "Sol kalça", group: "Gövde" },
    { id: "right_hip", label: "Sağ kalça", group: "Gövde" },
    { id: "left_knee", label: "Sol diz", group: "Bacaklar" },
    { id: "right_knee", label: "Sağ diz", group: "Bacaklar" },
    { id: "left_ankle", label: "Sol ayak bileği", group: "Bacaklar" },
    { id: "right_ankle", label: "Sağ ayak bileği", group: "Bacaklar" },
  ]);
  const COCO17_KEYPOINT_IDS = new Set(
    COCO17_KEYPOINTS.map((keypoint) => keypoint.id),
  );
  const DEFAULT_FENCE_KEYPOINTS = Object.freeze([
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
  ]);
  const DEFAULT_APPROACH_KEYPOINTS = Object.freeze([
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
  ]);
  const FORKLIFT_ALERT_ORDER = Object.freeze([
    "ppe_violation",
    "safe_walkway_violation",
  ]);
  const SCENARIOS = Object.freeze({
    person_office: Object.freeze({
      scenario: "fence_security",
      roiType: "restricted_zone",
      tabLabel: "Çit Güvenliği",
      badge: "ÇİT GÜVENLİĞİ",
      pipelineLabel: "İnsan + Alan İhlali",
      ruleTitle: "Yasak alana giriş kuralı",
      ruleText:
        "Seçilen iskelet noktalarının belirlenen yüzdesi kırmızı alana girdiğinde uyarı oluşturulur.",
      roiListTitle: "Yasak alanlar",
      roiTypeLabel: "YASAK ALAN",
      addLabel: "Yasak alan çiz",
      fullFrameLabel: "Tüm kareyi yasak alan yap",
      defaultName: "Yasak Alan",
      emptyStatus:
        "“Yasak alan çiz” ile başla ve girişte uyarı verecek bölgenin köşelerine tıkla.",
      drawingNoun: "yasak alan",
      instructionSubject: "Yasak alanın",
      minimumRois: 1,
      color: "#ff7087",
      fill: "rgba(255, 112, 135, 0.12)",
      activeFill: "rgba(255, 112, 135, 0.2)",
    }),
    ppe_safety: Object.freeze({
      scenario: "ppe_safety",
      roiType: "safe_walkway",
      tabLabel: "İSG / PPE",
      badge: "İSG / PPE",
      pipelineLabel: "İnsan + PPE + Yürüyüş Yolu",
      ruleTitle: "Kişi bazlı PPE ve güvenli yürüyüş kuralı",
      ruleText:
        "Kask ve yelek kontrolü tüm karede çalışır; doğrulanmış forklift sürücülerinin seçili alarmları bastırılır.",
      roiListTitle: "Güvenli yürüyüş yolları",
      roiTypeLabel: "YÜRÜYÜŞ YOLU",
      addLabel: "Yürüyüş yolu çiz",
      fullFrameLabel: "Tüm kareyi yürüyüş yolu yap",
      defaultName: "Yürüyüş Yolu",
      emptyStatus:
        "PPE kontrolü tüm karede aktiftir. İstersen “Yürüyüş yolu çiz” ile yol dışı yürüme alarmını etkinleştir.",
      drawingNoun: "güvenli yürüyüş yolu",
      instructionSubject: "Güvenli yürüyüş yolunun",
      minimumRois: 0,
      color: "#58dda6",
      fill: "rgba(88, 221, 166, 0.1)",
      activeFill: "rgba(88, 221, 166, 0.18)",
    }),
  });
  const els = Object.fromEntries(
    [
      "readyCount",
      "progressFill",
      "selectionCount",
      "catalogStatus",
      "categoryFilters",
      "personCategoryCount",
      "ppeCategoryCount",
      "videoCatalog",
      "emptyWorkspace",
      "editor",
      "activeVideoTitle",
      "activeVideoMeta",
      "activeVideoState",
      "scenarioRuleCard",
      "scenarioRuleBadge",
      "scenarioRuleTitle",
      "scenarioRuleText",
      "analysisOptionsPanel",
      "fenceAnalysisOptions",
      "ppeAnalysisOptions",
      "fenceCrossingEnabled",
      "fenceCrossingControls",
      "fenceLineSummary",
      "drawFenceLineButton",
      "deleteFenceLineButton",
      "fenceForbiddenSide",
      "fenceForbiddenLeft",
      "fenceForbiddenRight",
      "poseZoneEnabled",
      "poseControls",
      "poseInsidePercent",
      "poseInsidePercentNumber",
      "poseKeypointGrid",
      "poseRuleSummary",
      "poseEvidenceSummary",
      "forkliftSuppressionEnabled",
      "forkliftControls",
      "forkliftPpeViolation",
      "forkliftWalkwayViolation",
      "forkliftConfidence",
      "forkliftPersonIoa",
      "forkliftEnterFrames",
      "forkliftExitFrames",
      "forkliftRuleSummary",
      "videoStage",
      "previewVideo",
      "roiCanvas",
      "stageMessage",
      "playButton",
      "backButton",
      "forwardButton",
      "playhead",
      "timeReadout",
      "canvasInstructions",
      "addRoiButton",
      "addRoiLabel",
      "finishRoiButton",
      "undoButton",
      "fullFrameButton",
      "fullFrameLabel",
      "deleteRoiButton",
      "canvasStatus",
      "roiCount",
      "roiListTitle",
      "roiList",
      "vertexDetails",
      "vertexSummary",
      "vertexEditor",
      "clipSummary",
      "clipStart",
      "clipStartRange",
      "clipEnd",
      "clipEndRange",
      "useCurrentStart",
      "useCurrentEnd",
      "clipDuration",
      "resetClipButton",
      "saveHint",
      "saveButton",
      "globalMessage",
    ].map((id) => [id, document.getElementById(id)]),
  );

  const state = {
    catalog: [],
    byId: new Map(),
    catalogRevision: "",
    catalogRevisions: {},
    activeCategory: "person_office",
    selected: new Set(),
    drafts: {},
    activeVideoId: null,
    activeRoiId: null,
    drawingFenceLine: false,
    dragging: null,
    localDraftFound: false,
    saveTimer: null,
    messageTimer: null,
  };

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const clamp = (value, minimum, maximum) =>
    Math.min(maximum, Math.max(minimum, Number(value) || 0));

  const finiteOr = (value, fallback) => {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  };

  const round = (value, digits = 6) => {
    const factor = 10 ** digits;
    return Math.round((Number(value) + Number.EPSILON) * factor) / factor;
  };

  const formatTime = (value, milliseconds = false) => {
    const seconds = Math.max(0, Number(value) || 0);
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds - minutes * 60;
    return milliseconds
      ? `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`
      : `${String(minutes).padStart(2, "0")}:${String(Math.floor(remainder)).padStart(2, "0")}`;
  };

  const currentVideo = () => state.byId.get(state.activeVideoId) || null;
  const currentDraft = () =>
    state.activeVideoId ? state.drafts[state.activeVideoId] || null : null;
  const currentRoi = () => {
    const draft = currentDraft();
    return draft?.rois.find((roi) => roi.roi_id === state.activeRoiId) || null;
  };

  const scenarioForVideo = (video = currentVideo()) =>
    SCENARIOS[video?.category] || null;

  const scenarioForVideoId = (videoId) =>
    scenarioForVideo(state.byId.get(videoId));

  const scenarioForRoi = (roi) =>
    Object.values(SCENARIOS).find(
      (scenario) => scenario.roiType === roi?.roi_type,
    ) || scenarioForVideo();

  const isNewUserVideo = (video) =>
    Boolean(
      video &&
        ((video.category === "person_office" &&
          /^F\d+$/i.test(video.video_id)) ||
          (video.category === "ppe_safety" &&
            /^S\d+$/i.test(video.video_id))),
    );

  const frameStep = (video = currentVideo()) =>
    video ? 1 / Math.max(1, Number(video.fps) || 25) : 0.04;

  const normalizedPoint = (value) => {
    if (
      !value ||
      typeof value !== "object" ||
      Array.isArray(value) ||
      !Number.isFinite(Number(value.x)) ||
      !Number.isFinite(Number(value.y))
    ) {
      return null;
    }
    return {
      x: clamp(value.x, 0, 1),
      y: clamp(value.y, 0, 1),
    };
  };

  function deriveFenceBoundary(rois) {
    let longest = null;
    (Array.isArray(rois) ? rois : []).forEach((roi) => {
      const points = Array.isArray(roi?.points) ? roi.points : [];
      if (!roi?.closed || points.length < 3) return;
      points.forEach((start, index) => {
        const end = points[(index + 1) % points.length];
        const length = Math.hypot(end.x - start.x, end.y - start.y);
        if (!longest || length > longest.length) {
          longest = { start: { ...start }, end: { ...end }, points, length };
        }
      });
    });
    if (!longest || longest.length < 0.01) return null;
    const centroid = longest.points.reduce(
      (result, point) => ({
        x: result.x + point.x / longest.points.length,
        y: result.y + point.y / longest.points.length,
      }),
      { x: 0, y: 0 },
    );
    return {
      boundary_start: longest.start,
      boundary_end: longest.end,
      forbidden_side:
        orientation(longest.start, longest.end, centroid) >= 0
          ? "left"
          : "right",
    };
  }

  function normalizeAnalysisOptions(draft, scenario) {
    const source =
      draft.analysis_options &&
      typeof draft.analysis_options === "object" &&
      !Array.isArray(draft.analysis_options)
        ? draft.analysis_options
        : {};
    if (scenario.scenario === "fence_security") {
      const rawPose =
        source.fence_pose_roi &&
        typeof source.fence_pose_roi === "object" &&
        !Array.isArray(source.fence_pose_roi)
          ? source.fence_pose_roi
          : {};
      const selectedSource = Array.isArray(rawPose.selected_keypoints)
        ? rawPose.selected_keypoints
        : DEFAULT_FENCE_KEYPOINTS;
      const selected = [
        ...new Set(
          selectedSource
            .map((value) => String(value))
            .filter((value) => COCO17_KEYPOINT_IDS.has(value)),
        ),
      ];
      const minimumVisible = Math.max(
        1,
        Math.min(
          selected.length || 1,
          Math.round(finiteOr(rawPose.minimum_visible_keypoints, 4)),
        ),
      );
      const hasCrossingSource =
        source.fence_crossing_rule &&
        typeof source.fence_crossing_rule === "object" &&
        !Array.isArray(source.fence_crossing_rule);
      const rawCrossing = hasCrossingSource
        ? source.fence_crossing_rule
        : {};
      const derived = hasCrossingSource
        ? null
        : deriveFenceBoundary(draft.rois);
      const boundaryStart = normalizedPoint(
        rawCrossing.boundary_start || derived?.boundary_start,
      );
      const boundaryEnd = normalizedPoint(
        rawCrossing.boundary_end || derived?.boundary_end,
      );
      const approachSource = Array.isArray(
        rawCrossing.approach_keypoint_names,
      )
        ? rawCrossing.approach_keypoint_names
        : DEFAULT_APPROACH_KEYPOINTS;
      const approachKeypoints = [
        ...new Set(
          approachSource
            .map((value) => String(value))
            .filter((value) => COCO17_KEYPOINT_IDS.has(value)),
        ),
      ];
      draft.analysis_options = {
        fence_pose_roi: {
          enabled: rawPose.enabled !== false,
          selected_keypoints: selected,
          inside_ratio_threshold: clamp(
            finiteOr(rawPose.inside_ratio_threshold, 0.5),
            0.1,
            1,
          ),
          keypoint_confidence_threshold: clamp(
            finiteOr(rawPose.keypoint_confidence_threshold, 0.25),
            0,
            1,
          ),
          minimum_visible_keypoints: minimumVisible,
        },
        fence_crossing_rule: {
          enabled: rawCrossing.enabled !== false,
          boundary_start: boundaryStart,
          boundary_end: boundaryEnd,
          forbidden_side:
            rawCrossing.forbidden_side === "right" ||
            derived?.forbidden_side === "right"
              ? "right"
              : "left",
          contact_band: clamp(
            finiteOr(rawCrossing.contact_band, 0.03),
            0.001,
            0.25,
          ),
          minimum_confidence: clamp(
            finiteOr(rawCrossing.minimum_confidence, 0.3),
            0,
            1,
          ),
          minimum_core_visible: Math.round(
            clamp(finiteOr(rawCrossing.minimum_core_visible, 1), 1, 2),
          ),
          breach_enter_frames: Math.round(
            clamp(finiteOr(rawCrossing.breach_enter_frames, 4), 1, 30),
          ),
          breach_exit_frames: Math.round(
            clamp(finiteOr(rawCrossing.breach_exit_frames, 4), 1, 60),
          ),
          approach_keypoint_names: approachKeypoints.length
            ? approachKeypoints
            : [...DEFAULT_APPROACH_KEYPOINTS],
          approach_minimum_count: Math.round(
            clamp(
              finiteOr(rawCrossing.approach_minimum_count, 1),
              1,
              approachKeypoints.length || DEFAULT_APPROACH_KEYPOINTS.length,
            ),
          ),
          wrist_contact_required: Math.round(
            clamp(finiteOr(rawCrossing.wrist_contact_required, 1), 1, 2),
          ),
          hip_rise_ratio: clamp(
            finiteOr(rawCrossing.hip_rise_ratio, 0.08),
            0,
            1,
          ),
          raised_knee_ratio: clamp(
            finiteOr(rawCrossing.raised_knee_ratio, 0.1),
            0,
            1,
          ),
          climb_enter_frames: Math.round(
            clamp(finiteOr(rawCrossing.climb_enter_frames, 2), 1, 30),
          ),
          climb_exit_frames: Math.round(
            clamp(finiteOr(rawCrossing.climb_exit_frames, 2), 1, 60),
          ),
          history_window_frames: Math.round(
            clamp(finiteOr(rawCrossing.history_window_frames, 30), 2, 300),
          ),
        },
      };
      if (draft.analysis_options.fence_crossing_rule.enabled) {
        draft.analysis_options.fence_pose_roi.enabled = true;
      }
      return;
    }
    const raw =
      source.forklift_driver_suppression &&
      typeof source.forklift_driver_suppression === "object" &&
      !Array.isArray(source.forklift_driver_suppression)
        ? source.forklift_driver_suppression
        : {};
    const alertSource = Array.isArray(raw.suppressed_alerts)
      ? raw.suppressed_alerts
      : FORKLIFT_ALERT_ORDER;
    const alerts = FORKLIFT_ALERT_ORDER.filter((alert) =>
      alertSource.includes(alert),
    );
    draft.analysis_options = {
      forklift_driver_suppression: {
        enabled: raw.enabled !== false,
        suppressed_alerts: alerts,
        minimum_forklift_confidence: clamp(
          finiteOr(raw.minimum_forklift_confidence, 0.35),
          0.05,
          1,
        ),
        minimum_person_ioa: clamp(
          finiteOr(raw.minimum_person_ioa, 0.55),
          0.05,
          1,
        ),
        enter_debounce_frames: Math.round(
          clamp(finiteOr(raw.enter_debounce_frames, 4), 1, 30),
        ),
        exit_debounce_frames: Math.round(
          clamp(finiteOr(raw.exit_debounce_frames, 8), 1, 60),
        ),
      },
    };
  }

  function ensureDraft(videoId) {
    const video = state.byId.get(videoId);
    if (!video) return null;
    const scenario = scenarioForVideo(video);
    if (!scenario) return null;
    if (!state.drafts[videoId]) {
      state.drafts[videoId] = {
        scenario: scenario.scenario,
        start_seconds: 0,
        end_seconds: Number(video.duration_seconds),
        rois: [],
      };
    }
    const draft = state.drafts[videoId];
    draft.scenario = scenario.scenario;
    draft.start_seconds = clamp(
      draft.start_seconds,
      0,
      Number(video.duration_seconds),
    );
    draft.end_seconds = clamp(
      draft.end_seconds || video.duration_seconds,
      0,
      Number(video.duration_seconds),
    );
    if (draft.end_seconds <= draft.start_seconds) {
      draft.start_seconds = 0;
      draft.end_seconds = Number(video.duration_seconds);
    }
    draft.rois = Array.isArray(draft.rois) ? draft.rois : [];
    draft.rois.forEach((roi) => {
      // The catalog category is authoritative. This also migrates v1 local
      // drafts and server selections that predate typed ROI polygons.
      roi.roi_type = scenario.roiType;
      roi.closed = roi.closed !== false;
      roi.points = Array.isArray(roi.points) ? roi.points : [];
    });
    normalizeAnalysisOptions(draft, scenario);
    return draft;
  }

  function polygonArea(points) {
    if (points.length < 3) return 0;
    let area = 0;
    points.forEach((point, index) => {
      const next = points[(index + 1) % points.length];
      area += point.x * next.y - next.x * point.y;
    });
    return Math.abs(area / 2);
  }

  function orientation(a, b, c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  }

  function onSegment(point, start, end) {
    const epsilon = 1e-9;
    return (
      Math.abs(orientation(start, end, point)) <= epsilon &&
      point.x >= Math.min(start.x, end.x) - epsilon &&
      point.x <= Math.max(start.x, end.x) + epsilon &&
      point.y >= Math.min(start.y, end.y) - epsilon &&
      point.y <= Math.max(start.y, end.y) + epsilon
    );
  }

  function segmentsIntersect(a, b, c, d) {
    const epsilon = 1e-9;
    const abC = orientation(a, b, c);
    const abD = orientation(a, b, d);
    const cdA = orientation(c, d, a);
    const cdB = orientation(c, d, b);
    const proper =
      ((abC > epsilon && abD < -epsilon) ||
        (abC < -epsilon && abD > epsilon)) &&
      ((cdA > epsilon && cdB < -epsilon) ||
        (cdA < -epsilon && cdB > epsilon));
    return (
      proper ||
      onSegment(c, a, b) ||
      onSegment(d, a, b) ||
      onSegment(a, c, d) ||
      onSegment(b, c, d)
    );
  }

  function polygonError(points) {
    if (!Array.isArray(points) || points.length < 3) {
      return "En az 3 köşe gerekli.";
    }
    const unique = new Set(
      points.map((point) => `${round(point.x, 8)}:${round(point.y, 8)}`),
    );
    if (unique.size !== points.length) return "Köşeler birbirinden farklı olmalı.";
    if (polygonArea(points) < 1e-5) return "Alan çok küçük.";
    for (let first = 0; first < points.length; first += 1) {
      const firstNext = (first + 1) % points.length;
      for (let second = first + 1; second < points.length; second += 1) {
        const secondNext = (second + 1) % points.length;
        if (
          first === second ||
          firstNext === second ||
          secondNext === first ||
          (first === 0 && second === points.length - 1)
        ) {
          continue;
        }
        if (
          segmentsIntersect(
            points[first],
            points[firstNext],
            points[second],
            points[secondNext],
          )
        ) {
          return "Alan kenarları birbiriyle kesişemez.";
        }
      }
    }
    return "";
  }

  function roiReady(roi, expectedType) {
    return Boolean(
      roi?.closed &&
        roi.roi_type === expectedType &&
        !polygonError(roi.points),
    );
  }

  function analysisOptionsReady(draft, scenario) {
    if (scenario.scenario === "fence_security") {
      const pose = draft.analysis_options?.fence_pose_roi;
      const crossing = draft.analysis_options?.fence_crossing_rule;
      const poseReady = Boolean(
        pose &&
          (!pose.enabled ||
            (Array.isArray(pose.selected_keypoints) &&
          pose.selected_keypoints.length > 0 &&
          pose.selected_keypoints.every((keypoint) =>
            COCO17_KEYPOINT_IDS.has(keypoint),
          ) &&
          new Set(pose.selected_keypoints).size ===
            pose.selected_keypoints.length &&
          pose.inside_ratio_threshold > 0 &&
          pose.inside_ratio_threshold <= 1 &&
          pose.keypoint_confidence_threshold >= 0 &&
          pose.keypoint_confidence_threshold <= 1 &&
          pose.minimum_visible_keypoints >= 1 &&
              pose.minimum_visible_keypoints <=
                pose.selected_keypoints.length))
      );
      if (!poseReady) return false;
      if (!crossing || !crossing.enabled) return true;
      const start = normalizedPoint(crossing.boundary_start);
      const end = normalizedPoint(crossing.boundary_end);
      return Boolean(
        pose.enabled &&
          start &&
          end &&
          Math.hypot(end.x - start.x, end.y - start.y) >= 0.01 &&
          ["left", "right"].includes(crossing.forbidden_side) &&
          crossing.contact_band > 0 &&
          crossing.contact_band <= 0.25 &&
          crossing.minimum_confidence >= 0 &&
          crossing.minimum_confidence <= 1 &&
          crossing.minimum_core_visible >= 1 &&
          crossing.minimum_core_visible <= 2 &&
          crossing.breach_enter_frames >= 1 &&
          crossing.breach_exit_frames >= 1 &&
          Array.isArray(crossing.approach_keypoint_names) &&
          crossing.approach_keypoint_names.length >= 1 &&
          crossing.approach_minimum_count >= 1 &&
          crossing.approach_minimum_count <=
            crossing.approach_keypoint_names.length &&
          crossing.wrist_contact_required >= 1 &&
          crossing.wrist_contact_required <= 2 &&
          crossing.hip_rise_ratio >= 0 &&
          crossing.raised_knee_ratio >= 0 &&
          crossing.climb_enter_frames >= 1 &&
          crossing.climb_exit_frames >= 1 &&
          crossing.history_window_frames >=
            Math.max(
              crossing.breach_enter_frames,
              crossing.climb_enter_frames,
            )
      );
    }
    const suppression =
      draft.analysis_options?.forklift_driver_suppression;
    if (!suppression || !suppression.enabled) return true;
    return Boolean(
      Array.isArray(suppression.suppressed_alerts) &&
        suppression.suppressed_alerts.length > 0 &&
        suppression.suppressed_alerts.every((alert) =>
          FORKLIFT_ALERT_ORDER.includes(alert),
        ) &&
        new Set(suppression.suppressed_alerts).size ===
          suppression.suppressed_alerts.length &&
        suppression.minimum_forklift_confidence > 0 &&
        suppression.minimum_forklift_confidence <= 1 &&
        suppression.minimum_person_ioa > 0 &&
        suppression.minimum_person_ioa <= 1 &&
        suppression.enter_debounce_frames >= 1 &&
        suppression.exit_debounce_frames >= 1
    );
  }

  function videoReady(videoId) {
    if (!state.selected.has(videoId)) return false;
    const draft = ensureDraft(videoId);
    const video = state.byId.get(videoId);
    const scenario = scenarioForVideo(video);
    if (!draft || !video || !scenario) return false;
    if (draft.rois.length < scenario.minimumRois) return false;
    const clipOkay =
      draft.start_seconds >= 0 &&
      draft.end_seconds > draft.start_seconds &&
      draft.end_seconds <= Number(video.duration_seconds) + 0.05;
    return (
      clipOkay &&
      analysisOptionsReady(draft, scenario) &&
      draft.rois.every((roi) => roiReady(roi, scenario.roiType))
    );
  }

  function scheduleLocalSave() {
    window.clearTimeout(state.saveTimer);
    state.saveTimer = window.setTimeout(() => {
      try {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            schemaVersion: 2,
            catalogRevision: state.catalogRevision,
            catalogRevisions: state.catalogRevisions,
            selectedVideoIds: [...state.selected],
            activeVideoId: state.activeVideoId,
            drafts: state.drafts,
          }),
        );
      } catch {
        // The final server save remains available when local storage is blocked.
      }
    }, 120);
  }

  function categoryRevisionsMatch(savedRevisions) {
    if (
      !savedRevisions ||
      typeof savedRevisions !== "object" ||
      Array.isArray(savedRevisions)
    ) {
      return false;
    }
    const entries = Object.entries(savedRevisions);
    return (
      entries.length > 0 &&
      entries.every(
        ([category, revision]) =>
          String(state.catalogRevisions[category] || "") === String(revision),
      )
    );
  }

  function localRevisionMatches(payload) {
    if (payload.catalogRevisions) {
      return categoryRevisionsMatch(payload.catalogRevisions);
    }
    return (
      Boolean(payload.catalogRevision) &&
      String(payload.catalogRevision) === state.catalogRevision
    );
  }

  function restoreLocalDraft() {
    try {
      const raw =
        localStorage.getItem(STORAGE_KEY) ||
        localStorage.getItem(LEGACY_STORAGE_KEY);
      if (!raw) return false;
      const payload = JSON.parse(raw);
      if (!localRevisionMatches(payload)) return false;
      state.selected = new Set(
        (payload.selectedVideoIds || []).filter((id) => state.byId.has(id)),
      );
      const savedDrafts =
        payload.drafts && typeof payload.drafts === "object" ? payload.drafts : {};
      state.drafts = Object.fromEntries(
        Object.entries(savedDrafts).filter(([videoId]) =>
          state.byId.has(videoId),
        ),
      );
      state.activeVideoId = state.byId.has(payload.activeVideoId)
        ? payload.activeVideoId
        : null;
      // ensureDraft supplies scenario and roi_type for legacy v1 entries.
      [...state.selected].forEach(ensureDraft);
      state.localDraftFound = true;
      return true;
    } catch {
      return false;
    }
  }

  async function restoreLatestServerSelection() {
    try {
      const response = await fetch("/api/selection/latest", {
        headers: { Accept: "application/json" },
      });
      if (response.status === 404) return;
      if (!response.ok) throw new Error("Kayıtlı seçim okunamadı.");
      const payload = await response.json();
      const revisionMatches = payload.catalog_revisions
        ? categoryRevisionsMatch(payload.catalog_revisions)
        : String(payload.catalog_revision || "") === state.catalogRevision;
      if (!revisionMatches) return;
      for (const selected of payload.videos || []) {
        if (!state.byId.has(selected.video_id)) continue;
        const scenario = scenarioForVideoId(selected.video_id);
        if (!scenario) continue;
        state.selected.add(selected.video_id);
        state.drafts[selected.video_id] = {
          scenario: scenario.scenario,
          start_seconds: selected.start_seconds,
          end_seconds: selected.end_seconds,
          analysis_options:
            selected.analysis_options &&
            typeof selected.analysis_options === "object"
              ? JSON.parse(JSON.stringify(selected.analysis_options))
              : undefined,
          rois: (selected.rois || []).map((roi) => ({
            ...roi,
            roi_type: scenario.roiType,
            points: roi.points.map((point) => ({ ...point })),
            closed: true,
          })),
        };
        ensureDraft(selected.video_id);
      }
      state.activeVideoId = [...state.selected][0] || null;
    } catch (error) {
      showMessage(error.message || "Kayıtlı seçim okunamadı.", true);
    }
  }

  function renderCategoryFilters() {
    const counts = state.catalog.reduce((result, video) => {
      result[video.category] = (result[video.category] || 0) + 1;
      return result;
    }, {});
    els.personCategoryCount.textContent = String(counts.person_office || 0);
    els.ppeCategoryCount.textContent = String(counts.ppe_safety || 0);
    els.categoryFilters
      .querySelectorAll("[data-category-filter]")
      .forEach((button) => {
        const active = button.dataset.categoryFilter === state.activeCategory;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      });
  }

  function renderCatalog() {
    const visibleVideos = state.catalog
      .map((video, catalogIndex) => ({ video, catalogIndex }))
      .filter(({ video }) => video.category === state.activeCategory)
      .sort(
        (first, second) =>
          Number(isNewUserVideo(second.video)) -
            Number(isNewUserVideo(first.video)) ||
          first.catalogIndex - second.catalogIndex,
      )
      .map(({ video }) => video);
    els.videoCatalog.innerHTML = visibleVideos
      .map((video) => {
        const scenario = scenarioForVideo(video);
        const newUserVideo = isNewUserVideo(video);
        const selected = state.selected.has(video.video_id);
        const active = state.activeVideoId === video.video_id;
        const ready = videoReady(video.video_id);
        return `
          <article class="video-card${selected ? " selected" : ""}${active ? " active" : ""}${ready ? " ready" : ""}" data-video-card="${escapeHtml(video.video_id)}" data-category="${escapeHtml(video.category)}">
            <div class="video-poster">
              <img src="${escapeHtml(video.poster_url)}" alt="" loading="lazy" />
            </div>
            <div class="video-card-content">
              <div class="video-card-top">
                <p class="video-card-title">${escapeHtml(video.video_id)} · ${escapeHtml(video.title)}</p>
                <label class="video-select" title="Bu videoyu seç">
                  <span class="sr-only">${escapeHtml(video.title)} videosunu seç</span>
                  <input type="checkbox" data-select-video="${escapeHtml(video.video_id)}" ${selected ? "checked" : ""} />
                </label>
              </div>
              ${newUserVideo ? '<span class="new-video-badge">YENİ VİDEO</span>' : ""}
              <p class="video-card-meta">${escapeHtml(video.camera)}</p>
              <span class="pipeline-badge">${escapeHtml(scenario?.pipelineLabel || video.pipeline_label || video.pipeline)}</span>
              <button class="open-video" type="button" data-open-video="${escapeHtml(video.video_id)}">${selected ? "Alanı düzenle" : "Önizle"}</button>
              <span class="card-state${ready ? " ready" : ""}">${ready ? "Hazır" : selected ? "Alan bekliyor" : formatTime(video.duration_seconds)}</span>
            </div>
          </article>
        `;
      })
      .join("");
    els.videoCatalog.querySelectorAll("img").forEach((image) => {
      image.addEventListener("error", () => image.classList.add("image-error"));
    });
    renderCategoryFilters();
  }

  function renderRoiList() {
    const draft = currentDraft();
    const scenario = scenarioForVideo();
    const rois = draft?.rois || [];
    els.roiCount.textContent = `${rois.length} alan`;
    if (!rois.length) {
      els.roiList.innerHTML = '<p class="empty-list">Henüz alan çizilmedi.</p>';
    } else {
      els.roiList.innerHTML = rois
        .map((roi) => {
          const active = roi.roi_id === state.activeRoiId;
          const error = polygonError(roi.points);
          const typed = Boolean(
            scenario && roi.roi_type === scenario.roiType,
          );
          const label =
            roi.closed && !error && typed
              ? "Hazır"
              : roi.closed
                ? "Hatalı"
                : "Çiziliyor";
          const statusClass =
            roi.closed && !error && typed
              ? "ready"
              : roi.closed
                ? "error"
                : "open";
          return `
            <div class="roi-item ${escapeHtml(roi.roi_type)}${active ? " active" : ""}" data-roi-item="${escapeHtml(roi.roi_id)}">
              <div>
                <div class="roi-item-heading">
                  <button class="roi-select-button" type="button" data-select-roi="${escapeHtml(roi.roi_id)}">${escapeHtml(roi.name)}</button>
                  <span class="roi-type-badge ${escapeHtml(roi.roi_type)}">${escapeHtml(scenario?.roiTypeLabel || "ALAN")}</span>
                </div>
                <div class="roi-name-row">
                  <label>
                    <span class="sr-only">Alan adı</span>
                    <input type="text" maxlength="80" value="${escapeHtml(roi.name)}" data-rename-roi="${escapeHtml(roi.roi_id)}" />
                  </label>
                </div>
              </div>
              <span class="roi-item-state ${statusClass}">${label}<br />${roi.points.length} köşe</span>
            </div>
          `;
        })
        .join("");
    }
    renderVertexEditor();
  }

  function renderVertexEditor() {
    const roi = currentRoi();
    const points = roi?.points || [];
    els.vertexSummary.textContent = `${points.length} nokta`;
    if (!roi) {
      els.vertexEditor.innerHTML =
        '<p class="empty-list">Düzenlemek için bir alan seç.</p>';
      return;
    }
    els.vertexEditor.innerHTML = points
      .map(
        (point, index) => `
          <div class="vertex-row">
            <span class="vertex-index">${index + 1}</span>
            <label><span>X</span><input type="number" min="0" max="1" step="0.001" value="${round(point.x, 5)}" data-vertex-axis="x" data-vertex-index="${index}" /></label>
            <label><span>Y</span><input type="number" min="0" max="1" step="0.001" value="${round(point.y, 5)}" data-vertex-axis="y" data-vertex-index="${index}" /></label>
            <button class="vertex-delete" type="button" data-delete-vertex="${index}" aria-label="${index + 1}. köşeyi sil">×</button>
          </div>
        `,
      )
      .join("");
  }

  function updateClipUi() {
    const video = currentVideo();
    const draft = currentDraft();
    if (!video || !draft) return;
    const duration = Number(video.duration_seconds);
    const step = frameStep(video);
    const start = round(draft.start_seconds, 3);
    const end = round(draft.end_seconds, 3);
    [els.clipStart, els.clipStartRange].forEach((control) => {
      control.max = String(duration);
      control.step = String(step);
      control.value = String(start);
    });
    [els.clipEnd, els.clipEndRange].forEach((control) => {
      control.max = String(duration);
      control.step = String(step);
      control.value = String(end);
    });
    const full =
      Math.abs(start) < 0.001 && Math.abs(end - duration) <= Math.max(0.05, step);
    els.clipSummary.textContent = full
      ? "Tam video"
      : `${formatTime(start)}–${formatTime(end)}`;
    els.clipDuration.textContent = `${formatTime(end - start)} işlenecek`;
  }

  function setControlsDisabled(container, disabled) {
    container
      .querySelectorAll("input, button, select")
      .forEach((control) => {
        control.disabled = disabled;
      });
    container.classList.toggle("disabled", disabled);
  }

  function renderAnalysisOptions() {
    const video = currentVideo();
    const draft = currentDraft();
    const scenario = scenarioForVideo(video);
    if (!video || !draft || !scenario) return;
    els.analysisOptionsPanel.dataset.scenario = scenario.scenario;
    const fence = scenario.scenario === "fence_security";
    els.fenceAnalysisOptions.hidden = !fence;
    els.ppeAnalysisOptions.hidden = fence;
    if (fence) {
      const pose = draft.analysis_options.fence_pose_roi;
      const crossing = draft.analysis_options.fence_crossing_rule;
      const lineReady = Boolean(
        normalizedPoint(crossing.boundary_start) &&
          normalizedPoint(crossing.boundary_end),
      );
      els.fenceCrossingEnabled.checked = crossing.enabled;
      els.fenceForbiddenLeft.checked =
        crossing.forbidden_side === "left";
      els.fenceForbiddenRight.checked =
        crossing.forbidden_side === "right";
      els.fenceLineSummary.textContent = lineReady
        ? `Çit hattı hazır · yasak taraf ${
            crossing.forbidden_side === "left" ? "sol" : "sağ"
          }`
        : "Çit hattı gerekli: videoda iki noktaya tıkla";
      els.drawFenceLineButton.innerHTML = state.drawingFenceLine
        ? '<span aria-hidden="true">×</span> Çizimi iptal et'
        : lineReady
          ? '<span aria-hidden="true">╱</span> Hattı yeniden çiz'
          : '<span aria-hidden="true">╱</span> Çit hattı çiz';
      els.drawFenceLineButton.classList.toggle(
        "active",
        state.drawingFenceLine,
      );
      setControlsDisabled(
        els.fenceCrossingControls,
        !crossing.enabled,
      );
      els.deleteFenceLineButton.disabled =
        !crossing.enabled || !lineReady;
      els.poseZoneEnabled.checked = pose.enabled;
      els.poseZoneEnabled.disabled = crossing.enabled;
      const percentage = Math.round(pose.inside_ratio_threshold * 100);
      els.poseInsidePercent.value = String(percentage);
      els.poseInsidePercentNumber.value = String(percentage);
      const selected = new Set(pose.selected_keypoints);
      els.poseKeypointGrid.innerHTML = COCO17_KEYPOINTS.map(
        (keypoint) => `
          <label class="keypoint-option">
            <input
              type="checkbox"
              data-pose-keypoint="${escapeHtml(keypoint.id)}"
              ${selected.has(keypoint.id) ? "checked" : ""}
            />
            <span>
              <strong>${escapeHtml(keypoint.label)}</strong>
              <small>${escapeHtml(keypoint.group)}</small>
            </span>
          </label>
        `,
      ).join("");
      const required = pose.selected_keypoints.length
        ? Math.ceil(
            pose.selected_keypoints.length *
              pose.inside_ratio_threshold -
              1e-9,
          )
        : 0;
      els.poseRuleSummary.textContent = pose.selected_keypoints.length
        ? `${pose.selected_keypoints.length} noktadan en az ${required}'ü ROI içinde: yaklaşma kanıtı`
        : "Alarm için en az bir keypoint seç";
      els.poseEvidenceSummary.textContent =
        `Keypoint güveni ≥ %${Math.round(
          pose.keypoint_confidence_threshold * 100,
        )} · en az ${pose.minimum_visible_keypoints} görünür nokta`;
      setControlsDisabled(els.poseControls, !pose.enabled);
      return;
    }

    const suppression =
      draft.analysis_options.forklift_driver_suppression;
    els.forkliftSuppressionEnabled.checked = suppression.enabled;
    els.forkliftPpeViolation.checked =
      suppression.suppressed_alerts.includes("ppe_violation");
    els.forkliftWalkwayViolation.checked =
      suppression.suppressed_alerts.includes(
        "safe_walkway_violation",
      );
    els.forkliftConfidence.value = String(
      round(suppression.minimum_forklift_confidence, 2),
    );
    els.forkliftPersonIoa.value = String(
      round(suppression.minimum_person_ioa, 2),
    );
    els.forkliftEnterFrames.value = String(
      suppression.enter_debounce_frames,
    );
    els.forkliftExitFrames.value = String(
      suppression.exit_debounce_frames,
    );
    const labels = [];
    if (suppression.suppressed_alerts.includes("ppe_violation")) {
      labels.push("PPE");
    }
    if (
      suppression.suppressed_alerts.includes(
        "safe_walkway_violation",
      )
    ) {
      labels.push("yürüyüş yolu");
    }
    els.forkliftRuleSummary.textContent = labels.length
      ? `${labels.join(" ve ")} alarmı bastırılır`
      : "Bastırılacak en az bir alarm türü seç";
    setControlsDisabled(els.forkliftControls, !suppression.enabled);
  }

  function updateEditorState() {
    const video = currentVideo();
    const draft = currentDraft();
    const scenario = scenarioForVideo(video);
    if (!video || !draft || !scenario) return;
    els.scenarioRuleCard.dataset.scenario = scenario.scenario;
    els.scenarioRuleBadge.textContent = scenario.badge;
    els.scenarioRuleTitle.textContent = scenario.ruleTitle;
    renderAnalysisOptions();
    if (scenario.scenario === "fence_security") {
      const pose = draft.analysis_options.fence_pose_roi;
      const crossing = draft.analysis_options.fence_crossing_rule;
      els.scenarioRuleText.textContent = crossing.enabled
        ? "ROI yalnızca yaklaşmayı gösterir; tırmanma hareketi veya kalça merkezinin mor çit hattının yasak tarafına doğrulanmış geçişi alarm üretir."
        : pose.enabled
          ? "Aşamalı geçiş kapalıdır; seçilen iskelet noktalarının ROI içi oranı doğrudan alarm üretir."
        : "Pose oranı kapalıdır; takip edilen insanın ayak noktası kırmızı alana girdiğinde uyarı oluşturulur.";
    } else {
      const suppression =
        draft.analysis_options.forklift_driver_suppression;
      els.scenarioRuleText.textContent = suppression.enabled
        ? "Kask ve yelek kontrolü tüm karede çalışır; doğrulanmış forklift sürücülerinin seçili alarmları bastırılır."
        : "Kask ve yelek kontrolü tüm karede, takip edilen her insan için çalışır. Forklift sürücüsü alarm istisnası kapalıdır.";
    }
    els.roiListTitle.textContent = scenario.roiListTitle;
    els.addRoiLabel.textContent = scenario.addLabel;
    els.fullFrameLabel.textContent = scenario.fullFrameLabel;
    els.canvasInstructions.innerHTML =
      `${escapeHtml(scenario.instructionSubject)} köşelerine tıkla. ` +
      "Noktaları sürükleyerek düzeltebilir, <kbd>Enter</kbd> ile alanı kapatabilirsin.";
    const selected = state.selected.has(video.video_id);
    const ready = videoReady(video.video_id);
    els.activeVideoState.textContent = ready
      ? "Hazır"
      : selected
        ? "Alan bekliyor"
        : "Video seçilmedi";
    els.activeVideoState.className = `state-pill${ready ? " ready" : ""}`;
    const active = currentRoi();
    const drawing = Boolean(active && !active.closed);
    els.addRoiButton.disabled =
      !selected || drawing || state.drawingFenceLine;
    els.fullFrameButton.disabled =
      !selected || drawing || state.drawingFenceLine;
    els.finishRoiButton.disabled = !selected || !drawing || active.points.length < 3;
    els.undoButton.disabled = !selected || !active || active.points.length === 0;
    els.deleteRoiButton.disabled = !selected || !active;
    if (scenario.scenario === "fence_security") {
      const crossing = draft.analysis_options.fence_crossing_rule;
      const lineReady = Boolean(
        normalizedPoint(crossing.boundary_start) &&
          normalizedPoint(crossing.boundary_end),
      );
      els.drawFenceLineButton.disabled =
        !selected || !crossing.enabled;
      els.deleteFenceLineButton.disabled =
        !selected || !crossing.enabled || !lineReady;
    }
    els.roiCanvas.setAttribute("aria-disabled", String(!selected));
    if (!selected) {
      setCanvasStatus("Alan çizmek için önce soldaki seçim kutusunu işaretle.");
    } else if (state.drawingFenceLine) {
      const crossing = draft.analysis_options.fence_crossing_rule;
      setCanvasStatus(
        crossing.boundary_start
          ? "Çit hattının ikinci ucuna tıkla."
          : "Çitin görüntüdeki iki ucundan ilkine tıkla.",
      );
    } else if (!draft.rois.length) {
      setCanvasStatus(
        scenario.emptyStatus,
        false,
        scenario.minimumRois === 0,
      );
    } else if (
      scenario.scenario === "fence_security" &&
      draft.analysis_options.fence_crossing_rule.enabled &&
      (!normalizedPoint(
        draft.analysis_options.fence_crossing_rule.boundary_start,
      ) ||
        !normalizedPoint(
          draft.analysis_options.fence_crossing_rule.boundary_end,
        ))
    ) {
      setCanvasStatus(
        "Aşamalı alarm için iki noktalı çit hattını çiz.",
        true,
      );
    } else if (drawing) {
      setCanvasStatus(
        `${active.points.length} köşe eklendi. En az 3 köşeden sonra ${scenario.drawingNoun} çizimini tamamla.`,
      );
    } else if (active && polygonError(active.points)) {
      setCanvasStatus(polygonError(active.points), true);
    } else if (ready) {
      setCanvasStatus(
        scenario.scenario === "ppe_safety"
          ? "PPE tüm karede aktif; yürüyüş yolu alarm kuralı da hazır."
          : "ROI, çit hattı ve yasak taraf hazır. İstersen başka bir video seçebilirsin.",
        false,
        true,
      );
    } else {
      setCanvasStatus("Alanı tamamla veya hatalı köşeleri düzelt.");
    }
  }

  function updateProgress() {
    const selectedIds = [...state.selected];
    const readyCount = selectedIds.filter(videoReady).length;
    const total = selectedIds.length;
    els.selectionCount.textContent = `${total} seçili`;
    els.readyCount.textContent = `${readyCount} / ${total} hazır`;
    els.progressFill.style.width = `${total ? (readyCount / total) * 100 : 0}%`;
    const allReady = total > 0 && readyCount === total;
    els.saveButton.disabled = !allReady;
    els.saveHint.textContent = allReady
      ? `${total} video ve güvenlik kuralları hazır. Bu düğme yalnızca seçimi kaydeder.`
      : total
        ? `${total - readyCount} videonun alanı eksik veya açık.`
        : "Videoları seç; çit güvenliği videolarındaki yasak alanları tamamla.";
  }

  function refresh({ catalog = true, list = true } = {}) {
    if (catalog) renderCatalog();
    if (list) renderRoiList();
    updateClipUi();
    updateEditorState();
    updateProgress();
    drawCanvas();
    scheduleLocalSave();
  }

  function setCanvasStatus(message, error = false, success = false) {
    els.canvasStatus.textContent = message;
    els.canvasStatus.className = `canvas-status${error ? " error" : success ? " success" : ""}`;
  }

  function showMessage(message, error = false) {
    window.clearTimeout(state.messageTimer);
    els.globalMessage.textContent = message;
    els.globalMessage.className = `global-message${error ? " error" : ""}`;
    els.globalMessage.hidden = false;
    state.messageTimer = window.setTimeout(() => {
      els.globalMessage.hidden = true;
    }, error ? 6500 : 5000);
  }

  function videoContentRect() {
    const stageRect = els.videoStage.getBoundingClientRect();
    const video = currentVideo();
    if (!video || !stageRect.width || !stageRect.height) {
      return { x: 0, y: 0, width: stageRect.width, height: stageRect.height };
    }
    const mediaWidth =
      Number(video.width) || Number(els.previewVideo.videoWidth) || stageRect.width;
    const mediaHeight =
      Number(video.height) || Number(els.previewVideo.videoHeight) || stageRect.height;
    const videoAspect = mediaWidth / mediaHeight;
    const stageAspect = stageRect.width / stageRect.height;
    if (stageAspect > videoAspect) {
      const width = stageRect.height * videoAspect;
      return {
        x: (stageRect.width - width) / 2,
        y: 0,
        width,
        height: stageRect.height,
      };
    }
    const height = stageRect.width / videoAspect;
    return {
      x: 0,
      y: (stageRect.height - height) / 2,
      width: stageRect.width,
      height,
    };
  }

  function resizeCanvas() {
    const rect = els.roiCanvas.getBoundingClientRect();
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (els.roiCanvas.width !== width || els.roiCanvas.height !== height) {
      els.roiCanvas.width = width;
      els.roiCanvas.height = height;
    }
    return { width: rect.width, height: rect.height, ratio };
  }

  function forbiddenHalfPlanePolygon(start, end, side) {
    let polygon = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
    const signed = (point) =>
      (side === "left" ? 1 : -1) * orientation(start, end, point);
    const clipped = [];
    polygon.forEach((current, index) => {
      const next = polygon[(index + 1) % polygon.length];
      const currentValue = signed(current);
      const nextValue = signed(next);
      const currentInside = currentValue >= -1e-9;
      const nextInside = nextValue >= -1e-9;
      if (currentInside) clipped.push(current);
      if (currentInside !== nextInside) {
        const denominator = currentValue - nextValue;
        const ratio =
          Math.abs(denominator) < 1e-12
            ? 0
            : currentValue / denominator;
        clipped.push({
          x: current.x + (next.x - current.x) * ratio,
          y: current.y + (next.y - current.y) * ratio,
        });
      }
    });
    return clipped;
  }

  function drawCanvas() {
    const canvasSize = resizeCanvas();
    const context = els.roiCanvas.getContext("2d");
    context.setTransform(canvasSize.ratio, 0, 0, canvasSize.ratio, 0, 0);
    context.clearRect(0, 0, canvasSize.width, canvasSize.height);
    const draft = currentDraft();
    if (!draft) return;
    const content = videoContentRect();
    context.save();
    context.strokeStyle = "rgba(133, 187, 225, 0.35)";
    context.setLineDash([6, 6]);
    context.strokeRect(content.x + 0.5, content.y + 0.5, content.width - 1, content.height - 1);
    context.restore();

    const crossing = draft.analysis_options?.fence_crossing_rule;
    const boundaryStart = normalizedPoint(crossing?.boundary_start);
    const boundaryEnd = normalizedPoint(crossing?.boundary_end);
    if (crossing?.enabled && boundaryStart && boundaryEnd) {
      const forbidden = forbiddenHalfPlanePolygon(
        boundaryStart,
        boundaryEnd,
        crossing.forbidden_side,
      );
      if (forbidden.length >= 3) {
        context.save();
        context.beginPath();
        forbidden.forEach((point, index) => {
          const x = content.x + point.x * content.width;
          const y = content.y + point.y * content.height;
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.closePath();
        context.fillStyle = "rgba(159, 92, 224, 0.12)";
        context.fill();
        context.clip();
        context.strokeStyle = "rgba(210, 167, 255, 0.13)";
        context.lineWidth = 1;
        const span = canvasSize.width + canvasSize.height;
        for (let offset = -span; offset < span; offset += 18) {
          context.beginPath();
          context.moveTo(offset, canvasSize.height);
          context.lineTo(offset + canvasSize.height, 0);
          context.stroke();
        }
        context.restore();
      }
    }

    draft.rois.forEach((roi, roiIndex) => {
      if (!roi.points.length) return;
      const active = roi.roi_id === state.activeRoiId;
      const roiScenario = scenarioForRoi(roi);
      const color =
        roiScenario?.color ||
        ["#3daef5", "#58dda6", "#ffca70"][roiIndex % 3];
      const pixels = roi.points.map((point) => ({
        x: content.x + point.x * content.width,
        y: content.y + point.y * content.height,
      }));
      context.save();
      context.beginPath();
      context.moveTo(pixels[0].x, pixels[0].y);
      pixels.slice(1).forEach((point) => context.lineTo(point.x, point.y));
      if (roi.closed) context.closePath();
      context.lineWidth = active ? 3 : 2;
      context.strokeStyle = color;
      context.fillStyle = active
        ? roiScenario?.activeFill || "rgba(75, 214, 229, 0.17)"
        : roiScenario?.fill || "rgba(61, 174, 245, 0.11)";
      if (!roi.closed) context.setLineDash([8, 5]);
      if (roi.closed) context.fill();
      context.stroke();
      context.setLineDash([]);

      pixels.forEach((point, index) => {
        context.beginPath();
        context.arc(point.x, point.y, active ? 6 : 4.5, 0, Math.PI * 2);
        context.fillStyle = index === 0 && !roi.closed ? "#ffca70" : "#f5fbff";
        context.fill();
        context.lineWidth = 2;
        context.strokeStyle = color;
        context.stroke();
      });

      const label =
        `${roiScenario?.roiTypeLabel || "ALAN"} · ` +
        (roi.name || `Alan ${roiIndex + 1}`);
      context.font = "700 13px system-ui, sans-serif";
      const labelWidth = context.measureText(label).width + 18;
      const labelX = clamp(pixels[0].x + 9, 4, canvasSize.width - labelWidth - 4);
      const labelY = Math.max(5, pixels[0].y - 28);
      context.fillStyle = "rgba(3, 13, 27, 0.9)";
      context.fillRect(labelX, labelY, labelWidth, 23);
      context.fillStyle = color;
      context.fillText(label, labelX + 9, labelY + 16);
      context.restore();
    });

    if (crossing?.enabled && boundaryStart) {
      const start = {
        x: content.x + boundaryStart.x * content.width,
        y: content.y + boundaryStart.y * content.height,
      };
      const end = boundaryEnd
        ? {
            x: content.x + boundaryEnd.x * content.width,
            y: content.y + boundaryEnd.y * content.height,
          }
        : null;
      context.save();
      if (end) {
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.lineWidth = 7;
        context.strokeStyle = "rgba(3, 8, 18, 0.84)";
        context.stroke();
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.lineWidth = 4;
        context.strokeStyle = "#c690ff";
        context.stroke();

        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const length = Math.max(1, Math.hypot(dx, dy));
        const unitX = dx / length;
        const unitY = dy / length;
        const arrowX = end.x - unitX * 18;
        const arrowY = end.y - unitY * 18;
        context.beginPath();
        context.moveTo(end.x, end.y);
        context.lineTo(
          arrowX - unitY * 7,
          arrowY + unitX * 7,
        );
        context.lineTo(
          arrowX + unitY * 7,
          arrowY - unitX * 7,
        );
        context.closePath();
        context.fillStyle = "#e2c5ff";
        context.fill();

        const sideMultiplier =
          crossing.forbidden_side === "left" ? 1 : -1;
        const normalX = -unitY * sideMultiplier;
        const normalY = unitX * sideMultiplier;
        const middleX = (start.x + end.x) / 2;
        const middleY = (start.y + end.y) / 2;
        const sideX = middleX + normalX * 34;
        const sideY = middleY + normalY * 34;
        context.beginPath();
        context.moveTo(middleX, middleY);
        context.lineTo(sideX, sideY);
        context.lineWidth = 2;
        context.setLineDash([4, 4]);
        context.strokeStyle = "#e2c5ff";
        context.stroke();
        context.setLineDash([]);
        context.font = "800 11px system-ui, sans-serif";
        const sideLabel = "YASAK TARAF";
        const sideLabelWidth = context.measureText(sideLabel).width + 14;
        const labelX = clamp(
          sideX - sideLabelWidth / 2,
          content.x + 4,
          content.x + content.width - sideLabelWidth - 4,
        );
        const labelY = clamp(
          sideY - 12,
          content.y + 4,
          content.y + content.height - 24,
        );
        context.fillStyle = "rgba(42, 13, 66, 0.92)";
        context.fillRect(labelX, labelY, sideLabelWidth, 22);
        context.fillStyle = "#eddcff";
        context.fillText(sideLabel, labelX + 7, labelY + 15);
      }
      [start, end].filter(Boolean).forEach((point, index) => {
        context.beginPath();
        context.arc(point.x, point.y, 7, 0, Math.PI * 2);
        context.fillStyle = index === 0 ? "#f7edff" : "#c690ff";
        context.fill();
        context.lineWidth = 2.5;
        context.strokeStyle = "#5e278f";
        context.stroke();
      });
      context.restore();
    }
  }

  function pointFromEvent(event) {
    const canvasRect = els.roiCanvas.getBoundingClientRect();
    const content = videoContentRect();
    const x = event.clientX - canvasRect.left;
    const y = event.clientY - canvasRect.top;
    if (
      x < content.x ||
      y < content.y ||
      x > content.x + content.width ||
      y > content.y + content.height
    ) {
      return null;
    }
    return {
      x: clamp((x - content.x) / content.width, 0, 1),
      y: clamp((y - content.y) / content.height, 0, 1),
    };
  }

  function nearestVertex(event) {
    const draft = currentDraft();
    if (!draft) return null;
    const canvasRect = els.roiCanvas.getBoundingClientRect();
    const content = videoContentRect();
    const x = event.clientX - canvasRect.left;
    const y = event.clientY - canvasRect.top;
    let best = null;
    const hitRadius = event.pointerType === "touch" ? 24 : 14;
    draft.rois.forEach((roi) => {
      roi.points.forEach((point, index) => {
        const px = content.x + point.x * content.width;
        const py = content.y + point.y * content.height;
        const distance = Math.hypot(px - x, py - y);
        if (distance <= hitRadius && (!best || distance < best.distance)) {
          best = { roi_id: roi.roi_id, index, distance };
        }
      });
    });
    return best;
  }

  function nearestFenceEndpoint(event) {
    const crossing =
      currentDraft()?.analysis_options?.fence_crossing_rule;
    if (!crossing?.enabled) return null;
    const canvasRect = els.roiCanvas.getBoundingClientRect();
    const content = videoContentRect();
    const x = event.clientX - canvasRect.left;
    const y = event.clientY - canvasRect.top;
    const hitRadius = event.pointerType === "touch" ? 28 : 16;
    let best = null;
    [
      ["boundary_start", normalizedPoint(crossing.boundary_start)],
      ["boundary_end", normalizedPoint(crossing.boundary_end)],
    ].forEach(([field, point]) => {
      if (!point) return;
      const px = content.x + point.x * content.width;
      const py = content.y + point.y * content.height;
      const distance = Math.hypot(px - x, py - y);
      if (distance <= hitRadius && (!best || distance < best.distance)) {
        best = { field, distance };
      }
    });
    return best;
  }

  function toggleFenceLineDrawing() {
    const draft = currentDraft();
    const crossing = draft?.analysis_options?.fence_crossing_rule;
    if (
      !draft ||
      !crossing?.enabled ||
      !state.selected.has(state.activeVideoId)
    ) {
      return;
    }
    if (state.drawingFenceLine) {
      state.drawingFenceLine = false;
      refresh({ catalog: false, list: false });
      return;
    }
    crossing.boundary_start = null;
    crossing.boundary_end = null;
    state.drawingFenceLine = true;
    state.activeRoiId = null;
    els.previewVideo.pause();
    refresh();
  }

  function deleteFenceLine() {
    const crossing =
      currentDraft()?.analysis_options?.fence_crossing_rule;
    if (!crossing) return;
    crossing.boundary_start = null;
    crossing.boundary_end = null;
    state.drawingFenceLine = false;
    refresh();
  }

  function selectRoi(roiId) {
    state.drawingFenceLine = false;
    state.activeRoiId = roiId;
    refresh();
  }

  function addRoi() {
    if (!state.selected.has(state.activeVideoId)) return;
    const draft = currentDraft();
    const scenario = scenarioForVideo();
    if (!draft || !scenario || draft.rois.some((roi) => !roi.closed)) return;
    state.drawingFenceLine = false;
    const used = new Set(draft.rois.map((roi) => roi.roi_id));
    let number = draft.rois.length + 1;
    let roiId = `roi-${number}`;
    while (used.has(roiId)) {
      number += 1;
      roiId = `roi-${number}`;
    }
    draft.rois.push({
      roi_id: roiId,
      name: `${scenario.defaultName} ${number}`,
      roi_type: scenario.roiType,
      points: [],
      closed: false,
    });
    state.activeRoiId = roiId;
    els.previewVideo.pause();
    refresh();
  }

  function finishRoi() {
    const roi = currentRoi();
    if (!roi || roi.closed) return;
    const error = polygonError(roi.points);
    if (error) {
      setCanvasStatus(error, true);
      return;
    }
    roi.closed = true;
    const draft = currentDraft();
    const crossing = draft?.analysis_options?.fence_crossing_rule;
    if (
      crossing?.enabled &&
      (!crossing.boundary_start || !crossing.boundary_end)
    ) {
      const derived = deriveFenceBoundary(draft.rois);
      if (derived) Object.assign(crossing, derived);
    }
    refresh();
  }

  function undoPoint() {
    const roi = currentRoi();
    if (!roi || !roi.points.length) return;
    if (roi.closed) roi.closed = false;
    roi.points.pop();
    refresh();
  }

  function deleteRoi() {
    const draft = currentDraft();
    if (!draft || !state.activeRoiId) return;
    draft.rois = draft.rois.filter((roi) => roi.roi_id !== state.activeRoiId);
    state.activeRoiId = draft.rois.at(-1)?.roi_id || null;
    refresh();
  }

  function addFullFrameRoi() {
    if (!state.selected.has(state.activeVideoId)) return;
    const draft = currentDraft();
    const scenario = scenarioForVideo();
    if (!draft || !scenario || draft.rois.some((roi) => !roi.closed)) return;
    const used = new Set(draft.rois.map((roi) => roi.roi_id));
    let number = draft.rois.length + 1;
    let roiId = `roi-${number}`;
    while (used.has(roiId)) {
      number += 1;
      roiId = `roi-${number}`;
    }
    draft.rois.push({
      roi_id: roiId,
      name: `Tüm Kare ${scenario.defaultName} ${number}`,
      roi_type: scenario.roiType,
      points: [
        { x: 0.01, y: 0.01 },
        { x: 0.99, y: 0.01 },
        { x: 0.99, y: 0.99 },
        { x: 0.01, y: 0.99 },
      ],
      closed: true,
    });
    const crossing = draft.analysis_options?.fence_crossing_rule;
    if (
      crossing?.enabled &&
      (!crossing.boundary_start || !crossing.boundary_end)
    ) {
      const derived = deriveFenceBoundary(draft.rois);
      if (derived) Object.assign(crossing, derived);
    }
    state.activeRoiId = roiId;
    refresh();
  }

  function updateClip(which, rawValue) {
    const draft = currentDraft();
    const video = currentVideo();
    if (!draft || !video) return;
    const duration = Number(video.duration_seconds);
    const step = frameStep(video);
    if (which === "start") {
      draft.start_seconds = clamp(rawValue, 0, Math.max(0, draft.end_seconds - step));
    } else {
      draft.end_seconds = clamp(rawValue, draft.start_seconds + step, duration);
    }
    refresh({ catalog: false, list: false });
  }

  function toggleVideo(videoId, checked) {
    if (checked) {
      state.selected.add(videoId);
      ensureDraft(videoId);
      openVideo(videoId);
    } else {
      state.selected.delete(videoId);
      if (videoId === state.activeVideoId) {
        state.drawingFenceLine = false;
      }
      const roi = currentRoi();
      if (videoId === state.activeVideoId && roi && !roi.closed && !roi.points.length) {
        deleteRoi();
      }
      refresh();
    }
  }

  function openVideo(videoId) {
    const video = state.byId.get(videoId);
    if (!video) return;
    state.activeVideoId = videoId;
    state.drawingFenceLine = false;
    const draft = ensureDraft(videoId);
    state.activeRoiId =
      draft.rois.find((roi) => !roi.closed)?.roi_id ||
      draft.rois[0]?.roi_id ||
      null;
    els.emptyWorkspace.hidden = true;
    els.editor.hidden = false;
    els.activeVideoTitle.textContent = `${video.video_id} · ${video.title}`;
    els.activeVideoMeta.textContent = `${video.camera} · ${video.width}×${video.height} · ${Number(video.fps).toLocaleString("tr-TR")} FPS · ${formatTime(video.duration_seconds)}`;
    els.stageMessage.textContent = "Video hazırlanıyor…";
    els.stageMessage.hidden = false;
    els.previewVideo.pause();
    if (els.previewVideo.dataset.videoId !== videoId) {
      els.previewVideo.dataset.videoId = videoId;
      els.previewVideo.poster = video.poster_url;
      els.previewVideo.src = video.media_url;
      els.previewVideo.load();
    }
    refresh();
    window.setTimeout(() => els.editor.scrollIntoView({ behavior: "smooth", block: "start" }), 20);
  }

  function updateTransport() {
    const duration = Number(els.previewVideo.duration) || Number(currentVideo()?.duration_seconds) || 0;
    const current = Number(els.previewVideo.currentTime) || 0;
    els.playhead.max = String(duration);
    els.playhead.step = String(frameStep());
    els.playhead.value = String(Math.min(current, duration));
    els.timeReadout.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
    els.playButton.innerHTML = els.previewVideo.paused
      ? '<span aria-hidden="true">▶</span><span>Oynat</span>'
      : '<span aria-hidden="true">Ⅱ</span><span>Duraklat</span>';
  }

  async function saveSelection() {
    const selectedIds = state.catalog
      .map((video) => video.video_id)
      .filter((videoId) => state.selected.has(videoId));
    if (!selectedIds.length || !selectedIds.every(videoReady)) return;
    const payload = {
      videos: selectedIds.map((videoId) => {
        const draft = ensureDraft(videoId);
        const scenario = scenarioForVideoId(videoId);
        return {
          video_id: videoId,
          scenario: scenario.scenario,
          start_seconds: round(draft.start_seconds, 6),
          end_seconds: round(draft.end_seconds, 6),
          analysis_options:
            scenario.scenario === "fence_security"
              ? {
                  fence_pose_roi: {
                    enabled:
                      draft.analysis_options.fence_pose_roi.enabled,
                    selected_keypoints: [
                      ...draft.analysis_options.fence_pose_roi
                        .selected_keypoints,
                    ],
                    inside_ratio_threshold: round(
                      draft.analysis_options.fence_pose_roi
                        .inside_ratio_threshold,
                      4,
                    ),
                    keypoint_confidence_threshold: round(
                      draft.analysis_options.fence_pose_roi
                        .keypoint_confidence_threshold,
                      4,
                    ),
                    minimum_visible_keypoints:
                      draft.analysis_options.fence_pose_roi
                        .minimum_visible_keypoints,
                  },
                  ...(draft.analysis_options.fence_crossing_rule
                    .enabled
                    ? {
                        fence_crossing_rule: {
                          enabled: true,
                          boundary_start: {
                            x: round(
                              draft.analysis_options
                                .fence_crossing_rule.boundary_start.x,
                              8,
                            ),
                            y: round(
                              draft.analysis_options
                                .fence_crossing_rule.boundary_start.y,
                              8,
                            ),
                          },
                          boundary_end: {
                            x: round(
                              draft.analysis_options
                                .fence_crossing_rule.boundary_end.x,
                              8,
                            ),
                            y: round(
                              draft.analysis_options
                                .fence_crossing_rule.boundary_end.y,
                              8,
                            ),
                          },
                          forbidden_side:
                            draft.analysis_options
                              .fence_crossing_rule.forbidden_side,
                          contact_band: round(
                            draft.analysis_options
                              .fence_crossing_rule.contact_band,
                            4,
                          ),
                          minimum_confidence: round(
                            draft.analysis_options
                              .fence_crossing_rule.minimum_confidence,
                            4,
                          ),
                          minimum_core_visible:
                            draft.analysis_options
                              .fence_crossing_rule.minimum_core_visible,
                          breach_enter_frames:
                            draft.analysis_options
                              .fence_crossing_rule.breach_enter_frames,
                          breach_exit_frames:
                            draft.analysis_options
                              .fence_crossing_rule.breach_exit_frames,
                          approach_keypoint_names: [
                            ...draft.analysis_options
                              .fence_crossing_rule
                              .approach_keypoint_names,
                          ],
                          approach_minimum_count:
                            draft.analysis_options
                              .fence_crossing_rule.approach_minimum_count,
                          wrist_contact_required:
                            draft.analysis_options
                              .fence_crossing_rule.wrist_contact_required,
                          hip_rise_ratio: round(
                            draft.analysis_options
                              .fence_crossing_rule.hip_rise_ratio,
                            4,
                          ),
                          raised_knee_ratio: round(
                            draft.analysis_options
                              .fence_crossing_rule.raised_knee_ratio,
                            4,
                          ),
                          climb_enter_frames:
                            draft.analysis_options
                              .fence_crossing_rule.climb_enter_frames,
                          climb_exit_frames:
                            draft.analysis_options
                              .fence_crossing_rule.climb_exit_frames,
                          history_window_frames:
                            draft.analysis_options
                              .fence_crossing_rule.history_window_frames,
                        },
                      }
                    : {}),
                }
              : {
                  forklift_driver_suppression: {
                    enabled:
                      draft.analysis_options
                        .forklift_driver_suppression.enabled,
                    suppressed_alerts: [
                      ...draft.analysis_options
                        .forklift_driver_suppression
                        .suppressed_alerts,
                    ],
                    minimum_forklift_confidence: round(
                      draft.analysis_options
                        .forklift_driver_suppression
                        .minimum_forklift_confidence,
                      4,
                    ),
                    minimum_person_ioa: round(
                      draft.analysis_options
                        .forklift_driver_suppression
                        .minimum_person_ioa,
                      4,
                    ),
                    enter_debounce_frames:
                      draft.analysis_options
                        .forklift_driver_suppression
                        .enter_debounce_frames,
                    exit_debounce_frames:
                      draft.analysis_options
                        .forklift_driver_suppression
                        .exit_debounce_frames,
                  },
                },
          rois: draft.rois.map((roi) => ({
            roi_id: roi.roi_id,
            name: roi.name.trim(),
            roi_type: roi.roi_type,
            points: roi.points.map((point) => ({
              x: round(point.x, 8),
              y: round(point.y, 8),
            })),
          })),
        };
      }),
    };
    els.saveButton.disabled = true;
    els.saveButton.textContent = "Kaydediliyor…";
    let saved = false;
    try {
      const response = await fetch("/api/selection", {
        method: "PUT",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = Array.isArray(result.detail)
          ? result.detail.map((item) => item.msg).join(" · ")
          : result.detail;
        throw new Error(detail || "Seçimler kaydedilemedi.");
      }
      saved = true;
      showMessage("Seçimler ve ROI alanları kaydedildi. İşleme henüz başlamadı.");
    } catch (error) {
      showMessage(error.message || "Seçimler kaydedilemedi.", true);
    } finally {
      els.saveButton.textContent = "Seçimleri ve alanları kaydet";
      updateProgress();
      if (saved) {
        els.saveHint.textContent = `${selectedIds.length} video kaydedildi. İşleme henüz başlamadı.`;
      }
    }
  }

  function bindEvents() {
    els.categoryFilters.addEventListener("click", (event) => {
      const button = event.target.closest("[data-category-filter]");
      if (!button || button.dataset.categoryFilter === state.activeCategory) return;
      state.activeCategory = button.dataset.categoryFilter;
      renderCatalog();
    });
    els.videoCatalog.addEventListener("change", (event) => {
      const input = event.target.closest("[data-select-video]");
      if (input) toggleVideo(input.dataset.selectVideo, input.checked);
    });
    els.videoCatalog.addEventListener("click", (event) => {
      const button = event.target.closest("[data-open-video]");
      if (button) openVideo(button.dataset.openVideo);
    });
    els.roiList.addEventListener("click", (event) => {
      const select = event.target.closest("[data-select-roi]");
      const item = event.target.closest("[data-roi-item]");
      if (select || (item && !event.target.closest("input"))) {
        selectRoi((select || item).dataset.selectRoi || item.dataset.roiItem);
      }
    });
    els.roiList.addEventListener("change", (event) => {
      const input = event.target.closest("[data-rename-roi]");
      if (!input) return;
      const draft = currentDraft();
      const roi = draft?.rois.find((item) => item.roi_id === input.dataset.renameRoi);
      if (!roi) return;
      const name = input.value.trim();
      if (!name) {
        input.value = roi.name;
        showMessage("Alan adı boş olamaz.", true);
        return;
      }
      const duplicate = draft.rois.some(
        (item) =>
          item.roi_id !== roi.roi_id && item.name.trim().toLocaleLowerCase("tr") === name.toLocaleLowerCase("tr"),
      );
      if (duplicate) {
        input.value = roi.name;
        showMessage("Aynı videodaki alan adları farklı olmalı.", true);
        return;
      }
      roi.name = name;
      refresh();
    });
    els.vertexEditor.addEventListener("change", (event) => {
      const input = event.target.closest("[data-vertex-axis]");
      if (!input) return;
      const roi = currentRoi();
      const point = roi?.points[Number(input.dataset.vertexIndex)];
      if (!point) return;
      point[input.dataset.vertexAxis] = clamp(input.value, 0, 1);
      refresh();
    });
    els.vertexEditor.addEventListener("click", (event) => {
      const button = event.target.closest("[data-delete-vertex]");
      if (!button) return;
      const roi = currentRoi();
      if (!roi) return;
      roi.points.splice(Number(button.dataset.deleteVertex), 1);
      if (roi.points.length < 3) roi.closed = false;
      refresh();
    });

    els.addRoiButton.addEventListener("click", addRoi);
    els.finishRoiButton.addEventListener("click", finishRoi);
    els.undoButton.addEventListener("click", undoPoint);
    els.fullFrameButton.addEventListener("click", addFullFrameRoi);
    els.deleteRoiButton.addEventListener("click", deleteRoi);
    els.saveButton.addEventListener("click", saveSelection);

    els.fenceCrossingEnabled.addEventListener("change", () => {
      const draft = currentDraft();
      const crossing =
        draft?.analysis_options?.fence_crossing_rule;
      if (!crossing) return;
      crossing.enabled = els.fenceCrossingEnabled.checked;
      if (crossing.enabled) {
        draft.analysis_options.fence_pose_roi.enabled = true;
        if (!crossing.boundary_start || !crossing.boundary_end) {
          const derived = deriveFenceBoundary(draft.rois);
          if (derived) Object.assign(crossing, derived);
        }
      } else {
        state.drawingFenceLine = false;
      }
      refresh();
    });
    els.drawFenceLineButton.addEventListener(
      "click",
      toggleFenceLineDrawing,
    );
    els.deleteFenceLineButton.addEventListener(
      "click",
      deleteFenceLine,
    );
    els.fenceForbiddenSide.addEventListener("change", (event) => {
      const input = event.target.closest(
        'input[name="fenceForbiddenSide"]',
      );
      const crossing =
        currentDraft()?.analysis_options?.fence_crossing_rule;
      if (!input || !crossing) return;
      crossing.forbidden_side = input.value === "right" ? "right" : "left";
      refresh({ catalog: false, list: false });
    });

    els.poseZoneEnabled.addEventListener("change", () => {
      const draft = currentDraft();
      const pose = draft?.analysis_options?.fence_pose_roi;
      const crossing = draft?.analysis_options?.fence_crossing_rule;
      if (!pose || crossing?.enabled) return;
      pose.enabled = els.poseZoneEnabled.checked;
      refresh({ catalog: false, list: false });
    });
    const updatePosePercentage = (value) => {
      const pose = currentDraft()?.analysis_options?.fence_pose_roi;
      if (!pose) return;
      const percentage = Math.round(clamp(value, 10, 100));
      pose.inside_ratio_threshold = percentage / 100;
      refresh({ catalog: false, list: false });
    };
    els.poseInsidePercent.addEventListener("input", () =>
      updatePosePercentage(els.poseInsidePercent.value),
    );
    els.poseInsidePercentNumber.addEventListener("change", () =>
      updatePosePercentage(els.poseInsidePercentNumber.value),
    );
    els.poseKeypointGrid.addEventListener("change", (event) => {
      const input = event.target.closest("[data-pose-keypoint]");
      const pose = currentDraft()?.analysis_options?.fence_pose_roi;
      if (!input || !pose) return;
      const checked = new Set(
        Array.from(
          els.poseKeypointGrid.querySelectorAll(
            "[data-pose-keypoint]:checked",
          ),
          (control) => control.dataset.poseKeypoint,
        ),
      );
      pose.selected_keypoints = COCO17_KEYPOINTS.map(
        (keypoint) => keypoint.id,
      ).filter((keypoint) => checked.has(keypoint));
      pose.minimum_visible_keypoints = Math.max(
        1,
        Math.min(4, pose.selected_keypoints.length || 1),
      );
      refresh({ catalog: false, list: false });
    });

    els.forkliftSuppressionEnabled.addEventListener("change", () => {
      const suppression =
        currentDraft()?.analysis_options?.forklift_driver_suppression;
      if (!suppression) return;
      suppression.enabled = els.forkliftSuppressionEnabled.checked;
      refresh({ catalog: false, list: false });
    });
    const updateForkliftAlerts = () => {
      const suppression =
        currentDraft()?.analysis_options?.forklift_driver_suppression;
      if (!suppression) return;
      suppression.suppressed_alerts = FORKLIFT_ALERT_ORDER.filter(
        (alert) =>
          (alert === "ppe_violation" &&
            els.forkliftPpeViolation.checked) ||
          (alert === "safe_walkway_violation" &&
            els.forkliftWalkwayViolation.checked),
      );
      refresh({ catalog: false, list: false });
    };
    els.forkliftPpeViolation.addEventListener(
      "change",
      updateForkliftAlerts,
    );
    els.forkliftWalkwayViolation.addEventListener(
      "change",
      updateForkliftAlerts,
    );
    const updateForkliftNumber = (field, value, minimum, maximum) => {
      const suppression =
        currentDraft()?.analysis_options?.forklift_driver_suppression;
      if (!suppression) return;
      const next = clamp(finiteOr(value, suppression[field]), minimum, maximum);
      suppression[field] = field.endsWith("_frames")
        ? Math.round(next)
        : round(next, 2);
      refresh({ catalog: false, list: false });
    };
    els.forkliftConfidence.addEventListener("change", () =>
      updateForkliftNumber(
        "minimum_forklift_confidence",
        els.forkliftConfidence.value,
        0.05,
        1,
      ),
    );
    els.forkliftPersonIoa.addEventListener("change", () =>
      updateForkliftNumber(
        "minimum_person_ioa",
        els.forkliftPersonIoa.value,
        0.05,
        1,
      ),
    );
    els.forkliftEnterFrames.addEventListener("change", () =>
      updateForkliftNumber(
        "enter_debounce_frames",
        els.forkliftEnterFrames.value,
        1,
        30,
      ),
    );
    els.forkliftExitFrames.addEventListener("change", () =>
      updateForkliftNumber(
        "exit_debounce_frames",
        els.forkliftExitFrames.value,
        1,
        60,
      ),
    );

    els.roiCanvas.addEventListener("pointerdown", (event) => {
      if (!state.selected.has(state.activeVideoId)) return;
      const crossing =
        currentDraft()?.analysis_options?.fence_crossing_rule;
      if (state.drawingFenceLine && crossing?.enabled) {
        const point = pointFromEvent(event);
        if (!point) {
          setCanvasStatus("Çit hattını video görüntüsünün içinde çiz.", true);
          return;
        }
        if (!crossing.boundary_start) {
          crossing.boundary_start = point;
        } else {
          const distance = Math.hypot(
            point.x - crossing.boundary_start.x,
            point.y - crossing.boundary_start.y,
          );
          if (distance < 0.01) {
            setCanvasStatus(
              "Çit hattının ikinci ucu ilk uçtan daha uzakta olmalı.",
              true,
            );
            return;
          }
          crossing.boundary_end = point;
          state.drawingFenceLine = false;
        }
        refresh();
        event.preventDefault();
        return;
      }
      const fenceEndpoint = nearestFenceEndpoint(event);
      if (fenceEndpoint) {
        state.dragging = {
          kind: "fence",
          pointerId: event.pointerId,
          field: fenceEndpoint.field,
        };
        els.roiCanvas.setPointerCapture(event.pointerId);
        els.roiCanvas.classList.add("dragging");
        event.preventDefault();
        return;
      }
      const vertex = nearestVertex(event);
      if (vertex) {
        state.activeRoiId = vertex.roi_id;
        state.dragging = {
          kind: "roi",
          pointerId: event.pointerId,
          roi_id: vertex.roi_id,
          index: vertex.index,
        };
        els.roiCanvas.setPointerCapture(event.pointerId);
        els.roiCanvas.classList.add("dragging");
        event.preventDefault();
        refresh();
        return;
      }
      const roi = currentRoi();
      if (!roi || roi.closed) return;
      const point = pointFromEvent(event);
      if (!point) {
        setCanvasStatus("Köşeyi video görüntüsünün içinde seç.", true);
        return;
      }
      roi.points.push(point);
      refresh();
      event.preventDefault();
    });
    els.roiCanvas.addEventListener("pointermove", (event) => {
      if (!state.dragging || state.dragging.pointerId !== event.pointerId) return;
      const point = pointFromEvent(event);
      if (state.dragging.kind === "fence") {
        const crossing =
          currentDraft()?.analysis_options?.fence_crossing_rule;
        if (!point || !crossing) return;
        crossing[state.dragging.field] = point;
        drawCanvas();
        scheduleLocalSave();
        event.preventDefault();
        return;
      }
      const draft = currentDraft();
      const roi = draft?.rois.find((item) => item.roi_id === state.dragging.roi_id);
      if (!point || !roi?.points[state.dragging.index]) return;
      roi.points[state.dragging.index] = point;
      drawCanvas();
      scheduleLocalSave();
      event.preventDefault();
    });
    const stopDragging = (event) => {
      if (!state.dragging || state.dragging.pointerId !== event.pointerId) return;
      state.dragging = null;
      els.roiCanvas.classList.remove("dragging");
      try {
        els.roiCanvas.releasePointerCapture(event.pointerId);
      } catch {
        // Pointer capture may already be gone.
      }
      refresh();
    };
    els.roiCanvas.addEventListener("pointerup", stopDragging);
    els.roiCanvas.addEventListener("pointercancel", stopDragging);
    els.roiCanvas.addEventListener("dblclick", (event) => {
      event.preventDefault();
      if (!state.drawingFenceLine) finishRoi();
    });
    els.roiCanvas.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !state.drawingFenceLine) {
        finishRoi();
        event.preventDefault();
      } else if (event.key === "Escape" && state.drawingFenceLine) {
        state.drawingFenceLine = false;
        refresh({ catalog: false, list: false });
        event.preventDefault();
      } else if (event.key === "Escape" || (event.ctrlKey && event.key.toLowerCase() === "z")) {
        undoPoint();
        event.preventDefault();
      } else if (event.key === "Delete" && currentRoi()) {
        deleteRoi();
        event.preventDefault();
      }
    });

    els.previewVideo.addEventListener("loadedmetadata", () => {
      const video = currentVideo();
      const draft = currentDraft();
      if (video && draft) {
        const actualDuration = Number(els.previewVideo.duration);
        if (Number.isFinite(actualDuration) && actualDuration > 0) {
          const difference = Math.abs(draft.end_seconds - Number(video.duration_seconds));
          if (difference <= 0.1) {
            draft.end_seconds = Math.min(actualDuration, Number(video.duration_seconds));
          }
        }
      }
      els.stageMessage.hidden = true;
      updateTransport();
      refresh();
    });
    els.previewVideo.addEventListener("canplay", () => {
      els.stageMessage.hidden = true;
    });
    els.previewVideo.addEventListener("error", () => {
      els.stageMessage.textContent = "Video açılamadı.";
      els.stageMessage.hidden = false;
    });
    els.previewVideo.addEventListener("timeupdate", updateTransport);
    els.previewVideo.addEventListener("play", updateTransport);
    els.previewVideo.addEventListener("pause", updateTransport);
    els.playButton.addEventListener("click", () => {
      if (els.previewVideo.paused) {
        els.previewVideo.play().catch(() => showMessage("Video oynatılamadı.", true));
      } else {
        els.previewVideo.pause();
      }
    });
    els.backButton.addEventListener("click", () => {
      els.previewVideo.currentTime = Math.max(0, els.previewVideo.currentTime - 1);
    });
    els.forwardButton.addEventListener("click", () => {
      els.previewVideo.currentTime = Math.min(
        Number(els.previewVideo.duration) || Number(currentVideo()?.duration_seconds) || 0,
        els.previewVideo.currentTime + 1,
      );
    });
    els.playhead.addEventListener("input", () => {
      els.previewVideo.currentTime = Number(els.playhead.value);
      updateTransport();
    });

    els.clipStart.addEventListener("change", () => updateClip("start", els.clipStart.value));
    els.clipStartRange.addEventListener("input", () =>
      updateClip("start", els.clipStartRange.value),
    );
    els.clipEnd.addEventListener("change", () => updateClip("end", els.clipEnd.value));
    els.clipEndRange.addEventListener("input", () =>
      updateClip("end", els.clipEndRange.value),
    );
    els.useCurrentStart.addEventListener("click", () =>
      updateClip("start", els.previewVideo.currentTime),
    );
    els.useCurrentEnd.addEventListener("click", () =>
      updateClip("end", els.previewVideo.currentTime),
    );
    els.resetClipButton.addEventListener("click", () => {
      const draft = currentDraft();
      const video = currentVideo();
      if (!draft || !video) return;
      draft.start_seconds = 0;
      draft.end_seconds = Number(video.duration_seconds);
      refresh({ catalog: false, list: false });
    });

    if ("ResizeObserver" in window) {
      new ResizeObserver(drawCanvas).observe(els.videoStage);
    } else {
      window.addEventListener("resize", drawCanvas);
    }
  }

  async function initialize() {
    try {
      const response = await fetch("/api/videos", {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error("Video listesi yüklenemedi.");
      const payload = await response.json();
      state.catalog = Array.isArray(payload.videos) ? payload.videos : [];
      state.catalogRevision = String(payload.catalog_revision || "");
      state.catalogRevisions =
        payload.catalog_revisions &&
        typeof payload.catalog_revisions === "object" &&
        !Array.isArray(payload.catalog_revisions)
          ? Object.fromEntries(
              Object.entries(payload.catalog_revisions).map(
                ([category, revision]) => [category, String(revision)],
              ),
            )
          : {};
      state.byId = new Map(state.catalog.map((video) => [video.video_id, video]));
      if (!state.catalog.length) throw new Error("Gösterilecek video bulunamadı.");
      if (!state.catalog.some((video) => video.category === state.activeCategory)) {
        state.activeCategory = state.catalog[0].category;
      }
      const restored = restoreLocalDraft();
      if (!restored) await restoreLatestServerSelection();
      if (state.activeVideoId) {
        state.activeCategory =
          state.byId.get(state.activeVideoId)?.category || state.activeCategory;
      }
      els.catalogStatus.hidden = true;
      els.videoCatalog.hidden = false;
      bindEvents();
      renderCatalog();
      updateProgress();
      if (state.activeVideoId) {
        openVideo(state.activeVideoId);
      }
    } catch (error) {
      els.catalogStatus.textContent = error.message || "Video listesi yüklenemedi.";
      els.catalogStatus.classList.add("error");
    }
  }

  initialize();
})();
