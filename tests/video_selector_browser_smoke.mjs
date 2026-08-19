const cdpBaseUrl = process.env.CDP_BASE_URL || "http://127.0.0.1:9223";

const pages = await (await fetch(`${cdpBaseUrl}/json/list`)).json();
const page = pages.find((item) => item.type === "page");
if (!page) {
  throw new Error("CDP page bulunamadı");
}

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let nextId = 1;
const pending = new Map();

socket.addEventListener("message", (event) => {
  const message = JSON.parse(String(event.data));
  if (!message.id || !pending.has(message.id)) return;
  const waiter = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) waiter.reject(new Error(message.error.message));
  else waiter.resolve(message.result);
});

function send(method, params = {}) {
  const id = nextId;
  nextId += 1;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

async function evaluate(expression) {
  const reply = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (reply.exceptionDetails) {
    throw new Error(reply.exceptionDetails.text || "Tarayıcı değerlendirme hatası");
  }
  return reply.result.value;
}

async function waitFor(expression, label) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (await evaluate(expression)) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`${label} zaman aşımına uğradı`);
}

try {
  await send("Runtime.enable");
  await waitFor(
    'document.querySelectorAll("[data-video-card]").length > 0',
    "Katalog",
  );
  await evaluate(
    'document.querySelector("[data-category-filter=\\"person_office\\"]").click()',
  );
  await waitFor(
    'document.querySelector("[data-select-video=\\"01\\"]") !== null',
    "Çit güvenliği kataloğu",
  );
  await evaluate(`(() => {
    const input = document.querySelector("[data-select-video=\\"01\\"]");
    if (!input.checked) input.click();
    else document.querySelector("[data-open-video=\\"01\\"]").click();
    return true;
  })()`);
  await waitFor('!document.getElementById("editor").hidden', "Editör");
  await waitFor(
    'document.getElementById("previewVideo").readyState >= 1',
    "Video metadata",
  );
  const poseDefaults = await evaluate(`(() => ({
    enabled: document.getElementById("poseZoneEnabled").checked,
    selected: document.querySelectorAll(
      "#poseKeypointGrid [data-pose-keypoint]:checked"
    ).length,
    percentage: document.getElementById("poseInsidePercentNumber").value,
    fenceVisible: !document.getElementById("fenceAnalysisOptions").hidden,
    ppeHidden: document.getElementById("ppeAnalysisOptions").hidden
  }))()`);
  if (
    !poseDefaults.enabled ||
    poseDefaults.selected !== 8 ||
    poseDefaults.percentage !== "50" ||
    !poseDefaults.fenceVisible ||
    !poseDefaults.ppeHidden
  ) {
    throw new Error(`Pose varsayılanları geçersiz: ${JSON.stringify(poseDefaults)}`);
  }
  await evaluate(`(() => {
    const input = document.getElementById("poseInsidePercentNumber");
    input.value = "60";
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  await evaluate('document.getElementById("addRoiButton").click()');

  const rect = await evaluate(`(() => {
    const canvas = document.getElementById("roiCanvas");
    const bounds = canvas.getBoundingClientRect();
    return {
      left: bounds.left,
      top: bounds.top,
      width: bounds.width,
      height: bounds.height
    };
  })()`);
  const points = [
    [0.18, 0.22],
    [0.78, 0.23],
    [0.82, 0.76],
    [0.2, 0.79],
  ];
  for (const [x, y] of points) {
    const clientX = rect.left + rect.width * x;
    const clientY = rect.top + rect.height * y;
    await evaluate(`(() => {
      const canvas = document.getElementById("roiCanvas");
      canvas.dispatchEvent(new PointerEvent("pointerdown", {
        bubbles: true,
        pointerId: 7,
        clientX: ${clientX},
        clientY: ${clientY}
      }));
      return true;
    })()`);
  }

  await evaluate('document.getElementById("finishRoiButton").click()');
  await waitFor(
    '!document.getElementById("saveButton").disabled',
    "Kayıt düğmesi",
  );
  await evaluate('document.getElementById("saveButton").click()');
  await waitFor(
    'document.getElementById("globalMessage").textContent.includes("İşleme henüz başlamadı")',
    "Seçim kaydı",
  );

  const savedContract = await evaluate(`(async () => {
    const [selectionResponse, queueResponse] = await Promise.all([
      fetch("/api/selection/latest", {
        headers: { Accept: "application/json" }
      }),
      fetch("/api/queue/latest", {
        headers: { Accept: "application/json" }
      })
    ]);
    const payload = await selectionResponse.json();
    const queue = await queueResponse.json();
    const video = payload.videos.find((item) => item.video_id === "01");
    const queueItem = queue.items.find((item) => item.video_id === "01");
    return {
      scenario: video?.scenario,
      roiType: video?.rois?.[0]?.roi_type,
      poseEnabled: video?.analysis_options?.fence_pose_roi?.enabled,
      selectedKeypoints:
        video?.analysis_options?.fence_pose_roi?.selected_keypoints?.length,
      ratio:
        video?.analysis_options?.fence_pose_roi?.inside_ratio_threshold,
      minimumVisible:
        video?.analysis_options?.fence_pose_roi?.minimum_visible_keypoints,
      requestedModules: queueItem?.requested_modules,
      anchor: queueItem?.alert_policy?.person_anchor
    };
  })()`);
  if (
    savedContract.scenario !== "fence_security" ||
    savedContract.roiType !== "restricted_zone" ||
    !savedContract.poseEnabled ||
    savedContract.selectedKeypoints !== 8 ||
    savedContract.ratio !== 0.6 ||
    savedContract.minimumVisible !== 4 ||
    JSON.stringify(savedContract.requestedModules) !==
      JSON.stringify(["person_roi", "pose"]) ||
    savedContract.anchor !== "pose_keypoint_ratio"
  ) {
    throw new Error(`Senaryo sözleşmesi geçersiz: ${JSON.stringify(savedContract)}`);
  }

  await evaluate(
    'document.querySelector("[data-category-filter=\\"ppe_safety\\"]").click()',
  );
  await waitFor(
    'document.querySelector("[data-video-card]")?.dataset.category === "ppe_safety"',
    "İSG / PPE kataloğu",
  );
  const firstPpe = await evaluate(`(() => {
    const card = document.querySelector("[data-video-card]");
    return {
      videoId: card?.dataset.videoCard,
      isNew: Boolean(card?.querySelector(".new-video-badge"))
    };
  })()`);
  if (!/^S\d+$/.test(firstPpe.videoId || "") || !firstPpe.isNew) {
    throw new Error(`Yeni İSG videosu katalog başında değil: ${JSON.stringify(firstPpe)}`);
  }
  await evaluate(`(() => {
    const input = document.querySelector(
      "[data-select-video=\\"${firstPpe.videoId}\\"]"
    );
    if (!input.checked) input.click();
    else document.querySelector(
      "[data-open-video=\\"${firstPpe.videoId}\\"]"
    ).click();
    return true;
  })()`);
  await waitFor(
    'document.getElementById("scenarioRuleCard").dataset.scenario === "ppe_safety"',
    "İSG / PPE senaryosu",
  );
  const forkliftDefaults = await evaluate(`(() => ({
    enabled: document.getElementById("forkliftSuppressionEnabled").checked,
    ppe: document.getElementById("forkliftPpeViolation").checked,
    walkway: document.getElementById("forkliftWalkwayViolation").checked,
    confidence: document.getElementById("forkliftConfidence").value,
    ioa: document.getElementById("forkliftPersonIoa").value,
    fenceHidden: document.getElementById("fenceAnalysisOptions").hidden,
    ppeVisible: !document.getElementById("ppeAnalysisOptions").hidden
  }))()`);
  if (
    !forkliftDefaults.enabled ||
    !forkliftDefaults.ppe ||
    !forkliftDefaults.walkway ||
    forkliftDefaults.confidence !== "0.35" ||
    forkliftDefaults.ioa !== "0.55" ||
    !forkliftDefaults.fenceHidden ||
    !forkliftDefaults.ppeVisible
  ) {
    throw new Error(
      `Forklift varsayılanları geçersiz: ${JSON.stringify(forkliftDefaults)}`,
    );
  }
  await waitFor(
    '!document.getElementById("saveButton").disabled && document.getElementById("roiCount").textContent.startsWith("0")',
    "ROI olmadan İSG hazırlığı",
  );
  await evaluate('document.getElementById("saveButton").click()');
  await waitFor(
    `Promise.all([
      fetch("/api/selection/latest").then((response) => response.json()),
      fetch("/api/queue/latest").then((response) => response.json())
    ]).then(([selection, queue]) => {
      const item = selection.videos.find(
        (candidate) => candidate.video_id === "${firstPpe.videoId}"
      );
      const queueItem = queue.items.find(
        (candidate) => candidate.video_id === "${firstPpe.videoId}"
      );
      return (
        item?.scenario === "ppe_safety" &&
        Array.isArray(item?.rois) &&
        item.rois.length === 0 &&
        item?.analysis_options?.forklift_driver_suppression?.enabled === true &&
        item.analysis_options.forklift_driver_suppression
          .minimum_forklift_confidence === 0.35 &&
        item.analysis_options.forklift_driver_suppression
          .minimum_person_ioa === 0.55 &&
        JSON.stringify(queueItem?.requested_modules) ===
          JSON.stringify(["ppe", "forklift"]) &&
        queueItem?.alert_policy?.forklift_driver_suppression?.enabled === true
      );
    })`,
    "ROI olmadan İSG kaydı",
  );

  const outcome = await evaluate(`({
    cards: document.querySelectorAll("[data-video-card]").length,
    selected: document.querySelectorAll("[data-select-video]:checked").length,
    vertices: document.querySelectorAll(".vertex-row").length,
    ready: document.getElementById("readyCount").textContent.trim(),
    scenario: document.getElementById("scenarioRuleCard").dataset.scenario,
    firstPpe: document.querySelector("[data-video-card]")?.dataset.videoCard,
    firstPpeIsNew: Boolean(document.querySelector("[data-video-card] .new-video-badge")),
    message: document.getElementById("globalMessage").textContent.trim()
  })`);
  process.stdout.write(`${JSON.stringify(outcome)}\n`);
} finally {
  socket.close();
}
