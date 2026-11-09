"use strict";

// Same gates as the Python scanner: ORB nfeatures 1000, longest side 800,
// Lowe ratio 0.75, more than 15 good matches, RANSAC 5px, then a real quad
// with confidence above 85% or more than 20 inliers.
const MARGIN = 0.12;
const RATIO = 0.75;
const MIN_GOOD = 15;
const MIN_CONFIDENCE = 0.85;
const MIN_INLIERS = 20;
const RANSAC_REPROJ = 5.0;
const MAX_EDGE = 800;
const ORB_NFEATURES = 1000;
const MIN_FEATURES = 8;
const CRATE_COOLDOWN_MS = 5000;

const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const bannerEl = document.getElementById("banner");
const toastEl = document.getElementById("toast");
const view = document.getElementById("view");
const ctx = view.getContext("2d", { willReadFrequently: true });
const video = document.getElementById("camera");
const startBtn = document.getElementById("start-camera");
const stopBtn = document.getElementById("stop");
const ultraBtn = document.getElementById("try-ultra");
const fileInput = document.getElementById("file-input");
const exportBtn = document.getElementById("export-session");
const cratePanel = document.getElementById("crate-panel");
const crateList = document.getElementById("crate-list");
const crateEmpty = document.getElementById("crate-empty");
const crateCount = document.getElementById("crate-count");

let orb = null;
let matcher = null;
let records = [];
let ready = false;
let mode = "single";
let cameraOn = false;
let raf = 0;
let liveToken = 0;
let toastTimer = 0;
const session = [];
const lastAddedAt = new Map();

function setStatus(text) {
  statusEl.textContent = text;
}

function viewfinderBox(width, height) {
  let marginX = Math.round(width * MARGIN);
  let marginY = Math.round(height * MARGIN);
  marginX = Math.min(marginX, Math.max(0, Math.floor((width - 64) / 2)));
  marginY = Math.min(marginY, Math.max(0, Math.floor((height - 64) / 2)));
  const x0 = marginX;
  const y0 = marginY;
  const x1 = width - marginX;
  const y1 = height - marginY;
  if (x1 <= x0 || y1 <= y0) return [0, 0, width, height];
  return [x0, y0, x1, y1];
}

function deleteIf(mat) {
  if (mat && typeof mat.delete === "function") mat.delete();
}

function toGray(src) {
  const gray = new cv.Mat();
  const channels = src.channels();
  if (channels === 1) src.copyTo(gray);
  else if (channels === 4) cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  else cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
  return gray;
}

function resizeMax(src) {
  const longest = Math.max(src.rows, src.cols);
  if (longest <= MAX_EDGE) return src.clone();
  const scale = MAX_EDGE / longest;
  const dst = new cv.Mat();
  const size = new cv.Size(
    Math.max(1, Math.round(src.cols * scale)),
    Math.max(1, Math.round(src.rows * scale))
  );
  cv.resize(src, dst, size, 0, 0, cv.INTER_AREA);
  deleteIf(size);
  return dst;
}

function detectPoints(gray) {
  const height = gray.rows;
  const width = gray.cols;
  let scale = 1;
  let work = gray;
  let resized = null;
  const longest = Math.max(height, width);
  if (longest > MAX_EDGE) {
    scale = MAX_EDGE / longest;
    resized = new cv.Mat();
    const size = new cv.Size(
      Math.max(1, Math.round(width * scale)),
      Math.max(1, Math.round(height * scale))
    );
    cv.resize(gray, resized, size, 0, 0, cv.INTER_AREA);
    deleteIf(size);
    work = resized;
  }
  const keypoints = new cv.KeyPointVector();
  const descriptors = new cv.Mat();
  const noMask = new cv.Mat();
  try {
    orb.detectAndCompute(work, noMask, keypoints, descriptors);
    if (keypoints.size() < 2 || descriptors.rows < 2) {
      descriptors.delete();
      return null;
    }
    const pts = new Float32Array(keypoints.size() * 2);
    for (let i = 0; i < keypoints.size(); i += 1) {
      const kp = keypoints.get(i);
      pts[i * 2] = kp.pt.x / scale;
      pts[i * 2 + 1] = kp.pt.y / scale;
    }
    return { pts, descriptors };
  } catch (err) {
    descriptors.delete();
    throw err;
  } finally {
    keypoints.delete();
    noMask.delete();
    deleteIf(resized);
  }
}

function matAt(matrix, row, col) {
  const depth = matrix.type() & 7;
  if (depth === cv.CV_64F) return matrix.doubleAt(row, col);
  return matrix.floatAt(row, col);
}

function quadIsValid(pts, frameW, frameH) {
  if (!pts || pts.length !== 4) return false;
  for (const p of pts) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return false;
  }
  const rounded = [];
  for (const p of pts) rounded.push(Math.round(p.x), Math.round(p.y));
  const contour = cv.matFromArray(4, 1, cv.CV_32SC2, rounded);
  let convex = false;
  try {
    convex = cv.isContourConvex(contour);
  } catch (err) {
    contour.delete();
    return false;
  }
  contour.delete();
  if (!convex) return false;

  const contourF = cv.matFromArray(4, 1, cv.CV_32FC2, rounded);
  let area = 0;
  try {
    area = Math.abs(cv.contourArea(contourF));
  } finally {
    contourF.delete();
  }
  const minArea = Math.max(800, 0.015 * frameW * frameH);
  if (area < minArea) return false;

  let minSide = Infinity;
  let maxSide = 0;
  for (let i = 0; i < 4; i += 1) {
    const a = pts[i];
    const b = pts[(i + 1) % 4];
    const side = Math.hypot(b.x - a.x, b.y - a.y);
    if (side < minSide) minSide = side;
    if (side > maxSide) maxSide = side;
  }
  if (minSide < 20) return false;
  if (maxSide > 4 * Math.max(frameW, frameH)) return false;
  const cx = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4;
  const cy = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4;
  return cx >= 0 && cx <= frameW && cy >= 0 && cy <= frameH;
}

function tryRecord(queryPts, queryDes, record, offset, frameSize) {
  if (record.descriptors.rows < 2) return null;
  const matches = new cv.DMatchVectorVector();
  let srcMat = null;
  let dstMat = null;
  let mask = null;
  let homography = null;
  let normalized = null;
  let corners = null;
  let projected = null;
  try {
    matcher.knnMatch(queryDes, record.descriptors, matches, 2);
    const good = [];
    for (let i = 0; i < matches.size(); i += 1) {
      const pair = matches.get(i);
      if (pair.size() < 2) continue;
      const best = pair.get(0);
      const second = pair.get(1);
      if (best.distance < RATIO * second.distance) {
        good.push([best.queryIdx, best.trainIdx]);
      }
    }
    if (good.length <= MIN_GOOD) return null;

    const srcArr = [];
    const dstArr = [];
    for (const [queryIdx, trainIdx] of good) {
      srcArr.push(record.points[trainIdx * 2], record.points[trainIdx * 2 + 1]);
      dstArr.push(queryPts[queryIdx * 2], queryPts[queryIdx * 2 + 1]);
    }
    srcMat = cv.matFromArray(good.length, 1, cv.CV_32FC2, srcArr);
    dstMat = cv.matFromArray(good.length, 1, cv.CV_32FC2, dstArr);
    mask = new cv.Mat();
    try {
      homography = cv.findHomography(srcMat, dstMat, cv.RANSAC, RANSAC_REPROJ, mask);
    } catch (err) {
      return null;
    }
    if (!homography || homography.empty() || homography.rows !== 3 || homography.cols !== 3) {
      return null;
    }
    const h22 = matAt(homography, 2, 2);
    if (!Number.isFinite(h22) || Math.abs(h22) < 1e-8) return null;
    const vals = [];
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) {
        const value = matAt(homography, r, c) / h22;
        if (!Number.isFinite(value)) return null;
        vals.push(value);
      }
    }
    normalized = cv.matFromArray(3, 3, cv.CV_64FC1, vals);

    const maskData = mask.data;
    let inliers = 0;
    if (maskData) {
      for (let i = 0; i < maskData.length; i += 1) if (maskData[i]) inliers += 1;
    }
    const confidence = inliers / good.length;
    if (!(confidence > MIN_CONFIDENCE || inliers > MIN_INLIERS)) return null;

    corners = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      record.width, 0,
      record.width, record.height,
      0, record.height,
    ]);
    projected = new cv.Mat();
    try {
      cv.perspectiveTransform(corners, projected, normalized);
    } catch (err) {
      return null;
    }
    const data = projected.data32F;
    if (!data || data.length < 8) return null;
    const quad = [];
    for (let i = 0; i < 4; i += 1) {
      const x = data[i * 2] + offset[0];
      const y = data[i * 2 + 1] + offset[1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
      quad.push({ x, y });
    }
    const [frameW, frameH] = frameSize;
    if (!quadIsValid(quad, frameW, frameH)) return null;
    return { meta: record.meta, quad, inliers, confidence };
  } finally {
    matches.delete();
    deleteIf(srcMat);
    deleteIf(dstMat);
    deleteIf(mask);
    deleteIf(homography);
    deleteIf(normalized);
    deleteIf(corners);
    deleteIf(projected);
  }
}

function matchRoi(roi, offset, frameSize) {
  const gray = toGray(roi);
  let detected = null;
  try {
    detected = detectPoints(gray);
    if (!detected) return null;
    let best = null;
    for (const record of records) {
      const hit = tryRecord(detected.pts, detected.descriptors, record, offset, frameSize);
      if (!hit) continue;
      if (
        !best ||
        hit.inliers > best.inliers ||
        (hit.inliers === best.inliers && hit.confidence > best.confidence)
      ) {
        best = hit;
      }
    }
    return best;
  } finally {
    gray.delete();
    if (detected) detected.descriptors.delete();
  }
}

function matchFrame(frame) {
  const frameH = frame.rows;
  const frameW = frame.cols;
  const box = viewfinderBox(frameW, frameH);
  const [x0, y0, x1, y1] = box;
  const rect = new cv.Rect(x0, y0, x1 - x0, y1 - y0);
  const header = frame.roi(rect);
  const roi = header.clone();
  header.delete();
  deleteIf(rect);
  const started = performance.now();
  try {
    const hit = matchRoi(roi, [x0, y0], [frameW, frameH]);
    return { hit, ms: performance.now() - started, box };
  } finally {
    roi.delete();
  }
}

function ownershipText(meta) {
  const tags = [];
  if (Object.prototype.hasOwnProperty.call(meta, "spotify_owned")) {
    tags.push(meta.spotify_owned ? "[IN SPOTIFY]" : "[NOT IN SPOTIFY]");
  } else if (Object.prototype.hasOwnProperty.call(meta, "in_library")) {
    tags.push(meta.in_library ? "[IN SPOTIFY]" : "[NOT IN SPOTIFY]");
  }
  if (Object.prototype.hasOwnProperty.call(meta, "discogs_owned")) {
    tags.push(meta.discogs_owned ? "[DISCOGS OWNED]" : "[NOT ON DISCOGS]");
  }
  return tags.length ? tags.join(" ") : "ownership unknown";
}

function discogsPriceLine(meta) {
  for (const key of ["discogs_price", "price"]) {
    if (!Object.prototype.hasOwnProperty.call(meta, key)) continue;
    const value = meta[key];
    if (typeof value === "number" && Number.isFinite(value)) return `Discogs: ${value}`;
    if (typeof value === "string" && value.trim()) return `Discogs: ${value.trim()}`;
  }
  return null;
}

function bannerLines(meta) {
  const styles = Array.isArray(meta.styles) ? meta.styles : [];
  const primary = styles.length ? styles[0] : String(meta.genre || "");
  let hint = String(meta.genre || "");
  if (hint === primary && styles.length > 1) hint = styles[1];
  const lines = [
    String(meta.artist || ""),
    `${meta.album} (${meta.year})`,
    `Style: ${primary}`,
    `Related: ${hint}`,
    ownershipText(meta),
  ];
  const price = discogsPriceLine(meta);
  if (price) lines.push(price);
  return lines;
}

function formatOutcome(outcome) {
  if (!outcome.hit) return `NO MATCH time_ms=${outcome.ms.toFixed(1)}`;
  const meta = outcome.hit.meta;
  const percent = Math.round(outcome.hit.confidence * 100);
  return (
    `MATCH id=${meta.id} artist="${meta.artist}" album="${meta.album}" ` +
    `inliers=${outcome.hit.inliers} confidence=${percent}% time_ms=${outcome.ms.toFixed(1)}`
  );
}

function showBanner(hit) {
  bannerEl.replaceChildren();
  if (!hit) {
    bannerEl.hidden = true;
    return;
  }
  for (const line of bannerLines(hit.meta)) {
    const row = document.createElement("p");
    row.textContent = line;
    bannerEl.appendChild(row);
  }
  bannerEl.hidden = false;
}

function drawOverlay(outcome) {
  const [x0, y0, x1, y1] = outcome.box;
  ctx.save();
  ctx.strokeStyle = "#00ff00";
  ctx.lineWidth = 2;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  if (outcome.hit) {
    ctx.lineWidth = 3;
    ctx.beginPath();
    outcome.hit.quad.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

function renderCrate() {
  crateList.replaceChildren();
  for (const item of session) {
    const li = document.createElement("li");
    const percent = Math.round(item.confidence * 100);
    li.textContent = `${item.artist} — ${item.album} (${item.year}) · ${item.genre} · ${percent}%`;
    crateList.appendChild(li);
  }
  crateEmpty.hidden = session.length > 0;
  crateCount.textContent = String(session.length);
  exportBtn.disabled = session.length === 0;
  exportBtn.textContent = session.length ? `Експорт сесії (${session.length})` : "Експорт сесії";
}

function showToast(text) {
  toastEl.textContent = text;
  toastEl.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toastEl.hidden = true;
  }, 2200);
}

function vibrateTick() {
  if (typeof navigator.vibrate !== "function") return;
  try {
    navigator.vibrate(80);
  } catch (err) {
    // Some browsers expose vibrate and then reject it. The toast still stands.
  }
}

function considerCrate(hit) {
  const id = String(hit.meta.id);
  const now = performance.now();
  const previous = lastAddedAt.get(id);
  if (previous != null && now - previous < CRATE_COOLDOWN_MS) return;
  lastAddedAt.set(id, now);
  const styles = Array.isArray(hit.meta.styles) ? hit.meta.styles.slice() : [];
  session.push({
    id,
    artist: hit.meta.artist,
    album: hit.meta.album,
    year: hit.meta.year,
    genre: hit.meta.genre,
    styles,
    confidence: hit.confidence,
  });
  renderCrate();
  showToast(`Додано: ${hit.meta.artist} — ${hit.meta.album}`);
  vibrateTick();
}

function present(outcome) {
  resultEl.textContent = formatOutcome(outcome);
  resultEl.dataset.state = outcome.hit ? "match" : "nomatch";
  if (outcome.hit) resultEl.dataset.id = String(outcome.hit.meta.id);
  else delete resultEl.dataset.id;
  showBanner(outcome.hit);
  if (outcome.hit && mode === "crate") considerCrate(outcome.hit);
}

function sessionPayload() {
  return session.map((item) => ({
    artist: item.artist,
    album: item.album,
    year: item.year,
    genre: item.genre,
    styles: item.styles.slice(),
    confidence: Number(item.confidence.toFixed(4)),
  }));
}

function exportSession() {
  if (!session.length) return;
  const now = new Date();
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
  ].join("-");
  const blob = new Blob([`${JSON.stringify(sessionPayload(), null, 2)}\n`], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `crate_session_${stamp}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function setMode(next) {
  mode = next === "crate" ? "crate" : "single";
  document.body.dataset.mode = mode;
  document.getElementById("mode-single").setAttribute("aria-pressed", mode === "single" ? "true" : "false");
  document.getElementById("mode-crate").setAttribute("aria-pressed", mode === "crate" ? "true" : "false");
  cratePanel.hidden = mode !== "crate";
}

function drawPlaceholder() {
  view.width = 960;
  view.height = 720;
  ctx.fillStyle = "#070708";
  ctx.fillRect(0, 0, view.width, view.height);
  const box = viewfinderBox(view.width, view.height);
  ctx.strokeStyle = "#00ff00";
  ctx.lineWidth = 2;
  ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
}

function stopCamera() {
  cameraOn = false;
  liveToken += 1;
  if (raf) {
    cancelAnimationFrame(raf);
    raf = 0;
  }
  const stream = video.srcObject;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    video.srcObject = null;
  }
  stopBtn.disabled = true;
}

function tick() {
  if (!cameraOn) return;
  try {
    if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
      if (view.width !== video.videoWidth) view.width = video.videoWidth;
      if (view.height !== video.videoHeight) view.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, view.width, view.height);
      const frame = cv.imread(view);
      let outcome;
      try {
        outcome = matchFrame(frame);
      } finally {
        frame.delete();
      }
      if (!cameraOn) return;
      drawOverlay(outcome);
      present(outcome);
    }
  } catch (err) {
    stopCamera();
    setStatus(err && err.message ? err.message : String(err));
    return;
  }
  if (cameraOn) raf = requestAnimationFrame(tick);
}

async function startCamera() {
  stopCamera();
  const token = liveToken;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Камера недоступна. Оберіть зображення або обкладинку Ultra.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    if (token !== liveToken) {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    video.srcObject = stream;
    await video.play();
    if (token !== liveToken) {
      stopCamera();
      return;
    }
    cameraOn = true;
    stopBtn.disabled = false;
    setStatus("Камера увімкнена. Тримайте обкладинку в зеленій рамці.");
    raf = requestAnimationFrame(tick);
  } catch (err) {
    if (token !== liveToken) return;
    setStatus("Камеру відхилено. Оберіть зображення або обкладинку Ultra.");
  }
}

function runStill(image) {
  if (!ready) return;
  stopCamera();
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (!width || !height) {
    setStatus("Зображення ще не прочитане.");
    return;
  }
  view.width = width;
  view.height = height;
  ctx.drawImage(image, 0, 0, width, height);
  const frame = cv.imread(view);
  let outcome;
  try {
    outcome = matchFrame(frame);
  } finally {
    frame.delete();
  }
  drawOverlay(outcome);
  present(outcome);
  if (!outcome.hit) setStatus("Збігу немає. Обкладинка має бути в зеленій рамці.");
  else setStatus(mode === "crate" ? "Збіг додано до сесії, якщо id не був у останні 5 с." : "Збіг знайдено.");
}

function indexCover(image, meta) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const local = canvas.getContext("2d", { willReadFrequently: true });
  local.drawImage(image, 0, 0);
  const rgba = cv.imread(canvas);
  const resized = resizeMax(rgba);
  rgba.delete();
  const gray = toGray(resized);
  const width = resized.cols;
  const height = resized.rows;
  resized.delete();
  let detected = null;
  try {
    detected = detectPoints(gray);
  } finally {
    gray.delete();
  }
  const count = detected ? detected.descriptors.rows : 0;
  if (!detected || count < MIN_FEATURES) {
    if (detected) detected.descriptors.delete();
    throw new Error(`Замало ORB ознак (${count}) для ${meta.cover_filename}`);
  }
  return {
    meta,
    width,
    height,
    points: detected.pts,
    descriptors: detected.descriptors,
    count,
  };
}

function enableControls() {
  startBtn.disabled = false;
  ultraBtn.disabled = false;
  fileInput.disabled = false;
}

async function boot() {
  try {
    setStatus("Індексую обкладинки…");
    const harris = typeof cv.ORB_HARRIS_SCORE === "number" ? cv.ORB_HARRIS_SCORE : 0;
    orb = new cv.ORB(ORB_NFEATURES, 1.2, 8, 31, 0, 2, harris, 31, 20);
    if (typeof orb.setMaxFeatures === "function") orb.setMaxFeatures(ORB_NFEATURES);
    matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);

    const response = await fetch("database.json");
    if (!response.ok) throw new Error("database.json не прочитано");
    const database = await response.json();
    if (!Array.isArray(database) || !database.length) throw new Error("database.json порожній");

    const images = [...document.querySelectorAll("img[data-cover]")];
    await Promise.all(images.map((image) => (image.decode ? image.decode() : Promise.resolve())));

    const built = [];
    for (const meta of database) {
      const image = images.find((candidate) => candidate.dataset.cover === meta.cover_filename);
      if (!image) throw new Error(`Немає файлу ${meta.cover_filename}`);
      built.push(indexCover(image, meta));
    }
    records = built;
    ready = true;
    document.documentElement.dataset.ready = "1";
    enableControls();
    drawPlaceholder();
    const counts = built.map((record) => `${record.meta.album} ${record.count}`).join(", ");
    setStatus(`Готово. Індекс: ${counts}.`);
    startCamera();
  } catch (err) {
    setStatus(err && err.message ? err.message : String(err));
  }
}

document.getElementById("mode-single").addEventListener("click", () => setMode("single"));
document.getElementById("mode-crate").addEventListener("click", () => setMode("crate"));
startBtn.addEventListener("click", () => startCamera());
stopBtn.addEventListener("click", () => {
  stopCamera();
  setStatus("Камеру зупинено.");
});
ultraBtn.addEventListener("click", () => runStill(document.getElementById("img-ultra")));
fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    runStill(image);
    URL.revokeObjectURL(url);
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    setStatus("Не вдалося прочитати файл.");
  };
  image.src = url;
});
exportBtn.addEventListener("click", exportSession);
renderCrate();

if (typeof cv === "undefined") {
  setStatus("OpenCV.js не завантажився. Потрібна сторінка по http, не file://.");
} else if (cv.Mat) {
  boot();
} else {
  cv.onRuntimeInitialized = boot;
}
