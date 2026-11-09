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
const cratePanel = document.getElementById("crate-panel");
const crateList = document.getElementById("crate-list");
const crateEmpty = document.getElementById("crate-empty");
const crateCount = document.getElementById("crate-count");
const crateActions = document.getElementById("crate-actions");
const discogsBtn = document.getElementById("discogs-info");
const wishlistBtn = document.getElementById("wishlist-check");
const refreshBtn = document.getElementById("wantlist-refresh");
const exportAiBtn = document.getElementById("export-ai");
const usernameInput = document.getElementById("discogs-username");
const DISCOGS_UA = "VinylCoverScanner/1.0 (local static crate tool; http://127.0.0.1:8000)";
const DISCOGS_USER_KEY = "vinyl-scanner.discogs.username";
const DEFAULT_DISCOGS_USER = "Look_at_my_scarface";
const PRIVATE_WANTLIST = "Список бажань приватний. Потрібен токен Discogs.";

let orb = null;
let matcher = null;
let records = [];
let ready = false;
let mode = "single";
let cameraOn = false;
let raf = 0;
let liveToken = 0;
let toastTimer = 0;
let discogsBusy = false;
let wishlistBusy = false;
let sessionCreated = null;
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

function appendLine(parent, className, text) {
  const line = document.createElement("p");
  line.className = className;
  line.textContent = text;
  parent.appendChild(line);
  return line;
}

function renderCrate() {
  crateList.replaceChildren();
  for (const item of session) {
    const li = document.createElement("li");
    const percent = Math.round(item.confidence * 100);
    appendLine(
      li,
      "crate-title",
      `${item.artist} — ${item.album} (${item.year}) · ${item.genre} · ${percent}%`
    );
    appendLine(li, "crate-detail", `Стилі: ${item.styles.join(", ")}`);
    if (item.discogs) {
      const info = item.discogs;
      const bits = [info.title, info.year, info.country, info.label].filter((part) => part != null && part !== "");
      appendLine(li, "crate-detail", `Discogs: ${bits.join(" · ")}`);
      if (info.uri) {
        const linkLine = document.createElement("p");
        linkLine.className = "crate-detail";
        if (isDiscogsHttps(info.uri)) {
          const link = document.createElement("a");
          link.href = info.uri;
          link.target = "_blank";
          link.rel = "noreferrer";
          link.textContent = info.uri;
          linkLine.appendChild(link);
        } else {
          linkLine.textContent = info.uri;
        }
        li.appendChild(linkLine);
      }
      const extra = [];
      if (info.community && typeof info.community === "object") {
        if (info.community.have != null) extra.push(`мають ${info.community.have}`);
        if (info.community.want != null) extra.push(`хочуть ${info.community.want}`);
      }
      if (info.price != null && info.price !== "") extra.push(`ціна ${info.price}`);
      if (extra.length) appendLine(li, "crate-detail", extra.join(" · "));
    } else if (item.discogsError) {
      appendLine(li, "crate-error", item.discogsError);
    }
    const wishClass = item.on_wishlist === true ? "crate-detail crate-wish-yes" : "crate-detail";
    appendLine(li, wishClass, `Список бажань: ${wishlistLabel(item.on_wishlist)}`);
    crateList.appendChild(li);
  }
  const hasItems = session.length > 0;
  crateEmpty.hidden = hasItems;
  crateActions.hidden = !hasItems;
  crateCount.textContent = String(session.length);
  discogsBtn.disabled = !hasItems || discogsBusy;
  wishlistBtn.disabled = !hasItems || wishlistBusy;
  refreshBtn.disabled = !hasItems || wishlistBusy;
  exportAiBtn.disabled = !hasItems;
}

function wishlistLabel(value) {
  if (value === true) return "так";
  if (value === false) return "ні";
  return "не перевірено";
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
  if (!sessionCreated) sessionCreated = new Date().toISOString();
  const styles = Array.isArray(hit.meta.styles) ? hit.meta.styles.slice() : [];
  session.push({
    id,
    artist: hit.meta.artist,
    album: hit.meta.album,
    year: hit.meta.year,
    genre: hit.meta.genre,
    styles,
    confidence: Number(hit.confidence.toFixed(4)),
    discogs: null,
    on_wishlist: null,
    discogsError: "",
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

function sessionDocument() {
  return {
    created: sessionCreated,
    items: session.map((item) => ({
      id: item.id,
      artist: item.artist,
      album: item.album,
      year: item.year,
      genre: item.genre,
      styles: item.styles.slice(),
      confidence: item.confidence,
      discogs: item.discogs,
      on_wishlist: item.on_wishlist,
    })),
  };
}

function sessionText() {
  const blocks = session.map((item) => {
    const url = item.discogs && item.discogs.uri ? item.discogs.uri : "none";
    const wish = item.on_wishlist === null ? "null" : String(item.on_wishlist);
    return [
      `${item.artist} — ${item.album}`,
      `year: ${item.year}`,
      `genre: ${item.genre}`,
      `styles: ${item.styles.join(", ")}`,
      `confidence: ${item.confidence}`,
      `discogs: ${url}`,
      `on_wishlist: ${wish}`,
    ].join("\n");
  });
  return `${blocks.join("\n\n")}\n`;
}

function localStamp(date = new Date()) {
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function exportForAi() {
  if (!session.length) return;
  const stamp = localStamp();
  const json = `${JSON.stringify(sessionDocument(), null, 2)}\n`;
  downloadBlob(new Blob([json], { type: "application/json" }), `crate_session_${stamp}.json`);
  window.setTimeout(() => {
    downloadBlob(new Blob([sessionText()], { type: "text/plain;charset=utf-8" }), `crate_session_${stamp}.txt`);
  }, 300);
}

function isDiscogsHttps(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && /(^|\.)discogs\.com$/i.test(url.hostname);
  } catch (err) {
    return false;
  }
}

function discogsPageUrl(uri) {
  if (!uri || typeof uri !== "string") return null;
  if (isDiscogsHttps(uri)) return uri;
  if (uri.startsWith("/")) return `https://www.discogs.com${uri}`;
  return null;
}

function firstLabel(label) {
  if (Array.isArray(label)) {
    const found = label.find((part) => typeof part === "string" && part.trim());
    return found ? found.trim() : null;
  }
  if (typeof label === "string" && label.trim()) return label.trim();
  return null;
}

function discogsFromResult(result) {
  const info = {
    id: result.id,
    title: result.title ?? null,
    year: result.year ?? null,
    label: firstLabel(result.label),
    country: result.country ?? null,
    uri: discogsPageUrl(result.uri),
  };
  if (result.community && typeof result.community === "object") info.community = result.community;
  if (result.price != null && result.price !== "") info.price = result.price;
  else if (result.lowest_price != null && result.lowest_price !== "") info.price = result.lowest_price;
  return info;
}

async function lookupDiscogs(item) {
  const params = new URLSearchParams({
    q: `${item.artist} ${item.album}`,
    type: "release",
  });
  const response = await fetch(`https://api.discogs.com/database/search?${params}`, {
    headers: {
      Accept: "application/json",
      "User-Agent": DISCOGS_UA,
    },
  });
  if (!response.ok) {
    throw new Error(`Discogs: помилка HTTP ${response.status}`);
  }
  const data = await response.json();
  const result = Array.isArray(data.results) ? data.results[0] : null;
  if (!result) throw new Error("Discogs: нічого не знайдено");
  return discogsFromResult(result);
}

async function fillDiscogs() {
  if (!session.length || discogsBusy) return;
  discogsBusy = true;
  renderCrate();
  try {
    for (const item of session) {
      item.discogsError = "Запит до Discogs…";
      renderCrate();
      try {
        item.discogs = await lookupDiscogs(item);
        item.discogsError = "";
      } catch (err) {
        item.discogs = null;
        item.discogsError = err && err.message ? err.message : "Discogs: запит не вдався";
      }
      renderCrate();
    }
  } finally {
    discogsBusy = false;
    renderCrate();
  }
}

function sameText(left, right) {
  return String(left || "").trim().toLowerCase() === String(right || "").trim().toLowerCase();
}

function cleanArtistName(name) {
  return String(name || "").replace(/\s+\(\d+\)$/, "").trim();
}

function compactWant(want) {
  if (!want || typeof want !== "object") return null;
  const info = want.basic_information;
  if (info && typeof info === "object") {
    const artists = Array.isArray(info.artists) ? info.artists : [];
    let joined = "";
    artists.forEach((artist, index) => {
      const name = cleanArtistName(artist && artist.name);
      if (!name) return;
      joined += name;
      if (index < artists.length - 1) joined += (artist && artist.join) || ", ";
    });
    const id = info.id != null ? info.id : want.id;
    if (id == null) return null;
    return { id, artist: joined.trim(), title: String(info.title || "") };
  }
  if (want.id == null) return null;
  return { id: want.id, artist: String(want.artist || ""), title: String(want.title || "") };
}

function wantlistHit(item, wants) {
  const discogsId = item.discogs && item.discogs.id != null ? String(item.discogs.id) : "";
  return wants.some((want) => {
    if (discogsId && String(want.id) === discogsId) return true;
    if (!sameText(want.title, item.album)) return false;
    if (sameText(want.artist, item.artist)) return true;
    const scanned = cleanArtistName(item.artist).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    if (!scanned) return false;
    return new RegExp(`(?:^|\\s*(?:,|&)\\s*)${scanned}(?:$|\\s*(?:,|&))`, "i").test(want.artist || "");
  });
}

function discogsHeaders() {
  return {
    Accept: "application/json",
    "User-Agent": DISCOGS_UA,
  };
}

function nextDiscogsUrl(value) {
  if (!value || typeof value !== "string") return "";
  try {
    const url = new URL(value);
    if (url.protocol === "https:" && url.hostname === "api.discogs.com") return url.toString();
  } catch (err) {
    return "";
  }
  return "";
}

async function loadWantlist(username) {
  const headers = discogsHeaders();
  const wants = [];
  const seen = new Set();
  let url = `https://api.discogs.com/users/${encodeURIComponent(username)}/wants?${new URLSearchParams({
    page: "1",
    per_page: "100",
  })}`;
  while (url && seen.size < 100) {
    if (seen.has(url)) break;
    seen.add(url);
    let response;
    try {
      response = await fetch(url, { headers });
    } catch (err) {
      // A private wantlist is HTTP 401, and that response has no CORS header,
      // so the browser reports a failed fetch instead of the status.
      throw new Error(PRIVATE_WANTLIST);
    }
    if (response.status === 401 || response.status === 403) throw new Error(PRIVATE_WANTLIST);
    if (!response.ok) throw new Error(`Discogs: помилка HTTP ${response.status}`);
    const data = await response.json();
    if (Array.isArray(data.wants)) wants.push(...data.wants);
    url = nextDiscogsUrl(data.pagination && data.pagination.urls && data.pagination.urls.next);
  }
  return wants;
}

function persistDiscogsAccount() {
  try {
    const username = usernameInput.value.trim();
    if (username) localStorage.setItem(DISCOGS_USER_KEY, username);
    else localStorage.removeItem(DISCOGS_USER_KEY);
    localStorage.removeItem("vinyl-scanner.discogs.token");
  } catch (err) {
    // Storage can be blocked. The typed username is still used for a refresh.
  }
}

function loadDiscogsAccount() {
  try {
    const username = localStorage.getItem(DISCOGS_USER_KEY);
    usernameInput.value = typeof username === "string" && username.trim() ? username : DEFAULT_DISCOGS_USER;
    localStorage.removeItem("vinyl-scanner.discogs.token");
  } catch (err) {
    usernameInput.value = DEFAULT_DISCOGS_USER;
  }
}

function applyWantlist(wants, source) {
  const compact = wants.map(compactWant).filter(Boolean);
  for (const item of session) item.on_wishlist = wantlistHit(item, compact);
  const yes = session.filter((item) => item.on_wishlist).length;
  setStatus(`Список бажань (${source}): ${yes} із ${session.length}.`);
}

async function checkWishlist() {
  if (!session.length || wishlistBusy) return;
  wishlistBusy = true;
  renderCrate();
  setStatus("Читаю локальний список бажань…");
  try {
    const response = await fetch("wantlist.json");
    if (!response.ok) throw new Error("wantlist.json не прочитано");
    const data = await response.json();
    if (!Array.isArray(data)) throw new Error("wantlist.json має бути масивом");
    applyWantlist(data, "локальний файл");
  } catch (err) {
    setStatus(err && err.message ? err.message : "Список бажань не прочитано.");
  } finally {
    wishlistBusy = false;
    renderCrate();
  }
}

async function refreshWantlist() {
  if (!session.length || wishlistBusy) return;
  const username = usernameInput.value.trim() || DEFAULT_DISCOGS_USER;
  if (/[\s/?#]/.test(username)) {
    setStatus("Ім’я Discogs містить недопустимі символи.");
    return;
  }
  persistDiscogsAccount();
  wishlistBusy = true;
  renderCrate();
  setStatus("Оновлюю список бажань з Discogs…");
  try {
    const wants = await loadWantlist(username);
    applyWantlist(wants, "Discogs");
  } catch (err) {
    setStatus(err && err.message ? err.message : "Список бажань не прочитано.");
  } finally {
    wishlistBusy = false;
    renderCrate();
  }
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
discogsBtn.addEventListener("click", () => {
  fillDiscogs().catch((err) => setStatus(err && err.message ? err.message : String(err)));
});
wishlistBtn.addEventListener("click", () => {
  checkWishlist().catch((err) => setStatus(err && err.message ? err.message : String(err)));
});
refreshBtn.addEventListener("click", () => {
  refreshWantlist().catch((err) => setStatus(err && err.message ? err.message : String(err)));
});
usernameInput.addEventListener("input", persistDiscogsAccount);
exportAiBtn.addEventListener("click", exportForAi);
loadDiscogsAccount();
renderCrate();

if (typeof cv === "undefined") {
  setStatus("OpenCV.js не завантажився. Потрібна сторінка по http, не file://.");
} else if (cv.Mat) {
  boot();
} else {
  cv.onRuntimeInitialized = boot;
}
