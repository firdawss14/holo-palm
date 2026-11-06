/**
 * handTracking.js
 * ---------------------------------------------------------------------------
 * MediaPipe Hand Landmarker wrapper + everything that turns raw landmarks into
 * stable, usable interaction signals:
 *
 *   - Primary / secondary hand assignment that stays stable across frames
 *   - Palm centre smoothing (One-Euro filter -> low jitter AND low latency)
 *   - Landmark-geometry gesture classification (open palm / fist / pinch / spread)
 *     normalised by hand size, with hysteresis + multi-frame hold (debounce)
 *   - Swipe detection (min distance, min velocity, time window, cooldown)
 *   - Guards against accidental triggers when a hand enters / leaves the frame
 *   - ViewMapper: maps video-normalised landmarks to the (mirrored, object-fit:
 *     cover) camera panel so overlays and 3D content line up with the video.
 *
 * Coordinate conventions
 *   Landmarks from MediaPipe are normalised to the *raw, un-mirrored* video frame
 *   (x: 0..1 left->right, y: 0..1 top->bottom). The video is *displayed* mirrored
 *   (selfie view), so ViewMapper flips x when going to panel pixels.
 */

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// ---------------------------------------------------------------------------
// Configuration (all tuning knobs live here)
// ---------------------------------------------------------------------------
export const TRACKING_CONFIG = {
  wasmPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
  modelPath:
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  numHands: 2,
  minHandDetectionConfidence: 0.6,
  minHandPresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,

  /**
   * How a NEW hand gets its role:
   *  'first-seen'  -> first hand shown becomes PRIMARY, the next one SECONDARY.
   *                   (If both appear in the same frame, handedness decides:
   *                   the user's right hand = primary.)
   *  'handedness'  -> the user's right hand is always PRIMARY, left = SECONDARY.
   * Once assigned, a role sticks to the hand by position-tracking, not by label.
   */
  assignMode: 'first-seen',
  maxMatchDist: 0.35, // max wrist travel (in frame widths) between frames to be "the same hand"

  // Lifecycle (in detection frames, ~30/s)
  enterFrames: 4, // hand must be seen this long before it counts as "active"
  lostGraceFrames: 5, // tolerate brief detection dropouts without losing the hand
  releaseFrames: 14, // after this many missed frames the role slot is freed

  // Frames a candidate gesture must persist before it is committed
  hold: { open: 2, fist: 3, unknown: 4, pinch: 2, spread: 2, none: 2 }, // low = fast response

  // Finger extension = dist(tip, wrist) / dist(mcp, wrist). Hysteresis pair.
  fingerExtendOn: 1.5,
  fingerExtendOff: 1.3,

  // Thumb-tip <-> index-tip distance / hand size. Hysteresis pairs.
  pinchOn: 0.27,
  pinchOff: 0.38,
  spreadOn: 1.25,
  spreadOff: 1.05,

  // Swipe (horizontal, in mirrored/screen space; units = frame widths)
  swipe: {
    enabled: false, // swipe was replaced by the secondary-hand fist; skip the per-frame work
    windowMs: 500,
    minDist: 0.15,
    minSpeed: 0.6, // frame widths / second
    maxVertRatio: 0.6, // |dy| must be < ratio * |dx|
    monotonic: 0.7, // fraction of steps that must go the same way
    cooldownMs: 1000,
    minFrames: 12, // hand must have existed this long (ignores "entering" motion)
    edgeGuard: 0.05, // ignore swipes that start/end hugging the frame edge
  },

  edgeMargin: 0.03, // palm centre must be inside [margin, 1-margin] to trigger gestures
  filter: { minCutoff: 1.7, beta: 30.0, dCutoff: 1.0 }, // higher beta = far less lag when moving // One-Euro on palm centre
};

// Skeleton topology for drawing (MediaPipe hand connections)
export const PALM_CONNECTIONS = [
  [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
];
export const FINGER_CONNECTIONS = [
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
];

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

/** One-Euro filter: strong smoothing when still, little lag when moving fast. */
class OneEuro {
  constructor(minCutoff, beta, dCutoff) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.reset();
  }
  reset() {
    this.x = null;
    this.dx = 0;
    this.t = 0;
  }
  _alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(v, t) {
    if (this.x === null) {
      this.x = v;
      this.t = t;
      return v;
    }
    const dt = Math.max(1e-3, t - this.t);
    this.t = t;
    const dv = (v - this.x) / dt;
    this.dx += this._alpha(this.dCutoff, dt) * (dv - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += this._alpha(cutoff, dt) * (v - this.x);
    return this.x;
  }
}

/** Commits a new value only after it has been proposed for N consecutive frames. */
class Debounce {
  constructor(initial) {
    this.reset(initial);
  }
  reset(initial) {
    this.value = initial;
    this.pending = initial;
    this.count = 0;
  }
  step(candidate, holdFrames) {
    if (candidate === this.value) {
      this.pending = candidate;
      this.count = 0;
    } else if (candidate === this.pending) {
      if (++this.count >= holdFrames) {
        this.value = candidate;
        this.count = 0;
      }
    } else {
      this.pending = candidate;
      this.count = 1;
      if (holdFrames <= 1) this.value = candidate;
    }
    return this.value;
  }
}

// Landmark indices
const WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8;
const FINGERS = [ // [tip, mcp] for index, middle, ring, pinky
  [8, 5], [12, 9], [16, 13], [20, 17],
];
const PALM_IDS = [0, 5, 9, 13, 17];
const SWIPE_BUF = 16; // ring buffer length (power of two)

// ---------------------------------------------------------------------------
// Hand: persistent per-role state (reused every frame -> no per-frame allocs)
// ---------------------------------------------------------------------------
export class Hand {
  constructor(role, cfg) {
    this.role = role; // 'primary' | 'secondary'
    this.cfg = cfg;
    this.fx = new OneEuro(cfg.filter.minCutoff, cfg.filter.beta, cfg.filter.dCutoff);
    this.fy = new OneEuro(cfg.filter.minCutoff, cfg.filter.beta, cfg.filter.dCutoff);
    this.gestureDebounce = new Debounce('unknown');
    this.pinchDebounce = new Debounce('none');
    this.fingerExt = new Uint8Array(4);
    this.sx = new Float32Array(SWIPE_BUF);
    this.sy = new Float32Array(SWIPE_BUF);
    this.st = new Float64Array(SWIPE_BUF);
    this.reset();
  }

  reset() {
    this.visible = false; // detected (or within the dropout grace period)
    this.active = false; // visible long enough to be trusted
    this.inFrame = false;
    this.framesSeen = 0;
    this.lostFrames = 0;
    this.lm = null; // last raw landmarks (array of 21 {x,y,z})
    this.label = '';
    this.rawX = 0; this.rawY = 0; // raw palm centre (video-normalised)
    this.px = 0; this.py = 0; // smoothed palm centre (video-normalised, un-mirrored)
    this.prevMx = 0; this.prevPy = 0; this.hasPrev = false;
    this.dx = 0; this.dy = 0; // per-detection movement in screen space (frame widths)
    this.speed = 0; // frame widths / second
    this.size = 0.1; // hand size (frame-width units)
    this.pinchRatio = 1;
    this.ext = 0; // number of extended fingers (index..pinky)
    this.gesture = 'unknown'; // committed: 'open' | 'fist' | 'unknown'
    this.pinch = 'none'; // committed: 'pinch' | 'spread' | 'none'
    this.swipe = 0; // -1 left, +1 right (screen space); consumer must reset to 0
    this.lastSwipeMs = -1e9;
    this.lastT = 0;
    this.sCount = 0; this.sHead = 0;
    this.fingerExt.fill(0);
    this.fx.reset(); this.fy.reset();
    this.gestureDebounce.reset('unknown');
    this.pinchDebounce.reset('none');
  }

  /** Called when a detection was matched to this slot this frame. */
  update(lm, label, nowMs, ys) {
    const cfg = this.cfg;
    const wasVisible = this.visible;
    if (!wasVisible) { // fresh appearance -> start clean
      this.framesSeen = 0;
      this.hasPrev = false;
      this.sCount = 0;
      this.fx.reset(); this.fy.reset();
      this.gestureDebounce.reset('unknown');
      this.pinchDebounce.reset('none');
      this.fingerExt.fill(0);
    }
    this.visible = true;
    this.lostFrames = 0;
    this.framesSeen++;
    this.lm = lm;
    this.label = label;

    // --- palm centre (wrist + 4 knuckles), smoothed ------------------------
    let cx = 0, cy = 0;
    for (let i = 0; i < PALM_IDS.length; i++) {
      const p = lm[PALM_IDS[i]];
      cx += p.x; cy += p.y;
    }
    cx /= PALM_IDS.length; cy /= PALM_IDS.length;
    this.rawX = cx; this.rawY = cy;
    const t = nowMs / 1000;
    this.px = this.fx.filter(cx, t);
    this.py = this.fy.filter(cy, t);

    // movement in mirrored screen space
    const mx = 1 - this.px;
    const dt = Math.max(1e-3, t - this.lastT);
    this.lastT = t;
    if (this.hasPrev) {
      this.dx = mx - this.prevMx;
      this.dy = (this.py - this.prevPy) * ys; // y in frame-width units
      this.speed = Math.hypot(this.dx, this.dy) / dt;
    } else {
      this.dx = 0; this.dy = 0; this.speed = 0;
    }
    this.prevMx = mx; this.prevPy = this.py; this.hasPrev = true;

    const m = cfg.edgeMargin;
    this.inFrame = cx > m && cx < 1 - m && cy > m && cy < 1 - m;

    // --- geometry ------------------------------------------------------------
    const wrist = lm[WRIST];
    this.size = Math.max(1e-4, dist3(wrist, lm[9], ys));

    // finger extension w/ hysteresis (per finger)
    let ext = 0;
    for (let i = 0; i < 4; i++) {
      const tip = lm[FINGERS[i][0]], mcp = lm[FINGERS[i][1]];
      const ratio = dist3(tip, wrist, ys) / Math.max(1e-4, dist3(mcp, wrist, ys));
      if (this.fingerExt[i]) { if (ratio < cfg.fingerExtendOff) this.fingerExt[i] = 0; }
      else if (ratio > cfg.fingerExtendOn) this.fingerExt[i] = 1;
      ext += this.fingerExt[i];
    }

    this.ext = ext;
    let candidate = 'unknown';
    if (this.inFrame) {
      if (ext >= 4) candidate = 'open';
      else if (ext === 0) candidate = 'fist';
    }
    this.gesture = this.gestureDebounce.step(candidate, cfg.hold[candidate]);

    // pinch / spread w/ hysteresis on thumb-index distance
    this.pinchRatio = dist3(lm[THUMB_TIP], lm[INDEX_TIP], ys) / this.size;
    const r = this.pinchRatio;
    const cur = this.pinchDebounce.value;
    let pc;
    if (cur === 'pinch') pc = r < cfg.pinchOff ? 'pinch' : 'none';
    else if (cur === 'spread') pc = r > cfg.spreadOff ? 'spread' : 'none';
    else pc = r < cfg.pinchOn ? 'pinch' : r > cfg.spreadOn ? 'spread' : 'none';
    this.pinch = this.pinchDebounce.step(pc, cfg.hold[pc]);

    // --- swipe ----------------------------------------------------------------
    if (cfg.swipe.enabled) {
    const i = this.sHead;
    // raw (unfiltered) palm position: the smoothing filter would lag behind fast swipes
    this.sx[i] = 1 - cx; this.sy[i] = cy * ys; this.st[i] = nowMs;
    this.sHead = (i + 1) & (SWIPE_BUF - 1);
    if (this.sCount < SWIPE_BUF) this.sCount++;
    if (this.framesSeen >= cfg.swipe.minFrames && this.inFrame && this.pinch !== 'pinch' &&
        nowMs - this.lastSwipeMs > cfg.swipe.cooldownMs) {
      const dir = this._detectSwipe();
      if (dir !== 0) {
        this.swipe = dir;
        this.lastSwipeMs = nowMs;
        this.sCount = 0; // clear history so one motion = one swipe
      }
    }
    }

    this.active = this.framesSeen >= cfg.enterFrames;
  }

  /** Called when no detection matched this slot this frame. */
  miss() {
    if (!this.visible) return;
    this.lostFrames++;
    this.dx = 0; this.dy = 0; this.speed = 0;
    if (this.lostFrames > this.cfg.lostGraceFrames) {
      this.visible = false;
      this.active = false;
      this.swipe = 0;
    }
    if (this.lostFrames > this.cfg.releaseFrames) this.reset();
  }

  _detectSwipe() {
    const c = this.cfg.swipe;
    const n = this.sCount;
    if (n < 4) return 0;
    const newest = (this.sHead - 1) & (SWIPE_BUF - 1);
    const tN = this.st[newest];
    let idx = newest, count = 1;
    while (count < n) { // walk back while inside the time window
      const j = (idx - 1) & (SWIPE_BUF - 1);
      if (tN - this.st[j] > c.windowMs) break;
      idx = j; count++;
    }
    if (count < 4) return 0;
    const dx = this.sx[newest] - this.sx[idx];
    const dy = this.sy[newest] - this.sy[idx];
    const dtSec = (tN - this.st[idx]) / 1000;
    const adx = Math.abs(dx);
    if (adx < c.minDist || adx / dtSec < c.minSpeed || Math.abs(dy) > c.maxVertRatio * adx) return 0;
    // Edge guard: ignore a hand that is just entering / leaving the frame
    const x0 = this.sx[idx], x1 = this.sx[newest];
    if ((x0 < c.edgeGuard && dx > 0) || (x0 > 1 - c.edgeGuard && dx < 0) ||
        (x1 < c.edgeGuard && dx < 0) || (x1 > 1 - c.edgeGuard && dx > 0)) return 0;
    // Monotonic motion check (rejects wobbling)
    let agree = 0, total = 0, j = idx;
    for (let k = 0; k < count - 1; k++) {
      const nx = (j + 1) & (SWIPE_BUF - 1);
      if ((this.sx[nx] - this.sx[j]) * dx > 0) agree++;
      total++; j = nx;
    }
    if (total === 0 || agree / total < c.monotonic) return 0;
    return dx > 0 ? 1 : -1; // +1 = moving right on screen
  }
}

/** 3D distance in "frame width" units (y scaled by H/W, z already ~x-scale). */
function dist3(a, b, ys) {
  const dx = a.x - b.x, dy = (a.y - b.y) * ys, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// ViewMapper: video-normalised landmark -> camera-panel pixels (mirrored, cover)
// ---------------------------------------------------------------------------
export class ViewMapper {
  constructor() {
    this.pw = 1; this.ph = 1; this.vw = 1; this.vh = 1;
    this.scale = 1; this.ox = 0; this.oy = 0;
  }
  update(panelW, panelH, videoW, videoH) {
    this.pw = panelW; this.ph = panelH;
    this.vw = videoW || 1; this.vh = videoH || 1;
    this.scale = Math.max(panelW / this.vw, panelH / this.vh); // object-fit: cover
    this.ox = (panelW - this.vw * this.scale) / 2;
    this.oy = (panelH - this.vh * this.scale) / 2;
  }
  /** Writes panel pixel coordinates (CSS px) into out.x / out.y. */
  toPanel(nx, ny, out) {
    out.x = this.pw - (this.ox + nx * this.vw * this.scale); // mirror in X
    out.y = this.oy + ny * this.vh * this.scale;
    return out;
  }
}

// ---------------------------------------------------------------------------
// HandTracker
// ---------------------------------------------------------------------------
export class HandTracker {
  constructor(config = TRACKING_CONFIG) {
    this.cfg = config;
    this.landmarker = null;
    this.ready = false;
    this.delegate = 'GPU';
    this.primary = new Hand('primary', config);
    this.secondary = new Hand('secondary', config);
    this.numHands = 0; // raw number of hands detected this frame (0,1,2)
    this._dets = [
      { lm: null, label: '', cx: 0, cy: 0, used: false },
      { lm: null, label: '', cx: 0, cy: 0, used: false },
    ];
    this._lastTs = 0;
  }

  /** Download WASM + model and create the landmarker. */
  async init(onStatus = () => {}) {
    onStatus('Loading MediaPipe runtime…');
    const vision = await FilesetResolver.forVisionTasks(this.cfg.wasmPath);
    const make = (delegate) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: this.cfg.modelPath, delegate },
        runningMode: 'VIDEO',
        numHands: this.cfg.numHands,
        minHandDetectionConfidence: this.cfg.minHandDetectionConfidence,
        minHandPresenceConfidence: this.cfg.minHandPresenceConfidence,
        minTrackingConfidence: this.cfg.minTrackingConfidence,
      });
    onStatus('Loading hand landmark model…');
    try {
      this.landmarker = await make('GPU');
      this.delegate = 'GPU';
    } catch (err) {
      console.warn('GPU delegate unavailable, falling back to CPU', err);
      this.landmarker = await make('CPU');
      this.delegate = 'CPU';
    }
    // Warm-up: the first real inference compiles GPU programs (a visible freeze).
    // Do it now, on the loading screen, with a blank frame.
    try {
      const c = document.createElement('canvas');
      c.width = 256; c.height = 256;
      c.getContext('2d').fillRect(0, 0, 256, 256);
      this._lastTs = performance.now();
      this.landmarker.detectForVideo(c, this._lastTs);
    } catch (e) { /* non-fatal */ }
    this.ready = true;
  }

  /**
   * Run detection on the current video frame and update primary/secondary.
   * Call at most once per new video frame.
   */
  detect(video, nowMs) {
    if (!this.ready) return;
    const ts = Math.max(nowMs, this._lastTs + 1); // timestamps must strictly increase
    this._lastTs = ts;
    const res = this.landmarker.detectForVideo(video, ts);
    const ys = video.videoHeight / video.videoWidth; // y -> width units

    const lms = res.landmarks || [];
    const hnd = res.handedness || res.handednesses || [];
    const n = Math.min(lms.length, 2);
    this.numHands = n;

    for (let i = 0; i < 2; i++) {
      const d = this._dets[i];
      d.used = false;
      if (i < n) {
        const lm = lms[i];
        d.lm = lm;
        d.label = hnd[i] && hnd[i][0] ? hnd[i][0].categoryName : '';
        let cx = 0, cy = 0;
        for (let k = 0; k < PALM_IDS.length; k++) { cx += lm[PALM_IDS[k]].x; cy += lm[PALM_IDS[k]].y; }
        d.cx = cx / PALM_IDS.length; d.cy = cy / PALM_IDS.length;
      } else d.lm = null;
    }

    this._assign(n, ys, ts);
  }

  _assign(n, ys, ts) {
    const slots = [this.primary, this.secondary];
    const dets = this._dets;
    const slotUsed = [false, false];

    // 1) Keep identities: match existing (visible) slots to nearest detections.
    // At most 4 pairs -> tiny manual selection, no sorting / allocation.
    for (let pass = 0; pass < 2; pass++) {
      let best = Infinity, bs = -1, bd = -1;
      for (let s = 0; s < 2; s++) {
        if (slotUsed[s] || !slots[s].visible) continue;
        for (let d = 0; d < n; d++) {
          if (dets[d].used) continue;
          const dx = slots[s].rawX - dets[d].cx;
          const dy = (slots[s].rawY - dets[d].cy) * ys;
          const dist = Math.hypot(dx, dy);
          if (dist < best) { best = dist; bs = s; bd = d; }
        }
      }
      if (bs < 0 || best > this.cfg.maxMatchDist) break;
      slotUsed[bs] = true; dets[bd].used = true;
      slots[bs].update(dets[bd].lm, dets[bd].label, ts, ys);
    }

    // 2) Unmatched detections claim free slots (slots not currently tracked).
    const free0 = !slotUsed[0] && !slots[0].visible;
    const free1 = !slotUsed[1] && !slots[1].visible;
    const pending = [];
    for (let d = 0; d < n; d++) if (!dets[d].used) pending.push(d);

    if (pending.length === 2 && free0 && free1) {
      // Both appear together: use handedness (MediaPipe labels assume a mirrored
      // image; on the raw frame the user's RIGHT hand is reported as "Left").
      let a = pending[0], b = pending[1];
      const aRight = dets[a].label === 'Left', bRight = dets[b].label === 'Left';
      let primaryDet;
      if (aRight !== bRight) primaryDet = aRight ? a : b;
      else primaryDet = dets[a].cx < dets[b].cx ? a : b; // user's right = smaller raw x
      const other = primaryDet === a ? b : a;
      slotUsed[0] = slotUsed[1] = true;
      this.primary.update(dets[primaryDet].lm, dets[primaryDet].label, ts, ys);
      this.secondary.update(dets[other].lm, dets[other].label, ts, ys);
    } else {
      for (let k = 0; k < pending.length; k++) {
        const d = pending[k];
        let target = -1;
        const wantPrimary = this.cfg.assignMode === 'handedness' ? dets[d].label === 'Left' : true;
        const f0 = !slotUsed[0] && !slots[0].visible;
        const f1 = !slotUsed[1] && !slots[1].visible;
        if (wantPrimary) target = f0 ? 0 : f1 ? 1 : -1;
        else target = f1 ? 1 : f0 ? 0 : -1;
        if (target < 0) continue;
        slotUsed[target] = true;
        slots[target].update(dets[d].lm, dets[d].label, ts, ys);
      }
    }

    // 3) Slots with no detection this frame
    for (let s = 0; s < 2; s++) if (!slotUsed[s]) slots[s].miss();
  }

  close() {
    if (this.landmarker) this.landmarker.close();
    this.ready = false;
  }
}
