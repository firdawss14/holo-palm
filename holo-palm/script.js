/**
 * script.js — application entry point
 * ---------------------------------------------------------------------------
 * Wires everything together:
 *
 *   webcam  ->  HandTracker (MediaPipe)  ->  gesture state machine  ->  3D scene
 *                                    └─────────────────────────────->  dashboard
 *
 * One requestAnimationFrame loop does: detect (once per new video frame) ->
 * apply gestures -> animate scene -> render -> update dashboard.
 *
 * Roles
 *   PRIMARY hand   : open palm = show model above palm, move = follow,
 *                    fist = shatter into star particles, reopen = reassemble
 *   SECONDARY hand : pinch = scale up, spread = scale down,
 *                    swipe left/right = previous/next model, move = rotate
 */

import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { HandTracker, ViewMapper, PALM_CONNECTIONS, FINGER_CONNECTIONS } from './handTracking.js';
import { ObjectController, createNeonCube } from './objectController.js';
import { ParticleSystem } from './particleEffects.js';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const CONFIG = {
  fov: 40, // camera field of view (deg)
  sceneHeight: 10, // world units visible vertically at z = 0 (the palm plane)
  cubeFrac: 0.44, // cube edge as a fraction of the visible height
  modelFill: 0.62, // model size (at scale ×1) relative to the cube edge
  lift: 0.62, // how far above the palm the cube centre floats (in cube sizes)
  followDamping: 34, // higher = snappier follow (1/s); the One-Euro filter already removes jitter
  pinchGain: 2.4, // zoom per unit of thumb-index distance change (relative to hand size)
  pinchDeadband: 0.02, // finger-distance noise ignored (kept small so slow pinches still work)
  rotYawGain: 6.0, // radians per frame-width of horizontal hand travel
  rotPitchGain: 4.0,
  rotateMaxSpeed: 0.55, // hand speed (frame widths/s) above which we assume a swipe, not a rotate
  transitionCooldownMs: 450, // lock after fist / reassemble so it cannot flicker
};

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const dom = {
  stage: $('stage'), video: $('video'), three: $('three-canvas'), overlay: $('overlay-canvas'),
  toast: $('toast'), loading: $('loading'), loadingText: $('loading-text'),
  error: $('error'), errorTitle: $('error-title'), errorText: $('error-text'), retry: $('retry'),
  camChip: $('cam-chip'), camText: $('cam-text'), trackChip: $('track-chip'), trackText: $('track-text'),
  handCount: $('hand-count'), slotP: $('slot-primary'), slotS: $('slot-secondary'),
  trackStatus: $('track-status'), fps: $('fps-line'),
  objName: $('object-name'), objIndex: $('object-index'),
  scaleValue: $('scale-value'), scaleBar: $('scale-bar'),
  gP: $('gesture-primary'), gS: $('gesture-secondary'), lastAction: $('last-action'),
  steps: {
    camera: $('step-camera'), track: $('step-track'), models: $('step-models'),
  },
  how: [...document.querySelectorAll('.how li[data-key]')],
};
const ctx = dom.overlay.getContext('2d');

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
let renderer, scene, camera, object, cube, particles;
let W = 1, H = 1, visW = 1, cubeSize = 4, dpr = 1;
let stream = null;
let threeReady = false, modelsReady = false, loopStarted = false;

const tracker = new HandTracker();
const mapper = new ViewMapper();
const tmp = { x: 0, y: 0 }; // scratch (avoid per-frame allocations)
const pts = new Float32Array(42); // 21 landmarks in panel pixels

/** Primary-hand state machine: hidden -> shown -> shattered -> assembling -> shown */
let mode = 'hidden';
let lockUntil = 0; // transition cooldown timestamp
let lastSwipeAt = -1e9, lastSwipeDir = 0, prevSFist = true;
let rValid = false, rSmooth = 0, rAnchor = 0, lastScaleAt = -1e9, scaleDir = 0; // pinch-distance zoom
let useRVFC = false, rdpr = 1, lastW = 0, lastH = 0, lastDpr = 0, resizePending = false;
const DASH_A = [4, 5], DASH_NONE = [];
let lastAction = '—';
let cubeAlpha = 0.3;
let lastVideoTime = -1, lastDetect = 0, lastFrame = 0, time = 0;
let fpsFrames = 0, fpsDet = 0, fpsStamp = 0, renderFps = 0, detFps = 0;
let lastDash = 0, toastTimer = 0, modelStatus = '', detectErrorLogged = false;

// ---------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------
function setText(el, v) { if (el._v !== v) { el._v = v; el.textContent = v; } }
function setStep(name, state) { dom.steps[name].dataset.state = state; }
function setLoading(text) { dom.loadingText.textContent = text; }
function setChip(chip, textEl, state, text) { chip.dataset.state = state; setText(textEl, text); }

function toast(msg) {
  dom.toast.textContent = msg;
  dom.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => dom.toast.classList.remove('show'), 1300);
}
function action(text, withToast = true) {
  lastAction = text;
  if (withToast) toast(text.toUpperCase());
}

class AppError extends Error {
  constructor(title, text) { super(title); this.title = title; this.text = text; }
}
function showError(title, text) {
  dom.errorTitle.textContent = title;
  dom.errorText.textContent = text;
  dom.loading.classList.add('hidden');
  dom.error.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Three.js scene
// ---------------------------------------------------------------------------
function initThree() {
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: dom.three, alpha: true, antialias: (window.devicePixelRatio || 1) < 2, powerPreference: 'high-performance',
    });
  } catch (err) {
    throw new AppError('WebGL not available',
      'Your browser or GPU could not start WebGL, which is needed to render the 3D model. Try Chrome or Edge with hardware acceleration enabled.');
  }
  renderer.setClearColor(0x000000, 0); // transparent: the video shows through
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(CONFIG.fov, 1, 0.1, 100);
  // place the camera so that exactly `sceneHeight` world units fill the panel at z = 0
  camera.position.z = CONFIG.sceneHeight / 2 / Math.tan(THREE.MathUtils.degToRad(CONFIG.fov / 2));

  // image-based lighting so PBR models do not look flat
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();

  object = new ObjectController({
    onStatus: (s) => { modelStatus = s; },
    onModelReady: () => {
      // if the model changed while stars are gathering, aim them at the new shape
      if (mode === 'assembling') particles.retarget(object.getSamples());
    },
  });
  scene.add(object.anchor);

  cube = createNeonCube();
  object.anchor.add(cube);

  particles = new ParticleSystem();
  particles.onAssembled = () => {
    if (mode !== 'assembling') return;
    mode = 'shown';
    object.setVisible(true, true); // model replaces the stars at the exact same spot
    action('Object reassembled');
  };
  object.pivot.add(particles.points);

  threeReady = true;
  lastW = 0; // force renderer/overlay sizing on first call
  resize();
}

function resize() {
  const r = dom.stage.getBoundingClientRect();
  W = Math.max(1, r.width); H = Math.max(1, r.height);
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  rdpr = Math.min(window.devicePixelRatio || 1, 1.5); // 3D canvas is capped: big fill-rate saving, barely visible
  mapper.update(W, H, dom.video.videoWidth, dom.video.videoHeight);
  if (W === lastW && H === lastH && dpr === lastDpr) return; // nothing changed: do not reallocate buffers
  lastW = W; lastH = H; lastDpr = dpr;

  dom.overlay.width = Math.round(W * dpr);
  dom.overlay.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (!threeReady) return;
  renderer.setPixelRatio(rdpr);
  renderer.setSize(W, H, false);
  camera.aspect = W / H;
  camera.updateProjectionMatrix();

  visW = CONFIG.sceneHeight * camera.aspect;
  cubeSize = Math.min(CONFIG.sceneHeight * CONFIG.cubeFrac, visW * 0.62);
  cube.scale.setScalar(cubeSize);
  object.setFitSize(cubeSize * CONFIG.modelFill);
  particles.setViewport(H * rdpr, CONFIG.fov);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
async function startCamera() {
  setStep('camera', 'busy');
  setChip(dom.camChip, dom.camText, 'loading', 'Starting');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new AppError('Camera not supported',
      window.isSecureContext
        ? 'This browser does not support webcam access. Please use a recent version of Chrome, Edge, Firefox or Safari.'
        : 'Camera access only works on https:// or http://localhost. Serve this folder with a local web server (see the instructions in index.html) instead of opening the file directly.');
  }
  try {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 }, facingMode: 'user' }, audio: false,
      });
    } catch (e) {
      if (e && e.name === 'OverconstrainedError') stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      else throw e;
    }
  } catch (e) {
    const n = (e && e.name) || '';
    if (n === 'NotAllowedError' || n === 'SecurityError' || n === 'PermissionDeniedError') {
      throw new AppError('Camera permission denied',
        'Allow camera access for this site (click the camera icon in the address bar), then press “Try again”. Video never leaves your device.');
    }
    if (n === 'NotFoundError' || n === 'DevicesNotFoundError') {
      throw new AppError('No camera found', 'Connect a webcam and press “Try again”.');
    }
    if (n === 'NotReadableError' || n === 'TrackStartError') {
      throw new AppError('Camera is busy', 'Another application is using the camera. Close it and press “Try again”.');
    }
    throw new AppError('Could not start the camera', `${n || 'Unknown error'}: ${(e && e.message) || ''}`);
  }
  dom.video.srcObject = stream;
  await new Promise((res) => {
    if (dom.video.readyState >= 1) res(); else dom.video.addEventListener('loadedmetadata', res, { once: true });
  });
  await dom.video.play();
  mapper.update(W, H, dom.video.videoWidth, dom.video.videoHeight);
  stream.getVideoTracks()[0].addEventListener('ended', () => {
    setChip(dom.camChip, dom.camText, 'error', 'Lost');
    showError('Camera disconnected', 'The camera stopped sending video. Reconnect it and press “Try again”.');
  });
  setStep('camera', 'done');
  setChip(dom.camChip, dom.camText, 'on', 'Live');
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  dom.error.classList.add('hidden');
  dom.loading.classList.remove('hidden');
  for (const k in dom.steps) setStep(k, 'wait');
  setLoading('Starting…');

  try {
    if (!threeReady) initThree();

    const tasks = [];
    // camera (permission prompt runs in parallel with the downloads)
    if (!stream || !stream.active) tasks.push(startCamera());
    // hand-tracking model
    if (!tracker.ready) {
      setStep('track', 'busy');
      tasks.push(tracker.init(setLoading).then(() => setStep('track', 'done'))
        .catch((e) => {
          setStep('track', 'fail');
          console.error(e);
          throw new AppError('Hand tracking failed to load',
            'The MediaPipe model could not be downloaded. Check your internet connection (it is loaded from a CDN) and press “Try again”.');
        }));
    } else setStep('track', 'done');
    // 3D models
    if (!modelsReady) {
      setStep('models', 'busy');
      tasks.push(object.init().then(() => { modelsReady = true; setStep('models', 'done'); })
        .catch((e) => { console.error(e); setStep('models', 'fail'); }));
    } else setStep('models', 'done');

    const results = await Promise.allSettled(tasks);
    const failed = results.find((r) => r.status === 'rejected');
    if (failed) throw failed.reason;
  } catch (err) {
    console.error(err);
    if (err instanceof AppError) {
      if (dom.steps.camera.dataset.state === 'busy') setStep('camera', 'fail');
      setChip(dom.camChip, dom.camText, stream && stream.active ? 'on' : 'error', stream && stream.active ? 'Live' : 'Error');
      setChip(dom.trackChip, dom.trackText, 'error', 'Unavailable');
      showError(err.title, err.text);
    } else {
      showError('Unexpected error', String((err && err.message) || err));
    }
    return;
  }

  setLoading('Ready');
  dom.loading.classList.add('hidden');
  setChip(dom.trackChip, dom.trackText, 'idle', 'Searching');
  startVideoCallback();
  if (!loopStarted) { loopStarted = true; requestAnimationFrame(frame); }
}

// ---------------------------------------------------------------------------
// Gesture -> behaviour
// ---------------------------------------------------------------------------
function applyHands(now, dt) {
  const P = tracker.primary, S = tracker.secondary;
  P.swipe = 0; // only the secondary hand uses swipes

  // ---------------- PRIMARY: visibility state machine ----------------
  if (!P.active) {
    if (mode !== 'hidden') { // hand left the frame: hide everything cleanly
      mode = 'hidden';
      object.setVisible(false);
      particles.clear();
    }
  } else {
    const g = P.gesture;
    if (g === 'open') {
      if (mode === 'hidden') {
        mode = 'shown';
        object.setVisible(true);
        action('Object summoned');
      } else if (mode === 'shattered' && now >= lockUntil) {
        mode = 'assembling';
        lockUntil = now + CONFIG.transitionCooldownMs;
        particles.assemble(object.getSamples());
        action('Reassembling…');
      }
    } else if (g === 'fist' && (mode === 'shown' || mode === 'assembling') && now >= lockUntil) {
      particles.shatter(object.getSamples(), mode === 'assembling');
      object.setVisible(false, true); // model vanishes the instant the stars burst
      mode = 'shattered';
      lockUntil = now + CONFIG.transitionCooldownMs;
      action('Object shattered');
    }
  }

  // ---------------- SECONDARY: scale / switch / rotate ----------------
  if (S.active) {
    // Zoom follows the thumb-index distance in real time: closing the fingers grows the model,
    // opening them shrinks it. Only CHANGES in distance count (relative zoom), so a resting hand
    // never drifts, and the anchor/deadband ignores landmark noise while allowing slow pinches.
    if (S.inFrame && S.gesture !== 'fist' && S.ext >= 2 && S.speed < CONFIG.rotateMaxSpeed) {
      const r = S.pinchRatio;
      if (!rValid) { rValid = true; rSmooth = rAnchor = r; }
      else {
        rSmooth += (r - rSmooth) * 0.6; // light smoothing of the measured distance
        const e = rSmooth - rAnchor, dz = CONFIG.pinchDeadband;
        let d = 0;
        if (e > dz) d = e - dz; else if (e < -dz) d = e + dz;
        if (d !== 0) {
          rAnchor += d;
          if (d > 0.06) d = 0.06; else if (d < -0.06) d = -0.06; // ignore single-frame landmark jumps
          object.scaleBy(Math.exp(-CONFIG.pinchGain * d));
          scaleDir = d < 0 ? 1 : -1; lastScaleAt = now;
        }
      }
    } else rValid = false; // re-arm when the hand is steady/open again (avoids jumps after a fist)

    // closing the secondary hand into a fist = next model (fires once per closing)
    S.swipe = 0;
    const fist = S.gesture === 'fist';
    if (fist && !prevSFist && now - lastSwipeAt > 900) {
      lastSwipeAt = now; lastSwipeDir = 1;
      object.switchBy(1);
      action('Next model →');
    }
    prevSFist = fist;

    if (!fist && S.pinch === 'none' && now - lastSwipeAt > 500 && now - lastScaleAt > 250 && S.inFrame &&
        S.framesSeen > 8 && S.speed < CONFIG.rotateMaxSpeed) {
      // hand motion rotates the model (dead-zone removes residual jitter)
      const dx = Math.abs(S.dx) > 0.0012 ? S.dx : 0;
      const dy = Math.abs(S.dy) > 0.0012 ? S.dy : 0;
      if (dx !== 0 || dy !== 0) object.rotateBy(dx * CONFIG.rotYawGain, dy * CONFIG.rotPitchGain);
    }
  } else { S.swipe = 0; prevSFist = true; rValid = false; } // a hand must be seen OPEN before a fist counts
}

function updateScene(dt, t) {
  const P = tracker.primary, S = tracker.secondary;

  // --- where should the cube/model be? palm (mirrored) -> world coordinates ---
  let tx = 0, ty = 0, alphaTarget = 0.3;
  if (P.active) {
    mapper.toPanel(P.px, P.py, tmp); // smoothed palm centre -> panel pixels (mirror + cover crop)
    const wx = (tmp.x / W - 0.5) * visW;
    const wy = (0.5 - tmp.y / H) * CONFIG.sceneHeight;
    const half = cubeSize * 0.52;
    const mx = Math.max(0, visW / 2 - half), my = Math.max(0, CONFIG.sceneHeight / 2 - half);
    tx = THREE.MathUtils.clamp(wx, -mx, mx);
    ty = THREE.MathUtils.clamp(wy + cubeSize * CONFIG.lift, -my, my); // "floating above the palm"
    alphaTarget = mode === 'hidden' ? 0.6 : 1;
  }
  const k = 1 - Math.exp(-dt * CONFIG.followDamping);
  object.anchor.position.x += (tx - object.anchor.position.x) * k;
  object.anchor.position.y += (ty - object.anchor.position.y) * k;

  cubeAlpha += (alphaTarget - cubeAlpha) * (1 - Math.exp(-dt * 6));
  cube.userData.setAlpha(cubeAlpha);
  cube.rotation.set(0.22 + Math.sin(t * 0.6) * 0.05, 0.5 + Math.sin(t * 0.45) * 0.12, 0);
  cube.userData.update(t);

  object.autoSpin = !S.active; // gentle idle spin only when the secondary hand is not steering
  object.update(dt, t);
  particles.setWorldScale(object.pivot.scale.x);
  particles.update(dt);
}

// ---------------------------------------------------------------------------
// Skeleton overlay (pink + violet lines, glowing purple star landmarks)
// ---------------------------------------------------------------------------
const TIPS = new Set([4, 8, 12, 16, 20]);

function starPath(c, x, y, outer, inner, rot) {
  for (let i = 0; i < 10; i++) {
    const r = i % 2 === 0 ? outer : inner;
    const a = rot + (i * Math.PI) / 5;
    const px = x + Math.cos(a) * r, py = y + Math.sin(a) * r;
    i === 0 ? c.moveTo(px, py) : c.lineTo(px, py);
  }
  c.closePath();
}

function strokeConnections(list) {
  ctx.beginPath();
  for (let i = 0; i < list.length; i++) {
    const a = list[i][0] * 2, b = list[i][1] * 2;
    ctx.moveTo(pts[a], pts[a + 1]);
    ctx.lineTo(pts[b], pts[b + 1]);
  }
  ctx.stroke();
}

// Glowing star drawn ONCE (shadowBlur is expensive); per frame we only drawImage() it.
const starSprite = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.shadowColor = '#b04dff'; g.shadowBlur = 10;
  g.fillStyle = '#b86bff';
  g.beginPath(); starPath(g, 32, 32, 14, 6.3, -Math.PI / 2); g.fill();
  g.shadowBlur = 0; g.lineWidth = 1.2; g.strokeStyle = 'rgba(255,255,255,0.85)'; g.stroke();
  return c;
})();

function drawHand(h) {
  const lm = h.lm;
  for (let i = 0; i < 21; i++) {
    mapper.toPanel(lm[i].x, lm[i].y, tmp);
    pts[i * 2] = tmp.x; pts[i * 2 + 1] = tmp.y;
  }
  const isP = h.role === 'primary';
  const base = h.active ? 1 : 0.55; // dim while a new hand is still being confirmed
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';

  // neon glow = wide faint stroke under a crisp stroke (no blur filter)
  ctx.globalAlpha = base * 0.22; ctx.lineWidth = 9;
  ctx.strokeStyle = '#ff2fa8'; strokeConnections(PALM_CONNECTIONS);
  ctx.strokeStyle = '#9b5cff'; strokeConnections(FINGER_CONNECTIONS);
  ctx.globalAlpha = base; ctx.lineWidth = 3;
  ctx.strokeStyle = '#ff3cac'; strokeConnections(PALM_CONNECTIONS); // palm outline: neon pink
  ctx.strokeStyle = '#a56bff'; strokeConnections(FINGER_CONNECTIONS); // fingers: violet

  // glowing purple star landmarks
  const pulse = 1 + Math.sin(time * 5) * 0.08;
  for (let i = 0; i < 21; i++) {
    const o = (i === 0 ? 8 : TIPS.has(i) ? 8.5 : 5.5) * pulse;
    const sz = (o / 14) * 64;
    ctx.drawImage(starSprite, pts[i * 2] - sz / 2, pts[i * 2 + 1] - sz / 2, sz, sz);
  }

  // palm anchor ring (primary shows where the object is attached)
  mapper.toPanel(h.px, h.py, tmp);
  ctx.strokeStyle = isP ? 'rgba(255,122,217,0.9)' : 'rgba(201,182,255,0.8)';
  ctx.lineWidth = 1.5; ctx.setLineDash(DASH_A);
  ctx.beginPath(); ctx.arc(tmp.x, tmp.y, 16 + Math.sin(time * 3) * 1.5, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash(DASH_NONE);

  // role / gesture tag
  let tag = isP ? 'PRIMARY' : 'SECONDARY';
  if (h.active) {
    if (isP) tag += h.gesture === 'open' ? ' · OPEN' : h.gesture === 'fist' ? ' · FIST' : '';
    else tag += h.gesture === 'fist' ? ' · FIST' : h.pinch === 'pinch' ? ' · PINCH' : h.pinch === 'spread' ? ' · SPREAD' : '';
  }
  ctx.font = '700 10px Orbitron, sans-serif';
  ctx.fillStyle = isP ? '#ff9be3' : '#d9ccff';
  ctx.fillText(tag, pts[0] - 34, pts[1] + 34);
  ctx.globalAlpha = 1;
}

function drawOverlay() {
  ctx.clearRect(0, 0, W, H);
  let any = false;
  if (tracker.primary.visible && tracker.primary.lm) { drawHand(tracker.primary); any = true; }
  if (tracker.secondary.visible && tracker.secondary.lm) { drawHand(tracker.secondary); any = true; }
  if (!any) {
    ctx.font = '500 12px Orbitron, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(255, 190, 240, 0.85)';
    ctx.fillText('RAISE YOUR HAND · OPEN PALM TO BEGIN', W / 2, H - 46);
    ctx.textAlign = 'start';
  }
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
function primaryLabel(P) {
  if (!P.visible) return '— no hand —';
  if (!P.active) return 'Detecting…';
  if (mode === 'assembling') return 'Open · reassembling';
  return P.gesture === 'open' ? 'Open palm' : P.gesture === 'fist' ? 'Closed fist' : 'Transition…';
}
function secondaryLabel(S, now) {
  if (!S.visible) return '— no hand —';
  if (!S.active) return 'Detecting…';
  if (now - lastSwipeAt < 900 || S.gesture === 'fist') return 'Closed fist · next model';
  if (now - lastScaleAt < 250) return scaleDir > 0 ? 'Pinch · scale up' : 'Spread · scale down';
  if (S.pinch === 'pinch') return 'Pinch · scale up';
  if (S.pinch === 'spread') return 'Spread · scale down';
  return S.speed > 0.1 ? 'Moving · rotate' : 'Open · idle';
}

let lastHow = '';
function updateDashboard(now) {
  if (now - lastDash < 100) return;
  lastDash = now;
  const P = tracker.primary, S = tracker.secondary;
  const n = tracker.numHands;

  setText(dom.handCount, String(n));
  dom.slotP.classList.toggle('on', P.active);
  dom.slotS.classList.toggle('on', S.active);

  let status, chip = 'idle', chipText = 'Searching';
  if (n === 0) status = 'Searching for hands';
  else { status = n === 1 ? 'Tracking 1 hand' : 'Tracking 2 hands'; chip = 'on'; chipText = 'Tracking'; }
  setText(dom.trackStatus, status);
  setChip(dom.trackChip, dom.trackText, chip, chipText);
  setText(dom.fps, `${renderFps} fps render · ${detFps} fps detect · ${tracker.delegate}`);

  setText(dom.objName, object.name);
  setText(dom.objIndex, modelStatus || (object.count ? `${object.index + 1} / ${object.count}` : '—'));
  if (dom.dots.childElementCount !== object.count) {
    dom.dots.innerHTML = '<i></i>'.repeat(object.count);
  }
  for (let i = 0; i < dom.dots.children.length; i++) dom.dots.children[i].classList.toggle('on', i === object.index);
  setText(dom.scaleValue, `×${object.userScale.toFixed(2)}`);
  const pct = ((object.userScale - object.scaleMin) / (object.scaleMax - object.scaleMin)) * 100;
  dom.scaleBar.style.transform = `scaleX(${(pct / 100).toFixed(3)})`; // compositor-only, no layout

  setText(dom.gP, primaryLabel(P));
  setText(dom.gS, secondaryLabel(S, now));
  setText(dom.lastAction, lastAction);

  // highlight the instruction step that matches what is happening now
  const on = [];
  if (P.active) {
    if (mode === 'shown') { on.push('p-open'); if (P.speed > 0.15) on.push('p-move'); }
    if (mode === 'shattered') on.push('p-fist');
    if (mode === 'assembling') on.push('p-reopen');
  }
  if (S.active) {
    if (now - lastSwipeAt < 900 || S.gesture === 'fist') on.push('s-fist');
    else if (now - lastScaleAt < 250) on.push(scaleDir > 0 ? 's-pinch' : 's-spread');
    else if (S.pinch === 'pinch') on.push('s-pinch');
    else if (S.pinch === 'spread') on.push('s-spread');
    else if (S.speed > 0.1) on.push('s-move');
  }
  const key = on.join(',');
  if (key !== lastHow) {
    lastHow = key;
    for (const li of dom.how) li.classList.toggle('active', on.includes(li.dataset.key));
  }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
/** Detect hands on the newest camera frame and apply gestures. */
function processVideoFrame(now) {
  if (!tracker.ready) return;
  const v = dom.video;
  if (v.videoWidth !== mapper.vw || v.videoHeight !== mapper.vh) mapper.update(W, H, v.videoWidth, v.videoHeight);
  try {
    tracker.detect(v, now);
    fpsDet++;
    const dtDet = Math.min(0.1, (now - lastDetect) / 1000 || 0.033);
    lastDetect = now;
    applyHands(now, dtDet);
    drawOverlay();
  } catch (err) {
    if (!detectErrorLogged) { detectErrorLogged = true; console.error('Detection error:', err); }
  }
}

/** Run detection the instant the camera delivers a frame (lower latency than polling in rAF). */
function startVideoCallback() {
  const v = dom.video;
  if (useRVFC || !('requestVideoFrameCallback' in v)) return;
  useRVFC = true;
  const cb = () => { processVideoFrame(performance.now()); v.requestVideoFrameCallback(cb); };
  v.requestVideoFrameCallback(cb);
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - lastFrame) / 1000 || 0.016);
  lastFrame = now;
  time += dt;

  // Fallback only: browsers without requestVideoFrameCallback poll for new frames here.
  const v = dom.video;
  if (!useRVFC && v.readyState >= 2 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;
    processVideoFrame(now);
  }

  updateScene(dt, time);
  renderer.render(scene, camera);

  fpsFrames++;
  if (now - fpsStamp >= 1000) {
    renderFps = Math.round((fpsFrames * 1000) / (now - fpsStamp));
    detFps = Math.round((fpsDet * 1000) / (now - fpsStamp));
    fpsFrames = 0; fpsDet = 0; fpsStamp = now;
  }
  updateDashboard(now);
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
function scheduleResize() { // coalesce resize events into at most one per frame
  if (resizePending) return;
  resizePending = true;
  requestAnimationFrame(() => { resizePending = false; resize(); });
}
window.addEventListener('resize', scheduleResize);
new ResizeObserver(scheduleResize).observe(dom.stage);
dom.retry.addEventListener('click', () => {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  boot();
});
resize();
boot();
