/**
 * objectController.js
 * ---------------------------------------------------------------------------
 * Everything about the 3D object the user manipulates:
 *
 *   - Model discovery:   models/manifest.json  (or MODEL_CONFIG.fallbackModels)
 *   - GLB/GLTF loading:  GLTFLoader, one model resident at a time (+ the one
 *                        fading out), everything else is disposed
 *   - Normalisation:     every model is centred and scaled to a 1-unit box so any
 *                        file "just fits" the bounding cube
 *   - Transitions:       scale-in / scale-out crossfade when switching or hiding
 *   - Fallback:          procedural neon cube when nothing loads / list is empty
 *   - Surface samples:   points on the model surface used by the particle effect
 *   - Holographic look:  fresnel rim glow + faint scanlines patched into the
 *                        standard materials, pink/violet accent lights
 *   - createNeonCube():  the transparent glowing bounding cube
 *
 * Scene graph (all in world units):
 *
 *   anchor        <- follows the palm (position)         [+ lights]
 *    └─ pivot     <- user rotation + scale (fitSize * userScale)
 *        ├─ slot (per model)  <- show/hide scale + float bob
 *        │   └─ holder        <- normalisation (1-unit box), model yaw offset
 *        └─ (particle Points are added here by script.js)
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// ---------------------------------------------------------------------------
// CONFIGURATION — the one place to edit the model list
// ---------------------------------------------------------------------------
export const MODEL_CONFIG = {
  /**
   * Browsers cannot list a folder, so models are declared in models/manifest.json:
   *   { "models": [ { "name": "My Robot", "file": "robot.glb", "scale": 1, "yaw": 0 } ] }
   * ("scale" and "yaw" (degrees) are optional per-model tweaks. A plain array of
   *  file names also works.)  Add a .glb to models/, add a line to manifest.json,
   *  refresh the page - no code changes needed.
   */
  manifestUrl: 'models/manifest.json',
  modelsFolder: 'models/',

  /**
   * Optional explicit list used only if the manifest cannot be fetched/parsed.
   * Same shape as the manifest entries. Leave empty to fall back to the cube.
   */
  fallbackModels: [],

  surfaceSamples: 900, // points sampled per model for the particle effect
  holoRim: 0.55, // fresnel rim strength
  transitionSpeed: 6.5, // scale-in/out speed (1/seconds) - quick crossfade
  preloadDelayMs: 500, // wait this long after a switch before preloading the next model
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const easeOutBack = (x) => {
  const c1 = 1.4, c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
};
const smooth = (x) => x * x * (3 - 2 * x);

function prettyName(file) {
  return file
    .replace(/^.*\//, '')
    .replace(/\.(glb|gltf)$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Free GPU memory of an object tree: geometries, materials and every texture. */
export function disposeObject(root) {
  root.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.skeleton) o.skeleton.dispose();
    const mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
    for (const m of mats) {
      for (const key in m) {
        const v = m[key];
        if (v && v.isTexture) v.dispose();
      }
      m.dispose();
    }
  });
}

/** Random vertices of all meshes in holder-space (holder must have no parent). */
function sampleSurface(root, count) {
  root.updateMatrixWorld(true);
  const meshes = [];
  let total = 0;
  root.traverse((o) => {
    if (o.isMesh && o.geometry && o.geometry.attributes.position) {
      meshes.push(o);
      total += o.geometry.attributes.position.count;
    }
  });
  const out = new Float32Array(count * 3);
  if (!total) return out;
  const v = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    let r = Math.floor(Math.random() * total);
    let mesh = meshes[0];
    for (const m of meshes) {
      const c = m.geometry.attributes.position.count;
      if (r < c) { mesh = m; break; }
      r -= c;
    }
    v.fromBufferAttribute(mesh.geometry.attributes.position, r).applyMatrix4(mesh.matrixWorld);
    out[i * 3] = v.x + (Math.random() - 0.5) * 0.006;
    out[i * 3 + 1] = v.y + (Math.random() - 0.5) * 0.006;
    out[i * 3 + 2] = v.z + (Math.random() - 0.5) * 0.006;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Neon bounding cube (unit size; scale the returned group)
// ---------------------------------------------------------------------------
function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)');
  gr.addColorStop(0.25, 'rgba(255,90,208,0.75)');
  gr.addColorStop(1, 'rgba(255,47,168,0)');
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function createNeonCube() {
  const group = new THREE.Group();
  const mats = [];
  const reg = (m, base) => { m.userData.base = base; mats.push(m); return m; };
  const basic = (color, opacity, extra = {}) =>
    reg(new THREE.MeshBasicMaterial({
      color, transparent: true, opacity, depthWrite: false,
      blending: THREE.AdditiveBlending, ...extra,
    }), opacity);

  // translucent faces
  group.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
    basic(0xff2fa8, 0.045, { side: THREE.DoubleSide })));

  // 12 glowing edges: bright core + two wider, fainter halo layers
  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const layers = [
    { t: 0.009, color: 0xff7ad9, op: 1.0 },
    { t: 0.022, color: 0xff2fa8, op: 0.45 },
    { t: 0.045, color: 0xc04bff, op: 0.14 },
  ];
  const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _p = new THREE.Vector3(), _s = new THREE.Vector3();
  for (const L of layers) {
    const m = basic(L.color, L.op);
    const parts = [];
    for (let axis = 0; axis < 3; axis++) {
      for (const a of [-0.5, 0.5]) for (const b of [-0.5, 0.5]) {
        const sc = [L.t, L.t, L.t]; sc[axis] = 1 + L.t;
        const pp = [0, 0, 0];
        pp[(axis + 1) % 3] = a; pp[(axis + 2) % 3] = b;
        _m.compose(_p.set(pp[0], pp[1], pp[2]), _q.identity(), _s.set(sc[0], sc[1], sc[2]));
        parts.push(boxGeo.clone().applyMatrix4(_m));
      }
    }
    group.add(new THREE.Mesh(mergeGeometries(parts), m)); // 12 edges -> 1 draw call
    parts.forEach((g) => g.dispose());
  }

  // corner glows
  const gt = glowTexture();
  for (const x of [-0.5, 0.5]) for (const y of [-0.5, 0.5]) for (const z of [-0.5, 0.5]) {
    const sp = new THREE.Sprite(reg(new THREE.SpriteMaterial({
      map: gt, color: 0xffffff, transparent: true, opacity: 0.9,
      depthWrite: false, blending: THREE.AdditiveBlending,
    }), 0.9));
    sp.scale.setScalar(0.13);
    sp.position.set(x, y, z);
    group.add(sp);
  }

  // holographic base rings
  const ringMat = basic(0xff5ad0, 0.75, { side: THREE.DoubleSide });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.315, 72), ringMat);
  const ring2 = new THREE.Mesh(new THREE.RingGeometry(0.41, 0.4165, 72),
    basic(0xb36bff, 0.5, { side: THREE.DoubleSide }));
  const disk = new THREE.Mesh(new THREE.CircleGeometry(0.3, 48),
    basic(0xff2fa8, 0.08, { side: THREE.DoubleSide }));
  for (const m of [ring, ring2, disk]) { m.rotation.x = -Math.PI / 2; m.position.y = -0.49; group.add(m); }

  // scanner plane sweeping through the cube
  const scanner = new THREE.Group();
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), basic(0xff2fa8, 0.07, { side: THREE.DoubleSide }));
  plane.rotation.x = -Math.PI / 2;
  const lineMat = reg(new THREE.LineBasicMaterial({
    color: 0xff7ad9, transparent: true, opacity: 0.7, depthWrite: false, blending: THREE.AdditiveBlending,
  }), 0.7);
  const sq = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.5, 0, -0.5), new THREE.Vector3(0.5, 0, -0.5),
    new THREE.Vector3(0.5, 0, 0.5), new THREE.Vector3(-0.5, 0, 0.5),
  ]);
  scanner.add(plane, new THREE.LineLoop(sq, lineMat));
  group.add(scanner);

  group.userData.update = (t) => {
    scanner.position.y = ((t * 0.28) % 1) - 0.5;
    ring.rotation.z = t * 0.4;
    ring2.rotation.z = -t * 0.25;
  };
  /** Global opacity multiplier (0..1) */
  let lastA = -1;
  group.userData.setAlpha = (a) => {
    if (Math.abs(a - lastA) < 0.003) return; // skip redundant material updates
    lastA = a;
    for (const m of mats) m.opacity = m.userData.base * a;
  };
  group.userData.dispose = () => {
    group.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    for (const m of mats) { if (m.map) m.map.dispose(); m.dispose(); }
  };
  return group;
}

// ---------------------------------------------------------------------------
// ObjectController
// ---------------------------------------------------------------------------
export class ObjectController {
  constructor({ onStatus = () => {}, onModelReady = () => {} } = {}) {
    this.onStatus = onStatus;
    this.onModelReady = onModelReady;

    this.anchor = new THREE.Group(); // follows the palm
    this.pivot = new THREE.Group(); // rotation + scale
    this.anchor.add(this.pivot);

    // accent lights travel with the object
    const pink = new THREE.PointLight(0xff3db8, 55, 0, 2);
    pink.position.set(-2.6, 2.2, 3.2);
    const violet = new THREE.PointLight(0x8a5cff, 45, 0, 2);
    violet.position.set(2.8, -1.6, 2.6);
    const rim = new THREE.PointLight(0xffffff, 18, 0, 2);
    rim.position.set(0, 3.5, -2.5);
    this.anchor.add(pink, violet, rim);
    this.lights = [pink, violet, rim];

    this.loader = new GLTFLoader();
    this.defs = []; // [{name, url, scale, yaw}]
    this.index = 0;
    this.entries = []; // entries currently shown / fading (current + leaving ones)
    this.cache = new Map(); // index -> Promise<entry>   (loaded or loading)
    this.built = new Map(); // index -> entry            (loaded)
    this.standby = []; // preloaded entries being pre-warmed on the GPU
    this._preloadTimer = 0;
    this.current = null;
    this.loadToken = 0;
    this.loading = false;

    this.fitSize = 2.6; // world size of a 1-unit model at userScale 1
    this.userScale = 1;
    this.scaleMin = 0.35;
    this.scaleMax = 1.6;
    this._scaleSmooth = 1;

    this.yaw = 0; this.pitch = 0; // target rotation
    this._yaw = 0; this._pitch = 0; // smoothed
    this.autoSpin = true;
    this.visibleWanted = false;

    this.holo = { uTime: { value: 0 }, uRim: { value: MODEL_CONFIG.holoRim } };
  }

  // ---- public getters -------------------------------------------------------
  get count() { return this.defs.length; }
  get name() { return this.current ? this.current.name : this.loading ? 'Loading…' : '—'; }
  get hasModel() { return !!this.current; }

  setFitSize(s) { this.fitSize = s; }

  // ---- initialisation ---------------------------------------------------------
  async init() {
    this.defs = await this._readManifest();
    if (!this.defs.length) {
      this.onStatus('No models found – using the built-in cube');
      this.defs = [{ name: 'Neon Cube (built-in)', url: null, scale: 1, yaw: 0 }];
    }
    this.index = 0;
    await this._showIndex(0);
  }

  async _readManifest() {
    let list = [];
    try {
      const res = await fetch(MODEL_CONFIG.manifestUrl, { cache: 'no-cache' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      list = Array.isArray(json) ? json : json.models || [];
    } catch (err) {
      console.warn('[models] manifest unavailable, using fallbackModels:', err);
      list = MODEL_CONFIG.fallbackModels;
    }
    return list
      .map((m) => (typeof m === 'string' ? { file: m } : m))
      .filter((m) => m && m.file)
      .map((m) => ({
        name: m.name || prettyName(m.file),
        url: /^(https?:|\/|models\/)/i.test(m.file) ? m.file : MODEL_CONFIG.modelsFolder + m.file,
        scale: m.scale || 1,
        yaw: m.yaw || 0,
      }));
  }

  // ---- model loading / switching ---------------------------------------------------
  /** Switch to previous (-1) / next (+1) model, wrapping around. */
  switchBy(delta) {
    if (this.defs.length < 2) return;
    this.index = (this.index + delta + this.defs.length) % this.defs.length;
    return this._showIndex(this.index);
  }

  /** Entry for model i: instant if preloaded, otherwise loads (once) and caches. */
  _getEntry(i, onProgress) {
    let p = this.cache.get(i);
    if (p) return p;
    const def = this.defs[i];
    p = this._buildEntry(def, onProgress || (() => {}))
      .catch((err) => {
        console.warn(`[models] failed to load ${def.url}`, err);
        this.onStatus(`Could not load ${def.name} – showing fallback cube`);
        return this._makeCubeEntry(def.name + ' (fallback)');
      })
      .then((entry) => { entry.index = i; this.built.set(i, entry); return entry; });
    this.cache.set(i, p);
    return p;
  }

  async _showIndex(i) {
    const token = ++this.loadToken; // later requests invalidate earlier ones
    const def = this.defs[i];
    // Fast path: model already preloaded -> no await, the switch happens in this very call.
    let entry = this.built.get(i);
    if (!entry) {
      this.loading = true;
      this.onStatus(`Loading ${def.name}…`);
      entry = await this._getEntry(i, (p) => {
        if (token === this.loadToken) this.onStatus(`Loading ${def.name}… ${Math.round(p * 100)}%`);
      });
      if (token !== this.loadToken) { this._trim(); return; } // user already moved on
    }
    this.loading = false;
    // crossfade: old model scales out, new one scales in
    if (this.current && this.current !== entry) this.current.target = 0;
    const inScene = this.entries.includes(entry);
    if (!inScene) { entry.v = 0; this.entries.push(entry); }
    entry.target = this.visibleWanted ? 1 : 0;
    entry.slot.rotation.y = 0;
    if (entry.slot.parent !== this.pivot) this.pivot.add(entry.slot);
    this.current = entry;
    this.onStatus('');
    this.onModelReady(entry);
    this._trim();
    this._schedulePreload();
  }

  /** Indices worth keeping in memory: the current model and the one a switch goes to next. */
  _keep() {
    const n = this.defs.length;
    const k = new Set([this.index]);
    if (n > 1) k.add((this.index + 1) % n);
    return k;
  }

  /** Free GPU/CPU memory of every loaded model we do not need (and that is not on screen). */
  _trim() {
    const keep = this._keep();
    for (const [i, e] of this.built) {
      if (keep.has(i) || this.entries.includes(e)) continue;
      if (e.slot.parent) e.slot.parent.remove(e.slot);
      disposeObject(e.slot);
      this.built.delete(i);
      this.cache.delete(i);
      const si = this.standby.indexOf(e);
      if (si >= 0) this.standby.splice(si, 1);
    }
  }

  /** Load the next model in the background so the switch gesture is instant. */
  _schedulePreload() {
    clearTimeout(this._preloadTimer);
    const n = this.defs.length;
    if (n < 2) return;
    this._preloadTimer = setTimeout(() => {
      const i = (this.index + 1) % n;
      if (this.cache.has(i)) { const e = this.built.get(i); if (e && !this.entries.includes(e)) this._standby(e); return; }
      this._getEntry(i).then((e) => {
        if (!this._keep().has(i)) { this._trim(); return; }
        if (!this.entries.includes(e)) this._standby(e);
      });
    }, MODEL_CONFIG.preloadDelayMs);
  }

  /** Attach hidden in the scene and render it once (tiny) so textures/buffers upload now, not at switch time. */
  _standby(e) {
    if (e.warmed) return;
    e.warmed = true;
    e.warm = 3;
    if (e.slot.parent !== this.pivot) this.pivot.add(e.slot);
    this.standby.push(e);
  }

  async _buildEntry(def, onProgress) {
    if (!def.url) return this._makeCubeEntry(def.name);
    const gltf = await this.loader.loadAsync(def.url, (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    });
    const model = gltf.scene;

    // normalise: centre at origin, longest side = 1 unit
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    model.position.sub(center);

    const holder = new THREE.Group();
    holder.add(model);
    holder.scale.setScalar((1 / maxDim) * def.scale);
    holder.rotation.y = THREE.MathUtils.degToRad(def.yaw);

    model.traverse((o) => {
      if (!o.isMesh) return;
      o.frustumCulled = false; // skinned meshes often have stale bounds
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) this._applyHologram(m);
    });

    // The model's internal nodes never move: stop recomputing their local matrices every frame.
    holder.updateMatrixWorld(true);
    model.traverse((o) => { o.matrixAutoUpdate = false; });

    return this._wrap(def.name, holder);
  }

  _makeCubeEntry(name) {
    const holder = new THREE.Group();
    const geo = new THREE.BoxGeometry(0.7, 0.7, 0.7, 10, 10, 10);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x2a0a4a, metalness: 0.75, roughness: 0.28,
      emissive: 0xff2fa8, emissiveIntensity: 0.22,
    });
    this._applyHologram(mat);
    const mesh = new THREE.Mesh(geo, mat);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0xff7ad9 }));
    holder.add(mesh, edges);
    return this._wrap(name, holder);
  }

  _wrap(name, holder) {
    const samples = sampleSurface(holder, MODEL_CONFIG.surfaceSamples); // holder-space
    const slot = new THREE.Group();
    slot.add(holder);
    slot.visible = false;
    slot.scale.setScalar(0.0001);
    return { name, slot, holder, samples, v: 0, target: 0, index: -1 };
  }

  /** Fresnel rim + faint scanlines injected into MeshStandardMaterial. */
  _applyHologram(m) {
    if (!m || !m.isMeshStandardMaterial) return;
    m.envMapIntensity = 1.15;
    const uniforms = this.holo;
    m.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = uniforms.uTime;
      shader.uniforms.uRim = uniforms.uRim;
      shader.fragmentShader = shader.fragmentShader
        .replace('void main() {', 'uniform float uTime;\nuniform float uRim;\nvoid main() {')
        .replace(
          '#include <opaque_fragment>',
          `float holoRim = pow(1.0 - saturate(dot(normalize(normal), normalize(vViewPosition))), 3.0);
           outgoingLight += vec3(1.0, 0.25, 0.78) * holoRim * uRim;
           outgoingLight *= 0.95 + 0.05 * sin(gl_FragCoord.y * 1.4 + uTime * 3.0);
           #include <opaque_fragment>`);
    };
    m.customProgramCacheKey = () => 'holo-v1';
    m.needsUpdate = true;
  }

  // ---- control API used by script.js ------------------------------------------------
  /** Show/hide the model with a scale transition (instant = no animation). */
  setVisible(on, instant = false) {
    this.visibleWanted = on;
    if (this.current) {
      this.current.target = on ? 1 : 0;
      if (instant) this.current.v = on ? 1 : 0;
    }
  }
  get isVisible() { return !!this.current && this.current.v > 0.05 && this.visibleWanted; }

  scaleBy(factor) {
    this.userScale = THREE.MathUtils.clamp(this.userScale * factor, this.scaleMin, this.scaleMax);
  }
  rotateBy(dYaw, dPitch) {
    this.yaw += dYaw;
    this.pitch = THREE.MathUtils.clamp(this.pitch + dPitch, -1.0, 1.0);
  }

  /** Surface sample points (pivot-local space) for the particle effect. */
  getSamples() { return this.current ? this.current.samples : new Float32Array(MODEL_CONFIG.surfaceSamples * 3); }

  // ---- per-frame update -------------------------------------------------------------
  update(dt, t) {
    this.holo.uTime.value = t;

    // smooth rotation + scale
    if (this.autoSpin) this.yaw += dt * 0.35;
    const kr = 1 - Math.exp(-dt * 16);
    this._yaw += (this.yaw - this._yaw) * kr;
    this._pitch += (this.pitch - this._pitch) * kr;
    this._scaleSmooth += (this.userScale - this._scaleSmooth) * (1 - Math.exp(-dt * 22)); // follows the finger distance closely but stays smooth
    this.pivot.rotation.set(this._pitch, this._yaw, 0);
    this.pivot.scale.setScalar(this.fitSize * this._scaleSmooth);

    // light flicker (subtle hologram feel)
    this.lights[0].intensity = 55 + Math.sin(t * 7.1) * 2.5;
    this.lights[1].intensity = 45 + Math.sin(t * 5.3 + 1.0) * 2.5;

    // model transitions
    for (let i = this.entries.length - 1; i >= 0; i--) {
      const e = this.entries[i];
      const step = dt * MODEL_CONFIG.transitionSpeed;
      e.v = e.v < e.target ? Math.min(e.target, e.v + step) : Math.max(e.target, e.v - step);
      const s = e.target > 0.5 ? easeOutBack(e.v) : smooth(e.v);
      const leaving = e.target === 0 && e !== this.current;
      e.slot.visible = e.v > 0.001;
      e.slot.scale.setScalar(Math.max(0.0001, s));
      e.slot.position.y = Math.sin(t * 1.6) * 0.02; // gentle float
      if (leaving) e.slot.rotation.y += dt * 5; // spin away
      if (leaving && e.v <= 0.001) { // fully gone -> drop it (memory is freed unless it is still needed)
        this.pivot.remove(e.slot);
        this.entries.splice(i, 1);
        this._trim();
      }
    }

    // pre-warm preloaded models: render invisible-sized for a few frames, then hide
    for (let i = this.standby.length - 1; i >= 0; i--) {
      const e = this.standby[i];
      if (this.entries.includes(e)) { this.standby.splice(i, 1); continue; }
      if (e.warm > 0) { e.slot.visible = true; e.slot.scale.setScalar(0.0001); e.warm--; }
      else { e.slot.visible = false; this.standby.splice(i, 1); }
    }
  }

  dispose() {
    clearTimeout(this._preloadTimer);
    for (const e of this.built.values()) disposeObject(e.slot);
    this.built.clear(); this.cache.clear();
    this.entries.length = 0; this.standby.length = 0;
  }
}
