/**
 * particleEffects.js
 * ---------------------------------------------------------------------------
 * Star-particle "shatter / reassemble" effect, rendered as ONE THREE.Points
 * object (a single draw call) with a custom shader.
 *
 *   shatter(samples)  -> particles start on the model surface and burst outward,
 *                        drift, sparkle and gradually fade.
 *   assemble(samples) -> particles are pulled back (with a little swirl) onto the
 *                        model's surface; onAssembled() fires when they arrive.
 *   clear()           -> quick fade-out (e.g. when the hand leaves the frame).
 *
 * The system lives in the *model's local space* (add `points` to the model's
 * pivot group), so it automatically follows the model's position, rotation and
 * scale. `samples` is a Float32Array of xyz positions in that same space.
 *
 * Particle simulation runs on the CPU over a few hundred points (cheap), and the
 * position attribute is re-uploaded once per frame.
 */

import * as THREE from 'three';

export const PARTICLE_CONFIG = {
  count: 900, // "hundreds" of stars
  size: 0.06, // star size in model-local units (model is normalised to ~1 unit)
  burstSpeedMin: 0.8,
  burstSpeedMax: 2.6,
  drag: 1.5, // velocity damping while scattered
  swirl: 0.9, // gentle orbit while scattered
  maxRadius: 2.3, // soft boundary (model-local units)
  fadeDuration: 4.5, // seconds to fade from full to the floor while scattered
  fadeFloor: 0.18, // never fully vanish while scattered (so they can re-gather)
  gatherTime: 0.9, // spring ramp-up time
  arriveRatio: 0.9, // fraction of particles "arrived" that completes assembly
  arriveDist: 0.04,
};

/** Star-shaped, glowing sprite drawn on a canvas (5-point star + soft halo). */
export function createStarTexture(size = 128) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const cx = size / 2, cy = size / 2;

  // soft halo
  const halo = g.createRadialGradient(cx, cy, 0, cx, cy, size * 0.5);
  halo.addColorStop(0, 'rgba(255,255,255,0.55)');
  halo.addColorStop(0.35, 'rgba(255,255,255,0.14)');
  halo.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = halo;
  g.fillRect(0, 0, size, size);

  // 5-point star
  const outer = size * 0.42, inner = outer * 0.42;
  g.beginPath();
  for (let i = 0; i < 10; i++) {
    const r = i % 2 === 0 ? outer : inner;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
    i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
  }
  g.closePath();
  const fill = g.createRadialGradient(cx, cy, 0, cx, cy, outer);
  fill.addColorStop(0, 'rgba(255,255,255,1)');
  fill.addColorStop(1, 'rgba(255,255,255,0.85)');
  g.fillStyle = fill;
  g.fill();

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

const VERT = /* glsl */ `
  attribute float aSeed;
  attribute float aSize;
  uniform float uTime;
  uniform float uSize;
  uniform float uWorldScale;
  uniform float uPxScale;
  varying float vSeed;
  varying float vTw;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    // sparkle: each star twinkles at its own rate/phase
    float tw = 0.5 + 0.5 * sin(uTime * (2.5 + aSeed * 6.0) + aSeed * 40.0);
    tw = tw * tw * (3.0 - 2.0 * tw);
    vTw = tw;
    vSeed = aSeed;
    gl_PointSize = uSize * aSize * (0.7 + 0.6 * tw) * uWorldScale * uPxScale / max(0.1, -mv.z);
    gl_Position = projectionMatrix * mv;
  }
`;

const FRAG = /* glsl */ `
  uniform sampler2D uMap;
  uniform float uFade;
  uniform float uTime;
  varying float vSeed;
  varying float vTw;
  void main() {
    // spin the star sprite around its centre
    float a = vSeed * 6.2831853 + uTime * (0.3 + vSeed * 0.9);
    float c = cos(a), s = sin(a);
    vec2 uv = mat2(c, -s, s, c) * (gl_PointCoord - 0.5) + 0.5;
    float shape = texture2D(uMap, uv).a;
    vec3 violet   = vec3(0.64, 0.36, 1.00);
    vec3 lavender = vec3(0.86, 0.78, 1.00);
    vec3 col = mix(violet, lavender, fract(vSeed * 7.31) * 0.75) * (0.85 + 0.9 * vTw);
    col += vec3(1.0) * pow(shape, 5.0) * 0.45 * vTw;   // hot white core on sparkle
    float alpha = shape * uFade * (0.45 + 0.75 * vTw);
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(col, alpha);
  }
`;

export class ParticleSystem {
  constructor(config = PARTICLE_CONFIG) {
    this.cfg = config;
    const n = (this.count = config.count);

    this.pos = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.target = new Float32Array(n * 3);
    this.swirlDir = new Float32Array(n); // per-particle orbit direction/strength
    const seed = new Float32Array(n);
    const size = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      seed[i] = Math.random();
      size[i] = 0.55 + Math.random() * 0.9;
      this.swirlDir[i] = (Math.random() < 0.5 ? -1 : 1) * (0.5 + Math.random());
    }

    const geo = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(this.pos, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', posAttr);
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));

    this.texture = createStarTexture();
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: this.texture },
        uTime: { value: 0 },
        uSize: { value: config.size },
        uWorldScale: { value: 1 },
        uPxScale: { value: 600 },
        uFade: { value: 0 },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
    this.points.visible = false;
    this.points.renderOrder = 10;

    /** 'idle' | 'scattered' | 'gathering' | 'dissolving' | 'fading' */
    this.state = 'idle';
    this.t = 0; // time in current state
    this.fade = 0;
    this.progress = 0; // 0..1 gather progress
    this.onAssembled = null;
    this._time = 0;
  }

  /** Tell the shader how big things are on screen (call on resize / each frame). */
  setViewport(heightPx, fovDeg) {
    this.material.uniforms.uPxScale.value =
      heightPx / (2 * Math.tan(THREE.MathUtils.degToRad(fovDeg) / 2));
  }
  setWorldScale(s) {
    this.material.uniforms.uWorldScale.value = s;
  }

  /** Explode. `fromCurrent` keeps current positions (used when re-shattering). */
  shatter(samples, fromCurrent = false) {
    const { pos, vel, count } = this;
    const c = this.cfg;
    const startFresh = !fromCurrent || this.state === 'idle' || this.state === 'fading';
    for (let i = 0; i < count; i++) {
      const k = i * 3;
      if (startFresh) {
        const s = ((i * 7919) % (samples.length / 3)) * 3; // spread picks over the samples
        pos[k] = samples[s]; pos[k + 1] = samples[s + 1]; pos[k + 2] = samples[s + 2];
      }
      // outward direction + randomness + slight upward lift
      let dx = pos[k] + (Math.random() - 0.5) * 0.3;
      let dy = pos[k + 1] + (Math.random() - 0.5) * 0.3 + 0.1;
      let dz = pos[k + 2] + (Math.random() - 0.5) * 0.3;
      const l = Math.hypot(dx, dy, dz) || 1;
      const sp = c.burstSpeedMin + Math.random() * (c.burstSpeedMax - c.burstSpeedMin);
      vel[k] = (dx / l) * sp; vel[k + 1] = (dy / l) * sp; vel[k + 2] = (dz / l) * sp;
    }
    this.state = 'scattered';
    this.t = 0;
    this.fade = 1;
    this.points.visible = true;
    this._upload();
  }

  /** Pull particles back onto `samples` (model surface). */
  assemble(samples) {
    this.retarget(samples);
    if (this.state === 'idle') return; // nothing to gather
    this.state = 'gathering';
    this.t = 0;
    this.progress = 0;
    this.fade = 1;
    this.points.visible = true;
  }

  /** Update gather targets (e.g. the model changed while reassembling). */
  retarget(samples) {
    const { target, count } = this;
    const m = samples.length / 3;
    for (let i = 0; i < count; i++) {
      const s = ((i * 7919) % m) * 3;
      target[i * 3] = samples[s]; target[i * 3 + 1] = samples[s + 1]; target[i * 3 + 2] = samples[s + 2];
    }
  }

  /** Fade out quickly and hide. */
  clear() {
    if (this.state !== 'idle') { this.state = 'fading'; this.t = 0; }
  }

  update(dt) {
    this._time += dt;
    this.material.uniforms.uTime.value = this._time;
    if (this.state === 'idle') return;
    dt = Math.min(dt, 0.05);
    this.t += dt;
    const c = this.cfg, n = this.count, pos = this.pos, vel = this.vel;

    if (this.state === 'scattered') {
      const drag = Math.exp(-c.drag * dt);
      const r2max = c.maxRadius * c.maxRadius;
      for (let i = 0; i < n; i++) {
        const k = i * 3;
        // swirl around Y axis for a lively, orbiting dispersal
        const sw = c.swirl * this.swirlDir[i] * dt;
        vel[k] += -pos[k + 2] * sw; vel[k + 2] += pos[k] * sw;
        vel[k] *= drag; vel[k + 1] *= drag; vel[k + 2] *= drag;
        pos[k] += vel[k] * dt; pos[k + 1] += vel[k + 1] * dt; pos[k + 2] += vel[k + 2] * dt;
        // soft boundary so the cloud stays near the cube
        const d2 = pos[k] * pos[k] + pos[k + 1] * pos[k + 1] + pos[k + 2] * pos[k + 2];
        if (d2 > r2max) { const pull = 2.5 * dt; vel[k] -= pos[k] * pull; vel[k + 1] -= pos[k + 1] * pull; vel[k + 2] -= pos[k + 2] * pull; }
      }
      this.fade = Math.max(c.fadeFloor, 1 - (this.t / c.fadeDuration) * (1 - c.fadeFloor));
    } else if (this.state === 'gathering') {
      const ramp = Math.min(1, this.t / c.gatherTime);
      const k0 = 8 + 70 * ramp * ramp; // spring stiffness ramps up -> graceful swirl-in
      const damp = 2 * Math.sqrt(k0) * 0.75;
      const tgt = this.target;
      let arrived = 0;
      for (let i = 0; i < n; i++) {
        const k = i * 3;
        const ex = tgt[k] - pos[k], ey = tgt[k + 1] - pos[k + 1], ez = tgt[k + 2] - pos[k + 2];
        // a little tangential swirl that fades as they converge
        const sw = (1 - ramp) * 2.0 * this.swirlDir[i];
        vel[k] += (ex * k0 - vel[k] * damp - pos[k + 2] * sw) * dt;
        vel[k + 1] += (ey * k0 - vel[k + 1] * damp) * dt;
        vel[k + 2] += (ez * k0 - vel[k + 2] * damp + pos[k] * sw) * dt;
        pos[k] += vel[k] * dt; pos[k + 1] += vel[k + 1] * dt; pos[k + 2] += vel[k + 2] * dt;
        if (ex * ex + ey * ey + ez * ez < c.arriveDist * c.arriveDist) arrived++;
      }
      this.progress = arrived / n;
      if (this.t > 0.35 && this.progress >= c.arriveRatio) {
        this.state = 'dissolving'; // model takes over; stars melt away
        this.t = 0;
        if (this.onAssembled) this.onAssembled();
      }
    } else if (this.state === 'dissolving' || this.state === 'fading') {
      this.fade -= dt * (this.state === 'dissolving' ? 5 : 3.5);
      if (this.fade <= 0) {
        this.fade = 0; this.state = 'idle'; this.points.visible = false;
      }
      if (this.state === 'dissolving') this._shimmerInPlace(dt);
    }

    this.material.uniforms.uFade.value = this.fade;
    this._upload();
  }

  _shimmerInPlace() { /* particles already sit on the surface; nothing to move */ }

  _upload() {
    this.points.geometry.attributes.position.needsUpdate = true;
  }

  dispose() {
    this.points.geometry.dispose();
    this.material.dispose();
    this.texture.dispose();
  }
}
