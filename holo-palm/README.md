# HOLO·PALM — Hand-Gesture 3D Interface

A browser app that uses your **webcam + MediaPipe Hand Landmarker** to control a **Three.js** 3D model with hand gestures. The left half shows the live camera with a neon bounding cube and your hand skeleton; the right half is a glassmorphism dashboard with tracking state and instructions.

Everything is vanilla HTML/CSS/JS (ES modules). Libraries load from a CDN, so there is **no build step**.

---

## 1. Run it locally

> **Why a local server?** Browsers only allow webcam access (`getUserMedia`) on *secure contexts*: `https://` or `http://localhost`. Opening `index.html` by double-click (`file://`) blocks the camera, ES-module imports and the `fetch()` that reads `models/manifest.json`.

1. Open a terminal in the project folder (the one containing `index.html`).
2. Start any static server:
   ```bash
   python3 -m http.server 8000      # Python 3
   # or
   npx serve .                      # Node
   # or use the "Live Server" extension in VS Code
   ```
3. Open **http://localhost:8000** in Chrome or Edge (Firefox works too) and click **Allow** for the camera.

The page needs an internet connection the first time: three.js, MediaPipe (WASM + the `hand_landmarker.task` model) and the fonts come from CDNs.

---

## 2. Gestures

| Hand | Gesture | Result |
|---|---|---|
| **Primary** (first hand shown) | Open palm | Model appears above the palm, inside the cube |
| Primary | Move hand | Model follows the palm (smoothed) |
| Primary | Close fist | Model bursts into glowing violet star particles |
| Primary | Reopen hand | Particles gather back and rebuild the model |
| **Secondary** (second hand) | Pinch thumb + index together | Scale **up** |
| Secondary | Spread thumb + index apart | Scale **down** |
| Secondary | Close fist | Switch to the **next** model (wraps around) |
| Secondary | Move hand | Rotate the model |

Notes
- Show one hand first; it becomes the **primary**. The next hand you raise becomes the **secondary**. If both appear in the same frame, handedness decides (your right hand = primary). Roles stay locked to a hand by position tracking, not by MediaPipe's label.
- Zoom is **relative to finger distance**: only *changes* in thumb–index distance resize the model, so a resting hand never drifts.
- A hand must be seen **open** before a fist counts as "next model", so a hand entering the frame already closed does nothing.

---

## 3. Project structure

```
holo-palm/
├── index.html            Page layout, import map (CDN), run instructions in a comment
├── style.css             Pink-neon glassmorphism theme, responsive layout
├── script.js             App entry: boot, main loop, gesture → behaviour, overlay, dashboard
├── handTracking.js       MediaPipe wrapper, hand roles, filters, gesture classification
├── objectController.js   Model loading/cache/transitions, neon cube, hologram look
├── particleEffects.js    Star-particle shatter / reassemble (single THREE.Points)
├── README.md             This file
└── models/
    ├── manifest.json     List of models shown in the app
    └── *.glb             Your 3D models
```

---

## 4. Architecture

### 4.1 High-level data flow

```
 ┌──────────┐   frames    ┌────────────────────┐  landmarks   ┌────────────────────────┐
 │  Webcam  │───────────▶ │  HandTracker       │────────────▶ │  Hand (primary/second.)│
 │ <video>  │ (rVFC)      │  MediaPipe         │              │  filter • gestures     │
 └────┬─────┘             │  HandLandmarker    │              └───────────┬────────────┘
      │ shown mirrored    └────────────────────┘                          │ state
      ▼                                                                   ▼
 ┌───────────────────────────── script.js ──────────────────────────────────────────┐
 │  applyHands()  primary state machine • pinch zoom • fist→next • rotate           │
 │  updateScene() palm→world mapping • damped follow • cube alpha • animations      │
 │  drawOverlay() skeleton + star landmarks        updateDashboard() (10 Hz)        │
 └──────┬───────────────────────────────┬──────────────────────────────┬────────────┘
        ▼                               ▼                              ▼
 ┌───────────────────┐        ┌────────────────────┐          ┌──────────────────┐
 │ ObjectController  │        │  ParticleSystem    │          │  Dashboard DOM   │
 │ models • cube     │        │  shatter/assemble  │          │  counts, gesture,│
 │ scale/rotate      │        │  (THREE.Points)    │          │  scale, steps    │
 └─────────┬─────────┘        └─────────┬──────────┘          └──────────────────┘
           └──────────────┬─────────────┘
                          ▼
                ┌───────────────────┐   transparent canvas layered
                │  Three.js render  │   over the <video>
                └───────────────────┘
```

Layer order on the left panel (bottom → top): `<video>` (CSS-mirrored) → Three.js canvas (transparent) → 2D overlay canvas (skeleton) → vignette / HUD corners.

### 4.2 Main loop (`script.js`)

Two cooperating loops:

1. **Camera-frame callback** (`requestVideoFrameCallback`; falls back to polling inside `requestAnimationFrame`):
   `tracker.detect()` → `applyHands()` → `drawOverlay()`.
   Running detection exactly when a frame arrives removes up to one frame of latency.
2. **`requestAnimationFrame` loop** (display refresh rate):
   `updateScene()` (follow, animate, particles) → `renderer.render()` → FPS counter → `updateDashboard()` (throttled to 10 Hz, writes only changed values).

### 4.3 Modules

**`handTracking.js`**
- `HandTracker` — creates the landmarker (GPU delegate, CPU fallback, warm-up on the loading screen), runs `detectForVideo`, and assigns detections to the **primary/secondary slots**: existing roles are kept by nearest-wrist matching; new hands claim free slots (first-seen, or handedness when two appear together). Brief dropouts are tolerated (`lostGraceFrames`).
- `Hand` — persistent per-role state (reused every frame, no per-frame allocations):
  - palm centre (wrist + 4 knuckles) smoothed by a **One-Euro filter** (steady when still, low lag when moving);
  - **finger extension** = `dist(tip, wrist) / dist(mcp, wrist)` per finger, with hysteresis → *open palm* (4 extended) or *fist* (0 extended);
  - **pinch ratio** = `dist(thumb tip, index tip) / hand size` (hand size = wrist→middle-knuckle, so it is distance-independent);
  - a **debounce** commits a gesture only after it holds for a few frames, which prevents flicker;
  - `active` only after the hand has been visible for `enterFrames`.
- `ViewMapper` — converts video-normalised landmarks to panel pixels, accounting for the **mirrored** video and `object-fit: cover` cropping.
- Exports skeleton connection lists and `TRACKING_CONFIG` (all thresholds).

**`objectController.js`**
- `ObjectController` — owns the scene graph and model lifecycle:
  - reads `models/manifest.json`, loads GLB/GLTF with `GLTFLoader`, **normalises** each model (centred, longest side = 1 unit) so any file fits the cube;
  - **cache + preload**: keeps the current model and the next one; the next model is loaded in the background and pre-warmed on the GPU, so switching is instant; everything else is **disposed** (geometries, materials, textures);
  - scale-in / scale-out crossfade when showing, hiding or switching;
  - procedural **fallback cube** when the list is empty or a file fails;
  - surface **sample points** per model (used by the particles);
  - holographic look: fresnel rim glow + faint scanlines injected into `MeshStandardMaterial`, pink/violet accent lights, floating bob.
- `createNeonCube()` — the transparent bounding cube: faint faces, glowing edges (merged into 3 meshes), corner glows, base rings, a sweeping scan plane.

**`particleEffects.js`**
- `ParticleSystem` — one `THREE.Points` with a custom shader, ~900 stars. A canvas-generated **5-point star texture**, per-particle spin and twinkle, additive blending. CPU simulation:
  `idle → scattered` (burst, drag, swirl, slow fade) `→ gathering` (spring toward surface samples) `→ dissolving → idle`.
  Lives inside the model's pivot so it follows position, rotation and scale.

**`script.js`** also contains the camera start-up with clear error screens (denied, no camera, busy, unsupported, WebGL missing, model download failed), the loading overlay, and the skeleton drawing.

### 4.4 Scene graph (Three.js)

```
scene
└─ anchor            ← follows the palm (position, damped) + pink/violet/rim lights
   ├─ cube           ← neon bounding cube (scaled to cubeSize, gentle sway)
   └─ pivot          ← user rotation + zoom (fitSize × userScale)
      ├─ slot (per model)   ← show/hide scale + float bob
      │  └─ holder          ← normalisation to a 1-unit box
      │     └─ model (GLB)
      └─ particles.points   ← star particles (same local space as the model)
```

### 4.5 Palm → scene mapping

1. Smoothed palm centre (video-normalised, un-mirrored) → `ViewMapper.toPanel()` → panel pixels (x flipped for the mirror, cover-crop offset applied).
2. Panel pixels → world units on the z = 0 plane. The perspective camera is placed so exactly `sceneHeight` units fill the panel height, so screen position and 3D position match.
3. The cube/model is lifted `lift × cubeSize` above the palm and clamped to stay inside the view, then eased toward the target each frame.

### 4.6 Primary-hand state machine

```
          open (stable)                 fist (stable, cooldown ok)
 hidden ───────────────▶ shown ─────────────────────────────────▶ shattered
   ▲                       ▲                                           │
   │ hand leaves frame     │ particles arrived (model swaps in)        │ open (stable)
   └───────────────────────┴──────────────── assembling ◀──────────────┘
                                   (fist during assembling → shatters again)
```
A cooldown (`transitionCooldownMs`) after fist/reassemble transitions prevents flicker. Losing the hand hides everything cleanly.

### 4.7 Stability measures (anti-flicker)

- Multi-frame **hold** before any gesture commits (`hold` in `TRACKING_CONFIG`).
- **Hysteresis** on finger-extension and pinch thresholds.
- **Cooldowns** after discrete actions (fist transitions, model switch).
- **Enter guard:** a hand must be seen for several frames, and inside the frame margin, before it can trigger anything.
- **Zoom dead-band** and per-frame clamp ignore landmark noise and single-frame jumps; zoom is disarmed during fists so opening a fist never resizes the model.

---

## 5. Adding your own 3D models

Browsers cannot list a folder, so models are declared in `models/manifest.json`:

```json
{
  "models": [
    { "name": "Iron Man Mk VII", "file": "iron_man_mk7.glb" },
    { "name": "My Robot", "file": "robot.glb", "scale": 1.2, "yaw": 180 }
  ]
}
```

1. Copy the `.glb` (or `.gltf`) file into `models/`.
2. Add one line to `manifest.json`.
3. Refresh the page — no code changes.

Optional per-model fields: `scale` (size multiplier) and `yaw` (degrees; use `180` if a model faces away from you). A plain array of file names also works. If the manifest is missing or empty, the built-in neon cube is used (you can also set `MODEL_CONFIG.fallbackModels` in `objectController.js`).

---

## 6. Configuration cheat sheet

| What you want | Where |
|---|---|
| Gesture speed (frames to confirm) | `TRACKING_CONFIG.hold`, `enterFrames` — `handTracking.js` |
| Finger-curl / open thresholds | `fingerExtendOn` / `fingerExtendOff` — `handTracking.js` |
| Pinch / spread labels | `pinchOn/Off`, `spreadOn/Off` — `handTracking.js` |
| Palm smoothing vs lag | `TRACKING_CONFIG.filter` (`minCutoff` ↑ = snappier, `beta` ↑ = less lag when moving) |
| Primary/secondary assignment | `TRACKING_CONFIG.assignMode` (`'first-seen'` or `'handedness'`) |
| Zoom sensitivity | `CONFIG.pinchGain`, `pinchDeadband` — `script.js` |
| Rotation sensitivity | `CONFIG.rotYawGain`, `rotPitchGain` — `script.js` |
| How snappy the model follows | `CONFIG.followDamping` — `script.js` |
| Cube size / how high above the palm | `CONFIG.cubeFrac`, `lift` — `script.js` |
| Fist cooldown | `CONFIG.transitionCooldownMs` — `script.js` |
| Zoom limits | `scaleMin` / `scaleMax` — `ObjectController` |
| Particle count, size, speed, fade | `PARTICLE_CONFIG` — `particleEffects.js` |
| Camera / panel split | `getUserMedia` constraints in `script.js`; `grid-template-columns` in `style.css` |

---

## 7. Performance notes

- Detection runs on each camera frame; the dashboard updates at 10 Hz and only touches changed DOM nodes.
- Hand-landmark state objects are reused (no per-frame allocations in the tracker).
- The skeleton overlay avoids canvas blur filters: stars are a pre-drawn sprite, line glow is a double stroke.
- The 3D canvas pixel ratio is capped at 1.5; resize events are coalesced.
- One draw call for all particles; cube edges merged; static model nodes skip matrix updates.
- Next model is preloaded and pre-warmed; unused models are disposed.
- The dashboard shows render fps, detection fps and the delegate in use (GPU/CPU).

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Camera never starts / "secure context" message | Use `http://localhost`, not `file://` |
| "Camera permission denied" | Click the camera icon in the address bar → Allow → *Try again* |
| "No camera found" / "Camera is busy" | Connect a camera / close other apps using it |
| Stuck on loading / "Hand tracking failed to load" | Check your internet connection (CDN) and press *Try again* |
| Detection fps low, dashboard says CPU | Enable hardware acceleration in the browser; close heavy tabs |
| Model appears dark or faces away | Adjust `yaw` / `scale` for it in `manifest.json` |
| Gestures trigger too easily / not enough | Tune `hold`, `fingerExtendOn/Off` in `TRACKING_CONFIG` |
| Zoom too touchy / too slow | Lower / raise `CONFIG.pinchGain` in `script.js` |
| Model can't be switched | Use the **secondary** (second) hand; the first hand shown is the primary |

Tips for best tracking: good front lighting, plain background, hands about 40–80 cm from the camera with the palm facing it, and fingers clearly separated for the open-palm pose.

---

## 9. Tech stack

- [Three.js](https://threejs.org) r160 (core, `GLTFLoader`, `RoomEnvironment`, `BufferGeometryUtils`)
- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) 0.10.14 — Hand Landmarker (`numHands: 2`)
- Fonts: Orbitron + Outfit (Google Fonts)
- Pure ES modules via an import map — no bundler required
