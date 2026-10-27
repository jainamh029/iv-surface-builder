// Surface3D: a small Three.js engine for volatility surfaces.
// Free (trackball) and turntable rotation with momentum, smooth zoom/pan, idle auto-spin,
// hover read-out, click picking, back-wall grids, floor heat-shadow with contours, colour legend.
import * as THREE from "./vendor/three.module.min.js";

const rad = d => d * Math.PI / 180, clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const AX = new THREE.Vector3(1, 0, 0), AY = new THREE.Vector3(0, 1, 0);
const qAxis = (axis, a) => new THREE.Quaternion().setFromAxisAngle(axis, a);
const viewQ = (yaw, pitch) => qAxis(AX, rad(pitch)).multiply(qAxis(AY, rad(yaw)));   // q = Rx(pitch) * Ry(yaw)
const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

export const VIEWS = {def:[-34, 27], top:[0, 88], smile:[0, 5], term:[90, 5]};
export const SCALES = {
  plasma:["#2a2f9e", "#5b3fc4", "#a23aa8", "#e2506a", "#fb9a3c", "#ffd35a"],   // cool -> warm
  div:   ["#3156d3", "#7d9cf5", "#e9edf5", "#f4a35c", "#e8590c"],
};
const colorAt = (stops, t) => { const n = stops.length - 1, x = clamp(t, 0, 1) * n, i = Math.min(Math.floor(x), n - 1);
  return new THREE.Color(stops[i]).lerp(new THREE.Color(stops[i + 1]), x - i); };

function niceTicks(lo, hi, n) {
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function dot() { const c = document.createElement("canvas"); c.width = c.height = 32; const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.beginPath(); g.arc(16, 16, 13, 0, 7); g.fill(); const t = new THREE.CanvasTexture(c); return t; }

export class Surface3D {
  constructor(host, {onPick = () => {}, tip = () => ""} = {}) {
    this.host = host; this.onPick = onPick; this.tipFn = tip; this.mode = "free"; this.autoSpin = true;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches; if (this.reduced) this.autoSpin = false;
    this.q = viewQ(...VIEWS.def); this.dist = 5.9; this.distT = 5.9; this.pan = new THREE.Vector2(); this.panT = new THREE.Vector2();
    this.vx = 0; this.vy = 0; this.spinGain = 0; this.lastInput = -1e9; this.anim = null; this.ptrs = new Map(); this.drag = null;
    this.theme = {ink:"#e9eef8", muted:"#8493ad", line:"#26314a", accent:"#f5a524", violet:"#8b7cf6", bg:"#070a12"};
    this.labels = []; this.hover = null; this.raf = 0; this.visible = true; this.corner = -1; this.hasSurface = false;
    host.classList.add("s3d"); host.tabIndex = 0;
    try { this.renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, powerPreference:"high-performance"}); }
    catch (e) { host.innerHTML = '<div class="s3d-fail">The 3D view needs WebGL, which this browser has turned off.</div>'; this.failed = true; return; }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); this.renderer.setClearColor(0x000000, 0);
    this.el = this.renderer.domElement; this.el.style.cssText = "display:block;width:100%;height:100%;touch-action:none;cursor:grab;outline:none";
    this.scene = new THREE.Scene(); this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 60); this.camera.position.z = this.dist;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x6b7694, 1.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.15); key.position.set(2.5, 4, 3.5); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xb8a4ff, .55); rim.position.set(-3, 1.5, -2.5); this.scene.add(rim);
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.content = new THREE.Group(); this.root.add(this.content);
    this.surfG = new THREE.Group(); this.overG = new THREE.Group(); this.frameG = new THREE.Group(); this.wallG = new THREE.Group();
    this.content.add(this.frameG, this.wallG, this.surfG, this.overG);
    this.raycaster = new THREE.Raycaster(); this.dotTex = dot();
    this.labelLayer = document.createElement("div"); this.labelLayer.className = "s3d-labels";
    this.tipEl = document.createElement("div"); this.tipEl.className = "s3d-tip"; this.legendEl = document.createElement("div"); this.legendEl.className = "s3d-legend";
    host.append(this.el, this.labelLayer, this.legendEl, this.tipEl);
    this._bind(); this._observe(); this._resize();
  }

  /* ---------------- public API ---------------- */
  setTheme(t) { Object.assign(this.theme, t); if (this.hasSurface) { this._buildFrame(); this._buildOverlaysAgain(); } this._frame(); }
  setMode(m) { this.mode = m; this.el.style.cursor = m === "pan" ? "move" : "grab";
    if (m === "turntable") { const u = AY.clone().applyQuaternion(this.q), xv = AX.clone().applyQuaternion(this.q), e = clamp(Math.asin(clamp(u.z, -1, 1)), rad(8), rad(85));
      this._animateTo(qAxis(AX, e).multiply(qAxis(AY, Math.atan2(-xv.z / Math.cos(e), xv.x))), this.distT, null, 450); } }
  setAutoSpin(on) { this.autoSpin = on; this.lastInput = performance.now() - 5000; this._kick(); }
  view(name) { const [y, p] = VIEWS[name] || VIEWS.def; this._animateTo(viewQ(y, p), name === "def" ? 5.9 : this.distT, new THREE.Vector2(), 700); }
  resetView() { this.view("def"); }
  zoomBy(f) { this.distT = clamp(this.distT * f, 2.6, 11); this._kick(); }
  nudge(yaw, pitch) { this.vy += yaw; this.vx += pitch; this.lastInput = performance.now(); this._kick(); }
  dispose() { cancelAnimationFrame(this.raf); this.ro && this.ro.disconnect(); this.io && this.io.disconnect(); this.renderer && this.renderer.dispose(); }

  setSurface({kRows, days, z, scale = "plasma", title = "", fmt = v => v.toFixed(2), xTitle = "ln(K/F)", yTitle = "days", contours = true, clip = .995}) {
    if (this.failed) return; this.sf = {kRows, days, z, scale, title, fmt, xTitle, yTitle, contours};
    const flat = z.flat().sort((a, b) => a - b), pick = p => flat[Math.floor(p * (flat.length - 1))];
    const div = scale === "div"; let lo = pick(1 - clip), hi = pick(clip);
    if (div) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; } if (hi - lo < 1e-12) hi = lo + 1;
    this.vlo = lo; this.vhi = hi;
    const ks = kRows.flat(); this.kmid = (Math.min(...ks) + Math.max(...ks)) / 2; this.kh = (Math.max(...ks) - Math.min(...ks)) / 2 || 1;
    this.dmid = (days[0] + days[days.length - 1]) / 2; this.dh = (days[days.length - 1] - days[0]) / 2 || 1; this.H = .92;
    this.content.position.y = -this.H / 2; this.hasSurface = true;
    this._buildSurface(); this._buildFrame(); this._buildLegend(); this.setOverlays(this.ov || {}); this._drawNow();
  }
  map(k, d, v) { return new THREE.Vector3((k - this.kmid) / this.kh, clamp((v - this.vlo) / (this.vhi - this.vlo), 0, 1) * this.H, (this.dmid - d) / this.dh); }

  setOverlays(ov) {
    if (this.failed || !this.hasSurface) { this.ov = ov; return; } this.ov = ov;
    for (const c of [...this.overG.children]) { this.overG.remove(c); c.geometry && c.geometry.dispose(); c.material && c.material.dispose(); }
    const p = ov.points;
    if (p && p.k.length) { const g = new THREE.BufferGeometry(), a = new Float32Array(p.k.length * 3);
      p.k.forEach((k, i) => { const v = this.map(k, p.days[i], p.v[i]); a.set([Number.isFinite(v.x + v.y + v.z) ? v.x : 0, v.y + .006, v.z], i * 3); });
      g.setAttribute("position", new THREE.BufferAttribute(a, 3));
      this.overG.add(new THREE.Points(g, new THREE.PointsMaterial({size:3.4, sizeAttenuation:false, map:this.dotTex, alphaTest:.35, transparent:true, color:this.theme.ink, opacity:.8, depthWrite:false}))); }
    for (const c of ov.curves || []) { const pts = c.k.map((k, i) => this.map(k, c.days[i], c.v[i]).add(new THREE.Vector3(0, .008, 0))).filter(v => Number.isFinite(v.x + v.y + v.z)); if (pts.length < 2) continue;
      this.overG.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), Math.max(60, pts.length), c.radius || .008, 8), new THREE.MeshBasicMaterial({color:c.color}))); }
    if (ov.marker) { const m = new THREE.Mesh(new THREE.OctahedronGeometry(.045), new THREE.MeshBasicMaterial({color:ov.marker.color || this.theme.violet}));
      m.position.copy(this.map(ov.marker.k, ov.marker.days, ov.marker.v)).add(new THREE.Vector3(0, .05, 0)); this.overG.add(m); }
    this._drawNow();
  }
  _buildOverlaysAgain() { if (this.ov) this.setOverlays(this.ov); }

  /* ---------------- scene building ---------------- */
  _buildSurface() {
    for (const c of [...this.surfG.children]) { this.surfG.remove(c); c.geometry && c.geometry.dispose(); c.material && c.material.dispose(); }
    const {kRows, days, z, scale, contours} = this.sf, R = z.length, C = z[0].length, stops = SCALES[scale];
    const pos = new Float32Array(R * C * 3), col = new Float32Array(R * C * 3), floor = new Float32Array(R * C * 3);
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { const n = (i * C + j) * 3, v = z[i][j], p = this.map(kRows[i][j], days[i], v);
      pos.set([p.x, p.y, p.z], n); floor.set([p.x, 0, p.z], n); const c = colorAt(stops, (v - this.vlo) / (this.vhi - this.vlo)); col.set([c.r, c.g, c.b], n); }
    const idx = []; for (let i = 0; i < R - 1; i++) for (let j = 0; j < C - 1; j++) { const a = i * C + j, b = a + 1, c = a + C, d = c + 1; idx.push(a, c, b, b, c, d); }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(col, 3)); g.setIndex(idx); g.computeVertexNormals();
    this.mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({vertexColors:true, roughness:.55, metalness:.04, side:THREE.DoubleSide})); this.surfG.add(this.mesh);
    // fine wireframe texture
    const w = []; for (let i = 0; i < R; i += 3) for (let j = 0; j < C - 1; j++) { const a = (i * C + j) * 3, b = a + 3; w.push(pos[a], pos[a + 1] + .0015, pos[a + 2], pos[b], pos[b + 1] + .0015, pos[b + 2]); }
    for (let j = 0; j < C; j += 4) for (let i = 0; i < R - 1; i++) { const a = (i * C + j) * 3, b = a + C * 3; w.push(pos[a], pos[a + 1] + .0015, pos[a + 2], pos[b], pos[b + 1] + .0015, pos[b + 2]); }
    const wg = new THREE.BufferGeometry(); wg.setAttribute("position", new THREE.Float32BufferAttribute(w, 3));
    this.surfG.add(new THREE.LineSegments(wg, new THREE.LineBasicMaterial({color:0xffffff, transparent:true, opacity:.10, depthWrite:false})));
    // floor heat-shadow + contours
    const fg = new THREE.BufferGeometry(); fg.setAttribute("position", new THREE.BufferAttribute(floor, 3)); fg.setAttribute("color", new THREE.BufferAttribute(col, 3)); fg.setIndex(idx);
    this.surfG.add(new THREE.Mesh(fg, new THREE.MeshBasicMaterial({vertexColors:true, transparent:true, opacity:.42, side:THREE.DoubleSide, depthWrite:false})));
    if (contours) this._contours(floor, col, R, C);
  }
  _contours(floor, col, R, C) {
    const {z} = this.sf, segs = [], cols = [], T = [[], [3, 0], [0, 1], [3, 1], [1, 2], [3, 0, 1, 2], [0, 2], [3, 2], [2, 3], [0, 2], [0, 1, 2, 3], [1, 2], [1, 3], [0, 1], [3, 0], []];
    for (let l = 1; l < 11; l++) { const L = this.vlo + (this.vhi - this.vlo) * l / 11, c = colorAt(SCALES[this.sf.scale], l / 11);
      for (let i = 0; i < R - 1; i++) for (let j = 0; j < C - 1; j++) {
        const ids = [i * C + j, i * C + j + 1, (i + 1) * C + j + 1, (i + 1) * C + j], v = [z[i][j], z[i][j + 1], z[i + 1][j + 1], z[i + 1][j]];
        const k = (v[0] > L) | ((v[1] > L) << 1) | ((v[2] > L) << 2) | ((v[3] > L) << 3), t = T[k]; if (!t.length) continue;
        const e = n => { const a = n, b = (n + 1) % 4, f = (L - v[a]) / (v[b] - v[a] || 1e-12), pa = ids[a] * 3, pb = ids[b] * 3;
          return [floor[pa] + (floor[pb] - floor[pa]) * f, .002, floor[pa + 2] + (floor[pb + 2] - floor[pa + 2]) * f]; };
        for (let s = 0; s < t.length; s += 2) { segs.push(...e(t[s]), ...e(t[s + 1])); cols.push(c.r, c.g, c.b, c.r, c.g, c.b); } } }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3)); g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
    this.surfG.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({vertexColors:true, transparent:true, opacity:.85, depthWrite:false})));
  }
  _label(text, cls = "") { const d = document.createElement("span"); d.className = "s3d-l " + cls; d.textContent = text; this.labelLayer.append(d); const o = {el:d, p:new THREE.Vector3()}; this.labels.push(o); return o; }
  _buildFrame() {
    for (const c of [...this.frameG.children, ...this.wallG.children]) { c.parent.remove(c); c.geometry && c.geometry.dispose(); c.material && c.material.dispose(); }
    this.labelLayer.innerHTML = ""; this.labels = []; this.corner = -1;
    const {fmt, xTitle, yTitle, title} = this.sf, t = this.theme;
    this.xt = niceTicks(this.kmid - this.kh, this.kmid + this.kh, 5); this.dt = niceTicks(this.dmid - this.dh, this.dmid + this.dh, 5);
    this.vt = niceTicks(this.vlo, this.vhi, 4).filter(v => v >= this.vlo - 1e-9 && v <= this.vhi + 1e-9);
    this.xl = this.xt.map(k => ({o:this._label(k.toFixed(2)), k})); this.dl = this.dt.map(d => ({o:this._label(d.toFixed(0)), d})); this.vl = this.vt.map(v => ({o:this._label(fmt(v)), v}));
    this.xT = this._label(xTitle, "t"); this.dT = this._label(yTitle, "t"); this.vT = this._label(title, "t");
    // floor grid
    const f = [], X = k => (k - this.kmid) / this.kh, Z = d => (this.dmid - d) / this.dh;
    this.xt.forEach(k => f.push(X(k), 0, -1, X(k), 0, 1)); this.dt.forEach(d => f.push(-1, 0, Z(d), 1, 0, Z(d)));
    f.push(-1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 1, 1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, -1);
    const fg = new THREE.BufferGeometry(); fg.setAttribute("position", new THREE.Float32BufferAttribute(f, 3));
    this.frameG.add(new THREE.LineSegments(fg, new THREE.LineBasicMaterial({color:t.line, transparent:true, opacity:.9})));
    this.axisLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, this.H, 0)]), new THREE.LineBasicMaterial({color:t.muted})); this.frameG.add(this.axisLine);
  }
  _layoutCorner(c) {          // back corner decides which walls are drawn and where the labels sit
    this.corner = c; const sx = [-1, 1, 1, -1][c], sz = [-1, -1, 1, 1][c], X = k => (k - this.kmid) / this.kh, Z = d => (this.dmid - d) / this.dh, H = this.H;
    for (const ch of [...this.wallG.children]) { this.wallG.remove(ch); ch.geometry.dispose(); ch.material.dispose(); }
    const w = []; this.vt.forEach(v => { const y = clamp((v - this.vlo) / (this.vhi - this.vlo), 0, 1) * H; w.push(-1, y, sz, 1, y, sz, sx, y, -1, sx, y, 1); });
    this.xt.forEach(k => w.push(X(k), 0, sz, X(k), H, sz)); this.dt.forEach(d => w.push(sx, 0, Z(d), sx, H, Z(d)));
    w.push(-1, H, sz, 1, H, sz, sx, H, -1, sx, H, 1, sx, 0, sz, sx, H, sz);
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(w, 3));
    this.wallG.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({color:this.theme.line, transparent:true, opacity:.55})));
    this.axisLine.position.set(sx, 0, sz);
    this.xl.forEach(({o, k}) => o.p.set(X(k), -.02, -sz * 1.1)); this.dl.forEach(({o, d}) => o.p.set(-sx * 1.1, -.02, Z(d)));
    this.vl.forEach(({o, v}) => o.p.set(sx * 1.1, clamp((v - this.vlo) / (this.vhi - this.vlo), 0, 1) * H, sz * 1.1));
    this.xT.p.set(0, -.02, -sz * 1.42); this.dT.p.set(-sx * 1.42, -.02, 0); this.vT.p.set(sx * 1.1, H + .14, sz * 1.1);
  }
  _buildLegend() {
    const {scale, fmt} = this.sf, ts = niceTicks(this.vlo, this.vhi, 4).filter(v => v >= this.vlo && v <= this.vhi);
    this.legendEl.innerHTML = `<div class="bar" style="background:linear-gradient(to top,${SCALES[scale].join(",")})"></div><div class="ticks">${ts.map(v => `<span style="bottom:${(v - this.vlo) / (this.vhi - this.vlo) * 100}%">${fmt(v)}</span>`).join("")}</div>`;
  }

  /* ---------------- interaction ---------------- */
  _bind() {
    const el = this.el; this.pointerIn = false;
    const stamp = () => { this.lastInput = performance.now(); this.spinGain = 0; };
    el.addEventListener("pointerdown", e => { el.setPointerCapture(e.pointerId); this.ptrs.set(e.pointerId, {x:e.clientX, y:e.clientY}); stamp(); this.anim = null; this.vx = this.vy = 0;
      if (this.ptrs.size === 1) this.drag = {x:e.clientX, y:e.clientY, t:performance.now(), moved:0, btn:e.button, pan:e.button === 2 || e.shiftKey || e.ctrlKey || e.metaKey || this.mode === "pan", last:performance.now(), t0:performance.now()};
      else { const [a, b] = [...this.ptrs.values()]; this.pinch = {d:Math.hypot(a.x - b.x, a.y - b.y), cx:(a.x + b.x) / 2, cy:(a.y + b.y) / 2}; this.drag = null; }
      el.style.cursor = "grabbing"; this.tipEl.style.opacity = 0; this._kick(); });
    el.addEventListener("pointermove", e => {
      const pt = this.ptrs.get(e.pointerId);
      if (!pt) { this.pointer = {x:e.clientX, y:e.clientY}; this._hover(); return; }
      const dx = e.clientX - pt.x, dy = e.clientY - pt.y; pt.x = e.clientX; pt.y = e.clientY; stamp();
      if (this.ptrs.size === 2 && this.pinch) { const [a, b] = [...this.ptrs.values()], d = Math.hypot(a.x - b.x, a.y - b.y), cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
        this.distT = clamp(this.distT * this.pinch.d / Math.max(d, 1), 2.6, 11); this._panBy(cx - this.pinch.cx, cy - this.pinch.cy); this.pinch = {d, cx, cy}; this._kick(); return; }
      if (!this.drag) return; const D = this.drag; D.moved += Math.abs(dx) + Math.abs(dy);
      if (D.pan) this._panBy(dx, dy); else { const s = .0062, ax = dy * s, ay = dx * s, now = performance.now(), dt = Math.max(1 / 240, (now - D.last) / 1000); D.last = now;
        this._rotate(ay, ax); this.vy += (ay / dt - this.vy) * .45; this.vx += (ax / dt - this.vx) * .45; }
      this._kick(); });
    const up = e => { const had = this.ptrs.delete(e.pointerId); if (!had) return; try { el.releasePointerCapture(e.pointerId); } catch (_) {}
      el.style.cursor = this.mode === "pan" ? "move" : "grab"; if (this.ptrs.size < 2) this.pinch = null;
      const D = this.drag; if (D && this.ptrs.size === 0) { const held = performance.now() - D.last; if (held > 90 || this.reduced) { this.vx = this.vy = 0; }
        if (D.moved < 5 && performance.now() - D.t0 < 450 && D.btn === 0) this._click(e); this.drag = null; } this._kick(); };
    el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
    el.addEventListener("pointerleave", () => { this.pointer = null; this.hover = null; this.tipEl.style.opacity = 0; this._kick(); });
    el.addEventListener("wheel", e => { e.preventDefault(); stamp(); const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      this.distT = clamp(this.distT * Math.exp(clamp(dy, -240, 240) * .0011), 2.6, 11); this._kick(); }, {passive:false});
    el.addEventListener("dblclick", () => { stamp(); this.resetView(); });
    el.addEventListener("contextmenu", e => e.preventDefault());
    this.host.addEventListener("keydown", e => { const k = e.key; let used = true;
      if (k === "ArrowLeft") this.nudge(-1.1, 0); else if (k === "ArrowRight") this.nudge(1.1, 0); else if (k === "ArrowUp" && e.shiftKey) this.nudge(0, -1.1); else if (k === "ArrowDown" && e.shiftKey) this.nudge(0, 1.1);
      else if (k === "+" || k === "=") this.zoomBy(.85); else if (k === "-" || k === "_") this.zoomBy(1.18); else if (k === "0" || k === "r" || k === "R") this.resetView(); else used = false;
      if (used) e.preventDefault(); });
  }
  _rotate(yaw, pitch) {          // yaw about screen-vertical, pitch about screen-horizontal (free) / model-up + clamped elevation (turntable)
    if (this.mode === "turntable") { this.q.multiply(qAxis(AY, yaw));
      if (pitch) { const nq = qAxis(AX, pitch).multiply(this.q), uz = AY.clone().applyQuaternion(nq).z;
        if (uz >= Math.sin(rad(3)) && uz <= Math.sin(rad(89.5))) this.q.copy(nq); else this.vx = 0; } }
    else { const a = Math.hypot(yaw, pitch); if (a > 1e-9) this.q.premultiply(qAxis(new THREE.Vector3(pitch, yaw, 0).normalize(), a)); }
    this.q.normalize();
  }
  _panBy(dx, dy) { const k = 2 * this.dist * Math.tan(rad(this.camera.fov / 2)) / Math.max(this.el.clientHeight, 1); this.panT.x += dx * k; this.panT.y -= dy * k;
    const m = 2.2; this.panT.x = clamp(this.panT.x, -m, m); this.panT.y = clamp(this.panT.y, -m, m); }
  _animateTo(q1, d1, p1, ms) { this.anim = {t0:performance.now(), ms, q0:this.q.clone(), q1, d0:this.distT, d1, p1, p0:this.panT.clone()}; this.vx = this.vy = 0; this.lastInput = performance.now(); this.spinGain = 0; this._kick(); }
  _ndc(p) { const r = this.el.getBoundingClientRect(); return new THREE.Vector2((p.x - r.left) / r.width * 2 - 1, -((p.y - r.top) / r.height) * 2 + 1); }
  _pickAt(p) { if (!this.mesh) return null; this.raycaster.setFromCamera(this._ndc(p), this.camera); const h = this.raycaster.intersectObject(this.mesh)[0]; if (!h) return null;
    const cell = h.faceIndex >> 1, C = this.sf.z[0].length - 1, i0 = Math.floor(cell / C), j0 = cell % C, local = this.mesh.worldToLocal(h.point.clone()); let best = null, bd = 1e9;
    for (const [i, j] of [[i0, j0], [i0, j0 + 1], [i0 + 1, j0], [i0 + 1, j0 + 1]]) { const v = this.map(this.sf.kRows[i][j], this.sf.days[i], this.sf.z[i][j]), d = v.distanceToSquared(local); if (d < bd) { bd = d; best = {k:this.sf.kRows[i][j], days:this.sf.days[i], v:this.sf.z[i][j], p:v}; } }
    return best; }
  _hover() { const now = performance.now(); if (now - (this._lh || 0) < 28 || !this.pointer || this.drag) return; this._lh = now;
    this.hover = this._pickAt(this.pointer); this._tip(); this._kick(); }
  _tip() { const t = this.tipEl; if (!this.hover || !this.pointer) { t.style.opacity = 0; return; } const r = this.host.getBoundingClientRect();
    t.innerHTML = this.tipFn(this.hover); t.style.opacity = 1; t.style.transform = `translate(${Math.min(this.pointer.x - r.left + 14, r.width - 190)}px,${Math.max(this.pointer.y - r.top - 8, 4)}px)`; }
  _click(e) { const h = this._pickAt({x:e.clientX, y:e.clientY}); if (h) this.onPick(h); }

  /* ---------------- loop ---------------- */
  _observe() {
    this.ro = new ResizeObserver(() => this._resize()); this.ro.observe(this.host);
    this.io = new IntersectionObserver(es => { this.visible = es[0].isIntersecting; if (this.visible) this._kick(); }); this.io.observe(this.host);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) this._kick(); });
  }
  _resize() { const w = this.host.clientWidth, h = this.host.clientHeight; if (!w || !h) return; this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this._drawNow(); }
  _drawNow() { if (this.el && this.el.clientWidth && this.hasSurface) { this._frame(); } this._kick(); }
  _kick() { if (!this.raf && !this.failed) { this.last = performance.now(); this.raf = requestAnimationFrame(t => this._tick(t)); } }
  _tick(now) {
    this.raf = 0; if (!this.visible || document.hidden) return;
    const dt = Math.min(.05, Math.max(.001, (now - this.last) / 1000)); this.last = now; let busy = false;
    if (this.anim) { const A = this.anim, s = clamp((now - A.t0) / A.ms, 0, 1), e = ease(s); this.q.copy(A.q0).slerp(A.q1, e); this.distT = A.d0 + (A.d1 - A.d0) * e;
      if (A.p1) this.panT.copy(A.p0).lerp(A.p1, e); if (s >= 1) this.anim = null; busy = true; }
    else if (!this.drag && (Math.abs(this.vx) > .02 || Math.abs(this.vy) > .02)) { this._rotate(this.vy * dt, this.vx * dt); const k = Math.exp(-dt * (this.reduced ? 12 : 2.7)); this.vx *= k; this.vy *= k; busy = true; }
    const idle = (now - this.lastInput) > 1600;                     // ease the spin in after the user lets go
    this.spinGain += ((this.autoSpin && idle && !this.drag && !this.anim ? 1 : 0) - this.spinGain) * (1 - Math.exp(-dt * 2.2));
    if (this.spinGain > .004) { this.q.multiply(qAxis(AY, .24 * this.spinGain * dt)); this.q.normalize(); busy = true; if (this.pointer && !this.drag) { this.hover = this._pickAt(this.pointer); this._tip(); } }
    const kz = 1 - Math.exp(-dt * 11); const dd = this.distT - this.dist; if (Math.abs(dd) > 1e-4) { this.dist += dd * kz; busy = true; } else this.dist = this.distT;
    const px = this.panT.x - this.root.position.x, py = this.panT.y - this.root.position.y; if (Math.abs(px) + Math.abs(py) > 1e-4) { this.root.position.x += px * kz; this.root.position.y += py * kz; busy = true; }
    this._frame(); if (busy || this.drag || this.ptrs.size) this._kick();
  }
  _frame() {
    if (this.failed) return; this.camera.position.z = this.dist; this.root.quaternion.copy(this.q); this.root.updateMatrixWorld(true);
    if (this.hasSurface) { // back corner = the one farthest from the camera
      let best = 0, bz = 1e9; for (let c = 0; c < 4; c++) { const v = new THREE.Vector3([-1, 1, 1, -1][c], 0, [-1, -1, 1, 1][c]).applyMatrix4(this.content.matrixWorld); if (v.z < bz) { bz = v.z; best = c; } }
      if (best !== this.corner) this._layoutCorner(best); }
    this.renderer.render(this.scene, this.camera);
    const w = this.el.clientWidth, h = this.el.clientHeight, m = this.content.matrixWorld;
    for (const L of this.labels) { const v = L.p.clone().applyMatrix4(m), depth = v.z; v.project(this.camera); const hw = (L.hw || (L.hw = L.el.offsetWidth / 2 || 20)), x = clamp((v.x * .5 + .5) * w, hw + 8, w - 88 - hw), y = clamp((-v.y * .5 + .5) * h, 22, h - 64);
      L.el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px) translate(-50%,-50%)`; L.el.style.opacity = clamp(.35 + (depth + 1.6) * .4, .3, 1).toFixed(2); }
    if (this.hover) this._hoverMarker(); else if (this.dotEl) this.dotEl.style.opacity = 0;
  }
  _hoverMarker() { /* highlight the hovered grid vertex with a ring that is drawn as an HTML dot (cheap and crisp) */
    if (!this.dotEl) { this.dotEl = document.createElement("i"); this.dotEl.className = "s3d-dot"; this.labelLayer.append(this.dotEl); }
    const v = this.hover.p.clone().applyMatrix4(this.content.matrixWorld).project(this.camera), w = this.el.clientWidth, h = this.el.clientHeight;
    this.dotEl.style.transform = `translate(${((v.x * .5 + .5) * w).toFixed(1)}px,${((-v.y * .5 + .5) * h).toFixed(1)}px) translate(-50%,-50%)`; this.dotEl.style.opacity = 1; }
}
