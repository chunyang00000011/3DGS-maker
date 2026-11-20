"use strict";

const PART_PALETTE = [
  "56, 189, 248",
  "16, 185, 129",
  "245, 158, 11",
  "129, 140, 248",
  "244, 63, 94",
  "34, 211, 238",
  "168, 85, 247",
  "20, 184, 166",
  "251, 146, 60",
  "99, 102, 241",
  "132, 204, 22",
  "250, 204, 21",
];

function wrapAngleRad(a) {
  let x = (a + Math.PI) % (Math.PI * 2);
  if (x < 0) x += Math.PI * 2;
  return x - Math.PI;
}

class Real2SimViewport3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.canvas.style.touchAction = "none";
    this.ctx = canvas.getContext("2d");

    this.theme = "light";
    this.yaw = 0.72;
    this.pitch = 0.46;
    this.distance = 3.2;
    this.target = [0, 0, 0.28];
    this.fovScale = 1.05;
    this.cameraIndex = 0;

    this.layers = {
      neural: true,
      splats: true,
      mesh: false,
      convex: false,
      spawn: true,
    };

    this.scene = null;
    this.splatAxes = null;
    this.meshEdges = [];
    this.demoScene = this._createDemoShowcase();
    this.autoSpinDemo = true;
    this._rafId = null;

    this.neuralImage = null;
    this.neuralWorkspace = "";
    this._neuralInFlight = false;
    this._neuralDirty = false;
    this._neuralWantHQ = true;
    this._hqTimer = null;
    this._neuralFailed = false;
    this._neuralPausedByJob = false;
    this._neuralGen = 0;

    this.dragMode = null;
    this._activePointerId = null;
    this.lastMouse = { x: 0, y: 0 };

    // Interactive SAM 3 segmentation & 3D Scene Graph editing overlay state
    this.editorOpen = false;
    this.editorMode = "orbit";
    this.editorObjects = [];
    this.selectedObjectId = null;
    this.editorHighlight = true;
    this.editorDimBg = false;
    this.promptPoints = [];
    this.promptBox = null;
    this._maskOverlayImg = null;
    this._maskOverlayGen = 0;
    this.onEditorEvent = null;
    this._gizmoHandles = [];
    this._gizmoDrag = null;
    this._boxDragStart = null;
    this._pointerDownInfo = null;
    this._pointerMovedPx = 0;
    this.showAlignCrosshair = false;

    this._bindEvents();
    this.resize();
    this._startDemoLoop();
  }

  _createDemoShowcase() {
    const positions = [];
    const colors = [];
    const opacities = [];
    const scales = [];

    // Sculptural 3DGS radiance knot hovering above a Z=0 calibration pedestal.
    const numCurve = 280;
    for (let i = 0; i < numCurve; i++) {
      const t = (i / numCurve) * Math.PI * 2;
      const r = 0.56 + 0.2 * Math.cos(3 * t);
      const x = r * Math.cos(2 * t);
      const y = r * Math.sin(2 * t);
      const z = 0.42 + 0.2 * Math.sin(3 * t);
      positions.push([x, y, z]);
      const hueMix = 0.5 + 0.5 * Math.sin(t * 2.0);
      colors.push([
        0.18 + 0.35 * (1 - hueMix),
        0.72 + 0.22 * hueMix,
        0.95 - 0.18 * (1 - hueMix),
      ]);
      opacities.push(0.86);
      scales.push(0.032 + 0.012 * Math.abs(Math.cos(3 * t)));
    }

    // Subtle tabletop calibration disk splats on Z=0..0.04.
    const numBase = 140;
    for (let i = 0; i < numBase; i++) {
      const a = (i * 2.39996) % (Math.PI * 2);
      const rad = 0.92 * Math.sqrt((i + 0.5) / numBase);
      const x = rad * Math.cos(a);
      const y = rad * Math.sin(a);
      const z = 0.02 * Math.sin(rad * 6);
      positions.push([x, y, z]);
      const edge = rad / 0.92;
      colors.push([0.12 + 0.16 * edge, 0.42 + 0.38 * (1 - edge), 0.62 + 0.28 * edge]);
      opacities.push(0.42 + 0.28 * (1 - edge));
      scales.push(0.024);
    }

    // Orbiting multi-view capture camera frustums.
    const frustums = [];
    const numCams = 8;
    for (let c = 0; c < numCams; c++) {
      const ang = (c / numCams) * Math.PI * 2 + 0.25;
      const rad = 1.68;
      const z = 0.72 + 0.16 * Math.sin(c * 1.4);
      frustums.push({
        eye: [rad * Math.cos(ang), rad * Math.sin(ang), z],
        lookAt: [0, 0, 0.36],
        size: 0.16,
      });
    }

    return {
      splats: { positions, colors, opacities, scales },
      frustums,
      mesh: { vertices: [], faces: [] },
      convex_parts: [],
      scene_meta: { bbox_min: [-1.1, -1.1, 0], bbox_max: [1.1, 1.1, 0.9] },
    };
  }

  _startDemoLoop() {
    if (this._rafId !== null) return;
    const tick = () => {
      if (this.scene !== null || !this.autoSpinDemo) {
        this._rafId = null;
        return;
      }
      if (!this.dragMode) {
        this.yaw += 0.0032;
        this.render();
      }
      this._rafId = window.requestAnimationFrame(tick);
    };
    this._rafId = window.requestAnimationFrame(tick);
  }

  setTheme(theme) {
    this.theme = theme === "light" ? "light" : "dark";
    this.render();
    this._requestNeuralStream(true);
  }

  _clearNeuralImage() {
    if (this._hqTimer) {
      clearTimeout(this._hqTimer);
      this._hqTimer = null;
    }
    if (this.neuralImage && typeof this.neuralImage.close === "function") {
      this.neuralImage.close();
    }
    this.neuralImage = null;
    this.neuralWorkspace = "";
    this._neuralDirty = false;
    this._neuralFailed = false;
    this._neuralPausedByJob = false;
    this._neuralGen += 1;
  }

  _blobToImage(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = (err) => {
        URL.revokeObjectURL(url);
        reject(err);
      };
      img.src = url;
    });
  }

  _requestNeuralStream(highQuality = false) {
    if (!this.layers.neural || !this.scene?.neural_render_available || this._neuralPausedByJob) return;
    this._neuralFailed = false;
    if (this._hqTimer) {
      clearTimeout(this._hqTimer);
      this._hqTimer = null;
    }
    if (!highQuality) {
      this._hqTimer = setTimeout(() => {
        this._hqTimer = null;
        if (!this.dragMode) {
          this._requestNeuralStream(true);
        }
      }, 90);
    }
    this._neuralDirty = true;
    this._neuralWantHQ = Boolean(highQuality);
    this._pumpNeuralStream();
  }

  async _pumpNeuralStream() {
    if (this._neuralInFlight || !this._neuralDirty) return;
    if (!this.layers.neural || !this.scene?.neural_render_available || !this.scene?.workspace_dir) {
      this._neuralDirty = false;
      return;
    }

    this._neuralDirty = false;
    this._neuralInFlight = true;
    const gen = this._neuralGen;
    const wsDir = this.scene.workspace_dir;
    const isHQ = this._neuralWantHQ && !this.dragMode;

    const dpr = Math.min(window.devicePixelRatio || 1, 2.0);
    const w = isHQ
      ? Math.max(320, Math.min(1920, Math.round(this.width * dpr)))
      : Math.max(240, Math.min(960, Math.round(this.width * 0.72)));
    const h = isHQ
      ? Math.max(240, Math.min(1200, Math.round(this.height * dpr)))
      : Math.max(180, Math.min(640, Math.round(this.height * 0.72)));

    const params = new URLSearchParams({
      workspace_dir: wsDir,
      yaw: String(this.yaw),
      pitch: String(this.pitch),
      distance: String(this.distance),
      tx: String(this.target[0]),
      ty: String(this.target[1]),
      tz: String(this.target[2]),
      width: String(w),
      height: String(h),
      theme: this.theme,
      fov_scale: String(this.fovScale || 1.05),
      hq: isHQ ? "1" : "0",
      apply_edits: "1",
      highlight: this.editorOpen && this.editorHighlight ? "1" : "0",
      dim_bg: this.editorOpen && this.editorDimBg ? "1" : "0",
    });
    if (this.selectedObjectId !== null && this.selectedObjectId !== undefined) {
      params.set("selected_id", String(this.selectedObjectId));
    }

    try {
      const resp = await fetch(`/api/render-view?${params.toString()}`, { cache: "no-store" });
      if (resp.status === 409) {
        if (gen === this._neuralGen) {
          this._neuralPausedByJob = true;
          this._neuralDirty = false;
          this.render();
        }
        return;
      }
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const bmp = window.createImageBitmap ? await window.createImageBitmap(blob) : await this._blobToImage(blob);
      if (gen === this._neuralGen && this.scene?.workspace_dir === wsDir) {
        if (this.neuralImage && typeof this.neuralImage.close === "function") {
          this.neuralImage.close();
        }
        this.neuralImage = bmp;
        this.neuralWorkspace = wsDir;
        this._neuralFailed = false;
        this.render();
      } else if (bmp && typeof bmp.close === "function") {
        bmp.close();
      }
    } catch {
      if (gen === this._neuralGen && !this.neuralImage) {
        this._neuralFailed = true;
        this.render();
      }
    } finally {
      this._neuralInFlight = false;
      if (this._neuralDirty) {
        this._pumpNeuralStream();
      }
    }
  }

  showDemoShowcase() {
    this._clearNeuralImage();
    this.scene = null;
    this._invalidatePrompts(false);
    this.splatAxes = null;
    this.meshEdges = [];
    this.autoSpinDemo = true;
    this.resetView();
    this._startDemoLoop();
  }

  _invalidatePrompts(notify = true) {
    const had = Boolean(this.promptPoints.length || this.promptBox || this._maskOverlayImg);
    if (!had) return;
    this.promptPoints = [];
    this.promptBox = null;
    this._maskOverlayImg = null;
    this._maskOverlayGen += 1;
    if (notify && typeof this.onEditorEvent === "function") {
      this.onEditorEvent("prompts_invalidated", {});
    }
  }

  requestImmediateNeuralRefresh(highQuality = true) {
    this._neuralPausedByJob = false;
    this._neuralFailed = false;
    this._requestNeuralStream(highQuality);
  }

  setEditorState({ open, mode, objects, selectedId, highlight, dimBg } = {}) {
    let needNeural = false;
    if (open !== undefined && Boolean(open) !== this.editorOpen) {
      this.editorOpen = Boolean(open);
      if (!this.editorOpen) this._invalidatePrompts(false);
      needNeural = true;
    }
    if (mode !== undefined) {
      this.editorMode = String(mode || "orbit");
      const crossModes = new Set(["point", "box", "pick"]);
      this.canvas.style.cursor = this.editorOpen && crossModes.has(this.editorMode) ? "crosshair" : "grab";
    }
    if (objects !== undefined) {
      this.editorObjects = Array.isArray(objects) ? objects : [];
    }
    if (selectedId !== undefined && selectedId !== this.selectedObjectId) {
      this.selectedObjectId = selectedId;
      needNeural = true;
    }
    if (highlight !== undefined && Boolean(highlight) !== this.editorHighlight) {
      this.editorHighlight = Boolean(highlight);
      needNeural = true;
    }
    if (dimBg !== undefined && Boolean(dimBg) !== this.editorDimBg) {
      this.editorDimBg = Boolean(dimBg);
      needNeural = true;
    }
    this.render();
    if (needNeural) {
      this._requestNeuralStream(true);
    }
  }

  setPromptOverlay({ points, box, maskBase64 } = {}) {
    if (points !== undefined) {
      this.promptPoints = Array.isArray(points) ? points : [];
    }
    if (box !== undefined) {
      this.promptBox = box;
    }
    if (maskBase64 === null) {
      this._maskOverlayImg = null;
      this._maskOverlayGen += 1;
      this.render();
      return;
    }
    if (typeof maskBase64 === "string" && maskBase64.length > 0) {
      const gen = ++this._maskOverlayGen;
      const img = new Image();
      img.onload = () => {
        if (gen === this._maskOverlayGen) {
          this._maskOverlayImg = img;
          this.render();
        }
      };
      img.src = `data:image/png;base64,${maskBase64}`;
      return;
    }
    this.render();
  }

  getCameraSnapshot() {
    return {
      yaw: this.yaw,
      pitch: this.pitch,
      distance: this.distance,
      tx: this.target[0],
      ty: this.target[1],
      tz: this.target[2],
      width: this.width,
      height: this.height,
      fov_scale: this.fovScale || 1.05,
      theme: this.theme,
    };
  }

  focusOnBox(bboxMin, bboxMax) {
    if (!Array.isArray(bboxMin) || !Array.isArray(bboxMax) || bboxMin.length < 3 || bboxMax.length < 3) return;
    this._invalidatePrompts(true);
    this.autoSpinDemo = false;
    this.target = [
      (Number(bboxMin[0]) + Number(bboxMax[0])) * 0.5,
      (Number(bboxMin[1]) + Number(bboxMax[1])) * 0.5,
      (Number(bboxMin[2]) + Number(bboxMax[2])) * 0.5,
    ];
    const diag = Math.hypot(
      Number(bboxMax[0]) - Number(bboxMin[0]),
      Number(bboxMax[1]) - Number(bboxMin[1]),
      Number(bboxMax[2]) - Number(bboxMin[2])
    );
    this.distance = Math.max(0.32, Math.min(20.0, (diag || 0.28) * 2.05));
    this.render();
    this._requestNeuralStream(true);
  }

  _canvasLocalCoords(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      sx: e.clientX - rect.left,
      sy: e.clientY - rect.top,
    };
  }

  _selectedObject() {
    if (this.selectedObjectId === null || this.selectedObjectId === undefined) return null;
    return this.editorObjects.find((o) => Number(o.id) === Number(this.selectedObjectId)) || null;
  }

  _distToSegment2D(px, py, ax, ay, bx, by) {
    const vx = bx - ax;
    const vy = by - ay;
    const wx = px - ax;
    const wy = py - ay;
    const len2 = vx * vx + vy * vy;
    const t = len2 > 1e-6 ? Math.max(0, Math.min(1, (wx * vx + wy * vy) / len2)) : 0;
    return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
  }

  _intersectHorizontalPlane(sx, sy, z0, basis, focal) {
    const dx = ((sx - this.width * 0.5) / focal) * basis.right[0] - ((sy - this.height * 0.5) / focal) * basis.up[0] + basis.forward[0];
    const dy = ((sx - this.width * 0.5) / focal) * basis.right[1] - ((sy - this.height * 0.5) / focal) * basis.up[1] + basis.forward[1];
    const dz = ((sx - this.width * 0.5) / focal) * basis.right[2] - ((sy - this.height * 0.5) / focal) * basis.up[2] + basis.forward[2];
    if (Math.abs(dz) < 1e-3) return null;
    const t = (z0 - basis.eye[2]) / dz;
    if (t <= 0.05 || t > 500) return null;
    return [basis.eye[0] + t * dx, basis.eye[1] + t * dy, z0];
  }

  _hitTestGizmo(sx, sy) {
    if (!this.editorOpen || !this.scene || !["move", "rotate", "scale"].includes(this.editorMode)) return null;
    const obj = this._selectedObject();
    if (!obj || obj.deleted || obj.visible === false) return null;
    const handles = this._gizmoHandles || [];
    for (const h of handles) {
      if (h.kind === "plane" && Math.hypot(sx - h.cx, sy - h.cy) <= 13) {
        return h;
      }
      if (h.kind === "axis" && this._distToSegment2D(sx, sy, h.ax, h.ay, h.bx, h.by) <= 12) {
        return h;
      }
      if (h.kind === "ring" && Array.isArray(h.pts)) {
        for (let i = 0; i < h.pts.length - 1; i++) {
          const p0 = h.pts[i];
          const p1 = h.pts[i + 1];
          if (this._distToSegment2D(sx, sy, p0.x, p0.y, p1.x, p1.y) <= 13) {
            return h;
          }
        }
      }
    }
    return null;
  }

  _bindEvents() {
    window.addEventListener("resize", () => this.resize());
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    this.canvas.addEventListener("pointerdown", (e) => {
      if (e.button === 1) e.preventDefault();
      this.autoSpinDemo = false;
      const { sx, sy } = this._canvasLocalCoords(e);
      this._pointerDownInfo = {
        x: e.clientX,
        y: e.clientY,
        sx,
        sy,
        button: e.button,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
      };
      this._pointerMovedPx = 0;
      this._activePointerId = e.pointerId;
      this.lastMouse = { x: e.clientX, y: e.clientY };

      if (this.editorOpen && e.button === 0 && !e.altKey) {
        const hit = this._hitTestGizmo(sx, sy);
        if (hit) {
          const obj = this._selectedObject();
          const basis = this._cameraBasis();
          const focal = Math.min(this.width, this.height) * (this.scene ? (this.fovScale || 1.05) : 0.95);
          const piv = obj.transformed_pivot || obj.pivot || [0, 0, 0];
          const pivScr = this._project(piv, basis, focal) || { x: sx, y: sy };
          const planePt = this._intersectHorizontalPlane(sx, sy, piv[2], basis, focal);
          this._gizmoDrag = {
            handle: hit,
            objectId: Number(obj.id),
            startTrans: [...(obj.translation || [0, 0, 0])],
            startRot: [...(obj.rotation_euler_deg || [0, 0, 0])],
            startScale: Number(obj.scale ?? 1.0),
            startPivot: [...piv],
            startPlanePt: planePt,
            startAngle: Math.atan2(sy - pivScr.y, sx - pivScr.x),
            startClient: { x: e.clientX, y: e.clientY },
            pivScr,
            basis,
            focal,
            undoPending: true,
          };
          if (this.canvas.setPointerCapture) {
            try { this.canvas.setPointerCapture(e.pointerId); } catch {}
          }
          return;
        }

        if (this.editorMode === "box" && !e.shiftKey) {
          this._boxDragStart = { sx, sy };
          const nx = Math.max(0, Math.min(1, sx / Math.max(1, this.width)));
          const ny = Math.max(0, Math.min(1, sy / Math.max(1, this.height)));
          this.promptBox = { nx0: nx, ny0: ny, nx1: nx, ny1: ny };
          if (this.canvas.setPointerCapture) {
            try { this.canvas.setPointerCapture(e.pointerId); } catch {}
          }
          this.render();
          return;
        }
      }

      this.dragMode = e.button === 2 || e.button === 1 || e.shiftKey ? "pan" : "orbit";
      if (this.canvas.setPointerCapture) {
        try {
          this.canvas.setPointerCapture(e.pointerId);
        } catch {
          // Ignore synthetic events without active pointer ids.
        }
      }
    });

    const handleMove = (e) => {
      if (this._activePointerId !== null && e.pointerId !== undefined && e.pointerId !== this._activePointerId) {
        return;
      }
      if (this._pointerDownInfo) {
        this._pointerMovedPx = Math.max(
          this._pointerMovedPx,
          Math.hypot(e.clientX - this._pointerDownInfo.x, e.clientY - this._pointerDownInfo.y)
        );
      }

      if (this._gizmoDrag) {
        const g = this._gizmoDrag;
        const obj = this.editorObjects.find((o) => Number(o.id) === g.objectId);
        if (!obj) return;
        const { sx, sy } = this._canvasLocalCoords(e);
        const totalDx = e.clientX - g.startClient.x;
        const totalDy = e.clientY - g.startClient.y;

        if (g.handle.mode === "move") {
          let nextT = [...g.startTrans];
          if (g.handle.kind === "plane") {
            const curPt = this._intersectHorizontalPlane(sx, sy, g.startPivot[2], g.basis, g.focal);
            if (curPt && g.startPlanePt) {
              nextT[0] = g.startTrans[0] + (curPt[0] - g.startPlanePt[0]);
              nextT[1] = g.startTrans[1] + (curPt[1] - g.startPlanePt[1]);
            } else {
              const panScale = Math.max(0.0005, this.distance * 0.0016);
              nextT[0] = g.startTrans[0] + (g.basis.right[0] * totalDx - g.basis.up[0] * totalDy) * panScale;
              nextT[1] = g.startTrans[1] + (g.basis.right[1] * totalDx - g.basis.up[1] * totalDy) * panScale;
            }
          } else if (g.handle.kind === "axis") {
            const vx = g.handle.bx - g.handle.ax;
            const vy = g.handle.by - g.handle.ay;
            const len2 = Math.max(16, vx * vx + vy * vy);
            const proj = (totalDx * vx + totalDy * vy) / len2;
            const deltaWorld = proj * g.handle.arm;
            const axVec = g.handle.vec;
            nextT[0] = g.startTrans[0] + deltaWorld * axVec[0];
            nextT[1] = g.startTrans[1] + deltaWorld * axVec[1];
            nextT[2] = g.startTrans[2] + deltaWorld * axVec[2];
          }
          const diffT = [nextT[0] - (obj.translation?.[0] || 0), nextT[1] - (obj.translation?.[1] || 0), nextT[2] - (obj.translation?.[2] || 0)];
          obj.translation = nextT;
          if (obj.transformed_bbox_min && obj.transformed_bbox_max) {
            obj.transformed_bbox_min = [obj.transformed_bbox_min[0] + diffT[0], obj.transformed_bbox_min[1] + diffT[1], obj.transformed_bbox_min[2] + diffT[2]];
            obj.transformed_bbox_max = [obj.transformed_bbox_max[0] + diffT[0], obj.transformed_bbox_max[1] + diffT[1], obj.transformed_bbox_max[2] + diffT[2]];
          }
          if (obj.pivot) {
            obj.transformed_pivot = [obj.pivot[0] + nextT[0], obj.pivot[1] + nextT[1], obj.pivot[2] + nextT[2]];
          }
        } else if (g.handle.mode === "rotate") {
          const curAngle = Math.atan2(sy - g.pivScr.y, sx - g.pivScr.x);
          let da = curAngle - g.startAngle;
          while (da > Math.PI) da -= Math.PI * 2;
          while (da < -Math.PI) da += Math.PI * 2;
          const sign = g.basis.forward[2] <= 0 ? -1 : 1;
          const degDelta = (da * 180) / Math.PI * sign;
          obj.rotation_euler_deg = [g.startRot[0], g.startRot[1], Number((g.startRot[2] + degDelta).toFixed(2))];
        } else if (g.handle.mode === "scale") {
          const factor = Math.exp((totalDx - totalDy) * 0.008);
          obj.scale = Math.max(0.05, Math.min(20.0, Number((g.startScale * factor).toFixed(4))));
        }

        const recordUndo = Boolean(g.undoPending);
        g.undoPending = false;
        this.render();
        this.onEditorEvent?.("gizmo_drag", {
          objectId: g.objectId,
          translation: obj.translation,
          rotation_euler_deg: obj.rotation_euler_deg,
          scale: obj.scale,
          recordUndo,
        });
        return;
      }

      if (this._boxDragStart) {
        const { sx, sy } = this._canvasLocalCoords(e);
        const nx0 = Math.max(0, Math.min(1, this._boxDragStart.sx / Math.max(1, this.width)));
        const ny0 = Math.max(0, Math.min(1, this._boxDragStart.sy / Math.max(1, this.height)));
        const nx1 = Math.max(0, Math.min(1, sx / Math.max(1, this.width)));
        const ny1 = Math.max(0, Math.min(1, sy / Math.max(1, this.height)));
        this.promptBox = {
          nx0: Math.min(nx0, nx1),
          ny0: Math.min(ny0, ny1),
          nx1: Math.max(nx0, nx1),
          ny1: Math.max(ny0, ny1),
        };
        this.render();
        return;
      }

      if (!this.dragMode) return;
      const dx = e.clientX - this.lastMouse.x;
      const dy = e.clientY - this.lastMouse.y;
      this.lastMouse = { x: e.clientX, y: e.clientY };

      if (this._pointerMovedPx >= 5 && this.editorOpen) {
        this._invalidatePrompts(true);
      }

      if (this.dragMode === "orbit") {
        // Flip horizontal yaw direction when upside-down so continuous 360-deg vertical orbit feels natural.
        const yawDir = Math.cos(this.pitch) >= 0 ? -1 : 1;
        this.yaw = wrapAngleRad(this.yaw + yawDir * dx * 0.0075);
        this.pitch = wrapAngleRad(this.pitch + dy * 0.0075);
      } else {
        const basis = this._cameraBasis();
        const panScale = Math.max(0.0005, this.distance * 0.0016);
        this.target[0] += (-basis.right[0] * dx + basis.up[0] * dy) * panScale;
        this.target[1] += (-basis.right[1] * dx + basis.up[1] * dy) * panScale;
        this.target[2] += (-basis.right[2] * dx + basis.up[2] * dy) * panScale;
      }
      this.render();
      this._requestNeuralStream(false);
    };

    const endDrag = (e) => {
      if (this.canvas.releasePointerCapture && this._activePointerId !== null && e?.pointerId === this._activePointerId) {
        try {
          this.canvas.releasePointerCapture(this._activePointerId);
        } catch {
          // Pointer already released.
        }
      }

      if (this._gizmoDrag) {
        const g = this._gizmoDrag;
        const obj = this.editorObjects.find((o) => Number(o.id) === g.objectId);
        const didMove = !g.undoPending;
        this._gizmoDrag = null;
        this._activePointerId = null;
        this._pointerDownInfo = null;
        if (obj && didMove) {
          this.onEditorEvent?.("gizmo_end", {
            objectId: g.objectId,
            translation: obj.translation,
            rotation_euler_deg: obj.rotation_euler_deg,
            scale: obj.scale,
            recordUndo: false,
          });
        }
        this.render();
        return;
      }

      if (this._boxDragStart) {
        const box = this.promptBox;
        this._boxDragStart = null;
        this._activePointerId = null;
        this._pointerDownInfo = null;
        if (box && (box.nx1 - box.nx0) * this.width >= 8 && (box.ny1 - box.ny0) * this.height >= 8) {
          this.onEditorEvent?.("box_prompt", { box });
        } else {
          this.promptBox = null;
        }
        this.render();
        return;
      }

      if (!this.dragMode) return;
      const downInfo = this._pointerDownInfo;
      const moved = this._pointerMovedPx;
      this.dragMode = null;
      this._activePointerId = null;
      this._pointerDownInfo = null;

      if (moved < 5 && downInfo && this.editorOpen) {
        if (this.editorMode === "point" && (downInfo.button === 0 || downInfo.button === 2)) {
          const label = downInfo.button === 2 || downInfo.shiftKey ? 0 : 1;
          const nx = Math.max(0, Math.min(1, downInfo.sx / Math.max(1, this.width)));
          const ny = Math.max(0, Math.min(1, downInfo.sy / Math.max(1, this.height)));
          this.promptPoints.push({ nx, ny, label });
          this.render();
          this.onEditorEvent?.("point_prompt", { points: this.promptPoints });
          return;
        }
        if (this.editorMode === "pick" && downInfo.button === 0) {
          this.onEditorEvent?.("pick_click", {
            sx: downInfo.sx,
            sy: downInfo.sy,
            width: this.width,
            height: this.height,
          });
          return;
        }
      }

      this.render();
      this._requestNeuralStream(true);
    };

    this.canvas.addEventListener("pointermove", handleMove);
    this.canvas.addEventListener("pointerup", endDrag);
    this.canvas.addEventListener("pointercancel", endDrag);
    this.canvas.addEventListener("lostpointercapture", () => endDrag(null));
    window.addEventListener("blur", () => endDrag(null));

    this.canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.autoSpinDemo = false;
        if (this.editorOpen) this._invalidatePrompts(true);
        const factor = e.deltaY > 0 ? 1.12 : 0.89;
        this.distance = Math.max(0.05, Math.min(250.0, this.distance * factor));
        this.render();
        this._requestNeuralStream(false);
      },
      { passive: false }
    );

    this.canvas.addEventListener("dblclick", () => {
      if (this.editorOpen && (this.editorMode === "point" || this.editorMode === "pick")) return;
      this.resetView();
    });
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.width = Math.max(300, Math.floor(rect.width));
    this.height = Math.max(260, Math.floor(rect.height));
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
    this._requestNeuralStream(true);
  }

  setLayer(name, enabled) {
    this.layers[name] = Boolean(enabled);
    if (name === "neural" && this.layers.neural) {
      this._neuralFailed = false;
      this._neuralPausedByJob = false;
      this._requestNeuralStream(true);
    }
    this.render();
  }

  setAlignCrosshair(enabled) {
    this.showAlignCrosshair = Boolean(enabled);
    this.render();
  }

  _precomputeSplatAxes(splats) {
    const quats = splats?.quats;
    const scales3d = splats?.scales_3d;
    const n = splats?.positions?.length || 0;
    if (!quats || !scales3d || quats.length !== n || scales3d.length !== n) {
      return null;
    }
    const out = new Float32Array(n * 9);
    for (let i = 0; i < n; i++) {
      const q = quats[i];
      const norm = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
      const w = q[0] / norm, x = q[1] / norm, y = q[2] / norm, z = q[3] / norm;
      const xx = x * x, yy = y * y, zz = z * z;
      const xy = x * y, xz = x * z, yz = y * z;
      const wx = w * x, wy = w * y, wz = w * z;

      const s = scales3d[i];
      const base = i * 9;
      out[base + 0] = (1 - 2 * (yy + zz)) * s[0];
      out[base + 1] = (2 * (xy + wz)) * s[0];
      out[base + 2] = (2 * (xz - wy)) * s[0];

      out[base + 3] = (2 * (xy - wz)) * s[1];
      out[base + 4] = (1 - 2 * (xx + zz)) * s[1];
      out[base + 5] = (2 * (yz + wx)) * s[1];

      out[base + 6] = (2 * (xz + wy)) * s[2];
      out[base + 7] = (2 * (yz - wx)) * s[2];
      out[base + 8] = (1 - 2 * (xx + yy)) * s[2];
    }
    return out;
  }

  setSceneData(preview, { preserveCamera = false } = {}) {
    this._clearNeuralImage();
    this.scene = preview;
    this.autoSpinDemo = false;
    this.splatAxes = this._precomputeSplatAxes(preview?.splats);
    this.meshEdges = this._extractUniqueEdges(preview?.mesh?.faces || []);
    if (preserveCamera) {
      this._invalidatePrompts(true);
      this.render();
      this._requestNeuralStream(true);
    } else {
      this.resetView();
    }
  }

  _extractUniqueEdges(faces) {
    const seen = new Set();
    const edges = [];
    for (let i = 0; i < faces.length; i++) {
      const f = faces[i];
      const pairs = [
        [f[0], f[1]],
        [f[1], f[2]],
        [f[2], f[0]],
      ];
      for (let p = 0; p < 3; p++) {
        const a = Math.min(pairs[p][0], pairs[p][1]);
        const b = Math.max(pairs[p][0], pairs[p][1]);
        const key = `${a}_${b}`;
        if (!seen.has(key)) {
          seen.add(key);
          edges.push([a, b]);
        }
      }
    }
    return edges;
  }

  snapToCapturedCamera(index = 0) {
    const cams = this.scene?.cameras;
    if (!cams || cams.length === 0) return null;
    this._invalidatePrompts(true);
    this.cameraIndex = ((index % cams.length) + cams.length) % cams.length;
    const cam = cams[this.cameraIndex];
    this.autoSpinDemo = false;
    this.yaw = Number(cam.yaw) || 0;
    this.pitch = Number(cam.pitch) || 0;
    this.distance = Number(cam.distance) || 3.0;
    this.target = cam.target ? [...cam.target] : [0, 0, 0.25];
    this.fovScale = Number(cam.fov_scale) || 1.05;
    this.render();
    this._requestNeuralStream(true);
    return { index: this.cameraIndex + 1, total: cams.length, name: cam.name };
  }

  cycleCapturedCamera() {
    const cams = this.scene?.cameras;
    if (!cams || cams.length === 0) return null;
    return this.snapToCapturedCamera(this.cameraIndex + 1);
  }

  resetView() {
    if (this.scene?.cameras?.length > 0) {
      this.snapToCapturedCamera(0);
      return;
    }
    this._invalidatePrompts(true);
    this.yaw = 0.78;
    this.pitch = 0.42;
    this.fovScale = 0.95;
    const active = this.scene || this.demoScene;
    const meta = active?.scene_meta;
    if (meta && meta.bbox_min && meta.bbox_max) {
      const bmin = meta.bbox_min;
      const bmax = meta.bbox_max;
      this.target = [
        (bmin[0] + bmax[0]) * 0.5,
        (bmin[1] + bmax[1]) * 0.5,
        (bmin[2] + bmax[2]) * 0.5,
      ];
      const diag = Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
      this.distance = Math.max(1.0, diag * 0.68);
    } else if (active?.splats?.positions?.length > 0) {
      const pos = active.splats.positions;
      const xs = [], ys = [], zs = [];
      for (let i = 0; i < pos.length; i++) {
        xs.push(pos[i][0]);
        ys.push(pos[i][1]);
        zs.push(pos[i][2]);
      }
      xs.sort((a, b) => a - b);
      ys.sort((a, b) => a - b);
      zs.sort((a, b) => a - b);
      const lo = Math.floor(pos.length * 0.08);
      const hi = Math.max(lo + 1, Math.min(pos.length - 1, Math.floor(pos.length * 0.92)));
      this.target = [
        (xs[lo] + xs[hi]) * 0.5,
        (ys[lo] + ys[hi]) * 0.5,
        (zs[lo] + zs[hi]) * 0.5,
      ];
      const diag = Math.hypot(xs[hi] - xs[lo], ys[hi] - ys[lo], zs[hi] - zs[lo]);
      this.distance = Math.max(1.0, Math.min(25.0, diag * 0.65));
    } else {
      this.target = [0, 0, 0.25];
      this.distance = 2.6;
    }
    this.render();
    this._requestNeuralStream(true);
  }

  setPresetView(presetName) {
    const presets = {
      iso: { yaw: 0.72, pitch: 0.45 },
      top: { yaw: 0.0, pitch: 1.52 },
      front: { yaw: 0.0, pitch: 0.16 },
      side: { yaw: 1.57, pitch: 0.16 },
    };
    const p = presets[presetName];
    if (!p) return;
    this._invalidatePrompts(true);
    this.autoSpinDemo = false;
    this.yaw = p.yaw;
    this.pitch = p.pitch;
    this.render();
    this._requestNeuralStream(true);
  }

  _cameraBasis() {
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);

    const eye = [
      this.target[0] + this.distance * cp * cy,
      this.target[1] + this.distance * cp * sy,
      this.target[2] + this.distance * sp,
    ];
    // Singularity-free meridian basis allowing smooth unrestricted 360-deg horizontal & vertical orbit.
    const forward = [-cp * cy, -cp * sy, -sp];
    const up = [-sp * cy, -sp * sy, cp];
    const right = [-sy, cy, 0];
    return { eye, forward, right, up };
  }

  _project(pt, basis, focal) {
    const dx = pt[0] - basis.eye[0];
    const dy = pt[1] - basis.eye[1];
    const dz = pt[2] - basis.eye[2];
    const cz = dx * basis.forward[0] + dy * basis.forward[1] + dz * basis.forward[2];
    if (cz <= 0.05) return null;
    const cx = dx * basis.right[0] + dy * basis.right[1] + dz * basis.right[2];
    const cy = dx * basis.up[0] + dy * basis.up[1] + dz * basis.up[2];
    const scale = focal / cz;
    return {
      x: this.width * 0.5 + cx * scale,
      y: this.height * 0.5 - cy * scale,
      z: cz,
      scale,
    };
  }

  _drawLine3D(a, b, basis, focal, color, lineWidth = 1) {
    const pa = this._project(a, basis, focal);
    const pb = this._project(b, basis, focal);
    if (!pa || !pb) return;
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = lineWidth;
    this.ctx.beginPath();
    this.ctx.moveTo(pa.x, pa.y);
    this.ctx.lineTo(pb.x, pb.y);
    this.ctx.stroke();
  }

  _drawCameraFrustums(frustums, basis, focal, isLight) {
    const strokeCol = isLight ? "rgba(14, 116, 144, 0.55)" : "rgba(56, 189, 248, 0.52)";
    const fillDot = isLight ? "#0284c7" : "#38bdf8";
    for (let i = 0; i < frustums.length; i++) {
      const f = frustums[i];
      const e = f.eye;
      const lx = f.lookAt[0] - e[0];
      const ly = f.lookAt[1] - e[1];
      const lz = f.lookAt[2] - e[2];
      const len = Math.hypot(lx, ly, lz) || 1;
      const fwd = [lx / len, ly / len, lz / len];

      let rx = fwd[1], ry = -fwd[0], rz = 0;
      const rl = Math.hypot(rx, ry, rz) || 1;
      rx /= rl; ry /= rl;
      const ux = ry * fwd[2] - rz * fwd[1];
      const uy = rz * fwd[0] - rx * fwd[2];
      const uz = rx * fwd[1] - ry * fwd[0];

      const s = f.size || 0.15;
      const d = s * 1.25;
      const w = s * 0.72;
      const h = s * 0.48;
      const center = [e[0] + fwd[0] * d, e[1] + fwd[1] * d, e[2] + fwd[2] * d];
      const corners = [
        [center[0] - rx * w - ux * h, center[1] - ry * w - uy * h, center[2] - rz * w - uz * h],
        [center[0] + rx * w - ux * h, center[1] + ry * w - uy * h, center[2] + rz * w - uz * h],
        [center[0] + rx * w + ux * h, center[1] + ry * w + uy * h, center[2] + rz * w + uz * h],
        [center[0] - rx * w + ux * h, center[1] - ry * w + uy * h, center[2] - rz * w + uz * h],
      ];

      for (let k = 0; k < 4; k++) {
        this._drawLine3D(e, corners[k], basis, focal, strokeCol, 1);
        this._drawLine3D(corners[k], corners[(k + 1) % 4], basis, focal, strokeCol, 1.1);
      }
      const pe = this._project(e, basis, focal);
      if (pe) {
        this.ctx.fillStyle = fillDot;
        this.ctx.beginPath();
        this.ctx.arc(pe.x, pe.y, 2.6, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }
  }

  _drawOrientationGizmo(basis, isLight) {
    const cx = this.width - 46;
    const cy = this.height - 48;
    const len = 24;
    const ctx = this.ctx;

    ctx.save();
    ctx.fillStyle = isLight ? "rgba(255, 255, 255, 0.82)" : "rgba(13, 17, 26, 0.72)";
    ctx.strokeStyle = isLight ? "rgba(148, 163, 184, 0.5)" : "rgba(148, 163, 184, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 32, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    const axes = [
      { label: "X", vec: [1, 0, 0], color: "#fb7185" },
      { label: "Y", vec: [0, 1, 0], color: "#34d399" },
      { label: "Z", vec: [0, 0, 1], color: "#38bdf8" },
    ].map((ax) => {
      const sx = ax.vec[0] * basis.right[0] + ax.vec[1] * basis.right[1] + ax.vec[2] * basis.right[2];
      const sy = -(ax.vec[0] * basis.up[0] + ax.vec[1] * basis.up[1] + ax.vec[2] * basis.up[2]);
      const depth = ax.vec[0] * basis.forward[0] + ax.vec[1] * basis.forward[1] + ax.vec[2] * basis.forward[2];
      return { ...ax, x: cx + sx * len, y: cy + sy * len, depth };
    });

    axes.sort((a, b) => b.depth - a.depth);
    for (const ax of axes) {
      ctx.strokeStyle = ax.color;
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ax.x, ax.y);
      ctx.stroke();

      ctx.fillStyle = ax.color;
      ctx.beginPath();
      ctx.arc(ax.x, ax.y, 6.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#081018";
      ctx.font = "bold 9px Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(ax.label, ax.x, ax.y + 0.5);
    }
    ctx.restore();
  }

  _drawAlignCrosshair(isLight) {
    if (!this.showAlignCrosshair) return;
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    ctx.save();

    const subStroke = isLight ? "rgba(14, 165, 233, 0.28)" : "rgba(56, 189, 248, 0.28)";
    const mainStroke = isLight ? "rgba(14, 165, 233, 0.72)" : "rgba(56, 189, 248, 0.78)";

    ctx.strokeStyle = subStroke;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (const f of [0.25, 0.75]) {
      ctx.beginPath();
      ctx.moveTo(0, h * f);
      ctx.lineTo(w, h * f);
      ctx.moveTo(w * f, 0);
      ctx.lineTo(w * f, h);
      ctx.stroke();
    }

    ctx.strokeStyle = mainStroke;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([8, 4]);
    ctx.beginPath();
    ctx.moveTo(0, h * 0.5);
    ctx.lineTo(w, h * 0.5);
    ctx.moveTo(w * 0.5, 0);
    ctx.lineTo(w * 0.5, h);
    ctx.stroke();

    const yawDeg = ((this.yaw * 180) / Math.PI).toFixed(1);
    const pitchDeg = ((this.pitch * 180) / Math.PI).toFixed(1);
    const label = `Yaw: ${yawDeg}° · Pitch: ${pitchDeg}°`;
    ctx.setLineDash([]);
    ctx.font = "600 11px Consolas, monospace";
    const tw = ctx.measureText(label).width;
    const bw = tw + 18;
    const bx = (w - bw) * 0.5;
    const by = h - 54;
    ctx.fillStyle = isLight ? "rgba(255, 255, 255, 0.86)" : "rgba(13, 17, 26, 0.82)";
    ctx.strokeStyle = mainStroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(bx, by, bw, 22, 6) : ctx.rect(bx, by, bw, 22);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = isLight ? "#0369a1" : "#38bdf8";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, w * 0.5, by + 11.5);
    ctx.restore();
  }

  render() {
    const ctx = this.ctx;
    if (!ctx) return;
    const isLight = this.theme === "light";
    ctx.clearRect(0, 0, this.width, this.height);

    const basis = this._cameraBasis();
    const focal = Math.min(this.width, this.height) * (this.scene ? (this.fovScale || 1.05) : 0.95);
    const activeScene = this.scene || this.demoScene;
    const isDemo = this.scene === null;

    const hasNeuralFrame = Boolean(
      !isDemo &&
        this.layers.neural &&
        !this._neuralFailed &&
        activeScene.neural_render_available &&
        this.neuralImage &&
        this.neuralWorkspace === activeScene.workspace_dir
    );

    if (hasNeuralFrame) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(this.neuralImage, 0, 0, this.width, this.height);
    } else {
      // Optical center backdrop bloom
      const grad = ctx.createRadialGradient(
        this.width * 0.5,
        this.height * 0.45,
        10,
        this.width * 0.5,
        this.height * 0.5,
        Math.max(this.width, this.height) * 0.68
      );
      if (isLight) {
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(1, "#e6ecf5");
      } else {
        grad.addColorStop(0, "#131b2b");
        grad.addColorStop(0.55, "#0b101b");
        grad.addColorStop(1, "#07090f");
      }
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, this.width, this.height);

      // 1. Dual-density CAD Ground Grid on Z=0
      const gridExtent = 2.0;
      const minorColor = isLight ? "rgba(100, 116, 139, 0.14)" : "rgba(148, 163, 184, 0.08)";
      const majorColor = isLight ? "rgba(71, 85, 105, 0.28)" : "rgba(56, 189, 248, 0.18)";
      for (let g = -8; g <= 8; g++) {
        const v = g * 0.25;
        const isMajor = g % 4 === 0;
        const col = isMajor ? majorColor : minorColor;
        const lw = isMajor ? 1.15 : 0.75;
        this._drawLine3D([-gridExtent, v, 0], [gridExtent, v, 0], basis, focal, col, lw);
        this._drawLine3D([v, -gridExtent, 0], [v, gridExtent, 0], basis, focal, col, lw);
      }
    }

    if (this.layers.spawn) {
      ctx.save();
      if (hasNeuralFrame) ctx.globalAlpha = 0.75;
      this._drawLine3D([0, 0, 0], [0.58, 0, 0], basis, focal, "#fb7185", 2);
      this._drawLine3D([0, 0, 0], [0, 0.58, 0], basis, focal, "#34d399", 2);
      this._drawLine3D([0, 0, 0], [0, 0, 0.62], basis, focal, "#38bdf8", 2.2);
      ctx.restore();
    }

    if (isDemo && activeScene.frustums) {
      this._drawCameraFrustums(activeScene.frustums, basis, focal, isLight);
    }

    // 2. MuJoCo Convex Collision Parts (depth-sorted)
    if (this.layers.convex && activeScene.convex_parts && activeScene.convex_parts.length > 0) {
      const tris = [];
      for (let pIdx = 0; pIdx < activeScene.convex_parts.length; pIdx++) {
        const part = activeScene.convex_parts[pIdx];
        const rgb = PART_PALETTE[pIdx % PART_PALETTE.length];
        const verts = part.vertices || [];
        const proj = new Array(verts.length);
        for (let i = 0; i < verts.length; i++) {
          proj[i] = this._project(verts[i], basis, focal);
        }
        const faces = part.faces || [];
        for (let f = 0; f < faces.length; f++) {
          const [i0, i1, i2] = faces[f];
          const p0 = proj[i0], p1 = proj[i1], p2 = proj[i2];
          if (!p0 || !p1 || !p2) continue;
          tris.push({ p0, p1, p2, rgb, depth: (p0.z + p1.z + p2.z) / 3 });
        }
      }
      tris.sort((a, b) => b.depth - a.depth);
      ctx.lineWidth = 0.8;
      for (let i = 0; i < tris.length; i++) {
        const t = tris[i];
        ctx.fillStyle = `rgba(${t.rgb}, 0.17)`;
        ctx.strokeStyle = `rgba(${t.rgb}, 0.48)`;
        ctx.beginPath();
        ctx.moveTo(t.p0.x, t.p0.y);
        ctx.lineTo(t.p1.x, t.p1.y);
        ctx.lineTo(t.p2.x, t.p2.y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    // 3. Marching Cubes Collision Mesh Wireframe
    if (this.layers.mesh && activeScene.mesh && activeScene.mesh.vertices.length > 0) {
      const verts = activeScene.mesh.vertices;
      const proj = new Array(verts.length);
      for (let i = 0; i < verts.length; i++) {
        proj[i] = this._project(verts[i], basis, focal);
      }
      ctx.strokeStyle = isLight ? "rgba(14, 116, 144, 0.38)" : "rgba(56, 189, 248, 0.35)";
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      for (let e = 0; e < this.meshEdges.length; e++) {
        const [a, b] = this.meshEdges[e];
        const pa = proj[a];
        const pb = proj[b];
        if (!pa || !pb) continue;
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
      }
      ctx.stroke();
    }

    // 4. 3DGS Oriented Elliptical Splats (fast interactive preview when neural frame isn't active)
    if (!hasNeuralFrame && this.layers.splats && activeScene.splats && activeScene.splats.positions.length > 0) {
      const { positions, colors, opacities, scales } = activeScene.splats;
      const axes = isDemo ? null : this.splatAxes;
      const rx = basis.right[0], ry = basis.right[1], rz = basis.right[2];
      const ux = basis.up[0], uy = basis.up[1], uz = basis.up[2];

      const drawList = [];
      for (let i = 0; i < positions.length; i++) {
        const p = this._project(positions[i], basis, focal);
        if (!p) continue;
        if (p.x < -30 || p.x > this.width + 30 || p.y < -30 || p.y > this.height + 30) continue;

        if (axes !== null) {
          const base = i * 9;
          const s2 = p.scale * p.scale;
          const a0 = axes[base + 0] * rx + axes[base + 1] * ry + axes[base + 2] * rz;
          const b0 = axes[base + 0] * ux + axes[base + 1] * uy + axes[base + 2] * uz;
          const a1 = axes[base + 3] * rx + axes[base + 4] * ry + axes[base + 5] * rz;
          const b1 = axes[base + 3] * ux + axes[base + 4] * uy + axes[base + 5] * uz;
          const a2 = axes[base + 6] * rx + axes[base + 7] * ry + axes[base + 8] * rz;
          const b2 = axes[base + 6] * ux + axes[base + 7] * uy + axes[base + 8] * uz;

          const cxx = (a0 * a0 + a1 * a1 + a2 * a2) * s2 + 0.35;
          const cxy = (a0 * b0 + a1 * b1 + a2 * b2) * s2;
          const cyy = (b0 * b0 + b1 * b1 + b2 * b2) * s2 + 0.35;
          const tr = cxx + cyy;
          const diff = cxx - cyy;
          const h = Math.hypot(diff, 2 * cxy);
          const r1 = Math.max(1.4, Math.min(20.0, 2.1 * Math.sqrt(0.5 * (tr + h))));
          const r2 = Math.max(1.0, Math.min(16.0, 2.1 * Math.sqrt(Math.max(0.08, 0.5 * (tr - h)))));
          const ang = -0.5 * Math.atan2(2 * cxy, diff);
          drawList.push({ i, x: p.x, y: p.y, z: p.z, r1, r2, ang, iso: false });
        } else {
          const r = Math.max(1.8, Math.min(14.0, (scales[i] || 0.03) * p.scale * 1.35));
          drawList.push({ i, x: p.x, y: p.y, z: p.z, r1: r, r2: r, ang: 0, iso: true });
        }
      }
      drawList.sort((a, b) => b.z - a.z);

      const useHaloPass = drawList.length <= 2400;
      for (let k = 0; k < drawList.length; k++) {
        const d = drawList[k];
        const c = colors[d.i];
        const rC = Math.round(c[0] * 255);
        const gC = Math.round(c[1] * 255);
        const bC = Math.round(c[2] * 255);
        const a = Math.max(0.22, Math.min(0.95, opacities[d.i] || 0.8));

        if (useHaloPass && d.r1 > 2.2) {
          ctx.fillStyle = `rgba(${rC}, ${gC}, ${bC}, ${(a * 0.24).toFixed(3)})`;
          ctx.beginPath();
          ctx.ellipse(d.x, d.y, d.r1 * 1.55, d.r2 * 1.55, d.ang, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = `rgba(${rC}, ${gC}, ${bC}, ${a.toFixed(3)})`;
        ctx.beginPath();
        if (d.iso || d.r1 < 2.1) {
          ctx.arc(d.x, d.y, d.r1, 0, Math.PI * 2);
        } else {
          ctx.ellipse(d.x, d.y, d.r1, d.r2, d.ang, 0, Math.PI * 2);
        }
        ctx.fill();
      }
    }

    // 5. Robot Spawn Marker
    const spawn = activeScene.scene_meta?.robot_spawn_xyz;
    if (this.layers.spawn && spawn) {
      const ps = this._project(spawn, basis, focal);
      if (ps) {
        ctx.fillStyle = "#f59e0b";
        ctx.strokeStyle = isLight ? "#0f172a" : "#ffffff";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(ps.x, ps.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.font = "600 11.5px Segoe UI Variable, Segoe UI, sans-serif";
        ctx.fillStyle = isLight ? "#b45309" : "#fde68a";
        ctx.fillText(`Robot Spawn [${spawn.join(", ")}]`, ps.x + 10, ps.y + 4);
      }
    }

    // 6. Editor SAM 3 2D Overlays, Selected AABB & Interactive 3D Transform Gizmo
    this._drawEditorOverlays(basis, focal, isLight);

    // 7. Horizon Alignment Crosshair Grid
    this._drawAlignCrosshair(isLight);

    // 8. Synchronized 3D Orientation Gizmo
    this._drawOrientationGizmo(basis, isLight);
  }

  _drawEditorOverlays(basis, focal, isLight) {
    this._gizmoHandles = [];
    if (!this.editorOpen || !this.scene) return;
    const ctx = this.ctx;

    if (this._maskOverlayImg) {
      ctx.save();
      ctx.drawImage(this._maskOverlayImg, 0, 0, this.width, this.height);
      ctx.restore();
    }

    if (this.promptBox) {
      const b = this.promptBox;
      const x0 = b.nx0 * this.width;
      const y0 = b.ny0 * this.height;
      const bw = (b.nx1 - b.nx0) * this.width;
      const bh = (b.ny1 - b.ny0) * this.height;
      ctx.save();
      ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.fillRect(x0, y0, bw, bh);
      ctx.strokeRect(x0, y0, bw, bh);
      ctx.restore();
    }

    for (const pt of this.promptPoints || []) {
      const px = pt.nx * this.width;
      const py = pt.ny * this.height;
      const isPos = Number(pt.label) === 1;
      ctx.save();
      ctx.fillStyle = isPos ? "#10b981" : "#f43f5e";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 6.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(px - 3, py);
      ctx.lineTo(px + 3, py);
      if (isPos) {
        ctx.moveTo(px, py - 3);
        ctx.lineTo(px, py + 3);
      }
      ctx.stroke();
      ctx.restore();
    }

    const obj = this._selectedObject();
    if (!obj || obj.deleted || obj.visible === false) return;

    const bmin = obj.transformed_bbox_min || obj.bbox_min;
    const bmax = obj.transformed_bbox_max || obj.bbox_max;
    if (Array.isArray(bmin) && Array.isArray(bmax) && bmin.length === 3 && bmax.length === 3) {
      const c = obj.color_rgb || [0.14, 0.72, 0.96];
      const boxCol = `rgba(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)}, 0.88)`;
      const corners = [
        [bmin[0], bmin[1], bmin[2]],
        [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmax[1], bmin[2]],
        [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]],
        [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmax[2]],
        [bmin[0], bmax[1], bmax[2]],
      ];
      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];
      ctx.save();
      ctx.setLineDash([4, 3]);
      for (const [i0, i1] of edges) {
        this._drawLine3D(corners[i0], corners[i1], basis, focal, boxCol, 1.4);
      }
      ctx.restore();
    }

    if (!["move", "rotate", "scale"].includes(this.editorMode)) return;
    const piv = obj.transformed_pivot || obj.pivot;
    if (!Array.isArray(piv) || piv.length < 3) return;
    const pp = this._project(piv, basis, focal);
    if (!pp) return;

    const diag = Array.isArray(bmin) && Array.isArray(bmax)
      ? Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
      : 0.4;
    const arm = Math.max(0.14, Math.min(this.distance * 0.26, Math.max(0.24, diag * 0.7)));

    if (this.editorMode === "move" || this.editorMode === "scale") {
      const axes = [
        { id: "x", label: "X", vec: [1, 0, 0], col: "#fb7185" },
        { id: "y", label: "Y", vec: [0, 1, 0], col: "#34d399" },
        { id: "z", label: "Z", vec: [0, 0, 1], col: "#38bdf8" },
      ];
      for (const ax of axes) {
        const endW = [piv[0] + ax.vec[0] * arm, piv[1] + ax.vec[1] * arm, piv[2] + ax.vec[2] * arm];
        const pe = this._project(endW, basis, focal);
        if (!pe) continue;
        this._gizmoHandles.push({
          kind: "axis",
          mode: this.editorMode,
          axis: ax.id,
          vec: ax.vec,
          arm,
          ax: pp.x,
          ay: pp.y,
          bx: pe.x,
          by: pe.y,
        });
        ctx.save();
        ctx.strokeStyle = ax.col;
        ctx.fillStyle = ax.col;
        ctx.lineWidth = 2.8;
        ctx.beginPath();
        ctx.moveTo(pp.x, pp.y);
        ctx.lineTo(pe.x, pe.y);
        ctx.stroke();

        if (this.editorMode === "scale") {
          ctx.fillRect(pe.x - 4.5, pe.y - 4.5, 9, 9);
        } else {
          ctx.beginPath();
          ctx.arc(pe.x, pe.y, 5.2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.font = "bold 10px Consolas, monospace";
        ctx.fillStyle = isLight ? "#0f172a" : "#ffffff";
        ctx.fillText(ax.label, pe.x + 8, pe.y + 3);
        ctx.restore();
      }

      if (this.editorMode === "move") {
        const planeW = [piv[0] + arm * 0.36, piv[1] + arm * 0.36, piv[2]];
        const pPl = this._project(planeW, basis, focal);
        if (pPl) {
          this._gizmoHandles.push({
            kind: "plane",
            mode: "move",
            axis: "xy",
            cx: pPl.x,
            cy: pPl.y,
          });
          ctx.save();
          ctx.fillStyle = "rgba(245, 158, 11, 0.32)";
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 1.8;
          ctx.fillRect(pPl.x - 7, pPl.y - 7, 14, 14);
          ctx.strokeRect(pPl.x - 7, pPl.y - 7, 14, 14);
          ctx.restore();
        }
      }
    } else if (this.editorMode === "rotate") {
      const rRing = arm * 0.82;
      const pts = [];
      const segs = 48;
      for (let i = 0; i <= segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        const pw = this._project([piv[0] + rRing * Math.cos(a), piv[1] + rRing * Math.sin(a), piv[2]], basis, focal);
        if (pw) pts.push(pw);
      }
      if (pts.length >= 4) {
        this._gizmoHandles.push({ kind: "ring", mode: "rotate", axis: "z", pts });
        ctx.save();
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.8;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.stroke();
        ctx.fillStyle = "#38bdf8";
        ctx.beginPath();
        ctx.arc(pts[0].x, pts[0].y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
  }
}

window.Real2SimViewport3D = Real2SimViewport3D;
