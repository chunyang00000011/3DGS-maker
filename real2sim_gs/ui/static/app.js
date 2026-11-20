"use strict";

const STAGE_ORDER = ["ingest", "pose", "reconstruct", "postprocess", "export"];

const RECIPES = {
  pure_3dgs: {
    label: "纯 3DGS 重建 (旗舰)",
    summary:
      "极致纯视觉重建：抽帧(180帧) -> COLMAP (自动去畸变) -> 白盒 3DGS-MCMC (30,000 步 · 80万点 · SH3 完整球谐) -> 尺度感知去噪摆正 +Z -> 导出标准 .ply 与 cameras.json。",
    apply: {
      ingestName: "sharpness_ffmpeg",
      ingestFps: 2.0,
      ingestMaxFrames: 180,
      ingestKeepRatio: 0.85,
      poseName: "colmap",
      poseModel: "SIMPLE_RADIAL",
      reconstructName: "gsplat_whitebox",
      strategy: "mcmc",
      iterations: 30000,
      maxPoints: 800000,
      shDegree: 3,
      mipVar: 0.0,
      flatness: 0.0,
      depthSmooth: 0.0,
      selectiveAdam: true,
      bilateralGrid: false,
      poseOpt: false,
      ppClean: true,
      ppAlign: true,
      ppSegment: false,
      ppMesh: false,
      ppMinOpacity: 0.02,
      ppScaleFactor: 1.0,
      expPly: true,
      expMujoco: false,
      expUnity: false,
    },
  },
  tabletop_sim: {
    label: "桌面抓取 (推荐)",
    summary:
      "面向机械臂桌面操作：抽帧 -> COLMAP (自动去畸变) -> 3DGS-MCMC (20,000 步 · 50万点 · SH3) -> 尺度感知去噪 + RANSAC +Z 对齐 -> 3D Marching Cubes 网格 -> 导出纯 .ply 与 MuJoCo 12 块凸包。",
    apply: {
      ingestName: "sharpness_ffmpeg",
      ingestFps: 2.0,
      ingestMaxFrames: 160,
      ingestKeepRatio: 0.85,
      poseName: "colmap",
      poseModel: "SIMPLE_RADIAL",
      reconstructName: "gsplat_whitebox",
      strategy: "mcmc",
      iterations: 20000,
      maxPoints: 500000,
      shDegree: 3,
      mipVar: 0.0,
      flatness: 0.001,
      depthSmooth: 0.001,
      selectiveAdam: true,
      bilateralGrid: false,
      poseOpt: false,
      ppClean: true,
      ppAlign: true,
      ppSegment: false,
      ppMesh: true,
      ppMeshMethod: "marching_cubes",
      ppVoxelRes: 56,
      ppMinOpacity: 0.02,
      ppScaleFactor: 1.0,
      expPly: true,
      expMujoco: true,
      expUnity: false,
      expConvexParts: 12,
    },
  },
  indoor_nav: {
    label: "室内大场景导航",
    summary:
      "面向全屋/实验室高精漫游与机器狗导航：180 帧全对全匹配 -> 30,000 步 · 80万点 · SH3 高清重建 -> 尺度感知去噪防黑洞 -> 2.5D 连续地面防穿模 -> 导出 .ply 与 MuJoCo 8 块碰撞。",
    apply: {
      ingestName: "sharpness_ffmpeg",
      ingestFps: 1.5,
      ingestMaxFrames: 180,
      ingestKeepRatio: 0.85,
      poseName: "colmap",
      poseModel: "SIMPLE_RADIAL",
      reconstructName: "gsplat_whitebox",
      strategy: "mcmc",
      iterations: 30000,
      maxPoints: 800000,
      shDegree: 3,
      mipVar: 0.0,
      flatness: 0.0005,
      depthSmooth: 0.0005,
      selectiveAdam: true,
      bilateralGrid: false,
      poseOpt: false,
      ppClean: true,
      ppAlign: true,
      ppSegment: false,
      ppMesh: true,
      ppMeshMethod: "heightfield",
      ppVoxelRes: 64,
      ppMinOpacity: 0.02,
      ppScaleFactor: 1.0,
      expPly: true,
      expMujoco: true,
      expUnity: false,
      expConvexParts: 8,
    },
  },
  quick_check: {
    label: "快速真机验证",
    summary:
      "先快速验证拍摄视频与视角覆盖：5,000 步 · 20万点 · SH3 CUDA 快速训练 + 轻量网格 + 默认保留 pure_3dgs 与 MuJoCo 输出。",
    apply: {
      ingestName: "sharpness_ffmpeg",
      ingestFps: 1.5,
      ingestMaxFrames: 80,
      ingestKeepRatio: 0.85,
      poseName: "colmap",
      poseModel: "SIMPLE_RADIAL",
      reconstructName: "gsplat_whitebox",
      strategy: "mcmc",
      iterations: 5000,
      maxPoints: 200000,
      shDegree: 3,
      mipVar: 0.0,
      flatness: 0.0,
      depthSmooth: 0.0,
      selectiveAdam: true,
      bilateralGrid: false,
      poseOpt: false,
      ppClean: true,
      ppAlign: true,
      ppSegment: false,
      ppMesh: true,
      ppMeshMethod: "marching_cubes",
      ppVoxelRes: 36,
      ppMinOpacity: 0.02,
      ppScaleFactor: 1.0,
      expPly: true,
      expMujoco: true,
      expUnity: false,
      expConvexParts: 8,
    },
  },
};

const METRIC_LABELS = {
  num_frames: "筛选关键帧数",
  avg_sharpness: "平均清晰度 (Laplacian)",
  pose_backend: "SfM 求解引擎",
  registered_views: "成功注册相机视角",
  init_points: "初始稀疏点云",
  PSNR: "渲染保真度 PSNR (dB)",
  SSIM: "结构相似度 SSIM",
  test_PSNR: "测试视角 PSNR (dB)",
  test_SSIM: "测试视角 SSIM",
  train_PSNR: "训练视角 PSNR (dB)",
  train_SSIM: "训练视角 SSIM",
  registered_ratio: "视角注册率",
  num_gaussians: "最终高斯球数",
  final_loss: "末步损失 Loss",
  render_backend: "CUDA 光栅化后端",
  sh_degree: "球谐阶数 (SH)",
  strategy: "致密化策略",
  selective_adam: "SelectiveAdam 稀疏",
  bilateral_grid: "3D 双边网格",
  pose_opt: "SE(3) 位姿微调",
  removed_floaters: "剔除漂浮噪点",
  retained_gaussians: "保留有效高斯",
  scale_factor: "米制缩放系数",
  mesh_method: "碰撞网格算法",
  mesh_faces: "网格三角面片数",
  exported_3dgs_points: "纯 3DGS 点云导出数",
  mujoco_convex_parts: "MuJoCo 凸包分解块",
  unity_exported_splats: "Unity 导出点数",
  resumed_from_step: "断点续训起始步",
};

const state = {
  stagesCatalog: {},
  currentConfig: null,
  activeJobId: null,
  activeRecipe: "tabletop_sim",
  theme: "light",
  historyWorkspaces: [],
  totalStorageMb: 0,
  uploadsStorageMb: 0,
  pendingDeleteWorkspace: null,
  activeWorkspaceDir: null,
  pollTimer: null,
  elapsedTimer: null,
  jobStartedMs: 0,
  viewport: null,
  latestPreview: null,
  latestErrorReport: "",
  latestLogs: [],
  lastLogSig: "",
  allStagesExpanded: false,
  editorOpen: false,
  editorMode: "orbit",
  editorObjects: [],
  selectedObjectId: null,
  editorCanUndo: false,
  editorCanRedo: false,
  editorBakedDirty: false,
  editorHighlight: true,
  editorDimBg: false,
  editorAutoFocus: true,
  editorBusy: false,
  alignOpen: false,
  alignBusy: false,
  _tfSyncPending: null,
  _tfSyncInFlight: false,
  _refineTimer: null,
  _refineUndoArmed: true,
  _refineDirty: false,
};

function getStudioToken() {
  return document.querySelector('meta[name="studio-token"]')?.content || "";
}

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatDuration(secFloat) {
  const s = Math.max(0, Math.round(Number(secFloat) || 0));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

function normWsPath(p) {
  return String(p || "")
    .trim()
    .replaceAll("\\", "/")
    .replace(/^\.\//, "")
    .replace(/\/+$/, "")
    .toLowerCase();
}

async function fetchJSON(url, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const headers = new Headers(options.headers || {});
  if (["POST", "PUT", "PATCH", "DELETE"].includes(method)) {
    const token = getStudioToken();
    if (token && !headers.has("X-Studio-Token")) {
      headers.set("X-Studio-Token", token);
    }
  }
  let resp;
  try {
    resp = await fetch(url, { ...options, headers });
  } catch (err) {
    const msg = String(err?.message || err || "");
    const isNetErr = err instanceof TypeError || /failed to fetch|networkerror|load failed/i.test(msg);
    if (!isNetErr) {
      throw err;
    }
    let alive = false;
    try {
      const h = await fetch("/health", { cache: "no-store" });
      alive = Boolean(h.ok);
    } catch {
      alive = false;
    }
    if (alive) {
      if (method === "GET") {
        resp = await fetch(url, { ...options, headers });
      } else {
        throw new Error(`后台服务在线但请求中断 (${msg})，请再次点击重试。`);
      }
    } else {
      throw new Error(
        "无法连接 Studio 后台服务 (Failed to fetch)：后台 Python 服务已退出（例如关闭了旧窗口或直接关闭了黑色控制台窗口）。请关闭当前界面，重新双击 Real2Sim-GS Studio 启动（运行期间请将黑色控制台窗口最小化保留）。"
      );
    }
  }
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(txt || `HTTP ${resp.status}`);
  }
  return resp.json();
}

function applyTheme(themeName) {
  const theme = themeName === "light" ? "light" : "dark";
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem("r2s_studio_theme", theme);
  } catch {
    // Ignore restricted storage environments.
  }
  const label = el("themeToggleLabel");
  if (label) {
    label.textContent = theme === "light" ? "极夜流光" : "科研白昼";
  }
  if (state.viewport) {
    state.viewport.setTheme(theme);
  }
}

function populateSelect(selectEl, options, selected) {
  selectEl.innerHTML = "";
  for (const optVal of options) {
    const opt = document.createElement("option");
    opt.value = optVal;
    opt.textContent = optVal;
    if (optVal === selected) opt.selected = true;
    selectEl.appendChild(opt);
  }
}

function updateRangeFill(rangeEl) {
  if (!rangeEl) return;
  const min = Number(rangeEl.min) || 0;
  const max = Number(rangeEl.max) || 100;
  const val = Number(rangeEl.value) || min;
  const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
  rangeEl.style.setProperty("--val-pct", `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`);
}

function bindSliderPair(rangeId, numberId, minClamp, maxClamp) {
  const r = el(rangeId);
  const n = el(numberId);
  if (!r || !n) return;
  r.addEventListener("input", () => {
    n.value = r.value;
    updateRangeFill(r);
    refreshStageBadges();
    evaluateComboAdvisor();
  });
  n.addEventListener("input", () => {
    const v = Number(n.value);
    if (Number.isFinite(v)) {
      r.value = String(Math.max(minClamp, Math.min(maxClamp, v)));
      updateRangeFill(r);
    }
    refreshStageBadges();
    evaluateComboAdvisor();
  });
  updateRangeFill(r);
}

function syncSliderFromNumber(rangeId, numberId, minClamp, maxClamp) {
  const r = el(rangeId);
  const n = el(numberId);
  if (!r || !n) return;
  const v = Number(n.value);
  if (Number.isFinite(v)) {
    r.value = String(Math.max(minClamp, Math.min(maxClamp, v)));
  }
  updateRangeFill(r);
}

async function loadDoctor() {
  try {
    const data = await fetchJSON("/api/doctor");
    const pills = el("doctorPills");
    const tbody = el("doctorTableBody");
    pills.innerHTML = "";
    tbody.innerHTML = "";

    const quickList = ["PyTorch + CUDA", "FFmpeg (视频抽帧)", "COLMAP (SfM 位姿解算)", "gsplat (高速 CUDA 光栅化)"];
    for (const item of data.items) {
      if (quickList.includes(item.component)) {
        const pill = document.createElement("span");
        pill.className = `pill ${item.available ? "ok" : "warn"}`;
        const shortName = item.component.split(" ")[0];
        pill.innerHTML = `<span class="pill-dot"></span>${escapeHtml(shortName)}: ${item.available ? "就绪" : "回退"}`;
        pills.appendChild(pill);
      }
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(item.component)}</td>
        <td><span class="pill ${item.available ? "ok" : "warn"}">${item.available ? "可用" : "缺失"}</span></td>
        <td>${escapeHtml(item.detail)}</td>
        <td>${escapeHtml(item.install_hint || "-")}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (err) {
    console.error("Doctor load failed:", err);
  }
}

function formatPointCount(num) {
  const n = Number(num) || 0;
  if (n >= 10000) return `${(n / 10000).toFixed(1)}万点`;
  return n > 0 ? `${n}点` : "";
}

function renderHistoryCards() {
  const listEl = el("historyList");
  const countEl = el("historyCountBadge");
  const cleanupBtn = el("btnCleanupUploads");
  const workspaces = state.historyWorkspaces || [];
  if (countEl) {
    countEl.textContent = `${workspaces.length} 个场景 · ${Number(state.totalStorageMb || 0).toFixed(1)} MB`;
  }
  if (cleanupBtn) {
    const upMb = Number(state.uploadsStorageMb || 0);
    cleanupBtn.textContent = upMb > 0.05 ? `清理缓存 (${upMb.toFixed(1)}MB)` : "清理缓存";
  }
  if (!listEl) return;

  if (workspaces.length === 0) {
    listEl.innerHTML = '<div class="empty-hint">暂无历史训练记录。完成一次构建或 Mock 演练后将自动归档于此。</div>';
    return;
  }

  const byDir = new Map(workspaces.map((w) => [w.workspace_dir, w]));

  listEl.innerHTML = workspaces
    .map((ws) => {
      const isActive = state.activeWorkspaceDir && normWsPath(state.activeWorkspaceDir) === normWsPath(ws.workspace_dir);
      const sm = ws.summary_metrics || {};
      const psnrText = sm.PSNR !== undefined ? `PSNR ${Number(sm.PSNR).toFixed(1)}dB` : "";
      const ptsText = formatPointCount(sm.exported_3dgs_points ?? sm.retained_gaussians ?? sm.num_gaussians);
      const exportTags = [ws.has_3dgs ? "3DGS" : "", ws.has_mujoco ? "MuJoCo" : "", ws.has_unity ? "Unity" : ""]
        .filter(Boolean)
        .join("/");
      const sizeText = ws.size_mb !== undefined ? `${Number(ws.size_mb).toFixed(1)} MB` : "";
      const ckpt = ws.resumable_checkpoint;
      const resumeText = ckpt
        ? `断点 ${ckpt.step}/${ckpt.total_steps}步`
        : ws.is_incomplete
        ? `未完 ${ws.completed_stages.length}/5`
        : "";
      const thumbHtml = ws.thumbnail_url
        ? `<img src="${escapeHtml(ws.thumbnail_url)}" alt="${escapeHtml(ws.name)}" loading="lazy" />`
        : `<span>${ws.has_3dgs ? "3DGS" : `${ws.completed_stages.length}/5`}</span>`;

      return `
      <div class="history-card ${isActive ? "active" : ""}" data-ws-dir="${escapeHtml(ws.workspace_dir)}">
        <div class="history-thumb">${thumbHtml}</div>
        <div class="history-meta">
          <div class="history-title-row">
            <span class="history-name" title="${escapeHtml(ws.workspace_dir)}">${escapeHtml(ws.name)}</span>
            <span class="history-time">${escapeHtml(ws.updated_label || "")}</span>
          </div>
          <div class="history-badges">
            <span class="h-pill ${ws.is_mock ? "" : "real"}">${ws.is_mock ? "Mock" : "真实训练"}</span>
            ${resumeText ? `<span class="h-pill resume">${escapeHtml(resumeText)}</span>` : ""}
            ${psnrText ? `<span class="h-pill psnr">${escapeHtml(psnrText)}</span>` : ""}
            ${ptsText ? `<span class="h-pill">${escapeHtml(ptsText)}</span>` : ""}
            ${exportTags ? `<span class="h-pill">${escapeHtml(exportTags)}</span>` : ""}
            ${sizeText ? `<span class="h-pill size">${escapeHtml(sizeText)}</span>` : ""}
          </div>
          <div class="history-actions">
            ${ws.is_incomplete ? '<button type="button" class="btn-mini btn-hist-resume" data-action="resume">继续训练</button>' : ""}
            <button type="button" class="btn-mini btn-hist-load" data-action="load">查看结果</button>
            ${ws.has_saved_config ? '<button type="button" class="btn-mini btn-hist-cfg" data-action="config">复用参数</button>' : ""}
            <button type="button" class="btn-mini btn-hist-open" data-action="open">打开目录</button>
            <button type="button" class="btn-mini btn-hist-del" data-action="delete">清理/删除</button>
          </div>
        </div>
      </div>`;
    })
    .join("");

  listEl.querySelectorAll(".history-card").forEach((card) => {
    const wsDir = card.dataset.wsDir;
    const wsObj = byDir.get(wsDir);
    card.addEventListener("click", (e) => {
      const actionBtn = e.target.closest("button[data-action]");
      if (actionBtn) return;
      selectHistoryWorkspace(wsDir);
    });
    card.querySelector('[data-action="resume"]')?.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (wsObj) await resumeWorkspaceRun(wsObj);
    });
    card.querySelector('[data-action="load"]')?.addEventListener("click", (e) => {
      e.stopPropagation();
      selectHistoryWorkspace(wsDir);
    });
    card.querySelector('[data-action="config"]')?.addEventListener("click", async (e) => {
      e.stopPropagation();
      await restoreWorkspaceConfig(wsDir);
    });
    card.querySelector('[data-action="open"]')?.addEventListener("click", async (e) => {
      e.stopPropagation();
      await openWorkspaceFolder(wsDir);
    });
    card.querySelector('[data-action="delete"]')?.addEventListener("click", (e) => {
      e.stopPropagation();
      if (wsObj) openDeleteWorkspaceModal(wsObj);
    });
  });
}

async function resumeWorkspaceRun(ws) {
  if (!ws) return;
  if (state.pollTimer) {
    el("jobStatusLine").textContent = "当前已有任务运行中，请先停止当前任务再续训";
    return;
  }
  if (ws.has_saved_config) {
    await restoreWorkspaceConfig(ws.workspace_dir);
  }
  el("workspaceDir").value = ws.workspace_dir;
  el("fromStage").value = "";
  el("optForce").checked = false;
  el("optMock").checked = Boolean(ws.is_mock);
  state.activeWorkspaceDir = ws.workspace_dir;
  renderHistoryCards();
  await triggerPipelineRun(Boolean(ws.is_mock));
}

function openDeleteWorkspaceModal(ws) {
  state.pendingDeleteWorkspace = ws;
  const totalMb = Number(ws.size_mb || 0).toFixed(2);
  const reclaimMb = Number(ws.reclaimable_mb || 0).toFixed(2);
  el("deleteModalTitle").textContent = `清理空间与删除数据 · ${ws.name}`;
  const resumeWarn = ws.resumable_checkpoint
    ? "注意：该档案包含未完成训练断点，若清理中间抽帧将无法继续断点续训。"
    : "清理中间缓存可大幅释放磁盘，同时保留最终点云、渲染图和仿真导出包。";
  el("deleteModalSummary").textContent = `工作区路径：${ws.workspace_dir}（总占用 ${totalMb} MB）。${resumeWarn}`;
  el("deleteIntermediateNote").textContent = reclaimMb > 0 ? `约可释放 ${reclaimMb} MB` : "已无可清理中间缓存";
  el("deleteAllNote").textContent = `共释放 ${totalMb} MB`;
  el("btnConfirmCleanIntermediate").disabled = Number(ws.reclaimable_mb || 0) <= 0.005;
  el("btnConfirmDeleteAll").disabled = false;
  el("deleteWorkspaceModal").style.display = "flex";
}

function closeDeleteWorkspaceModal() {
  state.pendingDeleteWorkspace = null;
  el("deleteWorkspaceModal").style.display = "none";
}

async function executeWorkspaceDelete(mode) {
  const ws = state.pendingDeleteWorkspace;
  if (!ws) return;
  el("btnConfirmCleanIntermediate").disabled = true;
  el("btnConfirmDeleteAll").disabled = true;
  try {
    const res = await fetchJSON("/api/delete-workspace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_dir: ws.workspace_dir, mode }),
    });
    const wasActive = state.activeWorkspaceDir && normWsPath(state.activeWorkspaceDir) === normWsPath(ws.workspace_dir);
    closeDeleteWorkspaceModal();
    if (mode === "all" && wasActive) {
      state.activeWorkspaceDir = null;
    }
    const list = await loadHistoryWorkspaces(state.activeWorkspaceDir);
    if (mode === "all" && wasActive) {
      state.editorObjects = [];
      state.selectedObjectId = null;
      renderEditorPanel();
      if (list.length > 0) {
        await selectHistoryWorkspace(list[0].workspace_dir);
      } else {
        state.viewport.showDemoShowcase();
        const demoBadge = el("demoSceneBadge");
        if (demoBadge) demoBadge.style.display = "block";
        el("viewportStats").textContent = "全息演示场景 · 420 点 · 8 相机";
        renderExportPanel({ workspace_dir: el("workspaceDir").value, exports_available: {} });
      }
    } else if (mode === "intermediate" && wasActive) {
      await loadScenePreview(ws.workspace_dir).catch(() => {});
    }
    el("jobStatusLine").textContent =
      mode === "all"
        ? `已彻底删除工作区 ${ws.name}，释放 ${Number(res.freed_mb || 0).toFixed(2)} MB 空间`
        : `已清理 ${ws.name} 中间缓存，释放 ${Number(res.freed_mb || 0).toFixed(2)} MB（保留 ${Number(res.remaining_mb || 0).toFixed(2)} MB 导出成果）`;
  } catch (err) {
    closeDeleteWorkspaceModal();
    el("jobStatusLine").textContent = `清理失败: ${err.message}`;
  }
}

async function cleanupUploadsCache() {
  try {
    const keep = encodeURIComponent((el("inputPath")?.value || "").trim());
    const res = await fetchJSON(`/api/cleanup-uploads?grace_sec=0&keep_input_path=${keep}`, { method: "POST" });
    await loadHistoryWorkspaces(state.activeWorkspaceDir);
    el("jobStatusLine").textContent =
      res.removed_files > 0
        ? `已清理 ${res.removed_files} 个临时上传视频缓存，释放 ${Number(res.freed_mb || 0).toFixed(2)} MB`
        : "上传缓存目录已干净，无需清理";
  } catch (err) {
    el("jobStatusLine").textContent = `清理上传缓存失败: ${err.message}`;
  }
}

async function restoreWorkspaceConfig(wsDir) {
  try {
    const res = await fetchJSON(`/api/workspace-config?workspace_dir=${encodeURIComponent(wsDir)}`);
    if (res.has_config && res.config) {
      applyConfigToForm(res.config);
      el("jobStatusLine").textContent = `已载入历史参数快照: ${wsDir}`;
    } else {
      el("jobStatusLine").textContent = `该历史场景未保存参数快照: ${wsDir}`;
    }
  } catch (err) {
    el("jobStatusLine").textContent = `读取历史参数失败: ${err.message}`;
  }
}

async function openWorkspaceFolder(wsDir) {
  try {
    const res = await fetchJSON("/api/open-workspace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_dir: wsDir }),
    });
    el("jobStatusLine").textContent = res.opened
      ? `已在资源管理器中打开: ${res.path}`
      : `工作区路径: ${res.path}`;
  } catch (err) {
    el("jobStatusLine").textContent = `打开目录失败: ${err.message}`;
  }
}

function updateLaunchBtnLabel() {
  if (state.pollTimer) return;
  const wsNorm = normWsPath(el("workspaceDir").value);
  const ws = (state.historyWorkspaces || []).find((w) => normWsPath(w.workspace_dir) === wsNorm);
  const canAutoResume =
    ws &&
    ws.is_incomplete &&
    (ws.resumable_checkpoint || (ws.completed_stages && ws.completed_stages.length > 0)) &&
    !el("optForce").checked &&
    !el("fromStage").value;
  el("runBtnText").textContent = canAutoResume ? "继续断点训练" : "启动构建管线";
}

function syncInputPathFromWorkspace(inputPathStr) {
  const clean = String(inputPathStr || "").trim();
  if (!clean || clean === ".") return;
  const inputEl = el("inputPath");
  if (inputEl) inputEl.value = clean;
  const statusEl = el("uploadStatus");
  if (statusEl) {
    const baseName = clean.split(/[\\/]/).filter(Boolean).pop() || clean;
    statusEl.textContent = `当前绑定输入: ${baseName}`;
  }
}

async function selectHistoryWorkspace(wsDir) {
  if (!wsDir) return;
  state.activeWorkspaceDir = wsDir;
  el("workspaceDir").value = wsDir;
  const wsMeta = (state.historyWorkspaces || []).find((w) => normWsPath(w.workspace_dir) === normWsPath(wsDir));
  if (wsMeta?.input_path) {
    syncInputPathFromWorkspace(wsMeta.input_path);
  }
  const select = el("historyWorkspaceSelect");
  if (select) select.value = wsDir;
  renderHistoryCards();
  updateLaunchBtnLabel();
  try {
    await loadScenePreview(wsDir);
    el("jobStatusLine").textContent = `已加载历史结果: ${wsDir}`;
  } catch {
    el("jobStatusLine").textContent = `已选中未完成工作区: ${wsDir}（可直接点击继续训练）`;
  }
}

async function loadHistoryWorkspaces(selectWorkspaceDir = null) {
  try {
    const data = await fetchJSON("/api/workspaces");
    state.historyWorkspaces = data.workspaces || [];
    state.totalStorageMb = Number(data.total_size_mb || 0);
    state.uploadsStorageMb = Number(data.uploads_size_mb || 0);
    if (selectWorkspaceDir) {
      state.activeWorkspaceDir = selectWorkspaceDir;
    }

    const select = el("historyWorkspaceSelect");
    select.innerHTML = '<option value="">切换已有场景档案...</option>';
    for (const ws of state.historyWorkspaces) {
      const opt = document.createElement("option");
      opt.value = ws.workspace_dir;
      const tags = [ws.has_3dgs ? "3DGS" : "", ws.has_mujoco ? "MJCF" : "", ws.has_unity ? "Unity" : ""]
        .filter(Boolean)
        .join("/");
      const psnr = ws.summary_metrics?.PSNR !== undefined ? ` · ${Number(ws.summary_metrics.PSNR).toFixed(1)}dB` : "";
      opt.textContent = `${ws.name} (${ws.completed_stages.length}/5${tags ? " · " + tags : ""}${psnr})`;
      if (state.activeWorkspaceDir && ws.workspace_dir === state.activeWorkspaceDir) opt.selected = true;
      select.appendChild(opt);
    }
    renderHistoryCards();
    updateLaunchBtnLabel();
    return state.historyWorkspaces;
  } catch (err) {
    console.error("Load workspaces failed:", err);
    return [];
  }
}

function setRecipeActiveCard(recipeKey) {
  state.activeRecipe = recipeKey;
  document.querySelectorAll(".recipe-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.recipe === recipeKey);
  });
  const rec = RECIPES[recipeKey];
  el("recipeSummary").textContent = rec ? rec.summary : "当前为自定义 YAML / 手动调参配置。";
}

function applyRecipe(recipeKey) {
  const rec = RECIPES[recipeKey];
  if (!rec) return;
  // 切换官方智能配方时清空从旧快照带来的隐含超参覆盖（如旧的 mcmc_* / lr_* 覆写），确保按纯净旗舰规格执行。
  if (state.currentConfig?.pipeline?.reconstruct) {
    state.currentConfig.pipeline.reconstruct.params = {};
  }
  const a = rec.apply;
  setRecipeActiveCard(recipeKey);

  if (state.stagesCatalog.ingest?.includes(a.ingestName)) el("ingestName").value = a.ingestName;
  el("ingestFps").value = a.ingestFps;
  el("ingestMaxFrames").value = a.ingestMaxFrames;
  el("ingestKeepRatio").value = a.ingestKeepRatio;
  syncSliderFromNumber("ingestFpsRange", "ingestFps", 0.5, 10);

  if (state.stagesCatalog.pose?.includes(a.poseName)) el("poseName").value = a.poseName;
  el("poseModelOrBackend").value = a.poseModel;

  if (state.stagesCatalog.reconstruct?.includes(a.reconstructName)) el("reconstructName").value = a.reconstructName;
  el("reconstructStrategy").value = a.strategy;
  el("reconstructIters").value = a.iterations;
  syncSliderFromNumber("reconstructItersRange", "reconstructIters", 500, 40000);
  if (a.maxPoints !== undefined) {
    el("reconstructMaxPoints").value = a.maxPoints;
    syncSliderFromNumber("reconstructMaxPointsRange", "reconstructMaxPoints", 30000, 1500000);
  }
  if (a.shDegree !== undefined && el("reconstructShDegree")) {
    el("reconstructShDegree").value = String(a.shDegree);
  }
  el("reconstructMipVar").value = a.mipVar;
  el("reconstructFlatness").value = a.flatness;
  el("reconstructDepthSmooth").value = a.depthSmooth;
  el("optSelectiveAdam").checked = a.selectiveAdam;
  el("optBilateralGrid").checked = a.bilateralGrid;
  el("optPoseOpt").checked = a.poseOpt;

  el("ppClean").checked = a.ppClean;
  el("ppAlign").checked = a.ppAlign;
  el("ppSegment").checked = a.ppSegment;
  el("ppMesh").checked = a.ppMesh;
  el("ppMinOpacity").value = a.ppMinOpacity;
  el("ppScaleFactor").value = a.ppScaleFactor;
  if (a.ppMeshMethod) el("ppMeshMethod").value = a.ppMeshMethod;
  if (a.ppVoxelRes !== undefined) {
    el("ppVoxelRes").value = a.ppVoxelRes;
    syncSliderFromNumber("ppVoxelResRange", "ppVoxelRes", 20, 128);
  }

  el("expPly").checked = a.expPly;
  el("expMujoco").checked = a.expMujoco;
  el("expUnity").checked = a.expUnity;
  if (a.expConvexParts !== undefined) {
    el("expConvexParts").value = a.expConvexParts;
    syncSliderFromNumber("expConvexPartsRange", "expConvexParts", 1, 24);
  }

  syncConditionalFields();
  refreshStageBadges();
  evaluateComboAdvisor();
  updateLaunchBtnLabel();
}

function syncConditionalFields() {
  const poseName = el("poseName").value;
  el("poseCmdRow").style.display = poseName === "feedforward_slot" ? "flex" : "none";
  el("poseTipText").textContent =
    poseName === "feedforward_slot"
      ? "前馈槽位用于对接 VGGT / DUSt3R / MASt3R 外部推理脚本，需填写命令模板。"
      : poseName === "mock_pose"
      ? "Mock 位姿桩仅生成合成测试视角。"
      : "COLMAP 默认使用 OPENCV 径向畸变模型；运动相机大广角可换 OPENCV_FISHEYE。";

  const reconName = el("reconstructName").value;
  el("reconCmdRow").style.display =
    reconName === "twodgs_slot" || reconName === "external_subprocess" ? "flex" : "none";

  const strat = el("reconstructStrategy").value;
  el("strategyTipText").textContent =
    strat === "mcmc"
      ? "3DGS-MCMC：利用 SGLD 与死亡粒子重定位填补纹理弱区，在固定点数上限内画质更稳。"
      : "AbsGS + 透明度重置：按像素绝对梯度分裂并周期压低透明度清理相机前伪影。";

  el("ppPromptsRow").style.display = el("ppSegment").checked ? "flex" : "none";
  el("mujocoExportOpts").style.display = el("expMujoco").checked || el("expPly").checked ? "grid" : "none";
}

function evaluateComboAdvisor() {
  const tips = [];
  const fps = Number(el("ingestFps").value) || 2;
  const maxFrames = parseInt(el("ingestMaxFrames").value, 10) || 150;
  const poseName = el("poseName").value;
  const poseCmd = el("poseCmdTemplate").value.trim();
  const reconName = el("reconstructName").value;
  const reconCmd = el("reconCmdTemplate").value.trim();
  const iters = parseInt(el("reconstructIters").value, 10) || 30000;
  const ppClean = el("ppClean").checked;
  const ppAlign = el("ppAlign").checked;
  const ppMesh = el("ppMesh").checked;
  const ppSegment = el("ppSegment").checked;
  const expPly = el("expPly").checked;
  const expMujoco = el("expMujoco").checked;
  const expUnity = el("expUnity").checked;

  if (!expPly && !expMujoco && !expUnity) {
    tips.push({
      level: "warn",
      text: "尚未勾选任何导出目标，请在 Stage 05 至少开启「纯 3DGS (.ply)」或「MuJoCo / Unity」。",
    });
  }
  if (poseName === "feedforward_slot" && !poseCmd) {
    tips.push({
      level: "warn",
      text: "Stage 02 选择了 feedforward_slot，但命令模板为空；日常视频建议切回 colmap（pycolmap 已就绪）。",
    });
  }
  if ((reconName === "twodgs_slot" || reconName === "external_subprocess") && !reconCmd) {
    tips.push({
      level: "warn",
      text: `Stage 03 选择了 ${reconName}，但尚未填入外部命令行模板，运行会报错；本地训练请使用 gsplat_whitebox。`,
    });
  }
  if ((expMujoco || expUnity) && !ppMesh) {
    tips.push({
      level: "warn",
      text: "开启了 MuJoCo/Unity 导出但未勾选「提取物理碰撞网格」，系统会用占位几何替代真实碰撞体，建议在 Stage 04 勾选。",
    });
  }
  if (ppSegment) {
    tips.push({
      level: "info",
      text: "已启用语义分割槽位；交互式选物移动/删除/单物导出可直接在右侧 3D 视口点击「✂ 物体分割与编辑」操作。",
    });
  }
  if (fps > 5 || maxFrames > 220) {
    tips.push({
      level: "info",
      text: "抽帧率较高或帧数超过 220：COLMAP 特征匹配呈平方级耗时，通常 80–160 帧即可获得优质 3DGS。",
    });
  }
  if (expPly && !expMujoco && !expUnity && ppMesh) {
    tips.push({
      level: "info",
      text: "你当前仅需纯 3DGS，却启用了碰撞网格提取；可点击上方「纯 3DGS 重建」配方省去多余物理网格计算。",
    });
  }
  if ((expMujoco || expUnity) && ppMesh && ppAlign && reconName === "gsplat_whitebox") {
    tips.push({
      level: "good",
      text: `当前配置协调：${iters >= 10000 ? "高质量" : "快测"} 白盒 CUDA 重建 -> +Z 对齐与碰撞提取 -> ${
        expPly ? "纯 3DGS + " : ""
      }${expMujoco ? "MuJoCo " : ""}${expUnity ? "Unity" : ""}导出。`,
    });
  } else if (expPly && !expMujoco && !expUnity && reconName === "gsplat_whitebox") {
    tips.push({
      level: "good",
      text: `当前为轻量纯 3DGS 路线：去噪后直接导出标准 3DGS .ply（${ppClean ? "已开启去漂浮球" : "保留原始高斯"}）。`,
    });
  }

  const box = el("comboAdvisor");
  box.innerHTML = tips
    .slice(0, 3)
    .map((t) => `<div class="advisor-item ${t.level}">${escapeHtml(t.text)}</div>`)
    .join("");
}

function applyConfigToForm(cfg) {
  state.currentConfig = JSON.parse(JSON.stringify(cfg));
  const io = cfg.io || {};
  const pipe = cfg.pipeline || {};

  el("inputPath").value = io.input_path || "./sample_video.mp4";
  el("workspaceDir").value = io.workspace_dir || "./workspaces/default_scene";

  // Stage 1
  const ing = pipe.ingest || { name: "sharpness_ffmpeg", params: {} };
  populateSelect(el("ingestName"), state.stagesCatalog.ingest || [ing.name], ing.name);
  el("ingestFps").value = ing.params?.fps ?? 2.0;
  el("ingestMaxFrames").value = ing.params?.max_frames ?? 150;
  el("ingestKeepRatio").value = ing.params?.sharpness_keep_ratio ?? 0.8;
  syncSliderFromNumber("ingestFpsRange", "ingestFps", 0.5, 10);

  // Stage 2
  const pose = pipe.pose || { name: "colmap", params: {} };
  populateSelect(el("poseName"), state.stagesCatalog.pose || [pose.name], pose.name);
  el("poseModelOrBackend").value = pose.params?.camera_model || pose.params?.backend || "SIMPLE_RADIAL";
  el("poseCmdTemplate").value = pose.params?.command_template || "";

  // Stage 3
  const recon = pipe.reconstruct || { name: "gsplat_whitebox", params: {} };
  populateSelect(el("reconstructName"), state.stagesCatalog.reconstruct || [recon.name], recon.name);
  const rp = recon.params || {};
  el("reconstructStrategy").value = rp.strategy || "mcmc";
  el("reconstructIters").value = rp.iterations ?? 20000;
  syncSliderFromNumber("reconstructItersRange", "reconstructIters", 500, 40000);
  el("reconstructMaxPoints").value = rp.max_points ?? 500000;
  syncSliderFromNumber("reconstructMaxPointsRange", "reconstructMaxPoints", 30000, 1500000);
  if (el("reconstructShDegree")) {
    el("reconstructShDegree").value = String(rp.sh_degree ?? 3);
  }
  el("reconstructMipVar").value = rp.mip_3d_var ?? 0.0;
  el("reconstructFlatness").value = rp.flatness_weight ?? 0.001;
  el("reconstructDepthSmooth").value = rp.depth_smooth_weight ?? 0.001;
  el("optSelectiveAdam").checked = rp.selective_adam !== false;
  el("optBilateralGrid").checked = Boolean(rp.enable_bilateral_grid);
  el("optPoseOpt").checked = Boolean(rp.enable_pose_opt);
  el("reconCmdTemplate").value = rp.command_template || "";

  // Stage 4
  const ppList = pipe.postprocess || [];
  const byName = Object.fromEntries(ppList.map((s) => [s.name, s.params || {}]));
  el("ppClean").checked = "clean_floaters" in byName;
  el("ppMinOpacity").value = byName.clean_floaters?.min_opacity ?? 0.02;
  el("ppAlign").checked = "align_gravity_scale" in byName;
  el("ppScaleFactor").value = byName.align_gravity_scale?.scale_factor ?? 1.0;
  el("ppSegment").checked = "semantic_segment_slot" in byName;
  el("ppPrompts").value = (byName.semantic_segment_slot?.prompts || ["cup", "box"]).join(", ");
  el("ppMesh").checked = "extract_collision_mesh" in byName;
  el("ppMeshMethod").value = byName.extract_collision_mesh?.method || "marching_cubes";
  el("ppVoxelRes").value =
    byName.extract_collision_mesh?.voxel_resolution ?? byName.extract_collision_mesh?.grid_resolution ?? 56;
  syncSliderFromNumber("ppVoxelResRange", "ppVoxelRes", 20, 128);

  // Stage 5
  const expList = pipe.export || [];
  const expMap = Object.fromEntries(expList.map((s) => [s.name, s.params || {}]));
  el("expPly").checked = "ply_3dgs" in expMap || expList.length === 0;
  el("expMujoco").checked = "mujoco" in expMap;
  el("expUnity").checked = "unity" in expMap;
  el("expModelName").value = expMap.mujoco?.model_name || expMap.ply_3dgs?.scene_name || "real2sim_scene";
  el("expConvexParts").value = expMap.mujoco?.convex_parts ?? 12;
  syncSliderFromNumber("expConvexPartsRange", "expConvexParts", 1, 24);

  if (el("expPly").checked && !el("expMujoco").checked && !el("expUnity").checked) {
    setRecipeActiveCard("pure_3dgs");
  } else if (el("ppMeshMethod").value === "heightfield") {
    setRecipeActiveCard("indoor_nav");
  } else if ((parseInt(el("reconstructIters").value, 10) || 30000) <= 5000) {
    setRecipeActiveCard("quick_check");
  } else {
    setRecipeActiveCard("tabletop_sim");
  }

  syncConditionalFields();
  refreshStageBadges();
  evaluateComboAdvisor();
  updateLaunchBtnLabel();
}

function collectConfigFromForm() {
  const base = state.currentConfig ? JSON.parse(JSON.stringify(state.currentConfig)) : { io: {}, pipeline: {} };
  base.io = {
    input_path: el("inputPath").value.trim() || "./sample_video.mp4",
    workspace_dir: el("workspaceDir").value.trim() || "./workspaces/default_scene",
  };

  const ingParams = {
    ...(base.pipeline?.ingest?.params || {}),
    fps: Number(el("ingestFps").value) || 2.0,
    max_frames: parseInt(el("ingestMaxFrames").value, 10) || 150,
    sharpness_keep_ratio: Number(el("ingestKeepRatio").value) || 0.8,
  };

  const poseName = el("poseName").value;
  const poseParams = { ...(base.pipeline?.pose?.params || {}) };
  if (poseName === "feedforward_slot") {
    poseParams.backend = el("poseModelOrBackend").value.trim() || "vggt";
    poseParams.command_template = el("poseCmdTemplate").value.trim();
  } else {
    poseParams.camera_model = el("poseModelOrBackend").value.trim() || "SIMPLE_RADIAL";
  }

  const reconName = el("reconstructName").value;
  const reconParams = {
    ...(base.pipeline?.reconstruct?.params || {}),
    iterations: parseInt(el("reconstructIters").value, 10) || 20000,
    max_points: parseInt(el("reconstructMaxPoints").value, 10) || 500000,
    sh_degree: el("reconstructShDegree") ? parseInt(el("reconstructShDegree").value, 10) : 3,
    strategy: el("reconstructStrategy").value,
    selective_adam: el("optSelectiveAdam").checked,
    enable_bilateral_grid: el("optBilateralGrid").checked,
    enable_pose_opt: el("optPoseOpt").checked,
    background_mode: "random",
    mip_3d_var: Number(el("reconstructMipVar").value),
    flatness_weight: Number(el("reconstructFlatness").value),
    anisotropy_weight: 0.005,
    mcmc_opacity_weight: 0.005,
    mcmc_scale_weight: 0.005,
    mcmc_grow_ratio: 0.06,
    depth_smooth_weight: Number(el("reconstructDepthSmooth").value),
  };
  if (el("reconCmdTemplate").value.trim()) {
    reconParams.command_template = el("reconCmdTemplate").value.trim();
  }

  const postprocess = [];
  if (el("ppClean").checked) {
    postprocess.push({
      name: "clean_floaters",
      params: {
        min_opacity: Number(el("ppMinOpacity").value),
        nb_neighbors: 16,
        std_ratio: 4.5,
        sor_mode: "scale_aware",
        max_scale_ratio: 12.0,
      },
    });
  }
  if (el("ppAlign").checked) {
    postprocess.push({
      name: "align_gravity_scale",
      params: {
        scale_factor: Number(el("ppScaleFactor").value) || 1.0,
        ransac_trials: 200,
        distance_threshold: 0.03,
        seed: 42,
      },
    });
  }
  if (el("ppSegment").checked) {
    const prompts = el("ppPrompts")
      .value.split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    postprocess.push({ name: "semantic_segment_slot", params: { prompts } });
  }
  if (el("ppMesh").checked) {
    const res = parseInt(el("ppVoxelRes").value, 10) || 56;
    postprocess.push({
      name: "extract_collision_mesh",
      params: {
        method: el("ppMeshMethod").value,
        voxel_resolution: res,
        grid_resolution: res,
        iso_level: 0.15,
      },
    });
  }

  const sceneName = el("expModelName").value.trim() || "real2sim_scene";
  const exportList = [];
  if (el("expPly").checked) {
    exportList.push({
      name: "ply_3dgs",
      params: { scene_name: sceneName, include_raw: true },
    });
  }
  if (el("expMujoco").checked) {
    exportList.push({
      name: "mujoco",
      params: {
        model_name: sceneName,
        friction: [1.0, 0.005, 0.0001],
        convex_parts: parseInt(el("expConvexParts").value, 10) || 12,
      },
    });
  }
  if (el("expUnity").checked) {
    exportList.push({
      name: "unity",
      params: { scene_name: `${sceneName}_unity` },
    });
  }

  base.pipeline = {
    ingest: { name: el("ingestName").value, params: ingParams },
    pose: { name: poseName, params: poseParams },
    reconstruct: { name: reconName, params: reconParams },
    postprocess,
    export: exportList,
  };
  return base;
}

function refreshStageBadges() {
  el("badgeIngest").textContent = `${el("ingestName").value} · ${el("ingestFps").value}fps`;
  el("badgePose").textContent = el("poseName").value;
  const shDeg = el("reconstructShDegree") ? el("reconstructShDegree").value : "3";
  el("badgeReconstruct").textContent = `${el("reconstructStrategy").value} · SH${shDeg} · ${el("reconstructIters").value}步 · ${Math.round((Number(el("reconstructMaxPoints").value) || 500000) / 10000)}万点`;
  const ppCount = ["ppClean", "ppAlign", "ppSegment", "ppMesh"].filter((id) => el(id).checked).length;
  el("badgePostprocess").textContent = ppCount > 0 ? `${ppCount} 步启用` : "已跳过";
  const exps = [
    el("expPly").checked ? "3DGS" : "",
    el("expMujoco").checked ? "MuJoCo" : "",
    el("expUnity").checked ? "Unity" : "",
  ].filter(Boolean);
  el("badgeExport").textContent = exps.join(" + ") || "未选择";
}

function renderStepper(stageStates) {
  const statusLabels = {
    pending: "等待中",
    running: "进行中",
    completed: "已完成",
    skipped: "缓存跳过",
    cancelled: "已暂停",
    failed: "失败",
  };
  for (const stage of STAGE_ORDER) {
    const node = document.querySelector(`.step[data-stage="${stage}"]`);
    if (!node) continue;
    const st = stageStates?.[stage] || { status: "pending" };
    node.className = `step ${st.status}`;
    const small = node.querySelector("small");
    const dur = st.duration_sec !== undefined ? ` ${st.duration_sec}s` : "";
    let prog = 0;
    if (st.status === "completed" || st.status === "skipped") prog = 1;
    else if (st.status === "running") prog = Math.max(0.04, Math.min(1, Number(st.progress) || 0.04));
    small.textContent = (statusLabels[st.status] || st.status) + dur;
    const fill = node.querySelector(".step-progress-fill");
    if (fill) fill.style.transform = `scaleX(${prog})`;
  }
}

function renderLossSparkline(lossHistory) {
  const line = el("lossSparklineLine");
  const area = el("lossSparklineArea");
  const txt = el("lossLatestText");
  const emptyTxt = el("lossEmptyText");
  if (!lossHistory || lossHistory.length === 0) {
    line.setAttribute("points", "");
    if (area) area.setAttribute("points", "");
    txt.textContent = "--";
    if (emptyTxt) emptyTxt.style.display = "grid";
    return;
  }
  if (emptyTxt) emptyTxt.style.display = "none";
  const latest = lossHistory[lossHistory.length - 1].loss;
  txt.textContent = Number(latest).toFixed(4);
  if (lossHistory.length === 1) {
    line.setAttribute("points", "0,24 220,24");
    if (area) area.setAttribute("points", "0,48 0,24 220,24 220,48");
    return;
  }
  const vals = lossHistory.map((p) => Number(p.loss));
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const span = Math.max(1e-6, maxV - minV);
  const w = 220;
  const h = 44;
  const pad = 4;
  const pts = lossHistory.map((p, idx) => {
    const x = (idx / (lossHistory.length - 1)) * w;
    const y = h - pad - ((Number(p.loss) - minV) / span) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  line.setAttribute("points", pts.join(" "));
  if (area) {
    area.setAttribute("points", `0,48 ${pts.join(" ")} 220,48`);
  }
}

function renderJobTelemetry(cur) {
  const pct = Math.round((Number(cur.overall_progress) || 0) * 100);
  el("overallPct").textContent = `${pct}%`;
  el("overallBarFill").style.transform = `scaleX(${Math.max(0, Math.min(1, Number(cur.overall_progress) || 0))})`;
  el("telemetryDetail").textContent = cur.status_detail || "执行中...";

  const dot = el("liveDot");
  dot.className = `live-dot ${
    cur.status === "running"
      ? "running"
      : cur.status === "completed"
      ? "completed"
      : cur.status === "cancelled"
      ? "cancelled"
      : cur.status === "failed"
      ? "failed"
      : ""
  }`;

  const t = cur.telemetry || {};
  el("telemetrySpeed").textContent = t.it_per_sec !== undefined ? `${t.it_per_sec} it/s` : "--";
  el("telemetryEta").textContent = t.eta_sec !== undefined ? formatDuration(t.eta_sec) : "--";
  el("telemetrySplats").textContent =
    t.num_gaussians !== undefined ? Number(t.num_gaussians).toLocaleString() : cur.metrics?.num_gaussians?.toLocaleString() || "--";
  const vramEl = el("telemetryVram");
  if (vramEl) {
    if (t.vram_mb !== undefined && Number(t.vram_mb) > 0) {
      const mb = Number(t.vram_mb);
      vramEl.textContent = mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${Math.round(mb)} MB`;
    } else {
      vramEl.textContent = "--";
    }
  }

  renderLossSparkline(cur.loss_history || []);

  const logBox = el("jobLogBox");
  const logs = cur.logs || [];
  state.latestLogs = logs;
  const logSig = logs.length === 0 ? "0:" : `${logs.length}:${logs[logs.length - 1]}`;
  if (logSig !== state.lastLogSig) {
    const nearBottom = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 40 || !state.lastLogSig;
    state.lastLogSig = logSig;
    if (logs.length === 0) {
      logBox.innerHTML = '<div class="log-line muted">等待任务日志输出...</div>';
    } else {
      logBox.innerHTML = logs.map((ln) => `<div class="log-line">${escapeHtml(ln)}</div>`).join("");
      if (nearBottom) {
        logBox.scrollTop = logBox.scrollHeight;
      }
    }
  }
}

async function copyTextToClipboard(text) {
  const value = String(text || "");
  if (!value) return false;
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(value);
      return true;
    } catch {
      // Fallback below for embedded webview contexts.
    }
  }
  const ta = document.createElement("textarea");
  ta.value = value;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  document.body.removeChild(ta);
  return ok;
}

function showErrorBanner(title, shortMsg, fullReport) {
  state.latestErrorReport = fullReport || shortMsg || "";
  el("errorBannerTitle").textContent = title || "任务执行异常";
  el("errorBannerMsg").textContent = shortMsg || "未知错误";
  el("errorTracebackPre").textContent = state.latestErrorReport;
  el("errorDiagnosticBanner").style.display = "flex";
  el("copyErrorBtnText").textContent = "一键复制完整报错信息";
  el("btnCopyErrorReport").classList.remove("copied");
  setTimeout(() => state.viewport && state.viewport.resize(), 40);
}

function hideErrorBanner() {
  state.latestErrorReport = "";
  const banner = el("errorDiagnosticBanner");
  if (banner) banner.style.display = "none";
  setTimeout(() => state.viewport && state.viewport.resize(), 40);
}

function renderMetricsCards(metrics, preview) {
  const grid = el("metricsGrid");
  const merged = { ...(metrics || {}) };
  if (preview?.manifest?.stages) {
    for (const s of STAGE_ORDER) {
      Object.assign(merged, preview.manifest.stages[s]?.metrics || {});
    }
  }
  const entries = Object.entries(merged);
  if (entries.length === 0) {
    grid.innerHTML = '<div class="empty-hint">暂无指标数据。</div>';
    return;
  }
  grid.innerHTML = entries
    .map(([k, v]) => {
      const label = METRIC_LABELS[k] || k;
      const val = Array.isArray(v)
        ? v.map((item) => (typeof item === "number" ? Number(item.toFixed(2)) : item)).join(", ")
        : typeof v === "number"
        ? Number(v.toFixed(4))
        : typeof v === "boolean"
        ? v
          ? "开启"
          : "关闭"
        : v;
      return `
      <div class="metric-tile">
        <span title="${escapeHtml(k)}">${escapeHtml(label)}</span>
        <strong title="${escapeHtml(val)}">${escapeHtml(val)}</strong>
      </div>`;
    })
    .join("");
}

function renderFramesGallery(preview) {
  const grid = el("framesGrid");
  const renders = preview?.renders || [];
  const frames = preview?.frames || [];
  if (renders.length === 0 && frames.length === 0) {
    grid.innerHTML = '<div class="empty-hint">暂无关键帧或渲染结果。完成抽帧或重建后将自动显示。</div>';
    return;
  }
  const wsParam = encodeURIComponent(preview.workspace_dir);
  const renderHtml = renders
    .map((r) => {
      const imgUrl = `/api/workspace-file?workspace_dir=${wsParam}&rel_path=${encodeURIComponent(r.rel_path)}`;
      return `
      <div class="frame-card is-render" data-img-url="${imgUrl}" data-caption="CUDA 神经渲染视角 · ${escapeHtml(r.filename)}">
        <img src="${imgUrl}" alt="${escapeHtml(r.filename)}" loading="lazy" />
        <div>CUDA 渲染 · ${escapeHtml(r.filename)}</div>
      </div>`;
    })
    .join("");
  const framesHtml = frames
    .map((f) => {
      const imgUrl = `/api/workspace-file?workspace_dir=${wsParam}&rel_path=${encodeURIComponent(f.rel_path)}`;
      const sharp = escapeHtml(f.sharpness ?? "-");
      return `
      <div class="frame-card" data-img-url="${imgUrl}" data-caption="原始采集帧 ${escapeHtml(f.filename)} · 清晰度 ${sharp}">
        <img src="${imgUrl}" alt="${escapeHtml(f.filename)}" loading="lazy" />
        <div>${escapeHtml(f.filename)} · ${sharp}</div>
      </div>`;
    })
    .join("");
  grid.innerHTML = renderHtml + framesHtml;

  grid.querySelectorAll(".frame-card").forEach((card) => {
    card.addEventListener("click", () => {
      el("lightboxImg").src = card.dataset.imgUrl;
      el("lightboxCaption").textContent = card.dataset.caption;
      el("frameLightboxModal").style.display = "flex";
    });
  });
}

function setDownloadBtn(buttonEl, url) {
  if (!buttonEl) return;
  if (url) {
    buttonEl.href = url;
    buttonEl.classList.remove("is-disabled");
    buttonEl.removeAttribute("aria-disabled");
  } else {
    buttonEl.href = "#";
    buttonEl.classList.add("is-disabled");
    buttonEl.setAttribute("aria-disabled", "true");
  }
}

function renderExportPanel(preview) {
  const xmlBox = el("sceneXmlBox");
  xmlBox.textContent = preview.scene_xml || "<!-- 未生成 MuJoCo scene.xml（纯 3DGS 模式下不导出仿真场景） -->";
  const wsParam = encodeURIComponent(preview.workspace_dir);
  const avail = preview.exports_available || {};
  const fileUrl = (rel) => (rel ? `/api/workspace-file?workspace_dir=${wsParam}&rel_path=${encodeURIComponent(rel)}` : null);

  setDownloadBtn(el("btnDownloadPurePly"), fileUrl(avail.pure_3dgs_ply));
  setDownloadBtn(el("btnDownloadRawPly"), fileUrl(avail.raw_3dgs_ply));
  setDownloadBtn(el("btnDownloadCameras"), fileUrl(avail.cameras_json));
  setDownloadBtn(el("btnDownloadZip"), avail.bundle_zip_ready ? `/api/workspace-bundle.zip?workspace_dir=${wsParam}` : null);
  setDownloadBtn(el("btnDownloadPly"), fileUrl(avail.mujoco_splats_ply));
  setDownloadBtn(el("btnDownloadObj"), fileUrl(avail.collision_obj));
}

function syncDemoToggleUI() {
  const isDemoActive = state.viewport?.scene === null;
  const hasReal = Boolean(state.latestPreview);
  const btn = el("btnShowDemo");
  const badge = el("demoSceneBadge");
  const camBtn = el("btnCycleCapturedCam");

  if (isDemoActive && hasReal) {
    if (btn) {
      btn.textContent = "← 返回 .ply 实景";
      btn.classList.add("active-return");
    }
    if (badge) {
      badge.style.display = "block";
      badge.classList.add("clickable");
      badge.textContent = "全息演示场 · 点击此处或右上角「← 返回 .ply 实景」切回真实模型";
    }
    if (camBtn) camBtn.style.display = "none";
  } else if (isDemoActive) {
    if (btn) {
      btn.textContent = "演示";
      btn.classList.remove("active-return");
    }
    if (badge) {
      badge.style.display = "block";
      badge.classList.remove("clickable");
      badge.textContent = "全息演示场 · 拖拽可360°旋转 · 运行或选择历史工作区后自动切换真实 3DGS";
    }
    if (camBtn) camBtn.style.display = "none";
  } else {
    if (btn) {
      btn.textContent = "演示";
      btn.classList.remove("active-return");
    }
    if (badge) {
      badge.style.display = "none";
      badge.classList.remove("clickable");
    }
    if (camBtn) {
      const camCount = state.latestPreview?.cameras?.length || 0;
      camBtn.style.display = camCount > 0 ? "inline-block" : "none";
    }
  }
}

function restoreFromDemoToScene() {
  if (!state.latestPreview) return;
  const preview = state.latestPreview;
  state.viewport.setSceneData(preview);
  const splatCount = preview.splats?.total_points || 0;
  const shown = preview.splats?.shown_points || 0;
  const sparseTag = preview.splats?.is_sparse_init ? "SfM 稀疏点: " : preview.neural_render_available ? "CUDA 实景 + 高斯: " : "高斯点: ";
  const meshFaces = preview.mesh?.total_faces || 0;
  const partsCount = preview.convex_parts?.length || 0;
  el("viewportStats").textContent = `${sparseTag}${shown}/${splatCount} · 网格面: ${meshFaces} · 凸包: ${partsCount}`;
  const camBtn = el("btnCycleCapturedCam");
  if (camBtn) camBtn.textContent = "实拍视角";
  syncDemoToggleUI();
}

async function loadScenePreview(workspaceDir, jobMetrics = null, { preserveCamera = false } = {}) {
  const preview = await fetchJSON(`/api/scene-preview?workspace_dir=${encodeURIComponent(workspaceDir)}`);
  state.activeWorkspaceDir = preview.workspace_dir;
  state.latestPreview = preview;
  if (preview?.manifest?.input_path) {
    syncInputPathFromWorkspace(preview.manifest.input_path);
  }
  state.viewport.setSceneData(preview, { preserveCamera });
  const camBtn = el("btnCycleCapturedCam");
  if (camBtn) camBtn.textContent = "实拍视角";
  syncDemoToggleUI();

  const splatCount = preview.splats?.total_points || 0;
  const shown = preview.splats?.shown_points || 0;
  const sparseTag = preview.splats?.is_sparse_init ? "SfM 稀疏点: " : preview.neural_render_available ? "CUDA 实景 + 高斯: " : "高斯点: ";
  const meshFaces = preview.mesh?.total_faces || 0;
  const partsCount = preview.convex_parts?.length || 0;
  el("viewportStats").textContent = `${sparseTag}${shown}/${splatCount} · 网格面: ${meshFaces} · 凸包: ${partsCount}`;

  const manifestStages = {};
  let completedCount = 0;
  let totalSec = 0;
  for (const s of STAGE_ORDER) {
    const rec = preview.manifest?.stages?.[s];
    if (rec) {
      completedCount += 1;
      totalSec += Number(rec.duration_sec) || 0;
    }
    manifestStages[s] = rec ? { status: rec.status || "completed", duration_sec: rec.duration_sec } : { status: "pending" };
  }
  renderStepper(manifestStages);

  if (!state.pollTimer) {
    const frac = completedCount / STAGE_ORDER.length;
    el("overallPct").textContent = `${Math.round(frac * 100)}%`;
    el("overallBarFill").style.transform = `scaleX(${frac})`;
    el("telemetryElapsed").textContent = formatDuration(totalSec);
    el("liveDot").className = `live-dot ${completedCount === STAGE_ORDER.length ? "completed" : ""}`;
    el("telemetryDetail").textContent =
      completedCount === STAGE_ORDER.length
        ? `已载入历史档案 ${preview.workspace_dir} · ${splatCount.toLocaleString()} 个高斯`
        : `已载入工作区 ${preview.workspace_dir} (${completedCount}/5 阶段)`;
    el("telemetrySplats").textContent = splatCount ? Number(splatCount).toLocaleString() : "--";
  }
  renderMetricsCards(jobMetrics, preview);
  renderFramesGallery(preview);
  renderExportPanel(preview);
  state.viewport?.setPromptOverlay({ points: [], box: null, maskBase64: null });
  updateAlignStatusSummary(preview.alignment_transform || null);
  await loadEditorObjects(preview.workspace_dir);
}

function stopJobTimers() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
  if (state.elapsedTimer) {
    clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }
}

function setJobRunningUI(isRunning) {
  el("btnRunPipeline").disabled = isRunning;
  const stopBtn = el("btnCancelJob");
  if (stopBtn) {
    stopBtn.style.display = isRunning ? "inline-flex" : "none";
    stopBtn.disabled = false;
    el("cancelBtnText").textContent = "停止训练";
  }
}

async function cancelActiveJob() {
  const jid = state.activeJobId;
  if (!jid) return;
  const stopBtn = el("btnCancelJob");
  if (stopBtn) stopBtn.disabled = true;
  el("cancelBtnText").textContent = "正在停止...";
  el("jobStatusLine").textContent = "正在发送安全停止指令，等待保存训练断点...";
  try {
    await fetchJSON(`/jobs/${encodeURIComponent(jid)}/cancel`, { method: "POST" });
  } catch (err) {
    if (stopBtn) stopBtn.disabled = false;
    el("cancelBtnText").textContent = "停止训练";
    el("jobStatusLine").textContent = `停止请求失败: ${err.message}（若刚更新代码请重启桌面工作台）`;
  }
}

async function triggerPipelineRun(overrideMock = null) {
  stopJobTimers();
  hideErrorBanner();

  const inlineConfig = collectConfigFromForm();
  const useMock = overrideMock !== null ? overrideMock : el("optMock").checked;
  const payload = {
    inline_config: inlineConfig,
    from_stage: el("fromStage").value || null,
    force: el("optForce").checked,
    mock: useMock,
    run_async: true,
  };

  setJobRunningUI(true);
  el("runBtnText").textContent = "管线执行中...";
  el("jobStatusLine").textContent = "正在初始化工作区与调度任务...";
  state.jobStartedMs = Date.now();
  el("telemetryElapsed").textContent = "00:00";
  state.elapsedTimer = setInterval(() => {
    el("telemetryElapsed").textContent = formatDuration((Date.now() - state.jobStartedMs) / 1000);
  }, 500);

  renderStepper({});
  renderJobTelemetry({
    status: "running",
    overall_progress: 0.01,
    status_detail: "正在启动管线执行...",
    telemetry: {},
    loss_history: [],
    logs: [],
  });

  const completedStagesSeen = new Set();

  try {
    const job = await fetchJSON("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const jid = job.job_id;
    state.activeJobId = jid;
    state.lastLogSig = "";
    el("jobStatusLine").textContent = `任务 ${jid} 运行中...`;

    let pollInFlight = false;
    let terminalHandled = false;
    const pollOnce = async () => {
      if (state.activeJobId !== jid || terminalHandled || pollInFlight) return;
      pollInFlight = true;
      try {
        const cur = await fetchJSON(`/jobs/${jid}`);
        if (state.activeJobId !== jid || terminalHandled) return;
        renderStepper(cur.stage_states);
        renderJobTelemetry(cur);
        el("jobStatusLine").textContent = `${cur.status_detail || "执行中..."} (${Math.round((cur.overall_progress || 0) * 100)}%)`;

        if (cur.status === "running" && cur.workspace_dir) {
          let shouldRefreshMidPreview = false;
          for (const stage of ["ingest", "pose", "postprocess"]) {
            if (cur.stage_states?.[stage]?.status === "completed" && !completedStagesSeen.has(stage)) {
              completedStagesSeen.add(stage);
              shouldRefreshMidPreview = true;
            }
          }
          if (shouldRefreshMidPreview) {
            loadScenePreview(cur.workspace_dir, cur.metrics).catch(() => {});
          }
        }

        if (cur.status === "completed") {
          terminalHandled = true;
          stopJobTimers();
          setJobRunningUI(false);
          el("jobStatusLine").textContent = `任务 ${cur.job_id} 完成 · 已执行 [${cur.executed_stages.join(", ") || "无"}] · 跳过 [${cur.skipped_stages.join(", ") || "无"}]`;
          await loadHistoryWorkspaces(cur.workspace_dir);
          await loadScenePreview(cur.workspace_dir, cur.metrics);
        } else if (cur.status === "cancelled") {
          terminalHandled = true;
          stopJobTimers();
          setJobRunningUI(false);
          el("jobStatusLine").textContent = cur.status_detail || "已安全停止 · 下次点击将从断点继续";
          await loadHistoryWorkspaces(cur.workspace_dir);
          if (cur.workspace_dir) {
            await loadScenePreview(cur.workspace_dir, cur.metrics).catch(() => {});
          }
          updateLaunchBtnLabel();
        } else if (cur.status === "failed") {
          terminalHandled = true;
          stopJobTimers();
          setJobRunningUI(false);
          el("runBtnText").textContent = "启动构建管线";
          el("jobStatusLine").textContent = `任务失败: ${cur.error}`;
          showErrorBanner(
            `执行中断 · 阶段 [${cur.current_stage || "init"}]`,
            cur.error,
            cur.diagnostic_report || cur.error_traceback || cur.error
          );
        }
      } catch (pollErr) {
        console.error(pollErr);
      } finally {
        pollInFlight = false;
        if (state.activeJobId === jid && !terminalHandled) {
          state.pollTimer = setTimeout(pollOnce, 800);
        }
      }
    };
    state.pollTimer = setTimeout(pollOnce, 600);
  } catch (err) {
    stopJobTimers();
    setJobRunningUI(false);
    el("runBtnText").textContent = "启动构建管线";
    el("jobStatusLine").textContent = `提交失败: ${err.message}`;
    const clientReport = [
      "=== Real2Sim-GS 任务提交校验失败 ===",
      `时间: ${new Date().toLocaleString()}`,
      `错误: ${err.message}`,
      `配置快照:\n${JSON.stringify(inlineConfig, null, 2)}`,
    ].join("\n");
    showErrorBanner("任务提交失败", err.message, clientReport);
  }
}

async function handleVideoUpload(file) {
  if (!file) return;
  el("uploadStatus").textContent = `正在上传 ${file.name}...`;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetchJSON("/api/upload", { method: "POST", body: form });
    el("inputPath").value = res.input_path;
    const rawStem = String(res.filename || "scene_upload").replace(/\.[^/.]+$/, "");
    let safeStem = rawStem
      .replace(/[^a-zA-Z0-9_\-\u4e00-\u9fa5]+/g, "_")
      .replace(/^_+|_+$/g, "");
    if (!safeStem) safeStem = "scene_upload";

    const curWsNorm = normWsPath(el("workspaceDir").value);
    const knownWs = new Set((state.historyWorkspaces || []).map((w) => normWsPath(w.workspace_dir)));
    const shouldAutoAssign = !curWsNorm || curWsNorm === "workspaces/default_scene" || knownWs.has(curWsNorm);
    if (shouldAutoAssign) {
      let targetWs = `./workspaces/${safeStem}`;
      if (knownWs.has(normWsPath(targetWs))) {
        const d = new Date();
        const pad2 = (n) => String(n).padStart(2, "0");
        const suffix = `${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
        targetWs = `./workspaces/${safeStem}_${suffix}`;
      }
      el("workspaceDir").value = targetWs;
      el("fromStage").value = "";
      updateLaunchBtnLabel();
      el("uploadStatus").textContent = `已就绪: ${res.filename} → 输出至 ${targetWs}`;
    } else {
      updateLaunchBtnLabel();
      el("uploadStatus").textContent = `已就绪: ${res.filename}`;
    }
  } catch (err) {
    el("uploadStatus").textContent = `上传失败: ${err.message}`;
  }
}

function bindUIEvents() {
  el("btnThemeToggle").addEventListener("click", () => {
    applyTheme(state.theme === "dark" ? "light" : "dark");
  });

  el("btnToggleInspector").addEventListener("click", () => {
    const panel = el("inspectorPanel");
    const collapsed = panel.classList.toggle("collapsed");
    el("btnToggleInspector").textContent = collapsed ? "展开面板" : "最大化 3D 视口";
    setTimeout(() => state.viewport.resize(), 40);
  });

  el("presetSelect").addEventListener("change", async (e) => {
    const data = await fetchJSON(`/api/configs/${encodeURIComponent(e.target.value)}`);
    applyConfigToForm(data.config);
  });

  el("btnRefreshHistory").addEventListener("click", () => {
    loadHistoryWorkspaces(state.activeWorkspaceDir);
  });
  el("btnCleanupUploads").addEventListener("click", cleanupUploadsCache);
  el("workspaceDir").addEventListener("input", updateLaunchBtnLabel);
  el("fromStage").addEventListener("change", updateLaunchBtnLabel);
  el("optForce").addEventListener("change", updateLaunchBtnLabel);

  el("historyWorkspaceSelect").addEventListener("change", async (e) => {
    if (e.target.value) {
      await selectHistoryWorkspace(e.target.value);
    }
  });

  document.querySelectorAll(".recipe-card").forEach((btn) => {
    btn.addEventListener("click", () => applyRecipe(btn.dataset.recipe));
  });

  el("btnExpandStages").addEventListener("click", () => {
    state.allStagesExpanded = !state.allStagesExpanded;
    document.querySelectorAll(".stage-box").forEach((box) => {
      box.open = state.allStagesExpanded;
    });
    el("btnExpandStages").textContent = state.allStagesExpanded ? "全部折叠" : "展开全部";
  });

  bindSliderPair("ingestFpsRange", "ingestFps", 0.5, 10);
  bindSliderPair("reconstructItersRange", "reconstructIters", 500, 40000);
  bindSliderPair("reconstructMaxPointsRange", "reconstructMaxPoints", 30000, 1500000);
  bindSliderPair("ppVoxelResRange", "ppVoxelRes", 20, 128);
  bindSliderPair("expConvexPartsRange", "expConvexParts", 1, 24);

  const watchIds = [
    "ingestName",
    "ingestFps",
    "ingestMaxFrames",
    "ingestKeepRatio",
    "poseName",
    "poseModelOrBackend",
    "poseCmdTemplate",
    "reconstructName",
    "reconstructStrategy",
    "reconstructIters",
    "reconstructMaxPoints",
    "reconstructShDegree",
    "reconCmdTemplate",
    "ppClean",
    "ppAlign",
    "ppSegment",
    "ppMesh",
    "ppMeshMethod",
    "ppScaleFactor",
    "expPly",
    "expMujoco",
    "expUnity",
  ];
  for (const id of watchIds) {
    el(id).addEventListener("change", () => {
      syncConditionalFields();
      refreshStageBadges();
      evaluateComboAdvisor();
    });
  }

  const zone = el("uploadZone");
  const fileIn = el("videoFileInput");
  zone.addEventListener("click", () => fileIn.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileIn.click();
    }
  });
  fileIn.addEventListener("change", () => handleVideoUpload(fileIn.files[0]));
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    handleVideoUpload(e.dataTransfer.files[0]);
  });

  el("btnRunPipeline").addEventListener("click", () => triggerPipelineRun(null));
  el("btnCancelJob").addEventListener("click", cancelActiveJob);
  el("btnQuickMock").addEventListener("click", () => {
    el("optMock").checked = true;
    el("workspaceDir").value = "./workspaces/studio_demo";
    triggerPipelineRun(true);
  });

  el("btnResetCamera").addEventListener("click", () => {
    state.viewport.resetView();
    const camBtn = el("btnCycleCapturedCam");
    if (camBtn && state.latestPreview?.cameras?.length) camBtn.textContent = "实拍 1/" + state.latestPreview.cameras.length;
  });
  el("btnCycleCapturedCam").addEventListener("click", () => {
    const info = state.viewport.cycleCapturedCamera();
    if (info) {
      el("btnCycleCapturedCam").textContent = `实拍 ${info.index}/${info.total}`;
      el("jobStatusLine").textContent = `已切换真实拍摄机位 ${info.index}/${info.total} (${info.name})`;
    }
  });
  el("btnShowDemo").addEventListener("click", () => {
    if (state.viewport.scene === null && state.latestPreview) {
      restoreFromDemoToScene();
      return;
    }
    state.viewport.showDemoShowcase();
    el("viewportStats").textContent = "全息演示场景 · 420 点 · 8 相机";
    syncDemoToggleUI();
  });
  el("demoSceneBadge").addEventListener("click", () => {
    if (state.viewport.scene === null && state.latestPreview) {
      restoreFromDemoToScene();
    }
  });
  document.querySelectorAll(".btn-hud[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => state.viewport.setPresetView(btn.dataset.view));
  });
  el("layerNeural").addEventListener("change", (e) => state.viewport.setLayer("neural", e.target.checked));
  el("layerSplats").addEventListener("change", (e) => state.viewport.setLayer("splats", e.target.checked));
  el("layerMesh").addEventListener("change", (e) => state.viewport.setLayer("mesh", e.target.checked));
  el("layerConvex").addEventListener("change", (e) => state.viewport.setLayer("convex", e.target.checked));
  el("layerSpawn").addEventListener("change", (e) => state.viewport.setLayer("spawn", e.target.checked));

  for (const tabBtn of document.querySelectorAll(".tab")) {
    tabBtn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tabBtn.classList.add("active");
      el(tabBtn.dataset.tab).classList.add("active");
    });
  }

  el("btnCopyXml").addEventListener("click", async () => {
    await copyTextToClipboard(el("sceneXmlBox").textContent);
    el("jobStatusLine").textContent = "已复制 MuJoCo scene.xml 内容到剪贴板";
  });
  el("btnCopyViewerCmd").addEventListener("click", async () => {
    const ws = state.latestPreview?.workspace_dir || el("workspaceDir").value;
    await copyTextToClipboard(`python -m mujoco.viewer --mjcf="${ws}/05_export/mujoco/scene.xml"`);
    el("jobStatusLine").textContent = "已复制 MuJoCo 预览命令到剪贴板";
  });
  el("btnCopyLogs").addEventListener("click", async () => {
    const text = (state.latestLogs || []).join("\n") || el("jobLogBox").textContent;
    const ok = await copyTextToClipboard(text);
    const btn = el("btnCopyLogs");
    btn.textContent = ok ? "✓ 已复制" : "复制失败";
    setTimeout(() => {
      btn.textContent = "复制日志";
    }, 1800);
  });
  el("btnCopyErrorReport").addEventListener("click", async () => {
    const text = state.latestErrorReport || el("errorTracebackPre").textContent || el("errorBannerMsg").textContent;
    const ok = await copyTextToClipboard(text);
    const btn = el("btnCopyErrorReport");
    const lbl = el("copyErrorBtnText");
    if (ok) {
      btn.classList.add("copied");
      lbl.textContent = "✓ 已复制完整报错信息，可直接粘贴";
    } else {
      lbl.textContent = "请手动全选下方堆栈复制";
    }
    setTimeout(() => {
      btn.classList.remove("copied");
      lbl.textContent = "一键复制完整报错信息";
    }, 2400);
  });
  el("btnDismissError").addEventListener("click", hideErrorBanner);

  el("btnDoctorModal").addEventListener("click", () => {
    el("doctorModal").style.display = "flex";
  });
  el("btnCloseDoctor").addEventListener("click", () => {
    el("doctorModal").style.display = "none";
  });
  el("doctorModal").addEventListener("click", (e) => {
    if (e.target === el("doctorModal")) el("doctorModal").style.display = "none";
  });

  el("btnCloseLightbox").addEventListener("click", () => {
    el("frameLightboxModal").style.display = "none";
  });
  el("frameLightboxModal").addEventListener("click", (e) => {
    if (e.target === el("frameLightboxModal")) el("frameLightboxModal").style.display = "none";
  });

  el("btnConfirmCleanIntermediate").addEventListener("click", () => executeWorkspaceDelete("intermediate"));
  el("btnConfirmDeleteAll").addEventListener("click", () => executeWorkspaceDelete("all"));
  el("btnCloseDeleteModal").addEventListener("click", closeDeleteWorkspaceModal);
  el("deleteWorkspaceModal").addEventListener("click", (e) => {
    if (e.target === el("deleteWorkspaceModal")) closeDeleteWorkspaceModal();
  });

  bindEditorEvents();
}

/* =====================================================================
   SAM 3 物体分割与交互式场景图编辑控制器
   ===================================================================== */
const EDITOR_MODE_HINTS = {
  orbit: "浏览模式：左键拖拽 360° 旋转 · 右键/Shift+左键平移 · 滚轮缩放 · 双击居中重置",
  point: "SAM 3 点选：左键点击物体加前景点(+) · Shift+左键或右键短点击加排除点(-) · 拖拽可旋转视角",
  box: "SAM 3 框选：左键在画面拖出矩形框选择目标 · 右键/Alt+拖拽旋转平移视角",
  pick: "拾取模式：点击 3D 画面中的物体直接选中（开启自动聚焦将自动对准镜头）",
  move: "移动模式：拖拽红/绿/蓝坐标轴或橙色 XY 贴桌平面手柄移动物体 · 空白处拖拽可旋转视角",
  rotate: "旋转模式：拖拽蓝色圆环手柄绕 Z 轴旋转物体（0-3 阶 SH 高光同步旋转）· 空白处可旋转视角",
  scale: "缩放模式：拖拽坐标轴末端手柄等比缩放物体 · 空白处拖拽可旋转视角",
};

function selectedEditorObject() {
  if (state.selectedObjectId === null || state.selectedObjectId === undefined) return null;
  return state.editorObjects.find((o) => Number(o.id) === Number(state.selectedObjectId)) || null;
}

function setEditorFeedback(msg, isError = false) {
  const fb = el("edFeedbackText");
  if (fb) {
    fb.textContent = msg || "";
    fb.classList.toggle("error", Boolean(isError));
  }
  if (msg) {
    el("jobStatusLine").textContent = msg;
  }
}

function syncEditorViewport() {
  if (!state.viewport) return;
  state.viewport.setEditorState({
    open: state.editorOpen,
    mode: state.editorMode,
    objects: state.editorObjects,
    selectedId: state.selectedObjectId,
    highlight: state.editorHighlight,
    dimBg: state.editorDimBg,
  });
}

function updateCanvasHint() {
  const hint = el("canvasHintText");
  if (!hint) return;
  hint.textContent = state.editorOpen
    ? EDITOR_MODE_HINTS[state.editorMode] || EDITOR_MODE_HINTS.orbit
    : EDITOR_MODE_HINTS.orbit;
}

function setEditorOpen(open) {
  state.editorOpen = Boolean(open);
  if (state.editorOpen && state.alignOpen) {
    setAlignPopoverOpen(false);
  }
  const tb = el("editorToolbar");
  const drawer = el("editorDrawer");
  const toggleBtn = el("btnToggleEditor");
  if (tb) tb.style.display = state.editorOpen ? "flex" : "none";
  if (drawer) drawer.style.display = state.editorOpen ? "flex" : "none";
  if (toggleBtn) toggleBtn.classList.toggle("active", state.editorOpen);
  updateCanvasHint();
  syncEditorViewport();
  if (state.editorOpen) {
    loadSam3Status();
    if (state.activeWorkspaceDir) {
      loadEditorObjects(state.activeWorkspaceDir);
    }
  }
}

function setEditorMode(mode) {
  state.editorMode = String(mode || "orbit");
  document.querySelectorAll(".ed-tool-btn[data-ed-mode]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.edMode === state.editorMode);
  });
  updateCanvasHint();
  syncEditorViewport();
}

async function loadSam3Status() {
  try {
    const customPath = (el("edSam3ModelPath")?.value || "").trim();
    const q = customPath ? `?model_path=${encodeURIComponent(customPath)}` : "";
    const st = await fetchJSON(`/api/editor/status${q}`);
    const badge = el("edSam3Badge");
    const hint = el("edSam3StatusHint");
    if (badge) {
      badge.className = `ed-status-pill ${st.ready ? "ok" : "warn"}`;
      const loadedTag = st.loaded_models?.length ? ` · 已驻留` : "";
      badge.textContent = st.ready ? `SAM3 就绪${loadedTag}` : "需配置";
    }
    if (hint) {
      hint.textContent = `${st.hint || ""}（当前源: ${st.model_id || "facebook/sam3"} | 设备: ${st.device || "cuda"}）`;
    }
  } catch (err) {
    const badge = el("edSam3Badge");
    if (badge) {
      badge.className = "ed-status-pill warn";
      badge.textContent = "检查失败";
    }
  }
}

function applyEditorServerState(data, preferredSelectId = undefined) {
  state.editorObjects = Array.isArray(data?.objects) ? data.objects : [];
  state.editorCanUndo = Boolean(data?.can_undo);
  state.editorCanRedo = Boolean(data?.can_redo);
  state.editorBakedDirty = Boolean(data?.baked_dirty);

  if (preferredSelectId !== undefined) {
    state.selectedObjectId = preferredSelectId;
  }
  const stillExists = state.editorObjects.some((o) => Number(o.id) === Number(state.selectedObjectId));
  if (!stillExists) {
    const firstActive = state.editorObjects.find((o) => !o.deleted) || state.editorObjects[0] || null;
    state.selectedObjectId = firstActive ? Number(firstActive.id) : null;
  }
  renderEditorPanel();
}

async function loadEditorObjects(workspaceDir) {
  if (!workspaceDir) {
    applyEditorServerState({ objects: [] }, null);
    return;
  }
  try {
    const data = await fetchJSON(`/api/editor/objects?workspace_dir=${encodeURIComponent(workspaceDir)}`);
    applyEditorServerState(data);
  } catch {
    applyEditorServerState({ objects: [] }, null);
  }
}

function syncTransformInputsFromObj(obj) {
  if (!obj) return;
  const t = obj.translation || [0, 0, 0];
  const r = obj.rotation_euler_deg || [0, 0, 0];
  if (el("edTfTx")) el("edTfTx").value = Number(t[0] || 0).toFixed(3);
  if (el("edTfTy")) el("edTfTy").value = Number(t[1] || 0).toFixed(3);
  if (el("edTfTz")) el("edTfTz").value = Number(t[2] || 0).toFixed(3);
  if (el("edTfRx")) el("edTfRx").value = Number(r[0] || 0).toFixed(1);
  if (el("edTfRy")) el("edTfRy").value = Number(r[1] || 0).toFixed(1);
  if (el("edTfRz")) el("edTfRz").value = Number(r[2] || 0).toFixed(1);
  if (el("edTfScale")) el("edTfScale").value = Number(obj.scale ?? 1.0).toFixed(3);
}

function renderEditorPanel() {
  const objs = state.editorObjects || [];
  const activeCount = objs.filter((o) => !o.deleted).length;
  const countBadge = el("editorCountBadge");
  if (countBadge) countBadge.textContent = String(activeCount);

  const dirtyBadge = el("edUnbakedBadge");
  if (dirtyBadge) dirtyBadge.style.display = state.editorBakedDirty ? "inline-block" : "none";

  const sel = selectedEditorObject();
  const hasSel = Boolean(sel);
  if (el("btnEdFocus")) el("btnEdFocus").disabled = !hasSel;
  if (el("btnEdExportObjPly")) el("btnEdExportObjPly").disabled = !hasSel;
  if (el("btnEdDuplicate")) el("btnEdDuplicate").disabled = !hasSel;
  if (el("btnEdToggleVis")) {
    el("btnEdToggleVis").disabled = !hasSel || Boolean(sel?.deleted);
    el("btnEdToggleVis").textContent = sel && sel.visible === false ? "显示" : "隐藏";
  }
  if (el("btnEdDeleteRestore")) {
    el("btnEdDeleteRestore").disabled = !hasSel;
    el("btnEdDeleteRestore").textContent = sel?.deleted ? "恢复" : "删除";
  }
  if (el("btnEdUndo")) el("btnEdUndo").disabled = !state.editorCanUndo;
  if (el("btnEdRedo")) el("btnEdRedo").disabled = !state.editorCanRedo;
  if (el("btnEdBake")) el("btnEdBake").disabled = objs.length === 0;

  const listEl = el("edObjectsList");
  if (listEl) {
    if (objs.length === 0) {
      listEl.innerHTML = '<div class="empty-hint">暂无分割物体。使用左上方工具栏选点/框选或输入提示词提取物体。</div>';
    } else {
      listEl.innerHTML = objs
        .map((o) => {
          const isAct = Number(o.id) === Number(state.selectedObjectId);
          const c = o.color_rgb || [0.14, 0.72, 0.96];
          const rgbStr = `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
          const statusTag = o.deleted ? "已删除" : o.visible === false ? "已隐藏" : `${Number(o.point_count || 0).toLocaleString()}点`;
          return `
          <div class="ed-obj-item ${isAct ? "active" : ""} ${o.deleted ? "is-deleted" : ""}" data-oid="${Number(o.id)}">
            <div class="ed-obj-top">
              <div class="ed-obj-title">
                <span class="ed-swatch" style="background:${rgbStr};"></span>
                <span title="${escapeHtml(o.name)}">${escapeHtml(o.name)}</span>
              </div>
              <span class="ed-obj-meta">${escapeHtml(statusTag)}</span>
            </div>
            <div class="ed-obj-actions">
              <button type="button" class="btn-mini" data-ed-row-act="focus" title="聚焦镜头到该物体">聚焦</button>
              <button type="button" class="btn-mini" data-ed-row-act="export" title="单独导出该物体为 .ply">导出.ply</button>
              <button type="button" class="btn-mini" data-ed-row-act="rename" title="重命名">改名</button>
              <button type="button" class="btn-mini" data-ed-row-act="vis" ${o.deleted ? "disabled" : ""}>${o.visible === false ? "显示" : "隐藏"}</button>
              <button type="button" class="btn-mini btn-hist-del" data-ed-row-act="del">${o.deleted ? "恢复" : "删除"}</button>
            </div>
          </div>`;
        })
        .join("");

      listEl.querySelectorAll(".ed-obj-item").forEach((item) => {
        const oid = Number(item.dataset.oid);
        item.addEventListener("click", (e) => {
          if (e.target.closest("button[data-ed-row-act]")) return;
          selectEditorObject(oid, { focusIfAuto: true });
        });
        item.querySelector('[data-ed-row-act="focus"]')?.addEventListener("click", (e) => {
          e.stopPropagation();
          selectEditorObject(oid, { focusIfAuto: false });
          focusSelectedObject();
        });
        item.querySelector('[data-ed-row-act="export"]')?.addEventListener("click", (e) => {
          e.stopPropagation();
          selectEditorObject(oid, { focusIfAuto: false });
          exportSelectedObjectPly(oid);
        });
        item.querySelector('[data-ed-row-act="rename"]')?.addEventListener("click", async (e) => {
          e.stopPropagation();
          const targetObj = state.editorObjects.find((x) => Number(x.id) === oid);
          const nextName = window.prompt("输入物体新名称：", targetObj?.name || "");
          if (nextName && nextName.trim()) {
            await updateEditorObject(oid, { name: nextName.trim(), record_undo: true });
          }
        });
        item.querySelector('[data-ed-row-act="vis"]')?.addEventListener("click", async (e) => {
          e.stopPropagation();
          const targetObj = state.editorObjects.find((x) => Number(x.id) === oid);
          if (targetObj) {
            await updateEditorObject(oid, { visible: targetObj.visible === false, record_undo: true });
          }
        });
        item.querySelector('[data-ed-row-act="del"]')?.addEventListener("click", async (e) => {
          e.stopPropagation();
          const targetObj = state.editorObjects.find((x) => Number(x.id) === oid);
          if (targetObj) {
            await performEditorAction(targetObj.deleted ? "restore" : "delete", oid);
          }
        });
      });
    }
  }

  const refineBox = el("edRefineBox");
  if (refineBox) {
    if (sel && sel.has_score_cache && !sel.is_clone && !sel.deleted) {
      refineBox.style.display = "flex";
      const th = Number(sel.threshold ?? 0.5);
      const zc = Number(sel.z_clip ?? 0.0);
      el("edRefineThreshold").value = String(th);
      el("edRefineThreshVal").textContent = th.toFixed(2);
      el("edRefineZClip").value = String(zc);
      el("edRefineZClipVal").textContent = `${zc.toFixed(3)}m`;
    } else {
      refineBox.style.display = "none";
    }
  }

  const tfBox = el("edTransformBox");
  if (tfBox) {
    if (sel) {
      tfBox.style.display = "flex";
      el("edTransformTitle").textContent = `4. 精确空间变换 · ${sel.name}`;
      if (!state.viewport?._gizmoDrag) {
        syncTransformInputsFromObj(sel);
      }
    } else {
      tfBox.style.display = "none";
    }
  }

  syncEditorViewport();
}

function selectEditorObject(objectId, { focusIfAuto = true } = {}) {
  state.selectedObjectId = objectId;
  state._refineUndoArmed = true;
  state._refineDirty = false;
  renderEditorPanel();
  if (focusIfAuto && state.editorAutoFocus) {
    focusSelectedObject();
  }
}

function focusSelectedObject() {
  const sel = selectedEditorObject();
  if (!sel || !state.viewport) return;
  const bmin = sel.transformed_bbox_min || sel.bbox_min;
  const bmax = sel.transformed_bbox_max || sel.bbox_max;
  state.viewport.focusOnBox(bmin, bmax);
  setEditorFeedback(`镜头已聚焦至「${sel.name}」`);
}

function exportSelectedObjectPly(explicitId = null) {
  const targetId = explicitId !== null ? explicitId : state.selectedObjectId;
  const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
  if (!wsDir || targetId === null || targetId === undefined) return;
  const obj = state.editorObjects.find((o) => Number(o.id) === Number(targetId));
  const url = `/api/editor/export-object-ply?workspace_dir=${encodeURIComponent(wsDir)}&object_id=${encodeURIComponent(targetId)}`;
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setEditorFeedback(`正在下载独立物体点云「${obj?.name || targetId}」(.ply)...`);
}

function clearPromptsUI() {
  state.viewport?.setPromptOverlay({ points: [], box: null, maskBase64: null });
  if (el("edPromptInfo")) {
    el("edPromptInfo").textContent = "提示：切换到「点选+ / 框选」或在上方输入物体名";
  }
  setEditorFeedback("");
}

function _buildPromptPayloadFromViewport() {
  const snap = state.viewport.getCameraSnapshot();
  const pts = (state.viewport.promptPoints || []).map((p) => [p.nx * snap.width, p.ny * snap.height]);
  const labels = (state.viewport.promptPoints || []).map((p) => Number(p.label));
  const b = state.viewport.promptBox;
  const box = b ? [b.nx0 * snap.width, b.ny0 * snap.height, b.nx1 * snap.width, b.ny1 * snap.height] : null;
  const text = (el("edTextPrompt")?.value || "").trim() || null;
  return { snap, pts, labels, box, text };
}

async function runPreview2D() {
  const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
  if (!wsDir || !state.viewport?.scene || state.viewport.scene.splats?.is_sparse_init) {
    setEditorFeedback("请先加载已完成 3DGS 训练的真实场景工作区。", true);
    return;
  }
  const { snap, pts, labels, box, text } = _buildPromptPayloadFromViewport();
  if (pts.length === 0 && !box && !text) {
    state.viewport.setPromptOverlay({ maskBase64: null });
    return;
  }
  setEditorFeedback("SAM 3 正在预测当前视角 2D 掩码...");
  try {
    const res = await fetchJSON("/api/editor/preview-2d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        workspace_dir: wsDir,
        ...snap,
        points: pts.length ? pts : null,
        labels: labels.length ? labels : null,
        box,
        text,
      }),
    });
    state.viewport.setPromptOverlay({ maskBase64: res.mask_png_b64 });
    const areaPct = (Number(res.area_ratio || 0) * 100).toFixed(1);
    setEditorFeedback(
      `2D 预览就绪 (${res.backend} · 置信 ${Number(res.score || 0).toFixed(2)} · 占比 ${areaPct}% · ${res.elapsed_ms}ms)，确认后点「执行 3D 分割」`
    );
    loadSam3Status();
  } catch (err) {
    setEditorFeedback(`SAM 3 2D 预览失败: ${err.message}`, true);
  }
}

async function runSegment3D() {
  const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
  if (!wsDir || !state.viewport?.scene || state.viewport.scene.splats?.is_sparse_init) {
    setEditorFeedback("请先加载已完成 3DGS 训练的真实场景工作区。", true);
    return;
  }
  const { snap, pts, labels, box, text } = _buildPromptPayloadFromViewport();
  if (pts.length === 0 && !box && !text) {
    setEditorFeedback("请先在画面中点选/框选目标，或输入英文概念词后再执行 3D 分割。", true);
    return;
  }

  const btn = el("btnEdRunSeg3D");
  if (btn) btn.disabled = true;
  state.editorBusy = true;
  setEditorFeedback("正在运行 SAM 3 多视角追踪 + B³-Seg + FlashSplat + GaussianCut + SAGD 3D 分割...");

  try {
    const res = await fetchJSON("/api/editor/segment-3d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        workspace_dir: wsDir,
        ...snap,
        points: pts.length ? pts : null,
        labels: labels.length ? labels : null,
        box,
        text,
        name: (el("edNewObjName")?.value || "").trim() || null,
        num_views: parseInt(el("edNumViews")?.value || "12", 10),
        use_sagd: Boolean(el("edUseSagd")?.checked ?? true),
        use_gaussiancut: Boolean(el("edUseGraphCut")?.checked ?? true),
      }),
    });
    if (el("edNewObjName")) el("edNewObjName").value = "";
    state.viewport.setPromptOverlay({ points: [], box: null, maskBase64: null });
    applyEditorServerState(res, res.object?.id);
    setEditorMode("move");
    state.viewport.requestImmediateNeuralRefresh(true);
    const st = res.stats || {};
    setEditorFeedback(
      `已提取「${res.object?.name}」（${Number(res.object?.point_count || 0).toLocaleString()} 点 · SAGD 分裂 ${st.decomposed_gaussians || 0} 球），可直接拖拽移动/旋转或导出`
    );
    loadSam3Status();
  } catch (err) {
    setEditorFeedback(`3D 分割失败: ${err.message}`, true);
  } finally {
    state.editorBusy = false;
    if (btn) btn.disabled = false;
  }
}

async function updateEditorObject(objectId, patch, { refreshNeural = true } = {}) {
  const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
  if (!wsDir || objectId === null || objectId === undefined) return;
  try {
    const res = await fetchJSON("/api/editor/update-object", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_dir: wsDir, object_id: Number(objectId), ...patch }),
    });
    applyEditorServerState(res, Number(objectId));
    if (refreshNeural) {
      state.viewport.requestImmediateNeuralRefresh(patch.record_undo !== false);
    }
  } catch (err) {
    setEditorFeedback(`更新物体失败: ${err.message}`, true);
  }
}

function flushGizmoSync(payload, recordUndo, isFinal = false) {
  const prev = state._tfSyncPending;
  state._tfSyncPending = {
    ...payload,
    record_undo: Boolean(prev?.record_undo || recordUndo),
    is_final: Boolean(prev?.is_final || isFinal),
  };
  if (state._tfSyncInFlight) return;

  const pump = async () => {
    if (!state._tfSyncPending) {
      state._tfSyncInFlight = false;
      return;
    }
    const next = state._tfSyncPending;
    state._tfSyncPending = null;
    state._tfSyncInFlight = true;
    const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
    if (!wsDir) {
      state._tfSyncInFlight = false;
      return;
    }
    try {
      const res = await fetchJSON("/api/editor/update-object", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspace_dir: wsDir,
          object_id: next.objectId,
          translation: next.translation,
          rotation_euler_deg: next.rotation_euler_deg,
          scale: next.scale,
          record_undo: Boolean(next.record_undo),
        }),
      });
      if (next.is_final) {
        applyEditorServerState(res, next.objectId);
      } else {
        state.editorBakedDirty = Boolean(res.baked_dirty);
        state.editorCanUndo = Boolean(res.can_undo);
        state.editorCanRedo = Boolean(res.can_redo);
        if (el("btnEdUndo")) el("btnEdUndo").disabled = !state.editorCanUndo;
        if (el("btnEdRedo")) el("btnEdRedo").disabled = !state.editorCanRedo;
      }
      state.viewport.requestImmediateNeuralRefresh(Boolean(next.is_final));
    } catch (err) {
      setEditorFeedback(`同步变换失败: ${err.message}`, true);
    } finally {
      if (state._tfSyncPending) {
        pump();
      } else {
        state._tfSyncInFlight = false;
      }
    }
  };
  pump();
}

function scheduleRefineObject(isCommit) {
  const sel = selectedEditorObject();
  const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
  if (!sel || !wsDir) return;
  const threshold = Number(el("edRefineThreshold").value);
  const zClip = Number(el("edRefineZClip").value);
  el("edRefineThreshVal").textContent = threshold.toFixed(2);
  el("edRefineZClipVal").textContent = `${zClip.toFixed(3)}m`;

  if (state._refineTimer) clearTimeout(state._refineTimer);
  if (!isCommit) {
    state._refineDirty = true;
  }
  const runRefine = async (commitNow) => {
    if (commitNow && !state._refineDirty && !state._refineUndoArmed) {
      state._refineUndoArmed = true;
      return;
    }
    const recUndo = Boolean(state._refineUndoArmed);
    state._refineUndoArmed = false;
    if (commitNow) state._refineDirty = false;
    try {
      const res = await fetchJSON("/api/editor/refine-object", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspace_dir: wsDir,
          object_id: Number(sel.id),
          threshold,
          z_clip: zClip,
          record_undo: recUndo,
        }),
      });
      applyEditorServerState(res, Number(sel.id));
      state.viewport.requestImmediateNeuralRefresh(commitNow);
    } catch (err) {
      if (recUndo) state._refineUndoArmed = true;
      setEditorFeedback(err.message, true);
    } finally {
      if (commitNow) {
        state._refineUndoArmed = true;
        state._refineDirty = false;
      }
    }
  };
  if (isCommit) {
    runRefine(true);
  } else {
    state._refineTimer = setTimeout(() => runRefine(false), 85);
  }
}

async function performEditorAction(action, objectId = null) {
  const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
  if (!wsDir) return;
  const targetId = objectId !== null ? objectId : state.selectedObjectId;
  try {
    const res = await fetchJSON("/api/editor/object-action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_dir: wsDir, action, object_id: targetId }),
    });
    const prefId = res.object?.id !== undefined ? Number(res.object.id) : state.selectedObjectId;
    applyEditorServerState(res, prefId);
    state.viewport.requestImmediateNeuralRefresh(true);
    if (action === "duplicate" && res.object) {
      setEditorMode("move");
      setEditorFeedback(`已复制物体「${res.object.name}」`);
    } else if (action === "undo" || action === "redo") {
      setEditorFeedback(action === "undo" ? "已撤销上一步编辑" : "已重做编辑");
    } else if (action === "delete") {
      setEditorFeedback("已删除选中物体（仍可通过「恢复」或「撤销」找回，烘焙后将从导出移除）");
    } else if (action === "restore") {
      setEditorFeedback("已恢复选中物体");
    }
  } catch (err) {
    setEditorFeedback(`操作失败: ${err.message}`, true);
  }
}

async function bakeSceneEdits() {
  const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
  if (!wsDir) return;
  const btn = el("btnEdBake");
  if (btn) btn.disabled = true;
  setEditorFeedback("正在烘焙场景图修改至 edited_splats.ply 并同步更新标准 .ply 导出...");
  try {
    const res = await fetchJSON("/api/editor/bake", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_dir: wsDir, reexport_ply: true }),
    });
    applyEditorServerState(res, state.selectedObjectId);
    await loadScenePreview(wsDir, null, { preserveCamera: true });
    await loadHistoryWorkspaces(wsDir);
    setEditorFeedback(
      `烘焙成功！共保留 ${Number(res.total_points || 0).toLocaleString()} 个高斯（保留 ${res.kept_objects} / 移除 ${res.deleted_objects} 物体），导出下载已同步更新`
    );
  } catch (err) {
    setEditorFeedback(`烘焙失败: ${err.message}`, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function onViewportEditorEvent(type, detail) {
  if (type === "prompts_invalidated") {
    if (el("edPromptInfo")) {
      el("edPromptInfo").textContent = "视角已变动，已清除旧 2D 选区（可重新点击）";
    }
    return;
  }
  if (type === "point_prompt") {
    const pts = detail?.points || [];
    const pos = pts.filter((p) => Number(p.label) === 1).length;
    const neg = pts.length - pos;
    if (el("edPromptInfo")) {
      el("edPromptInfo").textContent = `已标 ${pos} 个前景点 / ${neg} 个排除点`;
    }
    await runPreview2D();
    return;
  }
  if (type === "box_prompt") {
    if (el("edPromptInfo")) {
      el("edPromptInfo").textContent = "已框选目标区域";
    }
    await runPreview2D();
    return;
  }
  if (type === "pick_click") {
    const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
    if (!wsDir) return;
    const snap = state.viewport.getCameraSnapshot();
    try {
      const res = await fetchJSON("/api/editor/pick-object", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspace_dir: wsDir,
          ...snap,
          u: detail.sx,
          v: detail.sy,
        }),
      });
      if (res.object_id !== null && res.object_id !== undefined) {
        selectEditorObject(Number(res.object_id), { focusIfAuto: true });
        const picked = selectedEditorObject();
        setEditorFeedback(`已拾取「${picked?.name || res.object_id}」`);
      } else {
        setEditorFeedback("点击位置未命中已分割物体（可先切换「点选+」分割该物体）");
      }
    } catch (err) {
      setEditorFeedback(`拾取失败: ${err.message}`, true);
    }
    return;
  }
  if (type === "gizmo_drag") {
    const obj = state.editorObjects.find((o) => Number(o.id) === Number(detail.objectId));
    if (obj) syncTransformInputsFromObj(obj);
    flushGizmoSync(detail, Boolean(detail.recordUndo), false);
    return;
  }
  if (type === "gizmo_end") {
    flushGizmoSync(detail, Boolean(detail.recordUndo), true);
  }
}

function formatAlignSummary(tr) {
  if (!tr || typeof tr !== "object") {
    return "未检测到 alignment_transform.json（可点击下方一键自动找平）";
  }
  const nz = Number(tr.aligned_normal?.[2] ?? 1.0).toFixed(4);
  const snap = Number(tr.manhattan_snapped_deg || 0).toFixed(1);
  const method = String(tr.alignment_method || "ransac");
  if (method === "surfel_irls_manhattan") {
    return `SOTA 曲面找平 + 曼哈顿摆正 (${snap}°) · nz=${nz}`;
  }
  if (method === "surfel_irls") {
    return `SOTA 曲面找平（无显著正交墙）· nz=${nz}`;
  }
  if (method === "manual_set_front") {
    return `已手动设定正前方 (-X / Yaw=0°) · nz=${nz}`;
  }
  if (method === "manual_manual") {
    return `已应用手动增量校准 · nz=${nz}`;
  }
  return `常规 RANSAC 对齐 (${method}) · nz=${nz}`;
}

function updateAlignStatusSummary(tr) {
  const sum = el("alignStatusSummary");
  if (sum) sum.textContent = formatAlignSummary(tr);
}

function setAlignFeedback(msg, isError = false) {
  const fb = el("alignFeedbackText");
  if (fb) {
    fb.textContent = msg || "";
    fb.classList.toggle("error", Boolean(isError));
  }
  if (msg) el("jobStatusLine").textContent = msg;
}

function resetAlignManualInputs() {
  for (const id of ["alignRollDeg", "alignPitchDeg", "alignYawDeg", "alignZOffset"]) {
    if (el(id)) el(id).value = "0";
  }
}

function setAlignPopoverOpen(open) {
  state.alignOpen = Boolean(open);
  if (state.alignOpen && state.editorOpen) {
    setEditorOpen(false);
  }
  const pop = el("alignPopover");
  const btn = el("btnToggleAlignPopover");
  if (pop) pop.style.display = state.alignOpen ? "flex" : "none";
  if (btn) btn.classList.toggle("active", state.alignOpen);
  if (state.alignOpen) {
    updateAlignStatusSummary(state.latestPreview?.alignment_transform || null);
  }
}

async function runSceneAlign(payload, { afterMode = "auto" } = {}) {
  const wsDir = state.activeWorkspaceDir || state.latestPreview?.workspace_dir;
  if (!wsDir || state.viewport?.scene === null) {
    setAlignFeedback("请先加载一个真实重建工作区（先退出全息演示模式）", true);
    return;
  }
  if (state.alignBusy) return;
  state.alignBusy = true;
  const btns = ["btnAlignAuto", "btnAlignSetFront", "btnAlignApplyManual"].map((id) => el(id)).filter(Boolean);
  btns.forEach((b) => { b.disabled = true; });
  setAlignFeedback(
    afterMode === "auto"
      ? "正在执行 SOTA 曲面法向找平 + 曼哈顿正方向摆正..."
      : "正在应用朝向与高度校准并同步更新 .ply..."
  );

  const prevYaw = Number(state.viewport?.yaw || 0);
  const prevTarget = [...(state.viewport?.target || [0, 0, 0])];

  try {
    const res = await fetchJSON("/api/editor/align-scene", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace_dir: wsDir, reexport_ply: true, ...payload }),
    });
    resetAlignManualInputs();
    if (afterMode === "set_front" && state.viewport) {
      const c = Math.cos(-prevYaw);
      const s = Math.sin(-prevYaw);
      state.viewport.target = [
        prevTarget[0] * c - prevTarget[1] * s,
        prevTarget[0] * s + prevTarget[1] * c,
        prevTarget[2],
      ];
      state.viewport.yaw = 0.0;
      await loadScenePreview(wsDir, null, { preserveCamera: true });
    } else if (afterMode === "manual") {
      await loadScenePreview(wsDir, null, { preserveCamera: true });
    } else {
      await loadScenePreview(wsDir, null, { preserveCamera: false });
      state.viewport?.setPresetView("front");
    }
    updateAlignStatusSummary(res.alignment_transform || state.latestPreview?.alignment_transform);
    setAlignFeedback(`校准成功：${formatAlignSummary(res.alignment_transform)}`);
  } catch (err) {
    setAlignFeedback(`校准失败: ${err.message}`, true);
  } finally {
    state.alignBusy = false;
    btns.forEach((b) => { b.disabled = false; });
  }
}

function bindAlignEvents() {
  el("btnToggleAlignPopover")?.addEventListener("click", () => setAlignPopoverOpen(!state.alignOpen));
  el("btnCloseAlignPopover")?.addEventListener("click", () => setAlignPopoverOpen(false));
  el("btnAlignAuto")?.addEventListener("click", () => runSceneAlign({ mode: "auto" }, { afterMode: "auto" }));
  el("btnAlignSetFront")?.addEventListener("click", () => {
    const curYaw = Number(state.viewport?.yaw || 0);
    runSceneAlign({ mode: "set_front", current_yaw_rad: curYaw }, { afterMode: "set_front" });
  });
  el("btnAlignApplyManual")?.addEventListener("click", () => {
    const roll_deg = Number(el("alignRollDeg")?.value || 0);
    const pitch_deg = Number(el("alignPitchDeg")?.value || 0);
    const yaw_deg = Number(el("alignYawDeg")?.value || 0);
    const z_offset = Number(el("alignZOffset")?.value || 0);
    if ([roll_deg, pitch_deg, yaw_deg, z_offset].every((v) => Math.abs(v) < 1e-6)) {
      setAlignFeedback("请至少输入一个非零的角度或高度微调值", true);
      return;
    }
    runSceneAlign({ mode: "manual", roll_deg, pitch_deg, yaw_deg, z_offset }, { afterMode: "manual" });
  });
  el("btnAlignResetInputs")?.addEventListener("click", () => {
    resetAlignManualInputs();
    setAlignFeedback("");
  });
  el("chkAlignCrosshair")?.addEventListener("change", (e) => {
    state.viewport?.setAlignCrosshair(Boolean(e.target.checked));
  });
  el("btnAlignLevelFront")?.addEventListener("click", () => {
    if (!state.viewport) return;
    state.viewport.autoSpinDemo = false;
    state.viewport.yaw = 0.0;
    state.viewport.pitch = 0.0;
    state.viewport.render();
    state.viewport.requestImmediateNeuralRefresh(true);
  });
}

function bindEditorEvents() {
  bindAlignEvents();
  el("btnToggleEditor")?.addEventListener("click", () => setEditorOpen(!state.editorOpen));
  el("btnEdCloseDrawer")?.addEventListener("click", () => setEditorOpen(false));

  document.querySelectorAll(".ed-tool-btn[data-ed-mode]").forEach((btn) => {
    btn.addEventListener("click", () => setEditorMode(btn.dataset.edMode));
  });

  el("btnEdFocus")?.addEventListener("click", focusSelectedObject);
  el("chkEdAutoFocus")?.addEventListener("change", (e) => {
    state.editorAutoFocus = Boolean(e.target.checked);
  });
  el("btnEdExportObjPly")?.addEventListener("click", () => exportSelectedObjectPly(null));
  el("btnEdDuplicate")?.addEventListener("click", () => performEditorAction("duplicate"));
  el("btnEdToggleVis")?.addEventListener("click", () => {
    const sel = selectedEditorObject();
    if (sel) updateEditorObject(sel.id, { visible: sel.visible === false, record_undo: true });
  });
  el("btnEdDeleteRestore")?.addEventListener("click", () => {
    const sel = selectedEditorObject();
    if (sel) performEditorAction(sel.deleted ? "restore" : "delete", sel.id);
  });
  el("btnEdUndo")?.addEventListener("click", () => performEditorAction("undo"));
  el("btnEdRedo")?.addEventListener("click", () => performEditorAction("redo"));
  el("chkEdHighlight")?.addEventListener("change", (e) => {
    state.editorHighlight = Boolean(e.target.checked);
    syncEditorViewport();
  });
  el("chkEdDimBg")?.addEventListener("change", (e) => {
    state.editorDimBg = Boolean(e.target.checked);
    syncEditorViewport();
  });
  el("btnEdBake")?.addEventListener("click", bakeSceneEdits);

  el("btnEdPreviewText")?.addEventListener("click", runPreview2D);
  el("edTextPrompt")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      runPreview2D();
    }
  });
  el("btnEdClearPrompt")?.addEventListener("click", clearPromptsUI);
  el("btnEdRunSeg3D")?.addEventListener("click", runSegment3D);

  el("btnEdApplySam3Path")?.addEventListener("click", async () => {
    const modelPath = (el("edSam3ModelPath")?.value || "").trim();
    try {
      await fetchJSON("/api/editor/sam3-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath, unload: false }),
      });
      await loadSam3Status();
      setEditorFeedback(
        modelPath
          ? `已切换 SAM 3 权重源至: ${modelPath}`
          : "已恢复自动权重源（优先本地 models/sam3，否则 facebook/sam3）"
      );
    } catch (err) {
      setEditorFeedback(`配置失败: ${err.message}`, true);
    }
  });

  el("btnEdUnloadSam3")?.addEventListener("click", async () => {
    try {
      await fetchJSON("/api/editor/sam3-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ unload: true }),
      });
      await loadSam3Status();
      setEditorFeedback("已卸载 SAM 3 模型权重并清理 GPU 显存缓存");
    } catch (err) {
      setEditorFeedback(`卸载失败: ${err.message}`, true);
    }
  });

  el("edRefineThreshold")?.addEventListener("input", () => scheduleRefineObject(false));
  el("edRefineThreshold")?.addEventListener("change", () => scheduleRefineObject(true));
  el("edRefineZClip")?.addEventListener("input", () => scheduleRefineObject(false));
  el("edRefineZClip")?.addEventListener("change", () => scheduleRefineObject(true));

  el("btnEdResetTransform")?.addEventListener("click", () => performEditorAction("reset_transform"));
  el("btnEdApplyTf")?.addEventListener("click", () => {
    const sel = selectedEditorObject();
    if (!sel) return;
    updateEditorObject(sel.id, {
      translation: [
        Number(el("edTfTx").value) || 0,
        Number(el("edTfTy").value) || 0,
        Number(el("edTfTz").value) || 0,
      ],
      rotation_euler_deg: [
        Number(el("edTfRx").value) || 0,
        Number(el("edTfRy").value) || 0,
        Number(el("edTfRz").value) || 0,
      ],
      scale: Number(el("edTfScale").value) || 1.0,
      record_undo: true,
    });
  });

  window.addEventListener("keydown", (e) => {
    if (!state.editorOpen) return;
    const tag = (document.activeElement?.tagName || "").toUpperCase();
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.shiftKey) {
      e.preventDefault();
      performEditorAction("undo");
    } else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey))) {
      e.preventDefault();
      performEditorAction("redo");
    } else if (e.key.toLowerCase() === "f") {
      e.preventDefault();
      focusSelectedObject();
    } else if (e.key === "Delete" && state.selectedObjectId !== null) {
      e.preventDefault();
      const sel = selectedEditorObject();
      if (sel) performEditorAction(sel.deleted ? "restore" : "delete", sel.id);
    }
  });
}

async function init() {
  state.viewport = new window.Real2SimViewport3D(el("viewport3d"));
  state.viewport.onEditorEvent = (type, detail) => {
    onViewportEditorEvent(type, detail);
  };
  let savedTheme = "light";
  try {
    savedTheme = window.localStorage.getItem("r2s_studio_theme") || "light";
  } catch {
    savedTheme = "light";
  }
  applyTheme(savedTheme);
  bindUIEvents();

  state.viewport.setLayer("neural", el("layerNeural").checked);
  state.viewport.setLayer("splats", el("layerSplats").checked);
  state.viewport.setLayer("mesh", el("layerMesh").checked);
  state.viewport.setLayer("convex", el("layerConvex").checked);
  state.viewport.setLayer("spawn", el("layerSpawn").checked);

  const [health, cfgInfo, , workspaces] = await Promise.all([
    fetchJSON("/health"),
    fetchJSON("/api/configs"),
    loadDoctor(),
    loadHistoryWorkspaces(),
  ]);
  state.stagesCatalog = health.stages || {};
  populateSelect(el("presetSelect"), cfgInfo.files || [], cfgInfo.default_name);
  if (cfgInfo.default_config) {
    applyConfigToForm(cfgInfo.default_config);
  }

  if (workspaces && workspaces.length > 0) {
    const preferred = workspaces.find((w) => !w.is_mock && w.has_3dgs) || workspaces.find((w) => w.has_3dgs) || workspaces[0];
    await selectHistoryWorkspace(preferred.workspace_dir);
  } else {
    renderExportPanel({ workspace_dir: el("workspaceDir").value, exports_available: {} });
  }
}

init().catch((err) => {
  el("jobStatusLine").textContent = `初始化异常: ${err.message}`;
});
