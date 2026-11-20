# Real2Sim-GS 工程 AI 协作准则与全景索引

---

## 第一部分：AI 协作铁律

1. **直上最前沿 SOTA，拒绝“低配过渡”路线（技术选型最高优先级原则）**
   - **一步到位选顶尖方案**：提出架构设计、算法选型、模型集成或开源依赖建议时，严禁推销“先做最基础/简陋玩具版，出问题再慢慢叠补丁升级”的保守路线。
   - **前沿验证优先**：必须主动检索并对齐最新顶会论文（CVPR / ICCV / ECCV / NeurIPS / SIGGRAPH / WACV 等）与 GitHub 已被社区验证的顶尖开源实现，结合实际软硬件边界（如本地 RTX 5070 Ti 12GB 显存与云端算力可扩展性）直接给出**当前业界最先进、经过验证且可工程落地的 SOTA 方案**。
2. **先读后思，改必同步（省 Token 与防失步原则）**
   - **前置阅读**：每次接到任务进行思考或检索代码前，必须优先依据本文档定位模块与源码位置，禁止盲目全局搜索以节省 Token。
   - **实时维护**：每次对项目代码、目录结构、功能逻辑或进度产生任何实质性修改后，必须在同一轮任务结束前同步更新 `AGENTS.md`，确保文档永远处于最新状态。
3. **零容忍“屎山”代码（代码洁癖原则）**
   - 修改 Bug、变更逻辑或重构代码时，**必须直接删除**原本错误、废弃或被替代的旧代码，严禁通过大段注释旧代码或打补丁式堆叠逻辑来凑合运行。
   - **唯一例外**：调试代码（Debug 日志、测试桩等）允许保留并通过注释启停，但必须加上明确的注释标记（如 `// [DEBUG]` 或 `# [DEBUG]`）。
4. **Git 操作绝对受控（手动门禁原则）**
   - 严禁 AI 在后台自动执行 `git commit`、`git merge`、`git push` 等提交、合并或上传操作。
   - 只有在我**明确、主动发出指令**要求合并或上传 Git 时，才允许执行对应的 Git 操作。
5. **文档与代码一致性熔断机制**
   - 在执行任务过程中，一旦发现 `AGENTS.md` 中的描述与实际项目代码、文件结构或逻辑存在不一致，必须**第一时间中止当前修改任务**，立即向我列出具体的冲突点，由我裁决以文档为准还是以现有代码为准，严禁自作主张覆盖任何一方。
6. **文档结构化演进与归并规则**
   - 维护更新本文档时，保持克制与整洁：若新内容与现有章节相关，必须优先在对应标题下直接合并描述，或新增子标题（如二级/三级标题）。
   - 只有当项目引入了**全新的核心概念或完全独立的新模块**时，才允许新增一级大标题。

---

## 第二部分：全景工程导览

### 1. 工程全局概览

#### 1.1 项目目标
构建一套**输入输出配置化、底层白盒可魔改、算法与新论文可热插拔**的端到端 Real-to-Sim 3D Gaussian Splatting（3DGS）终端工程管线（`real2sim_gs`）。核心解决将真实场景视频/多视角图像转化为机器人物理仿真平台（MuJoCo / MJWarp、Unity）可直接加载的**视觉+物理完备场景包**（标准 `.ply` 3DGS + 去噪对齐后碰撞网格 `.obj` + 场景配置文件 `scene.xml`），同时为科研实验提供可微训练、PSNR/SSIM 评测与外部顶会论文桥接能力。

#### 1.2 实现方案与架构
项目采用 **“微内核流水线 + 插件注册表（Registry）+ 标准化磁盘工作区契约”** 三层解耦架构：
- **配置、接口与 Studio 前端层（`configs/` + `real2sim_gs/api/` + `real2sim_gs/ui/`）**：通过 Pydantic v2 `_StrictModel`（`extra="forbid"`）强校验 YAML 配置并允许 CLI/SDK/FastAPI 仅覆写输入视频与输出目录即可启动；对外统一暴露 Typer 终端命令（`real2sim` 含 `run`/`doctor`/`stages`/`serve`/`ui`，其中 `ui --desktop` 采用 `reuse_existing=False` 独立端口隔离、Windows `SO_EXCLUSIVEADDRUSE` 与 `connect_ex` 通配监听探测及退出时 `uvicorn.Server.should_exit` 优雅收尾防止寄生旧服务断连）、Python SDK（`run_pipeline`）、FastAPI HTTP 调度服务（`/health`、`/jobs`、`/jobs/{id}/cancel`、`/api/*` 及 `/api/editor/*` 含 `/api/editor/align-scene`；内置单卡单活队列 `JobStore`、删除态互斥锁与严格清理 `_rmtree_strict`、`TrustedHostMiddleware`、`X-Studio-Token` 及远程 `--host 0.0.0.0`/`::` `?token=` Cookie 鉴权、输入/配置/工作区路径沙箱与 UNC 拒绝、流式上传格式/大小限流、`/api/cleanup-uploads` 自动保留活跃与未完成工作区引用输入、`_GPU_RENDER_LOCK` 并发锁、`_EDITOR_MANAGERS` 场景图编辑器缓存与训练期 409 冲突保护、带 UTF-8 RFC 5987 文件名与 `.ply` `ZIP_STORED` 打包临时 ZIP 下载）以及零 Node.js 内嵌式 **Real2Sim-GS Studio Web 3D 交互工作台**（自动注入 `studio-token` 请求头、`fetchJSON` 断连 `/health` 诊断与 GET 重试、递归低抖动增量轮询与差异日志渲染、`escapeHtml` XSS 防护、清理上传缓存保留活跃输入、默认开启**科研白昼 White Theme** 并支持一键切换极夜流光主题、左侧顶置**历史训练档案库**展示磁盘占用与断点状态并支持一键查看三维结果/继续中断训练/复用参数快照/打开本地目录/分档清理中间缓存或彻底删除档案、切换或加载历史工作区时自动回填关联 `input_path` 且一键切换智能配方时自动清空旧隐含超参覆盖、拖拽上传新视频自动隔离分配独立工作区路径、运行中实时监控显存占用 MB/GB 并可随时点击**「停止训练」**安全落盘断点、带背压单在途管线的 CUDA `/api/render-view` 连续流式神经渲染视口，在 409 训练抢占时自动平滑暂停并支持定向二维椭圆高斯回退与同步 3D 方位轴 Gizmo、顶部**`🧭 水平/正方向` 校准悬浮面板（`alignPopover`）**支持 SOTA 一键自动找平+曼哈顿摆正 / 设当前视角为正前 / Roll·Pitch·Yaw·ΔZ 增量微调与画布水平参考网格、顶部独立 **`✂ 物体分割与编辑` 工具栏（`editorToolbar`）** 与可折叠右侧 **物体编辑抽屉（`editorDrawer`）**——支持点/框/开放词汇文本交互式 SAM 3 分割、3D AABB 与 Move/Rotate/Scale 交互 Gizmo、一键聚焦/自动聚焦物体、单物体 `.ply` 独立导出、复制/隐藏/删除/Undo/Redo/高亮与场景烘焙回写、4 类场景一键智能配方与实时参数配合顾问、五阶段脉冲步进条+面积渐变 SVG Loss 曲线+一键复制报错诊断堆栈，以及纯 3DGS `.ply` 与 MuJoCo/Unity 分流下载导出）；根目录提供动态 `%~dp0` 与 Python 探测的 `start_studio.bat` 启动器（提示最小化保留后台黑框）。
- **交互式 3DGS 物体分割与场景编辑层（`real2sim_gs/editor/`）**：面向已训练 3DGS 模型提供 Windows 原生 PyTorch Meta SAM 3 多模态分割与 3DGS 抬升编辑能力，针对本地 RTX 5070 Ti 12 GB 显存 + 32 GB 系统内存优化：通过 Hugging Face `transformers` 按需懒加载 `bfloat16` SAM 3（支持本地权重路径或 HF Token、非 Mock 重建任务启动前自动卸载释放显存，`video_storage_device="cpu"` 将多视角视频跟踪特征缓存至 32 GB 内存）、结合 B³-Seg 解析式 EIG 主动视角规划、`gsplat` 反向传播与纯 PyTorch EWA 一致性遮挡感知 FlashSplat 权重求解、SciPy Dinic `s-t` 最小割 GaussianCut 实心体内部补全与 SAGD 边界主轴分解，并由 `SceneEditManager` 维护非破坏性场景图（40 步 Undo/Redo、阈值/Z-clip 分数缓存重算、选中聚焦拾取、O(3) 0..3 阶 SH 同步旋转、单物体 `.ply` 独立导出及全场景烘焙回写）。
- **微内核调度与契约层（`real2sim_gs/core/`）**：`StageRegistry` 通过 `@register_stage(stage_kind, name)` 解耦调度器与具体算法；`SceneWorkspace` 强制规定 `01_frames` → `02_pose` → `03_gaussians` → `04_processed` → `05_export` 五级目录、原子写文件（`atomic_write_text`/`atomic_write_json`）、损坏 `manifest.json` 备份恢复与 `04_processed/active_splat.json` 指针（`set_active_splat()`/`active_splat_path()`）、阶段上游链式签名失效与目录重置（`invalidate_from()`/`reset_stage_dir()`）、缺失输入视频前置 Fail-Fast 校验防误清已有缓存、已清理临时上传输入且 `01_frames` 完好时的指纹恢复跳过机制、`check_cancelled()` 协作式停止契约（`PipelineCancelledError`）及 `report_progress()` 轻量遥测钩子；提供跨平台 `shell=False` 进程组强杀树（`core/external.py`）、Unicode 路径 OpenCV 读写与自然排序（`core/image_io.py`）及 PEP 562 惰性顶层导入；配合 Stage 3 训练内步级断点 `03_gaussians/checkpoint.pt` 实现**“阶段跳过 + 步级续训”**双层断点续跑与跨机接力。
- **可插拔五阶段算法层（`real2sim_gs/stages/`）**：
  1. `ingest`（抽帧）：支持均匀抽帧（`uniform_ffmpeg`，无过采样与无效清晰度计算）、时间滑窗防模糊清晰度选帧（`sharpness_ffmpeg`，通过 `ffprobe` 自动按视频时长限流候选抽帧率、以高速 `raw_%06d.jpg` 临时解帧并实时响应取消信号，长边自动限幅 `1600px`，均采用 `-nostdin` + `DEVNULL` 与 Unicode 图像读写）及测试桩（`mock_ingest`）。
  2. `pose`（位姿解算）：支持 COLMAP SfM 解析（`colmap`，通过零 `torch`/`cv2` 依赖的轻量级受控子进程 `_colmap_worker.py` 运行 `pycolmap` 并限制 `max_image_size=1600`/`max_num_features=6144`/线程数防止 4K 爆内存且支持即时取消；`matcher="auto"` 智能区分顺序/全局匹配同时尊重显式 `"exhaustive"` 与非视频目录；`_COLMAP_CAMERA_LAYOUTS` 显式建模与空 `POINTS2D` 安全解析；CLI 自动选取最大连通子图并输出 `registered_ratio`/`num_models`；默认 `SIMPLE_RADIAL` 单焦距模型并导出畸变参数，亦可选 `OPENCV`/`OPENCV_FISHEYE`）及基于 `run_command_template` 的免 COLMAP 前馈论文槽位（`feedforward_slot`，面向 VGGT / DUSt3R / MASt3R）。
  3. `reconstruct`（重建核心）：默认采用 SOTA 白盒 PyTorch 3DGS（`gsplat_whitebox`，拆分为 `model`/`appearance`/`losses`/`strategy`/`trainer`，集成自动 4K 降采样缩放与 2 GiB 宿主 pinned-memory 预算的 `uint8` 显存压缩、图像/相机方向与数量一致性校验门卫、`CameraView.K` 缓存、鱼眼 `cv2.fisheye` 与针孔 `alpha=0.0` `valid_mask` 掩码去畸变（同步作用于 L1/SSIM 损失）、室内弱纹理低透明度探索种子点补充、尺度不变 `from_point_cloud()`/`scene_extent` 与相机中心尺度自适应学习率、Gaussian 尺度防爆上限 `clamp_scales_()`、3-axis 3D Mip-Splatting 抗混叠滤波并烘焙导出（`bake_mip_filter=True`）、k-NN 尺度初始化、默认 `classic` 光栅化对齐 PlayCanvas SuperSplat、渐进解锁 0..3 阶完整球谐系数（`sh_degree=3`，`45` 维 `sh_rest`，每 `1000` 步升阶）与旗舰 `800k`/`30k` 默认上限、默认 `random` 随机噪声背景（alpha 后合成绕过 `gsplat` 断言冲突）、`test_every=8` 留出测试视角评估 `train_PSNR`/`test_PSNR` 并冻结测试视角位姿与双边网格同时导出 `refined_cameras.json`、`MCMCStrategy` 重数感知梯度重定位（`bincount+1`）+ 0.05×低扰动/SGLD 与 `DensityControlStrategy` 旋转分裂（移除父高斯并保留未分裂存量动量）、`SparseAdam` + `Adam` 分组优化与原生 CUDA/零同步 `SelectiveAdam` 状态同步/迁移/局部重置、冷却期中段可愈合剪枝（`mid_cleanup_step`）+ 末步仅清死点、真针刺比 `s_max/s_mid` 各向异性正则与尺度归一化 `mcmc_reg_loss`/`depth_smooth`、带 Identity 锚点正则的稀疏 `BilateralGridModule` 曝光补偿（纯视觉默认关闭防止逐帧过拟合）、稀疏可微 `CameraPoseOptModule` 位姿残差精修、非截断 `ssim_tensor` 流式评测与 `ImagePairEvaluator`、支持 `gsplat` CUDA 加速与无上限 splat+pixel 分块纯 PyTorch EWA 回退、带 `params_signature` 校验的完整 `checkpoint.pt` 断点保存与恢复）；预留 `external_subprocess` 与 `twodgs_slot` 桥接 `third_party/` 下独立依赖仓库。
  4. `postprocess`（可叠加后处理链）：按列表顺序执行去漂浮噪点与飞针（`clean_floaters`：Opacity 过滤 + `s_max/s_mid` 真针刺剔除 + 并行 KDTree `scale_aware`/`global`/`depth_normalized` 尺度感知离群点去噪，避免误伤墙面/天花板大高斯，并更新 `active_splat.json`）、SOTA 曲面法向与曼哈顿世界对齐（`align_gravity_scale`：提取扁平高不透明 3DGS surfel 最短轴法向，通过双正交墙门控防止单一倾斜显示器劫持重力，在 $S^2$ 上执行 Cauchy-IRLS 联合全局水平面/门控双墙/多偏航相机右轴求解精确零俯仰/零横滚重力方向、面积加权最低实质地面峰精确定位 `Z=0`、以及曼哈顿 4 重正交墙体正方向 `-X` 对齐；合成/各向同性点云无缝回退面积加权下带 RANSAC；同步变换 0..3 阶全 SH 系数与输出 `alignment_transform.json`，并提供 `realign_workspace_scene()` 供 Studio 实时重校准与手动微调）、语义分割槽位（`semantic_segment_slot`：原生支持 `backend="sam3"` 批量文本开放词汇 3DGS 分割并同步支持 `command_template` 外部桥接）、前景核心 `[2%, 98%]` 分位聚焦的各向同性 3D Marching Cubes 等值面（`max_voxel_resolution=256`、`p99` 归一化）、自动剥离室内天花板顶盖的向量化 2.5D 高度场与核心点凸包物理碰撞网格提取（`extract_collision_mesh`）。
  5. `export`（多目标资产导出）：支持纯 3DGS 标准视觉点云包（`ply_3dgs`：基于 `alignment_transform.json` 准确判定对齐态，居中至前景视线焦点、裁剪极端边界飘浮噪点并经右手 `SO(3)` 旋转输出至 PlayCanvas SuperSplat / Inria COLMAP 标准 `.ply` 正立正面视角，完整保留 0..3 阶 SH、优先使用 `refined_cameras.json` 同步变换并输出 `splat_meta.json`）、MuJoCo/MJWarp 仿真包（`mujoco`：`scene.xml` + `visual_splats.ply` + `collision_mesh.obj` + `coacd` → `vhacd` → `normal_spatial_kmeans_hull` → `plane_only` 物理凸包分解、XML 属性转义、`inertia="shell"`、摩擦归一化、射线避让天花板出生点与 `scene_meta.json`）及 Unity 资产包（`unity`：默认 `"inria_rh"` 避免二次翻转并保留 `"unity_lh"` 显式模式）。

#### 1.3 软硬件环境清单

**硬件信息**

| 类别 | 配置/状态 | 说明 |
| :--- | :--- | :--- |
| 本地调试主机与 GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU (12227 MiB VRAM, Driver 595.79) + 32 GB 系统内存 | 已在当前 Windows 开发机验证 CUDA 可用；支持 SAM 3 `bfloat16` 与 `video_storage_device="cpu"` 内存分时卸载 |
| 云端训练算力 | `[待确认: 拟租用 AutoDL/实验室 Linux 服务器的 GPU 型号、数量及显存规格]` | 用于跑 30k 步 `gsplat` 及外部论文对比实验 |
| 场景采集前端 | `[待确认: 实际拍摄视频使用的相机/手机/运动相机型号，是否包含深度相机(RGB-D)或标定板用于真实米制尺度恢复]` | 当前管线默认支持单目 `.mp4` 视频或图片目录输入 |
| 下游仿真与实机硬件 | `[待确认: 下游 MuJoCo/Unity 拟接入的机器人本体型号(如 Unitree G1/Go2/机械臂)及是否涉及实机 Sim-to-Real 部署与外设通信接口]` | 当前已提供标准化 MJCF `scene.xml` 与碰撞网格导出 |

**软件信息**

| 维度 | 版本与路径 | 备注 |
| :--- | :--- | :--- |
| 运行系统 | Windows（本地开发，UTF-8 编码读写）/ Linux（AutoDL 云端训练兼容） | 已初始化 Git 仓库（`main` 分支，关联远端 `origin`） |
| Python 解释器 | Python 3.11.9（本地精确路径：`C:\Users\21241\AppData\Local\Programs\Python\Python311\python.exe`） | 本地注意区分默认 PATH 中无 pip/torch 的 Agent venv |
| 核心深度学习与计算库 | `torch 2.15.0+cu132`、`numpy 2.4.6`、`scipy 1.17.1`、`scikit-image 0.26.0`、`trimesh 5.1.0`、`opencv-python 4.13.0.92`、`pillow 12.2.0`、`transformers 5.23.0`、`accelerate 1.16.0`、`safetensors 0.8.0` | 核心纯白盒、SOTA 几何提取及 Windows 原生 SAM 3 分割库已安装就绪 |
| 配置与接口库 | `pydantic 2.12.5`、`PyYAML 6.0.3`、`typer 0.23.1`、`rich 13.9.4`、`fastapi 0.139.0`、`uvicorn 0.33.0`、`python-multipart 0.0.26`、`pywebview 6.2.1` | CLI / SDK / HTTP 服务与桌面/Web Studio 底座 |
| 外部工具与可选插件 | `ffmpeg 8.1.1`（已就绪）、`pycolmap 4.2.1`（已就绪，内置原生 C++ SfM）、`gsplat 1.5.3`（已在本机预编译生成 `gsplat/csrc.cp311-win_amd64.pyd`，针对 RTX 5070 Ti `sm_120` 开箱即用；本地编译链位于 `D:\DevEnv\VSBuildTools` 与 `D:\DevEnv\cuda-13.2`，无 CUDA/`gsplat` 机器仍自动回退内置分块 GPU/CPU EWA 渲染）、`mujoco 4.0.1` + `warp-lang 1.19.0` + `mujoco-warp 4.0.1`（已就绪，RTX 5070 Ti `sm_120` GPU 并行物理仿真已验证通过） | 使用 `real2sim doctor` 实时查看状态 |
| 构建安装与测试命令 | 安装：`python -m pip install -e . --no-build-isolation --no-deps`<br>桌面一键启动：双击 `start_studio.bat`（或桌面 `Real2Sim-GS Studio.bat`）<br>前端：`python -m real2sim_gs.api.cli ui`<br>测试：`python -m pytest -v`<br>演练：`python -m real2sim_gs.api.cli run -c configs/default_mujoco.yaml -o workspaces/demo --mock` | 完整 CLI 入口 `real2sim` 已注册至 `pyproject.toml` |

#### 1.4 当前工程进度
- **`[已完成]`**：工程基础骨架与包配置（`pyproject.toml` 含分功能 `[project.optional-dependencies]`（新增 `sam3`）与 `.gitignore` 过滤 `models/`/`workspaces/`、`configs/` 升级为旗舰 `SH3`/`20k~30k`/`500k~800k`/`scale_aware` 去噪与默认 `matcher: "auto"`）、`SceneWorkspace` 阶段（`_StrictModel`、上游链式签名 `invalidate_from`/`reset_stage_dir` 缓存失效、缺失输入前置阻断保护与临时上传清理后 `ingest` 指纹豁免、`atomic_write_text`/`atomic_write_json`、损坏 `manifest.json` `.corrupt` 备份恢复与 `active_splat.json` 指针）+训练步级双层断点续跑内核/`PipelineCancelledError` 协作式停止契约/`report_progress()` 遥测钩子/`core/external.py` 跨平台进程组强杀/`core/image_io.py` Unicode 路径与自然排序/`__init__.py` PEP 562 惰性导入与 `StageRegistry`、`doctor` 环境自检（含自动检测 `models/sam3` 与 HF Token 检查）、标准二进制/ASCII/big_endian 3DGS `.ply` 读写（完整 0..3 阶 SH 与 `O(3)` 变换）、近裁剪面裁剪 + 2D `cv2.remap` `sample_video.py` 合成器、Stage 1~5 全流程（4K 限幅 + `-nostdin` + Unicode 抽帧与无多余开销 `uniform_ffmpeg`；零 `torch`/`cv2` 隔离 `_colmap_worker.py` + `_COLMAP_CAMERA_LAYOUTS` 显式相机解析；白盒 CUDA 3DGS 完成 `CameraView.K` 缓存、鱼眼/针孔 `valid_mask` 去畸变、2 GiB pinned memory 限流、3-axis 3D Mip-Splatting 抗混叠与烘焙导出、0..3 阶渐进 SH 训练与 Studio 渲染、`s_max/s_mid` 真针刺正则与尺度归一化正则、冷却期中段可愈合剪枝、`test_every=8` 留出测试验证集、旋转扰动/重数感知 MCMC 与旋转分裂 ADC、`SparseAdam` + `SelectiveAdam` 状态同步/迁移/局部重置与零列 `sh_rest` 守卫、完整 `params_signature` 校验断点恢复、无上限分块纯 PyTorch EWA 回退、`ImagePairEvaluator`；SOTA 3DGS 曲面法向 Cauchy-IRLS 找平（含门控双正交墙防斜屏幕劫持、最低水平峰 `Z=0` 地面精定位与曼哈顿 4 重正交墙体正方向摆正，在实测场景 `IMG_2609` 与 `default_scene` 验证 `normal_z >= 0.9999` 与 24.3°/14.5° 自动摆正）、`scale_aware` 尺度感知 SOR、`SemanticSegmentSlotStage` 原生 `sam3` 文本开放词汇 3DGS 分割含显存卸载与重叠跳过、自动去天花板顶盖的 2.5D 高度场与各向同性 3D Marching Cubes；SuperSplat/Inria `.ply` 对齐态判定与完整 SH 导出、MuJoCo `coacd`/`vhacd`/`kmeans`/`plane_only` 凸包分解与避天花板出生点/XML 转义、Unity 默认 `"inria_rh"` 导出）、**交互式 3DGS 物体分割与场景编辑模块（`real2sim_gs/editor/`）**（Windows 原生 `transformers` Meta SAM 3 懒加载 `bfloat16`、单子模型显存驻留自动驱逐 `_evict_other_models()` 与 `video_storage_device="cpu"` 32 GB 内存分时卸载、ModelScope `facebook/sam3` Hugging Face 完整权重已校验落盘至 `models/sam3` 并实现零配置自动发现、全 4 个 SAM 3 子模型 `Sam3TrackerModel` / `Sam3TrackerVideoModel` / `Sam3Model` / `Sam3VideoModel` 已在 RTX 5070 Ti CUDA `bfloat16` 完成真机推理与双向多视角视频追踪验证、B³-Seg 解析式 EIG 主动视角规划、FlashSplat 遮挡感知加权求解、`gsplat` 反向传播与纯 PyTorch EWA Parity 一致性验证、SciPy Dinic GaussianCut 实心体内部补全与 SAGD 边界分裂及过期分数缓存同步清理、非破坏性 40 步 Undo/Redo `SceneEditManager`、Z 深度精确像素拾取、O(3) 全 SH 同步刚体变换与单物体独立 `.ply` / 场景烘焙导出）、CLI/SDK/FastAPI（`ui --desktop` 独立端口隔离与 Windows 通配监听探测/优雅收尾、单卡单活队列、删除互斥锁、`TrustedHostMiddleware` + `X-Studio-Token`/远程 `?token=` Cookie、路径与 UNC 沙箱、流式上传限流与未完成工作区引用保护、`_GPU_RENDER_LOCK` + `_EDITOR_MANAGERS` + 409 训练冲突隔离、高分辨率 4K 视口缩放/已变换物体遮挡感知 `segment-3d`/12 个 `/api/editor/*` 接口含 `align-scene` 实时找平摆正微调与场景图着色渲染、RFC 5987 UTF-8 临时 ZIP 下载）、`Real2Sim-GS Studio`（Token 请求头、`fetchJSON` 断连 `/health` 诊断与 GET 重试、递增稳定轮询、差异日志与 `escapeHtml` 防 XSS、409 自动暂停流式渲染、历史档案自动回填绑定输入、一键配方清旧隐含覆写、SH 阶数切换与 `30k/800k` 旗舰配方、专属 **`🧭 水平/正方向` 悬浮校准面板（含画布水平参考十字线与一键找平摆正/设当前视角为正前）** 与 **`✂ 物体分割与编辑` 工具栏（`editorToolbar`）及右侧交互抽屉（`editorDrawer`）** 支持点/框/文本 SAM 3 分割 + 单步 Undo 3D Gizmo 平移旋转缩放 + 滑动条无损微调 + 聚焦/自动聚焦/单物体 `.ply` 导出/撤销重做与烘焙保留视角、动态探测 `%~dp0` 与提示最小化黑框的 `start_studio.bat`），以及 `pytest` 全套 7 个测试文件共 56 项自动化测试用例全部通过。
- **`[进行中/半成品]`**：外部论文适配器（`FeedForwardPoseAdapter`、`TwoDGSSlotStage` 及 `SemanticSegmentSlotStage` 的 `command_template` 外部命令桥接模式）已完成标准化接口契约与子进程桥接骨架，待在 `third_party/` 克隆真实外部论文代码后填入具体 `command_template`；默认已启用本地 `models/sam3` 开箱即用推理，如需切换其他自定义模型路径可通过 `REAL2SIM_SAM3_MODEL_PATH` 或 Studio 抽屉配置（`hf auth login` 仅在拉取在线 gated 仓时使用）。
- **`[待开发/已知问题]`**：
  - FastAPI `server.py` 任务状态当前采用内存 `JobStore`，重启不持久化。
  - `[待确认: 大创团队首个拟深入魔改或对比的顶会论文方向（如 2DGS 表面法向约束、MCMC 致密化、VGGT 免位姿重建或物理参数反演），以及真实米制尺度 scale_factor 的自动标定方案]`。

---

### 2. 递归式项目目录分层解析

#### 2.1 项目根目录结构树
```text
d:\a临时下载\大创3dgs\
├── AGENTS.md                                      # AI 协作铁律与全景工程导览（本文档）
├── README.md                                      # 仓库说明文档
├── .gitignore                                     # Git 忽略规则（过滤 workspaces/、权重与缓存）
├── pyproject.toml                                 # 包元数据、依赖声明、CLI 入口与 pytest 配置
├── start_studio.bat                               # Windows 一键启动 Studio 桌面工作台批处理脚本
├── configs/                                       # 实验配置目录（输入输出路径与积木式算法编排）
│   ├── default_mujoco.yaml                        # 默认配置：视频 -> COLMAP -> 白盒3DGS -> 纯3DGS + MuJoCo
│   ├── pure_3dgs.yaml                             # 纯视觉重建预设：跳过物理网格，仅导出标准 3DGS .ply
│   └── exp_2dgs_vggt.yaml                         # 对比实验配置：VGGT 位姿 + 2DGS + SAM + Unity/MuJoCo
├── real2sim_gs/                                   # 核心 Python 源码包
│   ├── __init__.py                                # 包顶层公开 API 导出
│   ├── api/                                       # 对外统一接口层（CLI / SDK / HTTP Server）
│   │   ├── __init__.py
│   │   ├── cli.py                                 # Typer 终端命令行（run / doctor / stages / serve / ui）
│   │   ├── sdk.py                                 # Python 函数式调用入口
│   │   └── server.py                              # FastAPI 任务调度、3D 预览、上传下载与 Studio 托管服务
│   ├── ui/                                        # Real2Sim-GS Studio 内嵌式 Web 3D 可视化前端
│   │   ├── __init__.py                            # 导出 UI_STATIC_DIR 静态目录路径
│   │   └── static/                                # 原生零外部依赖前端资产
│   │       ├── app.js                             # 表单联动、异步任务轮询、画廊与导出控制器
│   │       ├── index.html                         # Studio 主控制台与 3D 视口布局
│   │       ├── styles.css                         # 深色工程风工作台样式
│   │       └── viewer3d.js                        # Z-up 交互式 3DGS 点云 + 网格线框 + 凸包渲染引擎
│   ├── core/                                      # 微内核调度、配置校验、工作区与插件管理
│   │   ├── __init__.py
│   │   ├── config.py                              # YAML -> Pydantic v2 强类型严格配置解析
│   │   ├── doctor.py                              # 运行环境与可选依赖惰性诊断器
│   │   ├── external.py                            # 跨平台安全无壳外部子进程桥接与进程组强杀树
│   │   ├── image_io.py                            # Unicode 路径兼容 OpenCV 图像读写与自然排序
│   │   ├── pipeline.py                            # 主流程编排器、链式失效与断点续跑逻辑
│   │   ├── registry.py                            # 阶段插件全局注册表与装饰器
│   │   ├── stage.py                               # 抽象阶段基类与标准返回值契约
│   │   └── workspace.py                           # 01~05 标准工作目录、active_splat 指针及 manifest.json 管理
│   ├── eval/                                      # 渲染质量与几何指标评测模块
│   │   ├── __init__.py
│   │   └── metrics.py                             # 非截断可微 SSIM 算子、ImagePairEvaluator 与 PSNR/SSIM 评测汇总
│   ├── editor/                                    # 交互式 3DGS 物体分割与场景图编辑引擎
│   │   ├── __init__.py                            # 导出编辑器状态、SAM 3 后端、3D 抬升与场景图接口
│   │   ├── sam3_backend.py                        # Windows 原生 transformers Meta SAM 3 多模态分割与 CPU 内存跟踪后端
│   │   ├── segment_3d.py                          # B³-Seg EIG 选视角 + FlashSplat + GaussianCut + SAGD 3DGS 抬升
│   │   └── scene_graph.py                         # 40 步 Undo/Redo 场景图、像素拾取、O(3) 变换、单物体 .ply 与烘焙导出
│   ├── stages/                                    # 五大可插拔算法阶段实现与外部论文槽位
│   │   ├── __init__.py                            # 统一触发内置阶段 @register_stage 注册
│   │   ├── ingest/                                # 阶段1：视频抽帧与图像筛帧
│   │   │   ├── __init__.py
│   │   │   ├── ffmpeg_ingest.py                   # 均匀抽帧与 Laplacian 清晰度筛帧
│   │   │   └── mock_ingest.py                     # 快速测试合成视角图像桩
│   │   ├── pose/                                  # 阶段2：相机内参/外参解算与初始稀疏点云
│   │   │   ├── __init__.py
│   │   │   ├── _colmap_worker.py                  # 零 torch/cv2 轻量级 pycolmap 隔离工作子进程
│   │   │   ├── colmap_pose.py                     # COLMAP SfM 调用编排与显式相机/TXT 位姿解析
│   │   │   ├── feed_forward_pose.py               # VGGT / DUSt3R 等免 COLMAP 外部论文适配器
│   │   │   └── mock_pose.py                       # 环绕视角合成相机位姿与初始点云桩
│   │   ├── reconstruct/                           # 阶段3：3DGS 重建算法核心与外部桥接
│   │   │   ├── __init__.py
│   │   │   ├── gsplat_whitebox/                   # SOTA 白盒 PyTorch 3DGS 主力科研代码
│   │   │   │   ├── __init__.py
│   │   │   │   ├── appearance.py                  # 逐视角 3D 双边网格曝光补偿 & SE(3) 位姿微调
│   │   │   │   ├── dataset.py                     # 图像与位姿配对数据加载器
│   │   │   │   ├── losses.py                      # 光度 Loss + 表面扁平/各向异性/深度平滑/MCMC 正则
│   │   │   │   ├── model.py                       # 显式高斯球、k-NN 初始化与 3D Mip-Splatting 滤波
│   │   │   │   ├── strategy.py                    # 3DGS-MCMC 重定位/SGLD 与 AbsGS 致密化剪枝策略
│   │   │   │   └── trainer.py                     # SelectiveAdam + gsplat CUDA / 纯 PyTorch EWA 渲染
│   │   │   ├── mock_reconstruct.py                # 带倾斜地面与漂浮噪点的测试高斯云生成器
│   │   │   ├── subprocess_adapter.py              # 通用独立环境外部论文命令行适配器
│   │   │   └── twodgs_slot.py                     # 2DGS / PGSR 预置槽位
│   │   ├── postprocess/                           # 阶段4：面向物理仿真的几何清洗、对齐与网格化
│   │   │   ├── __init__.py
│   │   │   ├── align_sim.py                       # 3DGS surfel Cauchy-IRLS 找平、曼哈顿正方向摆正与尺度/手动校准
│   │   │   ├── clean_floaters.py                  # 低透明度剔除与 KDTree 离群点去噪
│   │   │   ├── extract_mesh.py                    # 3D Marching Cubes / 2.5D 高度场 / 凸包碰撞网格提取
│   │   │   └── segment_slot.py                    # SAM / LangSplat 物体语义分割预留槽位
│   │   └── export/                                # 阶段5：多目标资产打包导出
│   │       ├── __init__.py
│   │       ├── mujoco_exporter.py                 # MuJoCo 多凸包碰撞体分解、scene.xml 与资产包生成
│   │       ├── ply_exporter.py                    # 纯 3DGS 标准 .ply + cameras.json 视觉资产导出
│   │       └── unity_exporter.py                  # Unity 左手系 Y-up 坐标系转换与资产包导出
│   └── utils/                                     # 跨阶段通用底层数据结构、示例场景生成与文件 I/O
│       ├── __init__.py
│       ├── ply_io.py                              # 标准 3DGS 二进制 PLY 与稀疏点云读写
│       └── sample_video.py                        # 开箱即用多视角纹理桌面环拍 sample_video.mp4 合成器
├── tests/                                         # pytest 自动化测试套件
│   ├── test_api_and_cli.py                        # CLI 命令、FastAPI 安全与接口测试
│   ├── test_core_pipeline.py                      # 微内核链式失效、严格配置、Unicode I/O、COLMAP 解析与子进程隔离单测
│   ├── test_editor_segmentation.py                # 3DGS 分割抬升（FlashSplat/GraphCut/SAGD）、CUDA 与 EWA 一致性及编辑接口单测
│   ├── test_pipeline_e2e.py                       # 全链路端到端执行、断点续跑与导出验证
│   ├── test_postprocess_export.py                 # SH 0..3 O(3) 不变性、40°俯拍对齐、各向同性网格与 MuJoCo 导出单测
│   ├── test_trainer_fixes.py                      # MCMC 重数/动量重置、留出测试集、旋转 ADC 分裂与 3D Mip 滤波单测
│   └── test_whitebox_gs.py                        # PLY 往返一致性、白盒短训与致密化单测
└── third_party/                                   # 外部克隆论文源码隔离存放区
    └── .gitkeep                                   # 占位与桥接规范说明
```
*(注：`workspaces/`、`models/`、`__pycache__/`、`.pytest_cache/`、`*.egg-info/` 为运行时实验输出、本地大模型权重或编译缓存目录，已配置在 `.gitignore` 中，不纳入源码模块索引。)*

#### 2.2 逐层目录递归职责说明
- **`configs/`**：集中存放实验编排 YAML 文件，将输入输出路径（`io`）与各阶段算法名、超参数（`pipeline`）完全从 Python 源码中剥离。
- **`real2sim_gs/`**：项目主 Python 包根目录，导出顶层 SDK 入口与核心数据类型。
  - **`real2sim_gs/api/`**：对外边界适配层，严禁直接编写算法计算细节，全部委托 `core/` 与 `stages/` 完成。
  - **`real2sim_gs/ui/`**：零 Node.js 的纯静态 Web 3D 前端工作台（`Real2Sim-GS Studio`），由 `server.py` 直接挂载托管并支持 `real2sim ui` 一键打开。
  - **`real2sim_gs/core/`**：整个管线的微内核引擎，定义所有阶段的抽象基类契约、YAML 配置结构、环境依赖检查、工作区路径规范及 `manifest.json` 幂等跳过机制。
  - **`real2sim_gs/eval/`**：集中存放渲染图像与几何指标计算，为白盒训练与外部论文对比提供统一度量尺。
  - **`real2sim_gs/editor/`**：交互式 3DGS 物体分割与场景图编辑引擎，将 Windows 原生 SAM 3、B³-Seg 主动选视角、FlashSplat/GaussianCut/SAGD 3DGS 抬升以及非破坏性多物体刚体编辑（支持 Undo/Redo、单物体独立 `.ply` 导出与全场景烘焙）封装为独立可复用核心库。
  - **`real2sim_gs/utils/`**：提供无阶段依赖的纯数据容器（`GaussianCloud`）与跨工具标准的二进制 PLY 序列化。
  - **`real2sim_gs/stages/`**：所有可插拔业务逻辑与算法创新所在地，按管线执行顺序拆分为 5 个子目录：
    - **`ingest/`**：将原始 `.mp4` 视频或图片目录标准化为 `01_frames/frame_XXXX.png` 与 `frames_meta.json`。
    - **`pose/`**：计算或接驳每帧相机的内参及 OpenCV `world_to_cam` 4×4 外参矩阵，产出 `02_pose/cameras.json` 与 `init_points.ply`。
    - **`reconstruct/`**：消费 `01_frames/` 与 `02_pose/` 产出 `03_gaussians/raw_splats.ply` 与 `metrics.json`；其下 **`gsplat_whitebox/`** 子目录为完全解耦的白盒 PyTorch 3DGS 训练核心（改模型参数去 `model.py`、改损失函数去 `losses.py`、改分裂剪枝去 `strategy.py`）。
    - **`postprocess/`**：消费原始或上游高斯云，按配置列表顺序执行清洗、对齐、分割与碰撞体生成，产出至 `04_processed/`。
    - **`export/`**：读取 `active_splat_path()` 与 `collision_mesh.obj`，按目标引擎坐标系规范打包至 `05_export/<target>/`。
- **`tests/`**：基于 `pytest` 的无显卡/快速回归测试层，验证全链路、白盒训练张量梯度与外部 API 契约。
- **`third_party/`**：用于存放需保留原始结构或存在独立 Conda 依赖的外部开源论文仓库（如 2DGS、VGGT），由适配插件桥接调用。

---

### 3. 模块 -> 文件映射表

| 业务/功能模块 | 归属文件（相对根目录完整路径） |
| :--- | :--- |
| 项目构建、忽略与说明文档 | `pyproject.toml`, `.gitignore`, `AGENTS.md`, `third_party/.gitkeep` |
| YAML 实验配置模板 | `configs/default_mujoco.yaml`, `configs/pure_3dgs.yaml`, `configs/exp_2dgs_vggt.yaml` |
| 微内核调度、配置与工作区管理 | `real2sim_gs/__init__.py`, `real2sim_gs/core/__init__.py`, `real2sim_gs/core/config.py`, `real2sim_gs/core/stage.py`, `real2sim_gs/core/registry.py`, `real2sim_gs/core/workspace.py`, `real2sim_gs/core/pipeline.py`, `real2sim_gs/core/doctor.py`, `real2sim_gs/core/external.py`, `real2sim_gs/core/image_io.py` |
| 对外三层接口（CLI / SDK / HTTP） | `start_studio.bat`, `real2sim_gs/api/__init__.py`, `real2sim_gs/api/sdk.py`, `real2sim_gs/api/cli.py`, `real2sim_gs/api/server.py` |
| Web 3D 可视化前端 (`ui`) | `real2sim_gs/ui/__init__.py`, `real2sim_gs/ui/static/index.html`, `real2sim_gs/ui/static/styles.css`, `real2sim_gs/ui/static/viewer3d.js`, `real2sim_gs/ui/static/app.js` |
| 通用 3DGS 数据结构、PLY I/O 与示例视频生成 | `real2sim_gs/utils/__init__.py`, `real2sim_gs/utils/ply_io.py`, `real2sim_gs/utils/sample_video.py` |
| 渲染与定量评测指标 | `real2sim_gs/eval/__init__.py`, `real2sim_gs/eval/metrics.py` |
| 交互式 3DGS 物体分割与场景图编辑 (`editor`) | `real2sim_gs/editor/__init__.py`, `real2sim_gs/editor/sam3_backend.py`, `real2sim_gs/editor/segment_3d.py`, `real2sim_gs/editor/scene_graph.py` |
| Stage 1：视频抽帧与筛帧 (`ingest`) | `real2sim_gs/stages/__init__.py`, `real2sim_gs/stages/ingest/__init__.py`, `real2sim_gs/stages/ingest/ffmpeg_ingest.py`, `real2sim_gs/stages/ingest/mock_ingest.py` |
| Stage 2：相机位姿估计 (`pose`) | `real2sim_gs/stages/pose/__init__.py`, `real2sim_gs/stages/pose/_colmap_worker.py`, `real2sim_gs/stages/pose/colmap_pose.py`, `real2sim_gs/stages/pose/feed_forward_pose.py`, `real2sim_gs/stages/pose/mock_pose.py` |
| Stage 3：SOTA 白盒 3DGS 与外部重建 (`reconstruct`) | `real2sim_gs/stages/reconstruct/__init__.py`, `real2sim_gs/stages/reconstruct/gsplat_whitebox/__init__.py`, `real2sim_gs/stages/reconstruct/gsplat_whitebox/appearance.py`, `real2sim_gs/stages/reconstruct/gsplat_whitebox/dataset.py`, `real2sim_gs/stages/reconstruct/gsplat_whitebox/model.py`, `real2sim_gs/stages/reconstruct/gsplat_whitebox/losses.py`, `real2sim_gs/stages/reconstruct/gsplat_whitebox/strategy.py`, `real2sim_gs/stages/reconstruct/gsplat_whitebox/trainer.py`, `real2sim_gs/stages/reconstruct/subprocess_adapter.py`, `real2sim_gs/stages/reconstruct/twodgs_slot.py`, `real2sim_gs/stages/reconstruct/mock_reconstruct.py` |
| Stage 4：去噪、仿真对齐与网格化 (`postprocess`) | `real2sim_gs/stages/postprocess/__init__.py`, `real2sim_gs/stages/postprocess/clean_floaters.py`, `real2sim_gs/stages/postprocess/align_sim.py`, `real2sim_gs/stages/postprocess/extract_mesh.py`, `real2sim_gs/stages/postprocess/segment_slot.py` |
| Stage 5：纯 3DGS / MuJoCo / Unity 导出 (`export`) | `real2sim_gs/stages/export/__init__.py`, `real2sim_gs/stages/export/ply_exporter.py`, `real2sim_gs/stages/export/mujoco_exporter.py`, `real2sim_gs/stages/export/unity_exporter.py` |
| 自动化测试套件 | `tests/test_api_and_cli.py`, `tests/test_core_pipeline.py`, `tests/test_editor_segmentation.py`, `tests/test_pipeline_e2e.py`, `tests/test_postprocess_export.py`, `tests/test_trainer_fixes.py`, `tests/test_whitebox_gs.py` |

---

### 4. 功能 -> 代码级精确定位索引

| 具体功能点 | 文件路径与精确符号锚点（禁止硬编码行号） |
| :--- | :--- |
| YAML 解析与命令行 IO 路径覆写 | `real2sim_gs/core/config.py` -> `_StrictModel`, `IOConfig`, `StageSpec`, `PipelineStagesConfig`, `PipelineConfig`, `load_config()` |
| 阶段基类契约与返回结构 | `real2sim_gs/core/stage.py` -> `StageKind`, `StageResult`, `BaseStage` |
| 算法插件全局注册与工厂实例化 | `real2sim_gs/core/registry.py` -> `StageRegistry`, `REGISTRY`, `register_stage()`；`real2sim_gs/stages/__init__.py` -> `load_builtin_stages()` |
| 磁盘目录规范、`manifest.json` 签名缓存、协作式停止与断点路径 | `real2sim_gs/core/workspace.py` -> `STAGE_ORDER`, `PipelineCancelledError`, `StageRecord`, `atomic_write_text()`, `atomic_write_json()`, `SceneWorkspace`（`check_cancelled()`, `report_progress()`, `reconstruct_checkpoint_path`, `reconstruct_checkpoint_meta_path`, `active_splat_pointer_path`, `set_active_splat()`, `is_completed()`, `mark_completed()`, `invalidate_from()`, `active_splat_path()`） |
| 全流程串联、输入指纹校验与链式失效双层断点续跑、`--mock` | `real2sim_gs/core/pipeline.py` -> `ProgressCallback`, `PipelineRunResult`, `_input_fingerprint()`, `_normalize_for_signature()`, `_signature_for_specs()`, `_validate_stage_plan()`, `PipelineRunner.run()` |
| 跨平台外部进程安全桥接、Unicode 路径 I/O 与自然排序 | `real2sim_gs/core/external.py` -> `resolve_repo_dir()`, `build_command()`, `format_command()`, `kill_process_tree()`, `popen_process_group_kwargs()`, `run_external_command()`, `run_command_template()`<br>`real2sim_gs/core/image_io.py` -> `imread_unicode()`, `imwrite_unicode()`, `natural_sort_key()`, `natural_sorted()` |
| 本地 CUDA / FFmpeg / pycolmap / gsplat 编译链 / MuJoCo / SAM 3 环境体检 | `real2sim_gs/core/doctor.py` -> `CheckItem`, `is_gsplat_cuda_ready()`, `diagnose_environment()` |
| Python SDK、Typer CLI 与 FastAPI HTTP 调度/预览/编辑/清理/安全认证服务 | `real2sim_gs/api/sdk.py` -> `run_pipeline()`, `list_available_stages()`, `check_environment()`<br>`real2sim_gs/api/cli.py` -> `app`, `run_cmd()`, `doctor_cmd()`, `stages_cmd()`, `serve_cmd()`, `_resolve_studio_endpoint()`, `_wait_until_server_ready()`, `ui_cmd()`（及根目录一键启动脚本 `start_studio.bat`）<br>`real2sim_gs/api/server.py` -> `JobCreateRequest`, `WorkspaceActionRequest`, `WorkspaceDeleteRequest`, `EditorAlignSceneRequest`, `JobRecord`, `JobStore`（含 `_exec_lock`, `has_running_job()`, `request_cancel()`, `has_active_job_for_workspace()`, `active_input_paths()`, `prune_terminal()`, `begin_delete()`, `end_delete()`）, `_EDITOR_MANAGERS`, `_evict_all_render_models()`, `_load_workspace_render_model()`, `_compact_mesh_payload()`, `_extract_aligned_camera_presets()`, `_safe_read_json()`, `_build_scene_preview()`, `_get_cached_render_model()`, `_render_novel_view_jpeg()`, `_dir_size_bytes()`, `_intermediate_cache_paths()`, `_evict_render_cache_for_workspace()`, `_resolve_workspace_path()`, `_resolve_job_input_path()`, `_resolve_job_config_path()`, `_rmtree_strict()`, `_corrupted_workspace_entry()`, `_summarize_workspace_entry()`, `_build_diagnostic_report()`, `_execute_job()`, `create_app()`（含 `TrustedHostMiddleware`、`X-Studio-Token`/Cookie 鉴权、`_GPU_RENDER_LOCK` 与 `/api/doctor`, `/api/configs`, `/api/upload`, `/api/workspaces`, `/api/delete-workspace`, `/api/cleanup-uploads`, `/api/workspace-config`, `/api/open-workspace`, `/api/scene-preview`, `/api/render-view`, `/api/workspace-file`, `/api/workspace-bundle.zip`, `/api/editor/*`（`status`, `sam3-config`, `objects`, `preview-2d`, `segment-3d`, `refine-object`, `pick-object`, `update-object`, `object-action`, `bake`, `align-scene`, `export-object-ply`）, `/jobs/{job_id}/cancel`） |
| Studio Web 3D 交互工作台（默认白昼/历史档案与清理/上传分工作区/显存遥测/360° 视口/水平正方向校准/物体分割与编辑工具栏） | `real2sim_gs/ui/__init__.py` -> `UI_STATIC_DIR`<br>`real2sim_gs/ui/static/viewer3d.js` -> `wrapAngleRad()`, `Real2SimViewport3D`（含 `_createDemoShowcase()`, `_precomputeSplatAxes()`, `_clearNeuralImage()`, `_blobToImage()`, `_requestNeuralStream()`, `_pumpNeuralStream()`, `showDemoShowcase()`, `snapToCapturedCamera()`, `cycleCapturedCamera()`, `resetView()`, `focusOnBox()`, `setAlignCrosshair()`, `_drawAlignCrosshair()`, `_cameraBasis()`, `_drawCameraFrustums()`, `_drawOrientationGizmo()`, `setTheme()`, `setPresetView()` 及 2D 掩膜/点框交互与 3D AABB/Move·Rotate·Scale Gizmo）<br>`real2sim_gs/ui/static/app.js` -> `RECIPES`, `normWsPath()`, `syncInputPathFromWorkspace()`, `applyTheme()`, `updateRangeFill()`, `renderHistoryCards()`, `resumeWorkspaceRun()`, `openDeleteWorkspaceModal()`, `closeDeleteWorkspaceModal()`, `executeWorkspaceDelete()`, `cleanupUploadsCache()`, `updateLaunchBtnLabel()`, `selectHistoryWorkspace()`, `restoreWorkspaceConfig()`, `openWorkspaceFolder()`, `syncDemoToggleUI()`, `restoreFromDemoToScene()`, `applyRecipe()`, `evaluateComboAdvisor()`, `copyTextToClipboard()`, `showErrorBanner()`, `hideErrorBanner()`, `renderJobTelemetry()`, `renderLossSparkline()`, `setJobRunningUI()`, `cancelActiveJob()`, `applyConfigToForm()`, `collectConfigFromForm()`, `triggerPipelineRun()`, `loadScenePreview()`, `handleVideoUpload()`, `setAlignPopoverOpen()`, `runSceneAlign()`, `init()`（及 `🧭 水平/正方向` `alignPopover`、顶部 `editorToolbar` / 右侧 `editorDrawer` 交互逻辑） |
| 标准 3DGS 二进制/ASCII `.ply` 序列化（含 0..3 阶 SH 与 O(3) 变换）及示例环拍视频合成 | `real2sim_gs/utils/ply_io.py` -> `SH_C0`, `eval_real_sh_band()`, `eval_sh_rest_color()`, `sh_degree_from_rest()`, `transform_sh_rest()`, `transform_sh_degree1()`, `GaussianCloud`（含 `sh_rest_degree1()`）, `transform_gaussian_cloud()`, `_read_ply_vertices()`, `save_gaussian_ply()`, `load_gaussian_ply()`, `save_point_cloud_ply()`, `load_point_cloud_ply()`<br>`real2sim_gs/utils/sample_video.py` -> `_make_studio_floor()`, `_build_scene_boxes()`, `generate_sample_orbiter_video()` |
| 可微非截断 SSIM 与 PSNR/SSIM 定量评测 | `real2sim_gs/eval/metrics.py` -> `_gaussian_1d_kernels()`, `ssim_tensor()`, `compute_psnr()`, `compute_ssim()`, `ImagePairEvaluator`, `evaluate_image_pairs()` |
| 交互式 3DGS 物体分割与场景图编辑（SAM 3 + B³-Seg + FlashSplat + GaussianCut + SAGD） | `real2sim_gs/editor/__init__.py`、`real2sim_gs/editor/sam3_backend.py` -> `DEFAULT_SAM3_MODEL_ID`, `SAM3_MODEL_PATH_ENV`, `Sam3MaskPrediction`, `default_local_sam3_dir()`, `is_valid_local_sam3_dir()`, `resolve_sam3_model_id()`, `get_sam3_status()`, `Sam3Backend`（`_evict_other_models()`, `predict_image_prompt()`, `track_multiview()`, `configure()`, `unload()`）, `get_shared_sam3_backend()`<br>`real2sim_gs/editor/segment_3d.py` -> `SegmentLiftConfig`, `SegmentLiftResult`, `cloud_to_gpu_model()`, `build_orbit_camera_view()`, `project_world_point()`, `render_camera_rgb()`, `estimate_target_anchor_3d()`, `plan_active_views()`, `accumulate_view_mask_weights()`, `solve_flashsplat_scores()`, `refine_gaussiancut()`, `_decompose_boundary_gaussians()`, `lift_masks_to_3dgs()`<br>`real2sim_gs/editor/scene_graph.py` -> `OBJECT_PALETTE_RGB`, `SceneObject`, `apply_scene_edits_to_gpu_model()`, `pick_object_at_pixel()`, `SceneEditManager`, `export_single_object_ply()`, `bake_edited_scene()` |
| 4K 长视频限流可中断抽帧与时间滑窗 Laplacian 清晰度选帧 | `real2sim_gs/stages/ingest/ffmpeg_ingest.py` -> `_probe_video_duration()`, `_collect_candidate_frames()`, `_compute_sharpness()`, `_select_windowed_sharp_frames()`, `_write_selected_frames()`, `UniformFFmpegIngest`, `SharpnessFFmpegIngest`<br>`real2sim_gs/stages/ingest/mock_ingest.py` -> `MockIngestStage` |
| COLMAP / 零 torch 隔离 pycolmap 工作进程与长视频顺序匹配解算、前馈新论文适配槽位与 Mock 位姿 | `real2sim_gs/stages/pose/_colmap_worker.py` -> `clean_pose_workdir()`, `resolve_use_sequential()`, `run_pycolmap_worker()`<br>`real2sim_gs/stages/pose/colmap_pose.py` -> `_parse_colmap_cameras()`, `_parse_colmap_images()`, `_parse_colmap_points3d()`, `_read_source_kind_and_count()`, `_run_pycolmap_sfm()`, `_pick_largest_colmap_model()`, `_run_colmap_cli_sfm()`, `ColmapPoseStage`<br>`real2sim_gs/stages/pose/feed_forward_pose.py` -> `FeedForwardPoseAdapter`<br>`real2sim_gs/stages/pose/mock_pose.py` -> `_look_at_w2c()`, `MockPoseStage` |
| SOTA 白盒 PyTorch 3DGS（4K 限幅 `uint8`/2 GiB 预算/鱼眼与 `valid_mask` 去畸变/3D Mip-Splatting 抗混叠与烘焙/渐进 0..3 阶完整 SH 与 `s_max/s_mid` 针刺正则/`test_every=8` 留出验证集/旋转分裂 ADC 与重数感知 MCMC/`SparseAdam`+`SelectiveAdam` 状态同步与局部重置/冷却中段可愈合清理/完整 `params_signature` 断点恢复） | `real2sim_gs/stages/reconstruct/gsplat_whitebox/dataset.py` -> `CameraView`, `_resolve_camera_model()`, `_valid_mask_from_remap()`, `_undistort()`, `PoseImageDataset`<br>`real2sim_gs/stages/reconstruct/gsplat_whitebox/appearance.py` -> `BilateralGridModule`（含 `tv_loss()`, `identity_loss()`）, `CameraPoseOptModule`（含 `delta_reg_loss`）<br>`real2sim_gs/stages/reconstruct/gsplat_whitebox/model.py` -> `quat_to_rotmat()`, `robust_point_extent()`, `GaussianModel`（`from_point_cloud()`, `max_log_scale`, `clamp_scales_()`, `sh_degree`, `sh_coeffs`, `covariance_3d()`, `compute_3d_filter()`, `mip_filtered_primitives()`, `select_and_append_()`, `to_gaussian_cloud()`）<br>`real2sim_gs/stages/reconstruct/gsplat_whitebox/losses.py` -> `l1_loss()`, `ssim_loss()`, `flatness_reg_loss()`, `anisotropy_reg_loss()`, `mcmc_reg_loss()`, `edge_aware_depth_smoothness_loss()`, `compute_total_loss()`<br>`real2sim_gs/stages/reconstruct/gsplat_whitebox/strategy.py` -> `compute_mcmc_relocation()`, `DensityControlStrategy`（含 AbsGS 与旋转分裂/`reset_opacities()`）, `MCMCStrategy`（重数与梯度引导 `relocate_and_grow()` 含 `last_reset_indices`, `inject_langevin_noise()`）<br>`real2sim_gs/stages/reconstruct/gsplat_whitebox/trainer.py` -> `RenderOutput`, `SelectiveAdam`（含 `reset_param_state()`）, `_render_splats_pytorch()`, `_render_view()`, `_extract_per_gaussian_absgrad()`, `_compute_params_signature()`, `_save_reconstruct_checkpoint()`, `_sync_gs_optimizer_state()`, `_sync_mcmc_optimizer_state()`, `_reset_opacity_optimizer_state()`, `_write_refined_cameras_json()`, `_exp_decay_lr()`, `_estimate_init_bg_color()`, `GsplatWhiteboxStage` |
| 外部重建论文子进程桥接（2DGS/PGSR）与 Mock 场景 | `real2sim_gs/stages/reconstruct/subprocess_adapter.py` -> `SubprocessReconstructAdapter`<br>`real2sim_gs/stages/reconstruct/twodgs_slot.py` -> `TwoDGSSlotStage`<br>`real2sim_gs/stages/reconstruct/mock_reconstruct.py` -> `MockReconstructStage` |
| 漂浮噪点与细长飞针过滤（透明度 + `s_max/s_mid` 真针刺剔除 + `scale_aware`/`global`/`depth_normalized` KDTree SOR） | `real2sim_gs/stages/postprocess/clean_floaters.py` -> `CleanFloatersStage` |
| SOTA 3DGS surfel Cauchy-IRLS 找平（门控双墙+最低水平峰 `Z=0`）、曼哈顿 4 重正交 `-X` 正方向摆正与手动校准 | `real2sim_gs/stages/postprocess/align_sim.py` -> `_atomic_save_gaussian_ply()`, `_active_cameras_path()`, `_extract_camera_axes()`, `_extract_camera_gravity_and_heading()`, `_extract_camera_centers()`, `_fit_ground_plane_ransac()`, `_rotation_matrix_to_z_up()`, `_orient_normal_upward()`, `_extract_flat_surfels()`, `_estimate_gravity_surfel_irls()`, `_find_lowest_floor_height()`, `_snap_heading_manhattan()`, `_build_upright_front_rotation()`, `AlignGravityScaleStage`, `realign_workspace_scene()` |
| 前景核心聚焦物理碰撞网格提取（各向同性 3D Marching Cubes / 去天花板顶盖 2.5D 高度场 / 凸包）与原生 SAM 3 / 外部语义分割槽位 | `real2sim_gs/stages/postprocess/extract_mesh.py` -> `_core_points_and_bounds()`, `_build_heightfield_mesh()`, `_build_marching_cubes_mesh()`, `ExtractCollisionMeshStage`<br>`real2sim_gs/stages/postprocess/segment_slot.py` -> `SemanticSegmentSlotStage`（支持 `backend="sam3"` 原生批量文本 3DGS 分割与 `command_template` 外部桥接） |
| 纯 3DGS / MuJoCo / Unity 多目标导出 | `real2sim_gs/stages/export/ply_exporter.py` -> `_R_ZUP_TO_SUPERSPLAT_PLY`, `_active_cameras_path()`, `_load_alignment_similarity()`, `_compute_aligned_foreground_pivot()`, `_trim_extreme_bound_outliers()`, `_export_transformed_cameras_json()`, `Ply3DGSExporterStage`<br>`real2sim_gs/stages/export/mujoco_exporter.py` -> `_esc_attr()`, `_normalize_friction()`, `_decompose_mesh_into_convex_parts()`, `_compute_robot_spawn()`, `_build_mjcf_xml()`, `MujocoExporterStage`<br>`real2sim_gs/stages/export/unity_exporter.py` -> `UnityExporterStage` |
| 端到端全链路、内核隔离/安全认证、SOTA 白盒训练修复、交互式 3DGS 分割编辑与几何后处理/导出全套单元测试（56 项） | `tests/test_pipeline_e2e.py` -> `test_mock_pipeline_end_to_end_and_checkpoint_resume()`, `test_unity_export_in_pipeline()`, `test_pure_3dgs_preset_pipeline()`, `test_supersplat_ply_export_orientation_and_centering()`, `test_input_change_invalidates_ingest_cache()`<br>`tests/test_core_pipeline.py` -> `test_chained_invalidation_and_mid_run_cancel()`, `test_corrupt_manifest_recovery_and_strict_config()`, `test_unicode_opencv_io_and_natural_sort()`, `test_colmap_camera_layouts_and_empty_points2d_images()`, `test_external_command_spaces_chinese_and_cancel()`, `test_active_splat_pointer_and_worker_no_torch()`, `test_pipeline_preserves_workspace_on_missing_input_and_skips_cleaned_upload()`<br>`tests/test_trainer_fixes.py` -> `test_mcmc_relocation_multiplicity_and_parent_momentum_reset()`, `test_train_test_holdout_split_and_refined_cameras()`, `test_checkpoint_complete_state_resume_and_param_change_reset()`, `test_adc_rotated_split_parent_removed_and_survivor_moments()`, `test_pytorch_fallback_renders_beyond_1024_splats()`, `test_mip_3d_filter_consistency_and_bake()`<br>`tests/test_whitebox_gs.py` -> `test_gaussian_ply_roundtrip()`, `test_whitebox_sota_mcmc_bilateral_pose_opt()`, `test_density_control_absgs_and_opacity_reset()`, `test_mcmc_scale_clamp_and_optimizer_momentum_sync()`, `test_chunked_pytorch_ewa_renderer_consistency()`, `test_gsplat_native_cuda_backend()`, `test_gsplat_packed_background_compositing()`, `test_whitebox_cancel_and_checkpoint_resume()`, `test_pose_image_dataset_4k_downscale_and_uint8_storage()`, `test_align_gravity_rejects_dominant_vertical_wall()`, `test_clean_floaters_prunes_elongated_needle_spikes()`, `test_anisotropy_loss_allows_flat_surfels_and_penalizes_needles()`, `test_clean_floaters_preserves_flat_surfels_and_sparse_walls()`<br>`tests/test_postprocess_export.py` -> `test_sh_degree1_and_degree3_o3_invariance()`, `test_covariance_invariance_align_ply_unity()`, `test_align_gravity_pitched_forward_cameras_40_deg()`, `test_postprocess_order_swap_align_before_clean()`, `test_isotropic_marching_cubes_and_depth_normalized_clean()`, `test_ply_io_degree3_big_endian_pre_element_and_truncated()`, `test_mujoco_xml_escaping_plane_only_and_ceiling_safe_spawn()`, `test_align_gravity_indoor_room_floor_not_desk_or_ceiling()`, `test_heightfield_strips_indoor_ceiling_lid()`, `test_align_gravity_surfel_irls_and_manhattan_heading()`, `test_align_gravity_tilted_monitor_desk_not_hijacked()`, `test_realign_set_front_and_manual_delta_preserves_sh3_and_cameras()`<br>`tests/test_api_and_cli.py` -> `test_cli_doctor_stages_and_mock_run()`, `test_fastapi_endpoints_and_studio_ui()`, `test_studio_port_resolution_and_launcher_script()`, `test_job_failure_diagnostic_report_and_sample_video()`, `test_workspace_cleanup_delete_and_cancel_endpoints()`, `test_api_security_path_traversal_and_token_guards()`, `test_chinese_workspace_zip_and_corrupt_manifest_resilience()`<br>`tests/test_editor_segmentation.py` -> `test_3dgs_lifting_occlusion_gaussiancut_and_sagd()`, `test_scene_graph_sh3_rotation_duplicate_refine_undo_export_bake()`, `test_fastapi_editor_endpoints_and_ui_toolbar()`, `test_editor_bugfix_regressions()`, `test_api_editor_align_scene_modes_and_auth()` |
