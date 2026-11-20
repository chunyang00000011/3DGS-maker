from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckItem:
    """Diagnostic status for a runtime tool, library, or hardware backend."""

    component: str
    available: bool
    detail: str
    install_hint: str = ""


@functools.lru_cache(maxsize=1)
def is_gsplat_cuda_ready() -> tuple[bool, bool, str]:
    """Check if gsplat Python package AND compiled/JIT CUDA extension toolchain are usable.

    Returns (pkg_installed, cuda_ready, detail_message). Cached per process.
    """
    pkg_installed = importlib.util.find_spec("gsplat") is not None
    if not pkg_installed:
        return False, False, "未安装（白盒训练器将自动使用内置分块 GPU EWA 渲染器）"

    import torch

    if importlib.util.find_spec("gsplat.csrc") is not None and torch.cuda.is_available():
        return True, True, "gsplat 预编译 CUDA 光栅化算子已就绪"

    has_nvcc = shutil.which("nvcc") is not None or bool(os.environ.get("CUDA_HOME"))
    if sys.platform == "win32":
        has_cxx = shutil.which("cl") is not None
    else:
        has_cxx = shutil.which("g++") is not None or shutil.which("c++") is not None

    if torch.cuda.is_available() and has_nvcc and has_cxx:
        return True, True, "gsplat 包与本地 CUDA/C++ JIT 编译环境已就绪"

    return (
        True,
        False,
        "已安装 gsplat 包，但本机缺少 CUDA Toolkit (nvcc) 或 VS C++ (cl.exe) 编译链（当前自动启用内置分块 GPU EWA 渲染）",
    )


def diagnose_environment() -> list[CheckItem]:
    """Inspect local Python, CUDA GPU, CLI tools, and optional simulation dependencies."""
    import torch

    items: list[CheckItem] = []

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    items.append(CheckItem("Python", True, f"{py_ver} ({sys.executable})"))

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        items.append(
            CheckItem(
                "PyTorch + CUDA",
                True,
                f"torch {torch.__version__}, GPU: {gpu_name} ({vram_gb:.1f} GB)",
            )
        )
    else:
        items.append(
            CheckItem(
                "PyTorch + CUDA",
                False,
                f"torch {torch.__version__} (未检测到可用 CUDA GPU，当前将回退 CPU 模式)",
                "安装带 CUDA 的 PyTorch，或在 AutoDL Linux GPU 服务器执行训练",
            )
        )

    ffmpeg_path = shutil.which("ffmpeg")
    items.append(
        CheckItem(
            "FFmpeg (视频抽帧)",
            ffmpeg_path is not None,
            ffmpeg_path or "未在 PATH 中找到 ffmpeg",
            "" if ffmpeg_path else "Windows 可运行 `winget install Gyan.FFmpeg`；Linux 运行 `apt install ffmpeg`",
        )
    )

    has_pycolmap = importlib.util.find_spec("pycolmap") is not None
    colmap_path = shutil.which("colmap")
    if has_pycolmap:
        import pycolmap

        items.append(
            CheckItem(
                "COLMAP (SfM 位姿解算)",
                True,
                f"pycolmap {pycolmap.__version__} (内置原生 C++ SfM 引擎已就绪)",
            )
        )
    else:
        items.append(
            CheckItem(
                "COLMAP (SfM 位姿解算)",
                colmap_path is not None,
                colmap_path or "未安装 pycolmap 或 colmap CLI（可用 mock 或 feedforward_slot 替代）",
                "" if colmap_path else "运行 `pip install pycolmap` 或安装官方 COLMAP 加入 PATH",
            )
        )

    gsplat_pkg, gsplat_ready, gsplat_detail = is_gsplat_cuda_ready()
    gsplat_hint = ""
    if not gsplat_pkg:
        gsplat_hint = "pip install gsplat（Linux/AutoDL 或装有 VS C++ + CUDA Toolkit 的 Windows 可编译加速）"
    elif not gsplat_ready:
        gsplat_hint = "若需本机 JIT 编译加速，需安装 VS2022「使用 C++ 的桌面开发」与对应 CUDA Toolkit；否则直接使用内置分块 GPU 渲染"
    items.append(CheckItem("gsplat (高速 CUDA 光栅化)", gsplat_ready, gsplat_detail, gsplat_hint))

    has_geom = all(importlib.util.find_spec(pkg) is not None for pkg in ("trimesh", "scipy"))
    items.append(
        CheckItem(
            "SciPy + Trimesh (后处理几何)",
            has_geom,
            "可用（支持 KDTree 去噪、RANSAC 对齐与碰撞网格提取）" if has_geom else "缺少 scipy 或 trimesh",
            "" if has_geom else "pip install scipy trimesh",
        )
    )

    has_mujoco = importlib.util.find_spec("mujoco") is not None
    items.append(
        CheckItem(
            "MuJoCo (本地仿真校验)",
            has_mujoco,
            "已安装（可直接加载导出的 scene.xml）" if has_mujoco else "未安装（不影响生成 scene.xml 资产包，仅跳过本地即时预览）",
            "" if has_mujoco else "pip install mujoco mujoco-warp",
        )
    )

    has_transformers = importlib.util.find_spec("transformers") is not None
    if not has_transformers:
        items.append(
            CheckItem(
                "SAM 3 (交互式 3DGS 分割)",
                False,
                "未安装 transformers（缺少 SAM 3 开放词汇与点选追踪推理引擎）",
                "运行 `pip install -U transformers accelerate safetensors`，并执行 `hf auth login` 授权访问 facebook/sam3",
            )
        )
    else:
        import transformers

        sam3_ready = all(
            hasattr(transformers, cls_name)
            for cls_name in ("Sam3Model", "Sam3TrackerModel")
        )
        local_sam3 = os.environ.get("REAL2SIM_SAM3_MODEL_PATH", "").strip()
        if not local_sam3:
            auto_dir = Path(__file__).resolve().parents[2] / "models" / "sam3"
            if (auto_dir / "config.json").is_file() and (auto_dir / "model.safetensors").is_file():
                local_sam3 = str(auto_dir)
        hf_token_present = False
        if importlib.util.find_spec("huggingface_hub") is not None:
            try:
                import huggingface_hub

                if hasattr(huggingface_hub, "get_token"):
                    hf_token_present = bool(huggingface_hub.get_token())
            except Exception:  # noqa: BLE001
                hf_token_present = False

        if not sam3_ready:
            items.append(
                CheckItem(
                    "SAM 3 (交互式 3DGS 分割)",
                    False,
                    f"已安装 transformers {transformers.__version__}，但未检测到 Sam3Model / Sam3TrackerModel",
                    "请升级至支持 SAM 3 的最新版本：`pip install -U transformers accelerate safetensors`",
                )
            )
        elif local_sam3:
            local_ok = os.path.isdir(local_sam3)
            items.append(
                CheckItem(
                    "SAM 3 (交互式 3DGS 分割)",
                    local_ok,
                    f"transformers {transformers.__version__} | 本地 SAM 3 权重已就绪: {local_sam3}"
                    if local_ok
                    else f"transformers {transformers.__version__} | REAL2SIM_SAM3_MODEL_PATH 路径不存在: {local_sam3}",
                    "" if local_ok else "检查 REAL2SIM_SAM3_MODEL_PATH 路径或在 Studio 中配置本地权重目录",
                )
            )
        elif hf_token_present:
            items.append(
                CheckItem(
                    "SAM 3 (交互式 3DGS 分割)",
                    True,
                    f"transformers {transformers.__version__} 已就绪（已检测到 Hugging Face 登录令牌，可直接拉取 facebook/sam3）",
                )
            )
        else:
            items.append(
                CheckItem(
                    "SAM 3 (交互式 3DGS 分割)",
                    True,
                    f"transformers {transformers.__version__} 已就绪（未检测到 HF Token；拉取 gated 模型需先登录或指定本地权重）",
                    "执行 `hf auth login` 并在 https://huggingface.co/facebook/sam3 同意模型协议，或在 Studio 填入本地权重路径",
                )
            )

    return items
