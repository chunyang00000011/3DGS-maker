from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
# 每颜色通道 f_rest 系数个数 K -> SH 阶数（Inria 布局：f_rest 按通道主序 [R(K), G(K), B(K)] 排列）
_SH_REST_K_TO_DEGREE = {3: 1, 8: 2, 15: 3}
# 一阶实球谐（gsplat/Inria 约定 [-y, z, -x]）与 xyz 之间的置换矩阵
_SH_P = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
# 采样最小二乘求解高阶 SH 旋转矩阵所用的固定方向集合（确定性种子，保证可复现）
_SH_FIT_DIRS = np.random.default_rng(20240607).normal(size=(96, 3))
_SH_FIT_DIRS /= np.linalg.norm(_SH_FIT_DIRS, axis=1, keepdims=True)


def eval_real_sh_band(degree: int, dirs: np.ndarray) -> np.ndarray:
    """Evaluate the gsplat/Inria real SH basis of a single band `degree` (1..3) at unit directions (M,3)."""
    d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    if degree == 1:
        return np.stack([-SH_C1 * y, SH_C1 * z, -SH_C1 * x], axis=1)
    xx, yy, zz = x * x, y * y, z * z
    if degree == 2:
        return np.stack(
            [
                SH_C2[0] * x * y,
                SH_C2[1] * y * z,
                SH_C2[2] * (2.0 * zz - xx - yy),
                SH_C2[3] * x * z,
                SH_C2[4] * (xx - yy),
            ],
            axis=1,
        )
    if degree == 3:
        return np.stack(
            [
                SH_C3[0] * y * (3.0 * xx - yy),
                SH_C3[1] * x * y * z,
                SH_C3[2] * y * (4.0 * zz - xx - yy),
                SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                SH_C3[4] * x * (4.0 * zz - xx - yy),
                SH_C3[5] * z * (xx - yy),
                SH_C3[6] * x * (xx - 3.0 * yy),
            ],
            axis=1,
        )
    raise ValueError(f"Unsupported SH band degree {degree} (expected 1..3).")


def eval_sh_rest_color(sh_rest: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Evaluate the view-dependent (non-DC) SH colour term for every splat along matching directions (N,3)."""
    arr = np.asarray(sh_rest, dtype=np.float64)
    n = arr.shape[0]
    k = arr.shape[1] // 3
    degree = _SH_REST_K_TO_DEGREE[k]
    coeffs = arr.reshape(n, 3, k)
    basis = np.concatenate([eval_real_sh_band(level, dirs) for level in range(1, degree + 1)], axis=1)
    return np.einsum("nck,nk->nc", coeffs, basis.reshape(n, k))


def _sh_band_transform(degree: int, linear_3x3: np.ndarray) -> np.ndarray:
    """Return X such that coef' = X @ coef satisfies c'(M d) == c(d) for orthogonal M (rotation or reflection)."""
    m = np.asarray(linear_3x3, dtype=np.float64).reshape(3, 3)
    if degree == 1:
        return _SH_P @ m @ _SH_P.T
    # c(M^T d) = coef · Y(M^T d)；Y(M^T d) 与 Y(d) 同属该阶不可约表示空间，最小二乘即可精确求解
    a = eval_real_sh_band(degree, _SH_FIT_DIRS)
    b = eval_real_sh_band(degree, _SH_FIT_DIRS @ m)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    return x


def sh_degree_from_rest(sh_rest: np.ndarray | None) -> int:
    """Infer SH degree (0..3) from flattened channel-major f_rest coefficients (N, 3K)."""
    if sh_rest is None:
        return 0
    k = int(np.asarray(sh_rest).shape[-1]) // 3
    return _SH_REST_K_TO_DEGREE.get(k, 0)


def transform_sh_rest(sh_rest: np.ndarray | None, linear_3x3: np.ndarray) -> np.ndarray | None:
    """Transform Inria f_rest SH coefficients (N, 3K), K in {3, 8, 15}, under an orthogonal 3x3 transform."""
    if sh_rest is None:
        return None
    arr = np.asarray(sh_rest, dtype=np.float64)
    n, cols = arr.shape[0], arr.shape[1] if arr.ndim == 2 else 0
    k = cols // 3
    if k not in _SH_REST_K_TO_DEGREE:
        raise ValueError(f"Unsupported f_rest coefficient count {cols}; expected 9, 24 or 45.")
    if n == 0:
        return np.zeros((0, cols), dtype=np.float32)
    coeffs = arr.reshape(n, 3, k)
    out = np.empty_like(coeffs)
    start = 0
    for level in range(1, _SH_REST_K_TO_DEGREE[k] + 1):
        width = 2 * level + 1
        x_mat = _sh_band_transform(level, linear_3x3)
        out[:, :, start : start + width] = np.einsum("ij,ncj->nci", x_mat, coeffs[:, :, start : start + width])
        start += width
    return np.ascontiguousarray(out.reshape(n, cols), dtype=np.float32)


def transform_sh_degree1(sh_rest: np.ndarray | None, linear_3x3: np.ndarray) -> np.ndarray | None:
    """Transform Inria degree-1 SH coefficients (N, 9) under an orthogonal 3x3 coordinate transform."""
    if sh_rest is None:
        return None
    arr = np.asarray(sh_rest, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 9).copy()
    return transform_sh_rest(arr.reshape(-1, 9), linear_3x3)


def _quats_wxyz_to_matrices(quats_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quats_wxyz, dtype=np.float64).reshape(-1, 4)
    norms = np.linalg.norm(q, axis=1)
    q = q.copy()
    q[norms < 1e-12] = np.array([1.0, 0.0, 0.0, 0.0])
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def _matrices_to_quats_wxyz(mats: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(mats).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    wxyz[wxyz[:, 0] < 0.0] *= -1.0
    return wxyz.astype(np.float32)


@dataclass
class GaussianCloud:
    """Explicit 3D Gaussian Splatting cloud container matching standard PLY attributes.

    `sh_rest` 保存完整 f_rest 系数，形状 (N, 3K)，按 Inria 通道主序 [R(K), G(K), B(K)] 展平，
    K ∈ {3, 8, 15} 分别对应 SH 阶数 1/2/3；可用 `sh_rest_coeffs()` 取得 (N, 3, K) 视图，
    仅支持一阶的渲染路径请使用 `sh_rest_degree1()`。
    """

    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh_dc: np.ndarray
    sh_rest: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.means = np.ascontiguousarray(self.means, dtype=np.float32).reshape(-1, 3)
        n = self.means.shape[0]
        self.scales = np.ascontiguousarray(self.scales, dtype=np.float32).reshape(-1, 3)
        self.quats = np.ascontiguousarray(self.quats, dtype=np.float32).reshape(-1, 4)
        self.opacities = np.ascontiguousarray(self.opacities, dtype=np.float32).reshape(-1)
        self.sh_dc = np.ascontiguousarray(self.sh_dc, dtype=np.float32).reshape(-1, 3)
        if self.sh_rest is not None:
            rest = np.ascontiguousarray(self.sh_rest, dtype=np.float32).reshape(n, -1)
            if rest.shape[1] // 3 not in _SH_REST_K_TO_DEGREE or rest.shape[1] % 3 != 0:
                raise ValueError(f"sh_rest must have 9, 24 or 45 columns per splat, got {rest.shape[1]}.")
            self.sh_rest = rest

    @property
    def num_points(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        return sh_degree_from_rest(self.sh_rest)

    def sh_rest_coeffs(self) -> np.ndarray | None:
        """Return f_rest as (N, 3, K) channel-major view."""
        if self.sh_rest is None:
            return None
        return self.sh_rest.reshape(self.num_points, 3, -1)

    def sh_rest_degree1(self) -> np.ndarray | None:
        """Return degree-1 truncated f_rest (N, 9) for renderers that only support SH degree 1."""
        coeffs = self.sh_rest_coeffs()
        if coeffs is None:
            return None
        return np.ascontiguousarray(coeffs[:, :, :3].reshape(self.num_points, 9))

    def sigmoid_opacities(self) -> np.ndarray:
        """Convert logit opacities to [0, 1] alpha values."""
        clipped = np.clip(self.opacities, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def rgb_colors(self) -> np.ndarray:
        """Convert degree-0 spherical harmonics to [0, 1] RGB colors."""
        return np.clip(self.sh_dc * SH_C0 + 0.5, 0.0, 1.0)

    def subset(self, mask: np.ndarray) -> GaussianCloud:
        """Return a filtered copy keeping entries where mask is True."""
        return GaussianCloud(
            means=self.means[mask].copy(),
            scales=self.scales[mask].copy(),
            quats=self.quats[mask].copy(),
            opacities=self.opacities[mask].copy(),
            sh_dc=self.sh_dc[mask].copy(),
            sh_rest=self.sh_rest[mask].copy() if self.sh_rest is not None else None,
        )

    def clone(self) -> GaussianCloud:
        return GaussianCloud(
            means=self.means.copy(),
            scales=self.scales.copy(),
            quats=self.quats.copy(),
            opacities=self.opacities.copy(),
            sh_dc=self.sh_dc.copy(),
            sh_rest=self.sh_rest.copy() if self.sh_rest is not None else None,
        )


def transform_gaussian_cloud(
    cloud: GaussianCloud,
    linear_3x3: np.ndarray,
    translation: np.ndarray | None = None,
    scale: float = 1.0,
) -> GaussianCloud:
    """Apply x' = scale * (M x) + t to every Gaussian, with M orthogonal (det = +1 rotation or det = -1 reflection).

    协方差严格满足 Σ' = scale² · M Σ Mᵀ：旋转 R' = M R；镜像时 R' = M R diag(-1,1,1) 保持 det=+1，
    因 diag(±1) S² diag(±1) = S² 不改变尺度。SH 系数按 O(3) 表示逐阶变换（支持 1~3 阶）。
    """
    m = np.asarray(linear_3x3, dtype=np.float64).reshape(3, 3)
    if not np.allclose(m @ m.T, np.eye(3), atol=1e-4):
        raise ValueError("transform_gaussian_cloud expects an orthogonal 3x3 matrix.")
    if scale <= 0.0:
        raise ValueError("scale must be strictly positive.")
    t = np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation, dtype=np.float64).reshape(3)

    means = (cloud.means.astype(np.float64) @ m.T) * scale + t[None, :]
    if cloud.num_points > 0:
        rot_new = np.einsum("ij,njk->nik", m, _quats_wxyz_to_matrices(cloud.quats))
        if np.linalg.det(m) < 0.0:
            rot_new = rot_new @ np.diag([-1.0, 1.0, 1.0])
        quats = _matrices_to_quats_wxyz(rot_new)
    else:
        quats = cloud.quats.copy()
    return GaussianCloud(
        means=means,
        scales=cloud.scales + np.float32(np.log(scale)),
        quats=quats,
        opacities=cloud.opacities.copy(),
        sh_dc=cloud.sh_dc.copy(),
        sh_rest=transform_sh_rest(cloud.sh_rest, m),
    )


_PLY_BASE_TYPES = {
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "u2",
    "uint16": "u2",
    "short": "i2",
    "int16": "i2",
    "uint": "u4",
    "uint32": "u4",
    "int": "i4",
    "int32": "i4",
}
_PLY_ENDIAN = {"binary_little_endian": "<", "binary_big_endian": ">", "ascii": "<"}


def _read_ply_vertices(path: Path) -> np.ndarray:
    """Parse ASCII / binary little- or big-endian PLY and return the vertex table as native-endian structured array."""
    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Invalid PLY file (missing end_header): {path}")
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        body = f.read()

    if not header_lines or header_lines[0] != "ply":
        raise ValueError(f"Invalid PLY file (missing 'ply' magic): {path}")

    fmt = ""
    # elements: [(name, count, [(prop_name, base_type)], has_list)]
    elements: list[tuple[str, int, list[tuple[str, str]], bool]] = []
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            elements.append((parts[1], int(parts[2]), [], False))
        elif parts[0] == "property" and elements:
            name, count, props, has_list = elements[-1]
            if parts[1] == "list":
                elements[-1] = (name, count, props, True)
                continue
            ply_type, prop_name = parts[1], parts[2]
            if ply_type not in _PLY_BASE_TYPES:
                raise ValueError(f"Unsupported PLY property type '{ply_type}' in {path}")
            props.append((prop_name, _PLY_BASE_TYPES[ply_type]))

    if fmt not in _PLY_ENDIAN:
        raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")
    vertex_idx = next((i for i, e in enumerate(elements) if e[0] == "vertex"), None)
    if vertex_idx is None:
        raise ValueError(f"PLY file has no vertex element: {path}")
    _, vertex_count, vertex_props, vertex_has_list = elements[vertex_idx]
    if vertex_has_list:
        raise ValueError(f"Unsupported list property in vertex element: {path}")

    endian = _PLY_ENDIAN[fmt]
    dtype = np.dtype([(name, endian + base) for name, base in vertex_props])

    if fmt == "ascii":
        lines = body.decode("ascii", errors="replace").splitlines()
        offset = sum(e[1] for e in elements[:vertex_idx])
        rows_txt = lines[offset : offset + vertex_count]
        if len(rows_txt) < vertex_count:
            raise ValueError(
                f"Truncated ASCII PLY {path}: expected {vertex_count} vertex rows, found {len(rows_txt)}."
            )
        rows = [tuple(r.split()[: len(vertex_props)]) for r in rows_txt]
        return np.array(rows, dtype=dtype) if rows else np.zeros(0, dtype=dtype)

    offset = 0
    for name, count, props, has_list in elements[:vertex_idx]:
        if has_list:
            raise ValueError(
                f"Unsupported PLY layout in {path}: element '{name}' with list properties precedes vertex data."
            )
        offset += count * np.dtype([(p, endian + b) for p, b in props]).itemsize
    need = offset + vertex_count * dtype.itemsize
    if len(body) < need:
        raise ValueError(
            f"Truncated PLY file {path}: vertex data needs {need} bytes after header, only {len(body)} present."
        )
    data = np.frombuffer(body, dtype=dtype, count=vertex_count, offset=offset)
    return data.astype(dtype.newbyteorder("="))


def save_gaussian_ply(path: Path | str, cloud: GaussianCloud) -> Path:
    """Write a GaussianCloud to a standard Inria/SuperSplat/MJWarp binary little-endian PLY."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = cloud.num_points
    num_rest = 0 if cloud.sh_rest is None else int(cloud.sh_rest.shape[1])
    prop_names = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        *[f"f_rest_{i}" for i in range(num_rest)],
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = np.dtype([(name, "<f4") for name in prop_names])
    arr = np.zeros(n, dtype=dtype)
    if n > 0:
        arr["x"], arr["y"], arr["z"] = cloud.means[:, 0], cloud.means[:, 1], cloud.means[:, 2]
        for c in range(3):
            arr[f"f_dc_{c}"] = cloud.sh_dc[:, c]
        for b in range(num_rest):
            arr[f"f_rest_{b}"] = cloud.sh_rest[:, b]
        arr["opacity"] = cloud.opacities
        for c in range(3):
            arr[f"scale_{c}"] = cloud.scales[:, c]
        for c in range(4):
            arr[f"rot_{c}"] = cloud.quats[:, c]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        *[f"property float {name}" for name in prop_names],
        "end_header\n",
    ]
    with open(out_path, "wb") as f:
        f.write("\n".join(header_lines).encode("ascii"))
        f.write(arr.tobytes())
    return out_path


def load_gaussian_ply(path: Path | str) -> GaussianCloud:
    """Load a standard 3DGS PLY file into a GaussianCloud (keeps full SH up to degree 3)."""
    in_path = Path(path)
    arr = _read_ply_vertices(in_path)
    names = set(arr.dtype.names or ())
    n = len(arr)

    means = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32) if n else np.zeros((0, 3), dtype=np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh_dc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float32)
    elif {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.float32) / 255.0
        sh_dc = (rgb - 0.5) / SH_C0
    else:
        sh_dc = np.zeros((n, 3), dtype=np.float32)

    opacities = arr["opacity"].astype(np.float32) if "opacity" in names else np.zeros(n, dtype=np.float32)

    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        scales = np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1).astype(np.float32)
    else:
        scales = np.full((n, 3), -4.0, dtype=np.float32)

    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        quats = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], axis=1).astype(np.float32)
    else:
        quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))

    rest_indices = sorted(
        int(name.removeprefix("f_rest_"))
        for name in names
        if name.startswith("f_rest_") and name.removeprefix("f_rest_").isdigit()
    )
    sh_rest: np.ndarray | None = None
    if len(rest_indices) >= 9 and len(rest_indices) % 3 == 0 and rest_indices == list(range(len(rest_indices))):
        k_file = len(rest_indices) // 3
        k_keep = max(k for k in _SH_REST_K_TO_DEGREE if k <= k_file)
        if k_keep != k_file:
            logger.warning(
                "PLY %s carries %d SH coefficients per channel; truncating to degree %d (%d coefficients).",
                in_path, k_file, _SH_REST_K_TO_DEGREE[k_keep], k_keep,
            )
        pick = [ch * k_file + b for ch in range(3) for b in range(k_keep)]
        sh_rest = (
            np.stack([arr[f"f_rest_{idx}"] for idx in pick], axis=1).astype(np.float32)
            if n
            else np.zeros((0, 3 * k_keep), dtype=np.float32)
        )

    return GaussianCloud(means=means, scales=scales, quats=quats, opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest)


def save_point_cloud_ply(path: Path | str, points: np.ndarray, colors: np.ndarray | None = None) -> Path:
    """Save an XYZ(+RGB) sparse or dense point cloud to binary PLY."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = pts.shape[0]
    if colors is None:
        rgb = np.full((n, 3), 180, dtype=np.uint8)
    else:
        col_arr = np.asarray(colors)
        if np.issubdtype(col_arr.dtype, np.floating):
            rgb = np.clip(col_arr * 255.0, 0.0, 255.0).astype(np.uint8).reshape(-1, 3)
        else:
            rgb = np.clip(col_arr, 0, 255).astype(np.uint8).reshape(-1, 3)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "<u1"),
            ("green", "<u1"),
            ("blue", "<u1"),
        ]
    )
    arr = np.zeros(n, dtype=dtype)
    if n > 0:
        arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n",
        ]
    )
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
    return out_path


def load_point_cloud_ply(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load an XYZ(+RGB) PLY point cloud returning (points[N,3] float32, rgb_0_1[N,3] float32)."""
    arr = _read_ply_vertices(Path(path))
    n = len(arr)
    names = set(arr.dtype.names or ())
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.float32) / 255.0
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip(sh * SH_C0 + 0.5, 0.0, 1.0)
    else:
        rgb = np.full((n, 3), 0.7, dtype=np.float32)
    return pts, rgb
