from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np

from real2sim_gs.core.image_io import imwrite_unicode


def _look_at_w2c(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / np.linalg.norm(right)
    cam_down = np.cross(forward, right)
    r = np.stack([right, cam_down, forward], axis=0).astype(np.float32)
    t = (-r @ eye).astype(np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = r
    w2c[:3, 3] = t
    return w2c


def _clip_polygon_near_z(poly: np.ndarray, z_near: float = 0.08) -> np.ndarray:
    """Sutherland-Hodgman clip of a 3D convex polygon against camera half-space z >= z_near."""
    pts = [poly[i] for i in range(len(poly))]
    out: list[np.ndarray] = []
    n = len(pts)
    for i in range(n):
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        cur_in = float(cur[2]) >= z_near
        nxt_in = float(nxt[2]) >= z_near
        if cur_in and nxt_in:
            out.append(nxt)
        elif cur_in and not nxt_in:
            t = (z_near - cur[2]) / (nxt[2] - cur[2])
            out.append(cur + t * (nxt - cur))
        elif not cur_in and nxt_in:
            t = (z_near - cur[2]) / (nxt[2] - cur[2])
            out.append(cur + t * (nxt - cur))
            out.append(nxt)
    return np.stack(out, axis=0) if len(out) >= 3 else np.zeros((0, 3), dtype=np.float32)


def _make_face_texture(seed: int, base_bgr: tuple[int, int, int], size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = np.full((size, size, 3), base_bgr, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    cell = ((xx // 16) + (yy // 16)) % 2 == 0
    tex[cell] = np.clip(tex[cell].astype(np.int32) - 32, 0, 255).astype(np.uint8)
    tex = np.clip(
        tex.astype(np.int32) + rng.integers(-22, 23, size=(size, size, 3)),
        0,
        255,
    ).astype(np.uint8)
    for _ in range(18):
        cx, cy = int(rng.integers(10, size - 10)), int(rng.integers(10, size - 10))
        rad = int(rng.integers(4, 11))
        col = tuple(int(c) for c in rng.integers(30, 245, size=3))
        cv2.circle(tex, (cx, cy), rad, col, -1)
        cv2.circle(tex, (cx, cy), max(2, rad // 2), (255 - col[0], 255 - col[1], 255 - col[2]), -1)
    cv2.rectangle(tex, (2, 2), (size - 3, size - 3), (235, 240, 248), 2)
    return tex


def _make_ground_texture(size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(101)
    tex = np.full((size, size, 3), (78, 86, 98), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    tile = ((xx // 32) + (yy // 32)) % 2 == 0
    tex[tile] = (118, 128, 142)
    tex = np.clip(tex.astype(np.int32) + rng.integers(-28, 29, size=(size, size, 3)), 0, 255).astype(np.uint8)
    for idx in range(70):
        cx = int(rng.integers(16, size - 16))
        cy = int(rng.integers(16, size - 16))
        rad = int(rng.integers(5, 14))
        col = tuple(int(c) for c in rng.integers(40, 240, size=3))
        cv2.circle(tex, (cx, cy), rad, col, -1)
        if idx % 2 == 0:
            cv2.drawMarker(tex, (cx, cy), (245, 245, 245), cv2.MARKER_TILTED_CROSS, rad + 4, 2)
    return tex


def _build_scene_boxes() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (face_quad_3d[4,3], face_texture_bgr) for textured 3D tabletop objects."""
    boxes = [
        ((-0.24, -0.14), (0.18, 0.16, 0.24), (210, 140, 50), 11),
        ((0.26, -0.05), (0.16, 0.20, 0.34), (70, 185, 130), 22),
        ((0.02, 0.28), (0.22, 0.14, 0.19), (85, 110, 225), 33),
    ]
    faces_spec = [
        ([0, 1, 2, 3], 1),
        ([4, 5, 1, 0], 2),
        ([5, 6, 2, 1], 3),
        ([6, 7, 3, 2], 4),
        ([7, 4, 0, 3], 5),
    ]
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for (cx, cy), (hx, hy, hz), base_bgr, seed in boxes:
        z0, z1 = 0.0, hz * 2.0
        verts = np.array(
            [
                [cx - hx, cy - hy, z1],
                [cx + hx, cy - hy, z1],
                [cx + hx, cy + hy, z1],
                [cx - hx, cy + hy, z1],
                [cx - hx, cy - hy, z0],
                [cx + hx, cy - hy, z0],
                [cx + hx, cy + hy, z0],
                [cx - hx, cy + hy, z0],
            ],
            dtype=np.float32,
        )
        for idx_list, f_off in faces_spec:
            quad = verts[idx_list]
            tex = _make_face_texture(seed * 10 + f_off, base_bgr)
            result.append((quad, tex))
    return result


def _make_studio_floor(size: int = 512) -> np.ndarray:
    """Subtle textured studio cyclorama floor surrounding the tabletop."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    tex = np.full((size, size, 3), (218, 224, 232), dtype=np.float32)
    tile = ((xx.astype(np.int32) // 32) + (yy.astype(np.int32) // 32)) % 2 == 0
    tex[tile] = (204, 212, 222)
    dist = np.hypot((xx - size * 0.5) / (size * 0.5), (yy - size * 0.5) / (size * 0.5))
    shade = np.clip(1.04 - 0.16 * dist, 0.78, 1.04)[..., None]
    return np.clip(tex * shade, 0, 255).astype(np.uint8)


def generate_sample_orbiter_video(
    out_path: Path | str,
    num_frames: int = 42,
    width: int = 480,
    height: int = 360,
    fps: int = 3,
) -> Path:
    """Synthesize a 360-degree anti-aliased z-buffered tabletop orbital MP4 for SfM and 3DGS."""
    target_path = Path(out_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    w_even = max(2, int(width) - (int(width) % 2))
    h_even = max(2, int(height) - (int(height) % 2))
    if (w_even, h_even) != (int(width), int(height)):
        warnings.warn(
            f"libx264 yuv420p 要求偶数分辨率，已将 ({width}, {height}) 调整为 ({w_even}, {h_even})。",
            RuntimeWarning,
            stacklevel=2,
        )
    width, height = w_even, h_even

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("FFmpeg not found in PATH; cannot encode sample_video.mp4.")

    studio_floor_tex = _make_studio_floor(512)
    ground_tex = _make_ground_texture(512)
    floor_quad = np.array(
        [[-1.32, -1.32, -0.015], [1.32, -1.32, -0.015], [1.32, 1.32, -0.015], [-1.32, 1.32, -0.015]],
        dtype=np.float32,
    )
    ground_quad = np.array(
        [[-0.95, -0.95, 0.0], [0.95, -0.95, 0.0], [0.95, 0.95, 0.0], [-0.95, 0.95, 0.0]],
        dtype=np.float32,
    )
    box_faces = _build_scene_boxes()

    ss = 2
    ss_w, ss_h = width * ss, height * ss
    fx = fy = 420.0 * (ss_w / 480.0)
    cx, cy = ss_w * 0.5, ss_h * 0.5
    k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    look_target = np.array([0.0, 0.0, 0.14], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="real2sim_sample_vid_") as tmp:
        tmp_dir = Path(tmp)
        for i in range(num_frames):
            ang = 2.0 * np.pi * (i / num_frames)
            elev = 0.88 + 0.12 * np.sin(ang * 2.0)
            rad = 1.78
            eye = np.array([rad * np.cos(ang), rad * np.sin(ang), elev], dtype=np.float32)
            w2c = _look_at_w2c(eye, look_target)
            r_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3]

            canvas = np.full((ss_h, ss_w, 3), (224, 230, 238), dtype=np.uint8)
            z_buffer = np.full((ss_h, ss_w), np.inf, dtype=np.float32)

            def _rasterize_quad(quad: np.ndarray, tex: np.ndarray) -> None:
                p_cam = quad @ r_w2c.T + t_w2c
                clipped = _clip_polygon_near_z(p_cam, z_near=0.08)
                if len(clipped) < 3:
                    return
                n_cam = np.cross(p_cam[1] - p_cam[0], p_cam[2] - p_cam[0])
                n_len = float(np.linalg.norm(n_cam))
                if n_len < 1e-8:
                    return
                n_cam = n_cam / n_len
                d_plane = float(np.dot(n_cam, p_cam[0]))

                uv = clipped @ k_mat.T
                dst_pts = (uv[:, :2] / uv[:, 2:3]).astype(np.float32)
                x0 = max(0, int(np.floor(dst_pts[:, 0].min())))
                y0 = max(0, int(np.floor(dst_pts[:, 1].min())))
                x1 = min(ss_w, int(np.ceil(dst_pts[:, 0].max())) + 1)
                y1 = min(ss_h, int(np.ceil(dst_pts[:, 1].max())) + 1)
                if x1 <= x0 or y1 <= y0:
                    return

                poly_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                shifted_pts = np.round(dst_pts - np.array([[x0, y0]], dtype=np.float32)).astype(np.int32)
                cv2.fillConvexPoly(poly_mask, shifted_pts, 255)
                sub_mask = poly_mask > 0
                if not np.any(sub_mask):
                    return

                gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                rx = (gx - cx) / fx
                ry = (gy - cy) / fy
                denom = n_cam[0] * rx + n_cam[1] * ry + n_cam[2]
                valid_z = np.where(np.abs(denom) > 1e-5, d_plane / np.where(np.abs(denom) > 1e-5, denom, 1.0), np.inf)
                update = sub_mask & (valid_z > 0.05) & (valid_z < z_buffer[y0:y1, x0:x1])
                if not np.any(update):
                    return

                # 精确反投影到四边形 3D 平面求纹理 UV，近平面裁剪时依然透视正确且无整块丢弃。
                e_u = (p_cam[1] - p_cam[0]).astype(np.float64)
                e_v = (p_cam[3] - p_cam[0]).astype(np.float64)
                gram = np.array(
                    [[np.dot(e_u, e_u), np.dot(e_u, e_v)], [np.dot(e_u, e_v), np.dot(e_v, e_v)]],
                    dtype=np.float64,
                )
                basis = np.linalg.solve(gram + 1e-12 * np.eye(2), np.vstack([e_u, e_v]))
                z_upd = valid_z[update].astype(np.float64)
                p3_rel = np.column_stack([rx[update] * z_upd, ry[update] * z_upd, z_upd]) - p_cam[0][None, :]
                uv_frac = p3_rel @ basis.T
                th, tw = tex.shape[:2]
                bh, bw = y1 - y0, x1 - x0
                map_x = np.zeros((bh, bw), dtype=np.float32)
                map_y = np.zeros((bh, bw), dtype=np.float32)
                map_x[update] = np.clip(uv_frac[:, 0], 0.0, 1.0).astype(np.float32) * float(tw - 1)
                map_y[update] = np.clip(uv_frac[:, 1], 0.0, 1.0).astype(np.float32) * float(th - 1)
                remapped = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                z_buffer[y0:y1, x0:x1][update] = valid_z[update]
                canvas[y0:y1, x0:x1][update] = remapped[update]

            _rasterize_quad(floor_quad, studio_floor_tex)
            _rasterize_quad(ground_quad, ground_tex)

            for quad, tex in box_faces:
                normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
                center = quad.mean(axis=0)
                if float(np.dot(normal, eye - center)) <= 0.0:
                    continue
                _rasterize_quad(quad, tex)

            frame_out = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
            imwrite_unicode(tmp_dir / f"frame_{i:04d}.png", frame_out)

        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            str(target_path),
        ]
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed encoding sample video:\n{proc.stderr}")

    return target_path
