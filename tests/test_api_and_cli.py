from __future__ import annotations

import io
import json
import socket
import time
import zipfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from real2sim_gs.api.cli import _resolve_studio_endpoint
from real2sim_gs.api.cli import app as cli_app
from real2sim_gs.api.server import STUDIO_TOKEN_HEADER, create_app
from real2sim_gs.utils.sample_video import generate_sample_orbiter_video


def _authed_client(app) -> TestClient:
    return TestClient(app, headers={STUDIO_TOKEN_HEADER: app.state.studio_token})


def test_cli_doctor_stages_and_mock_run(tmp_path: Path) -> None:
    runner = CliRunner()

    doc_res = runner.invoke(cli_app, ["doctor"])
    assert doc_res.exit_code == 0
    assert "PyTorch" in doc_res.output

    stages_res = runner.invoke(cli_app, ["stages"])
    assert stages_res.exit_code == 0
    assert "gsplat_whitebox" in stages_res.output
    assert "mujoco" in stages_res.output

    ws_dir = tmp_path / "cli_ws"
    run_res = runner.invoke(
        cli_app,
        ["run", "-c", "configs/default_mujoco.yaml", "-o", str(ws_dir), "--mock"],
    )
    assert run_res.exit_code == 0
    assert (ws_dir / "05_export" / "mujoco" / "scene.xml").exists()


def test_fastapi_endpoints_and_studio_ui(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspaces"
    app = create_app(workspaces_root=ws_root, allowed_input_roots=[tmp_path])
    client = _authed_client(app)

    # Studio static UI pages & token injection
    home = client.get("/")
    assert home.status_code == 200
    assert "Real2Sim-GS Studio" in home.text
    assert 'data-theme="light"' in home.text
    assert "历史训练档案" in home.text
    assert app.state.studio_token in home.text
    assert client.get("/static/viewer3d.js").status_code == 200
    assert client.get("/static/app.js").status_code == 200

    # Doctor & configs catalog
    doc_resp = client.get("/api/doctor")
    assert doc_resp.status_code == 200
    doc_items = doc_resp.json()["items"]
    assert len(doc_items) >= 5
    colmap_item = next(item for item in doc_items if "COLMAP" in item["component"])
    assert colmap_item["available"] is True
    assert "pycolmap" in colmap_item["detail"]

    cfg_resp = client.get("/api/configs")
    assert cfg_resp.status_code == 200
    assert "default_mujoco.yaml" in cfg_resp.json()["files"]
    assert "pure_3dgs.yaml" in cfg_resp.json()["files"]

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "gsplat_whitebox" in body["stages"]["reconstruct"]

    # Inline config job execution
    inline_cfg = yaml.safe_load(Path("configs/default_mujoco.yaml").read_text(encoding="utf-8"))
    ws_dir = ws_root / "api_ws"
    inline_cfg["io"]["workspace_dir"] = str(ws_dir)

    post_res = client.post(
        "/jobs",
        json={
            "inline_config": inline_cfg,
            "mock": True,
            "run_async": False,
        },
    )
    assert post_res.status_code == 200
    job = post_res.json()
    assert job["status"] == "completed"
    assert job["overall_progress"] == 1.0
    assert len(job["logs"]) >= 1
    assert job["gaussian_ply_path"] is not None
    assert job["mujoco_xml_path"] is not None
    assert job["stage_states"]["reconstruct"]["status"] == "completed"

    get_res = client.get(f"/jobs/{job['job_id']}")
    assert get_res.status_code == 200
    assert get_res.json()["job_id"] == job["job_id"]

    # 3D scene preview API & live CUDA/PyTorch novel-view render endpoint
    prev_res = client.get("/api/scene-preview", params={"workspace_dir": str(ws_dir)})
    assert prev_res.status_code == 200
    preview = prev_res.json()
    assert preview["splats"]["shown_points"] > 0
    assert len(preview["splats"]["scales_3d"]) == preview["splats"]["shown_points"]
    assert len(preview["splats"]["quats"]) == preview["splats"]["shown_points"]
    assert preview["neural_render_available"] is True
    assert len(preview["mesh"]["faces"]) > 0
    assert len(preview["convex_parts"]) >= 1
    assert "<mujoco" in preview["scene_xml"]
    assert len(preview["frames"]) >= 1
    assert preview["exports_available"]["pure_3dgs_ply"] == "05_export/gaussian_splats/scene_3dgs.ply"

    render_res = client.get(
        "/api/render-view",
        params={"workspace_dir": str(ws_dir), "width": 320, "height": 240, "theme": "light"},
    )
    assert render_res.status_code == 200
    assert render_res.headers["content-type"].startswith("image/jpeg")
    assert render_res.content[:2] == b"\xff\xd8"

    cfg_snap_res = client.get("/api/workspace-config", params={"workspace_dir": str(ws_dir)})
    assert cfg_snap_res.status_code == 200
    cfg_snap = cfg_snap_res.json()
    assert cfg_snap["has_config"] is True
    assert Path(cfg_snap["config"]["io"]["workspace_dir"]).resolve() == ws_dir.resolve()

    # Workspace file serving + path traversal protection
    frame_rel = preview["frames"][0]["rel_path"]
    file_ok = client.get(
        "/api/workspace-file",
        params={"workspace_dir": str(ws_dir), "rel_path": frame_rel},
    )
    assert file_ok.status_code == 200

    file_blocked = client.get(
        "/api/workspace-file",
        params={"workspace_dir": str(ws_dir), "rel_path": "../pyproject.toml"},
    )
    assert file_blocked.status_code == 403

    # Export bundle ZIP
    zip_res = client.get("/api/workspace-bundle.zip", params={"workspace_dir": str(ws_dir)})
    assert zip_res.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_res.content), "r") as zf:
        names = set(zf.namelist())
        assert "05_export/mujoco/scene.xml" in names
        assert "05_export/mujoco/visual_splats.ply" in names


def test_studio_port_resolution_and_launcher_script() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        occupied_port = sock.getsockname()[1]

        resolved_port, already_running = _resolve_studio_endpoint("127.0.0.1", occupied_port)
        assert already_running is False
        assert resolved_port != occupied_port

    free_port, already_running2 = _resolve_studio_endpoint("127.0.0.1", occupied_port)
    assert free_port == occupied_port
    assert already_running2 is False

    # Wildcard 0.0.0.0 listener must not trick _resolve_studio_endpoint on Windows.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as wildcard_sock:
        wildcard_sock.bind(("0.0.0.0", 0))
        wildcard_sock.listen(1)
        wildcard_port = wildcard_sock.getsockname()[1]
        resolved_wc, already_wc = _resolve_studio_endpoint("127.0.0.1", wildcard_port)
        assert already_wc is False
        assert resolved_wc != wildcard_port

    # reuse_existing=True reuses active /health endpoint; reuse_existing=False isolates desktop runs.
    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                body = json.dumps({"status": "ok", "stages": {}}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        health_port = int(srv.server_address[1])
        reused_port, already_reused = _resolve_studio_endpoint(
            "127.0.0.1", health_port, reuse_existing=True
        )
        assert reused_port == health_port
        assert already_reused is True

        isolated_port, already_isolated = _resolve_studio_endpoint(
            "127.0.0.1", health_port, reuse_existing=False
        )
        assert isolated_port != health_port
        assert already_isolated is False
    finally:
        srv.shutdown()
        srv.server_close()
        srv_thread.join(timeout=2.0)

    launcher = Path("start_studio.bat")
    assert launcher.exists()
    raw = launcher.read_bytes()
    assert raw.startswith(b"@echo off\r\nchcp 65001 >nul\r\n")
    assert b"%~dp0" in raw


def test_job_failure_diagnostic_report_and_sample_video(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspaces"
    app = create_app(workspaces_root=ws_root, allowed_input_roots=[tmp_path])
    client = _authed_client(app)
    missing_vid = tmp_path / "missing_custom_video.mp4"
    res = client.post(
        "/jobs",
        json={
            "config_path": "configs/default_mujoco.yaml",
            "input_path": str(missing_vid),
            "output_dir": str(ws_root / "fail_ws"),
            "mock": False,
            "run_async": False,
        },
    )
    assert res.status_code == 200
    job = res.json()
    assert job["status"] == "failed"
    assert "missing_custom_video.mp4" in (job["error"] or "")
    assert job["error_traceback"] and "FileNotFoundError" in job["error_traceback"]
    assert job["diagnostic_report"] and "Real2Sim-GS 错误诊断报告" in job["diagnostic_report"]
    assert "Traceback" in job["diagnostic_report"]

    sample_out = tmp_path / "sample_video.mp4"
    generate_sample_orbiter_video(sample_out, num_frames=8, width=160, height=120, fps=4)
    assert sample_out.is_file() and sample_out.stat().st_size > 1024


def test_workspace_cleanup_delete_and_cancel_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inline_cfg = yaml.safe_load(Path("configs/default_mujoco.yaml").read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)

    app = create_app(upload_grace_sec=0.0)
    client = _authed_client(app)

    ws_a = Path("workspaces") / "scene_a"
    inline_cfg["io"]["workspace_dir"] = str(ws_a)
    post_res = client.post("/jobs", json={"inline_config": inline_cfg, "mock": True, "run_async": False})
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "completed"

    uploads_dir = Path("workspaces") / "_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dummy_up = uploads_dir / "temp_clip.mp4"
    dummy_up.write_bytes(b"0" * 65536)

    list_res = client.get("/api/workspaces")
    assert list_res.status_code == 200
    payload = list_res.json()
    assert payload["uploads_size_mb"] >= 0.05
    assert payload["total_size_mb"] >= payload["uploads_size_mb"]
    assert len(payload["workspaces"]) == 1
    ws_entry = payload["workspaces"][0]
    assert ws_entry["name"] == "scene_a"
    assert ws_entry["size_bytes"] > 0
    assert ws_entry["reclaimable_bytes"] > 0
    assert ws_entry["is_incomplete"] is False

    # Guard against path escape & system dirs
    assert (
        client.post("/api/delete-workspace", json={"workspace_dir": str(tmp_path / "outside"), "mode": "all"}).status_code
        == 403
    )
    assert (
        client.post("/api/delete-workspace", json={"workspace_dir": str(uploads_dir), "mode": "all"}).status_code == 400
    )

    # Mode "intermediate": frees 01_frames/ while keeping 05_export assets intact
    slim_res = client.post(
        "/api/delete-workspace",
        json={"workspace_dir": str(ws_a), "mode": "intermediate"},
    )
    assert slim_res.status_code == 200
    assert slim_res.json()["deleted"] is True
    assert not (ws_a / "01_frames").exists()
    assert (ws_a / "05_export" / "mujoco" / "scene.xml").is_file()
    assert (ws_a / "05_export" / "gaussian_splats" / "scene_3dgs.ply").is_file()

    # Mode "all": deletes workspace completely
    del_res = client.post(
        "/api/delete-workspace",
        json={"workspace_dir": str(ws_a), "mode": "all"},
    )
    assert del_res.status_code == 200
    assert not ws_a.exists()
    assert client.get("/api/workspaces").json()["workspaces"] == []

    # Uploads cleanup
    clean_up_res = client.post("/api/cleanup-uploads")
    assert clean_up_res.status_code == 200
    assert clean_up_res.json()["removed_files"] == 1
    assert not dummy_up.exists()

    # Job cancel endpoint on a real async job (non-mock pytorch whitebox)
    cancel_cfg = yaml.safe_load(yaml.safe_dump(inline_cfg))
    ws_b = Path("workspaces") / "scene_cancel"
    cancel_cfg["io"]["workspace_dir"] = str(ws_b)
    cancel_cfg["pipeline"]["ingest"] = {"name": "mock_ingest", "params": {"mock_num_views": 4, "mock_image_size": 32}}
    cancel_cfg["pipeline"]["pose"] = {"name": "mock_pose", "params": {}}
    cancel_cfg["pipeline"]["reconstruct"] = {
        "name": "gsplat_whitebox",
        "params": {"iterations": 200, "backend": "pytorch", "strategy": "mcmc"},
    }
    async_job = client.post("/jobs", json={"inline_config": cancel_cfg, "mock": False, "run_async": True}).json()
    jid = async_job["job_id"]

    time.sleep(0.25)
    cancel_res = client.post(f"/jobs/{jid}/cancel")
    assert cancel_res.status_code == 200
    deadline = time.time() + 15.0
    final_job = client.get(f"/jobs/{jid}").json()
    while final_job["status"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.1)
        final_job = client.get(f"/jobs/{jid}").json()
    assert final_job["status"] in ("cancelled", "completed")
    assert final_job["error"] is None


def test_api_security_path_traversal_and_token_guards(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspaces"
    outside_dir = tmp_path / "outside_secret"
    outside_dir.mkdir(parents=True, exist_ok=True)
    secret_file = outside_dir / "secret.txt"
    secret_file.write_text("top-secret", encoding="utf-8")

    app = create_app(workspaces_root=ws_root, max_upload_bytes=128)
    raw_client = TestClient(app)
    authed = _authed_client(app)

    # 1. Missing or invalid X-Studio-Token rejected on all mutating endpoints.
    assert raw_client.post("/api/cleanup-uploads").status_code == 403
    assert raw_client.post("/api/delete-workspace", json={"workspace_dir": "workspaces/x"}).status_code == 403
    assert raw_client.post("/api/upload", files={"file": ("clip.mp4", b"123", "video/mp4")}).status_code == 403
    assert raw_client.post("/jobs", json={"mock": True}).status_code == 403
    assert (
        raw_client.post(
            "/api/cleanup-uploads",
            headers={STUDIO_TOKEN_HEADER: "wrong-token"},
        ).status_code
        == 403
    )

    # 2. TrustedHostMiddleware blocks DNS rebinding Host header on loopback.
    bad_host_client = TestClient(app, headers={"Host": "evil.attacker.com"})
    assert bad_host_client.get("/health").status_code == 400

    # 3. Path traversal & UNC paths blocked across every workspace_dir endpoint.
    for bad_ws in (str(outside_dir), r"\\evil\share\ws"):
        assert authed.get("/api/workspace-file", params={"workspace_dir": bad_ws, "rel_path": "secret.txt"}).status_code == 403
        assert authed.get("/api/workspace-bundle.zip", params={"workspace_dir": bad_ws}).status_code == 403
        assert authed.get("/api/workspace-config", params={"workspace_dir": bad_ws}).status_code == 403
        assert authed.get("/api/scene-preview", params={"workspace_dir": bad_ws}).status_code == 403
        assert authed.get("/api/render-view", params={"workspace_dir": bad_ws}).status_code == 403
        assert authed.post("/api/open-workspace", json={"workspace_dir": bad_ws}).status_code == 403

    # 4. /jobs restricts both workspace_dir and arbitrary input_path.
    inline_cfg = yaml.safe_load(Path("configs/default_mujoco.yaml").read_text(encoding="utf-8"))
    inline_cfg["io"]["workspace_dir"] = str(outside_dir)
    assert authed.post("/jobs", json={"inline_config": inline_cfg, "mock": True}).status_code == 403

    inline_cfg["io"]["workspace_dir"] = str(ws_root / "ok_ws")
    inline_cfg["io"]["input_path"] = str(Path("pyproject.toml").resolve())
    assert authed.post("/jobs", json={"inline_config": inline_cfg, "mock": True}).status_code == 403

    # 5. Upload validates file extension and byte size limit.
    bad_ext = authed.post("/api/upload", files={"file": ("malware.exe", b"MZ", "application/octet-stream")})
    assert bad_ext.status_code == 400
    too_big = authed.post("/api/upload", files={"file": ("big.mp4", b"x" * 256, "video/mp4")})
    assert too_big.status_code == 413

    # 6. Remote 0.0.0.0 bind enforces token auth on GET routes too (while keeping /health public).
    remote_app = create_app(workspaces_root=ws_root, bind_host="0.0.0.0", studio_token="remote_secret_123")
    remote_raw = TestClient(remote_app)
    assert remote_raw.get("/health").status_code == 200
    assert remote_raw.get("/api/configs").status_code == 403
    login_res = remote_raw.get("/?token=remote_secret_123")
    assert login_res.status_code == 200
    assert remote_raw.get("/api/configs").status_code == 200


def test_chinese_workspace_zip_and_corrupt_manifest_resilience(tmp_path: Path) -> None:
    ws_root = tmp_path / "workspaces"
    app = create_app(workspaces_root=ws_root, upload_grace_sec=1800.0)
    client = _authed_client(app)

    # 1. Chinese workspace ZIP download (RFC 5987 + ZIP_STORED for .ply)
    cn_ws = ws_root / "测试场景_01"
    inline_cfg = yaml.safe_load(Path("configs/default_mujoco.yaml").read_text(encoding="utf-8"))
    inline_cfg["io"]["workspace_dir"] = str(cn_ws)
    run_res = client.post("/jobs", json={"inline_config": inline_cfg, "mock": True, "run_async": False})
    assert run_res.status_code == 200

    zip_res = client.get("/api/workspace-bundle.zip", params={"workspace_dir": str(cn_ws)})
    assert zip_res.status_code == 200
    assert "filename*=UTF-8''" in zip_res.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(zip_res.content), "r") as zf:
        ply_info = zf.getinfo("05_export/mujoco/visual_splats.ply")
        assert ply_info.compress_type == zipfile.ZIP_STORED

    # 2. Corrupted manifest.json in sibling workspace does not 500 /api/workspaces or /api/scene-preview
    bad_ws = ws_root / "坏掉场景"
    bad_ws.mkdir(parents=True, exist_ok=True)
    (bad_ws / "manifest.json").write_text("{corrupt-json", encoding="utf-8")

    list_res = client.get("/api/workspaces")
    assert list_res.status_code == 200
    by_name = {w["name"]: w for w in list_res.json()["workspaces"]}
    assert by_name["坏掉场景"]["is_corrupted"] is True
    assert by_name["测试场景_01"]["is_corrupted"] is False

    prev_bad = client.get("/api/scene-preview", params={"workspace_dir": str(bad_ws)})
    assert prev_bad.status_code == 200
    assert prev_bad.json()["manifest"]["corrupted"] is True
    assert prev_bad.json()["warnings"]

    # 3. Grace window protects freshly uploaded file until grace_sec=0
    up_res = client.post("/api/upload", files={"file": ("中文视频.mp4", b"mp4bytes", "video/mp4")})
    assert up_res.status_code == 200
    uploaded_file = Path(up_res.json()["input_path"])
    assert uploaded_file.is_file()
    assert client.post("/api/cleanup-uploads").json()["removed_files"] == 0
    assert uploaded_file.is_file()
    assert client.post("/api/cleanup-uploads?grace_sec=0").json()["removed_files"] == 1
    assert not uploaded_file.exists()

    # 4. /api/render-view returns 409 while any job is running
    rec = app.state.job_store.create(workspace_dir=cn_ws.resolve(), display_workspace=str(cn_ws))
    rec_ref = app.state.job_store.jobs[rec.job_id]
    rec_ref.status = "running"
    try:
        render_busy = client.get("/api/render-view", params={"workspace_dir": str(cn_ws)})
        assert render_busy.status_code == 409
    finally:
        rec_ref.status = "completed"
