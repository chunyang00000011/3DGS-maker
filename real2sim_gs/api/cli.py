from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from real2sim_gs.api.sdk import check_environment, list_available_stages, run_pipeline

app = typer.Typer(
    name="real2sim",
    help="Real2Sim-GS: 面向 MuJoCo / Unity 仿真的模块化白盒 3DGS 场景生成管线",
    no_args_is_help=True,
)
console = Console()


@app.command("run")
def run_cmd(
    config: Path = typer.Option(
        Path("configs/default_mujoco.yaml"),
        "--config",
        "-c",
        help="YAML 配置文件路径",
    ),
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="覆盖配置中的输入视频或图片文件夹路径",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="覆盖配置中的工作区输出目录",
    ),
    from_stage: Optional[str] = typer.Option(
        None,
        "--from-stage",
        help="从指定阶段开始重跑 (ingest | pose | reconstruct | postprocess | export)",
    ),
    force: bool = typer.Option(False, "--force", help="忽略 manifest 缓存，从头强制重跑"),
    mock: bool = typer.Option(False, "--mock", help="使用快速模拟数据演练整条管线"),
) -> None:
    """运行或断点续跑 3DGS Real-to-Sim 管线。"""
    result = run_pipeline(
        config=config,
        input_path=input_path,
        output_dir=output_dir,
        from_stage=from_stage,
        force=force,
        mock=mock,
    )

    table = Table(title="Real2Sim-GS 执行汇总")
    table.add_column("项目", style="cyan")
    table.add_column("内容", style="white")
    table.add_row("工作区目录", str(result.workspace_dir))
    table.add_row("本次执行阶段", ", ".join(result.executed_stages) or "无")
    table.add_row("缓存跳过阶段", ", ".join(result.skipped_stages) or "无")
    table.add_row("最终高斯文件", str(result.splat_ply_path or "无"))
    table.add_row("纯 3DGS 导出", str(result.gaussian_ply_path or "未导出"))
    table.add_row("MuJoCo XML", str(result.mujoco_xml_path or "未导出"))
    table.add_row("Unity 资产目录", str(result.unity_dir or "未导出"))
    for k, v in result.metrics.items():
        table.add_row(f"指标 · {k}", str(v))
    console.print(table)


@app.command("doctor")
def doctor_cmd() -> None:
    """检查本地 GPU、FFmpeg、COLMAP、gsplat、MuJoCo 等依赖环境。"""
    items = check_environment()
    table = Table(title="Real2Sim-GS 环境体检报告")
    table.add_column("组件", style="cyan")
    table.add_column("状态", style="bold")
    table.add_column("详情", style="white")
    table.add_column("安装建议", style="yellow")

    for item in items:
        status_text = "[green]可用[/green]" if item.available else "[yellow]缺失/可选[/yellow]"
        table.add_row(item.component, status_text, item.detail, item.install_hint)
    console.print(table)


@app.command("stages")
def stages_cmd() -> None:
    """列出当前所有已注册的可插拔算法模块。"""
    grouped = list_available_stages()
    table = Table(title="已注册可插拔阶段与算法槽位")
    table.add_column("阶段 (StageKind)", style="cyan")
    table.add_column("可用实现 (name)", style="green")
    for kind, names in grouped.items():
        table.add_row(kind, ", ".join(names))
    console.print(table)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="监听主机地址（设为 0.0.0.0 时强制校验 X-Studio-Token / ?token=）"),
    port: int = typer.Option(8000, "--port", help="监听端口"),
) -> None:
    """启动 FastAPI HTTP 调度服务与 Web Studio 工作台。"""
    import uvicorn

    from real2sim_gs.api.server import create_app

    fastapi_app = create_app(bind_host=host)
    if host in ("0.0.0.0", "::"):
        tok = fastapi_app.state.studio_token
        console.print(f"[yellow]远程监听模式已启用令牌鉴权:[/yellow] X-Studio-Token={tok}")
        console.print(f"[yellow]浏览器访问链接:[/yellow] http://127.0.0.1:{port}/?token={tok}")
    uvicorn.run(fastapi_app, host=host, port=port)


def _resolve_studio_endpoint(
    host: str,
    preferred_port: int,
    *,
    reuse_existing: bool = True,
) -> tuple[int, bool]:
    """Return `(port, already_running)` reusing an existing Studio or finding a free port."""
    check_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    bind_target = "0.0.0.0" if host in ("0.0.0.0", "::") else check_host

    def _is_port_free(candidate: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.15)
            if probe.connect_ex((check_host, candidate)) == 0:
                return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                sock.bind((bind_target, candidate))
                return True
            except OSError:
                return False

    if _is_port_free(preferred_port):
        return preferred_port, False

    if reuse_existing:
        health_url = f"http://{check_host}:{preferred_port}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("status") == "ok" and "stages" in payload:
                        return preferred_port, True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            pass

    for candidate in range(preferred_port + 1, preferred_port + 50):
        if _is_port_free(candidate):
            return candidate, False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_target, 0))
        return int(sock.getsockname()[1]), False


def _wait_until_server_ready(
    url: str,
    timeout_sec: float = 15.0,
    server_thread: threading.Thread | None = None,
) -> bool:
    """Poll Studio `/health` until the local HTTP server starts accepting requests."""
    deadline = time.monotonic() + timeout_sec
    health_url = f"{url}/health"
    while time.monotonic() < deadline:
        if server_thread is not None and not server_thread.is_alive():
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.1)
    return False


@app.command("ui")
def ui_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="监听主机地址（AutoDL 远程设 0.0.0.0 时自动开启 Token 鉴权）"),
    port: int = typer.Option(8000, "--port", help="监听端口"),
    no_browser: bool = typer.Option(False, "--no-browser", help="仅启动服务，不自动弹开浏览器"),
    desktop: bool = typer.Option(False, "--desktop", help="使用本地独立桌面窗口打开 (pywebview)"),
) -> None:
    """一键启动 Real2Sim-GS Studio 可视化 3D 前端工作台。"""
    import uvicorn

    from real2sim_gs.api.server import create_app

    resolved_port, already_running = _resolve_studio_endpoint(host, port, reuse_existing=not desktop)
    open_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    is_remote = host in ("0.0.0.0", "::")
    url = f"http://{open_host}:{resolved_port}"

    if desktop:
        fastapi_app = create_app(bind_host=host)
        if is_remote:
            console.print(f"[yellow]远程令牌 (?token= / X-Studio-Token):[/yellow] {fastapi_app.state.studio_token}")
        uvi_server = uvicorn.Server(
            uvicorn.Config(fastapi_app, host=host, port=resolved_port, log_level="warning")
        )
        server_thread = threading.Thread(target=uvi_server.run, daemon=True)
        server_thread.start()
        if not _wait_until_server_ready(url, timeout_sec=15.0, server_thread=server_thread):
            if not server_thread.is_alive():
                console.print("[red]错误: Studio 后台服务启动失败，端口可能被系统占用。[/red]")
                raise typer.Exit(code=1)
            console.print(f"[yellow]警告: Studio 后台服务在 15s 内未就绪 ({url})，窗口加载可能较慢。[/yellow]")

        try:
            import webview

            webview.settings["ALLOW_DOWNLOADS"] = True
            console.print(f"[green]Real2Sim-GS Studio 桌面软件已启动:[/green] {url}")
            webview.create_window("Real2Sim-GS Studio", url, width=1440, height=900)
            webview.start()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]独立桌面窗口启动失败 ({exc})，自动切换至系统浏览器...[/yellow]")
            if not no_browser:
                webbrowser.open(url)
            try:
                while server_thread.is_alive():
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
        finally:
            uvi_server.should_exit = True
            server_thread.join(timeout=2.0)
        return

    if already_running:
        console.print(f"[green]检测到 Real2Sim-GS Studio 已在运行，直接打开工作台:[/green] {url}")
        if not no_browser:
            webbrowser.open(url)
        return

    fastapi_app = create_app(bind_host=host)
    tok = fastapi_app.state.studio_token
    launch_url = f"{url}/?token={tok}" if is_remote else url
    if is_remote:
        console.print(f"[yellow]远程监听模式已启用令牌鉴权:[/yellow] X-Studio-Token={tok}")
        console.print(f"[yellow]完整带令牌访问链接:[/yellow] {launch_url}")
    if not no_browser:

        def _open_when_ready() -> None:
            if _wait_until_server_ready(url):
                webbrowser.open(launch_url)
            else:
                console.print(f"[yellow]警告: Studio 服务启动超时，请手动访问 {launch_url}[/yellow]")

        threading.Thread(target=_open_when_ready, daemon=True).start()

    console.print(f"[green]Real2Sim-GS Studio 已启动:[/green] {launch_url}")
    uvicorn.run(fastapi_app, host=host, port=resolved_port)


if __name__ == "__main__":
    app()
