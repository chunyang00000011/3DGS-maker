"""Shared helpers for launching external paper repositories / CLI tools as subprocesses.

- Placeholder paths are resolved to absolute paths before formatting.
- Commands are built as argv lists (shell=False) so paths with spaces / non-ASCII chars are safe.
- Merged stdout/stderr is drained by a background thread into a bounded tail buffer (+ optional log file).
- The workspace cancellation predicate is polled; on cancel/timeout the whole process tree is killed.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from real2sim_gs.core.workspace import SceneWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_dir(repo_dir: str | Path, required: bool = True) -> Path:
    """Resolve a (possibly relative) third_party repo dir: CWD first, then project root."""
    p = Path(str(repo_dir)).expanduser()
    if p.is_absolute():
        candidate = p
    else:
        cwd_candidate = (Path.cwd() / p).resolve()
        candidate = cwd_candidate if cwd_candidate.exists() else (PROJECT_ROOT / p).resolve()
    if required and not candidate.is_dir():
        raise FileNotFoundError(
            f"外部论文仓库目录不存在 (repo_dir not found): {repo_dir} -> {candidate}。"
            f"请先将对应论文代码克隆到该目录，或在 YAML params 中修正 repo_dir。"
        )
    return candidate


def _split_template(template: str) -> list[str]:
    """POSIX-style split honouring quotes but keeping backslashes literal (Windows paths)."""
    lex = shlex.shlex(template, posix=True)
    lex.whitespace_split = True
    lex.escape = ""
    lex.commenters = ""
    return list(lex)


def build_command(template: str, placeholders: dict[str, Any]) -> list[str]:
    """Split the template into argv tokens, then format each token with absolute-path placeholders."""
    values: dict[str, Any] = {}
    for key, val in placeholders.items():
        if isinstance(val, Path):
            values[key] = str(val.resolve())
        else:
            values[key] = val
    tokens = _split_template(template)
    if not tokens:
        raise ValueError("command_template 为空，无法构建外部命令。")
    argv: list[str] = []
    for tok in tokens:
        try:
            argv.append(tok.format(**values))
        except KeyError as exc:
            raise ValueError(
                f"command_template 中存在未知占位符 {exc}；可用占位符: {sorted(values)}"
            ) from exc
        except (IndexError, ValueError) as exc:
            raise ValueError(f"command_template 令牌格式化失败: {tok!r} ({exc})") from exc
    return argv


def format_command(argv: list[str]) -> str:
    """Human-readable, correctly quoted command line for logs and error messages."""
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def kill_process_tree(proc: subprocess.Popen[Any], wait_s: float = 5.0) -> None:
    """Kill a subprocess and all its children; never raises on timeout."""
    if proc.poll() is None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=wait_s,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        pass


def popen_process_group_kwargs() -> dict[str, Any]:
    """Start the child in its own process group so the whole tree can be killed."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def run_external_command(
    argv: list[str],
    *,
    label: str,
    cwd: str | Path | None = None,
    ws: SceneWorkspace | None = None,
    timeout_s: float | None = None,
    log_path: str | Path | None = None,
    progress: float = 0.5,
    tail_size: int = 60,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> str:
    """Run argv (shell=False) with drained merged output, cancellation and optional timeout.

    Returns the tail of combined output. Raises RuntimeError (non-zero exit), TimeoutError, or
    PipelineCancelledError (propagated from ``ws.check_cancelled()``) after killing the process tree.
    """
    from real2sim_gs.core.workspace import PipelineCancelledError

    run_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if env:
        run_env.update(env)
    log_fh = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        log_fh.write(f"$ {format_command(argv)}\n")
        log_fh.flush()

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=run_env,
            shell=False,
            **popen_process_group_kwargs(),
        )
    except OSError as exc:
        if log_fh is not None:
            log_fh.close()
        raise RuntimeError(f"[{label}] 无法启动外部命令: {format_command(argv)} ({exc})") from exc

    tail: deque[str] = deque(maxlen=tail_size)
    lock = threading.Lock()
    last_line: list[str] = [""]

    def _drain() -> None:
        if proc.stdout is None:
            return
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            if log_fh is not None:
                try:
                    log_fh.write(line + "\n")
                except ValueError:
                    pass
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:  # noqa: BLE001
                    pass
            if line:
                with lock:
                    tail.append(line)
                    last_line[0] = line

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    def _tail_text() -> str:
        with lock:
            return "\n".join(tail)

    t0 = time.monotonic()
    last_report = 0.0
    try:
        while proc.poll() is None:
            if ws is not None:
                ws.check_cancelled()
                now = time.monotonic()
                if now - last_report >= 1.0:
                    last_report = now
                    with lock:
                        msg = last_line[0]
                    if msg:
                        ws.report_progress(progress, f"[{label}] {msg[-160:]}")
            if timeout_s is not None and timeout_s > 0 and time.monotonic() - t0 > timeout_s:
                kill_process_tree(proc)
                reader.join(timeout=2.0)
                raise TimeoutError(
                    f"[{label}] 外部命令超时 ({timeout_s:.0f}s) 已强制终止: {format_command(argv)}\n"
                    f"最近输出:\n{_tail_text()}"
                )
            time.sleep(0.1)
    except PipelineCancelledError:
        kill_process_tree(proc)
        reader.join(timeout=2.0)
        raise
    finally:
        if proc.poll() is None:
            kill_process_tree(proc)
        reader.join(timeout=5.0)
        if log_fh is not None:
            log_fh.close()

    if proc.returncode != 0:
        raise RuntimeError(
            f"[{label}] 外部命令异常退出 (exit_code={proc.returncode}): {format_command(argv)}\n"
            f"最近输出:\n{_tail_text()}"
        )
    return _tail_text()


def run_command_template(
    ws: SceneWorkspace,
    params: dict[str, Any],
    placeholders: dict[str, Any],
    *,
    label: str,
    default_repo_dir: str,
    log_path: str | Path | None = None,
) -> str:
    """Standard path for third_party adapters: resolve repo_dir, build argv from template, run it.

    Recognised params: ``command_template`` (required), ``repo_dir``, ``timeout_s`` (optional, seconds).
    The repo dir is the working directory and is also available as ``{repo_dir}``.
    """
    template = str(params.get("command_template", "")).strip()
    if not template:
        raise RuntimeError(f"[{label}] 缺少 'command_template' 参数。")
    repo = resolve_repo_dir(params.get("repo_dir", default_repo_dir), required=True)
    argv = build_command(template, {**placeholders, "repo_dir": repo})
    timeout_raw = params.get("timeout_s")
    timeout_s = float(timeout_raw) if timeout_raw not in (None, "", 0) else None
    return run_external_command(
        argv,
        label=label,
        cwd=repo,
        ws=ws,
        timeout_s=timeout_s,
        log_path=log_path,
    )
