"""Subprocess orchestration for pipeline runs.

Spawns ``python run_pipeline.py --config <path>``, tees its combined
stdout/stderr into an in-memory scrollback buffer, and fans each line
out to any number of live WebSocket subscribers (plus late joiners get
the buffer replayed).

Deliberately uses the classic blocking ``subprocess`` module inside a
worker thread (``loop.run_in_executor``) rather than
``asyncio.create_subprocess_exec``. On Windows, uvicorn's reload /
multi-worker supervisor forces ``asyncio.WindowsSelectorEventLoopPolicy``,
and ``SelectorEventLoop`` does not implement subprocess transports at all
(only ``ProactorEventLoop`` does) -- so ``asyncio.create_subprocess_exec``
raises ``NotImplementedError`` there. Plain ``subprocess.Popen`` has no
such restriction on any platform/event-loop combination, so we use it
everywhere and marshal output back onto the event loop with
``loop.call_soon_threadsafe``.

Single-user, local-only tool -> a plain in-memory dict of runs is fine;
no database, no multi-process coordination.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

logger = logging.getLogger("webui.process_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_BUFFER_LINES = 8000

# Strips ANSI color/cursor escape codes (tqdm progress bars, colored
# logging, etc.) so the browser terminal panel gets clean text. Covers
# both plain CSI color codes ("\x1b[31m") and cursor-visibility codes
# tqdm also emits ("\x1b[?25l" / "\x1b[?25h").
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _clean_line(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = _ANSI_RE.sub("", text)
    return text.rstrip("\r\n")


def _next_terminal_chunk(buf: bytes) -> tuple[bytes, bool, bytes] | None:
    """Split ``buf`` at its first line boundary, terminal-emulator style.

    Returns ``(chunk, is_progress_update, remainder)`` or ``None`` if no
    boundary has arrived yet. A trailing ``\\n`` (or ``\\r\\n``) means a
    finished line (``is_progress_update=False``); a bare ``\\r`` means an
    in-place rewrite, which is exactly how tqdm/Ultralytics progress bars
    update their line -- they never emit ``\\n`` until the bar finishes.
    Splitting only on ``\\n`` (e.g. via ``readline()``) would leave those
    updates stuck in the OS pipe buffer, invisible until the process
    exits and the remaining buffered bytes are flushed all at once.
    """
    idx_r = buf.find(b"\r")
    idx_n = buf.find(b"\n")
    if idx_r == -1 and idx_n == -1:
        return None
    if idx_n != -1 and (idx_r == -1 or idx_n < idx_r):
        return buf[:idx_n], False, buf[idx_n + 1 :]
    if idx_n == idx_r + 1:  # "\r\n" pair -> a single finalized line
        return buf[:idx_r], False, buf[idx_n + 1 :]
    return buf[:idx_r], True, buf[idx_r + 1 :]


@dataclass
class PipelineRun:
    run_id: str
    config_path: Path
    status: str = "starting"  # starting | running | success | failed | stopped
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    process: subprocess.Popen | None = None
    buffer: list[str] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    _reader_task: Any = None  # asyncio.Future wrapping the executor thread

    _last_was_progress: bool = False

    def push_line(self, line: str, is_progress: bool = False) -> None:
        # A progress-bar rewrite (bare "\r", no "\n" yet) replaces the
        # previous buffered line instead of appending a new one, mirroring
        # how a real terminal redraws the current line in place. Once a
        # "\n" finalizes the line, the next line always appends.
        replacing = self._last_was_progress and bool(self.buffer)
        if replacing:
            self.buffer[-1] = line
        else:
            self.buffer.append(line)
            if len(self.buffer) > MAX_BUFFER_LINES:
                del self.buffer[: len(self.buffer) - MAX_BUFFER_LINES]
        self._last_was_progress = is_progress
        frame_type = "replace" if replacing else "line"
        for q in list(self.subscribers):
            q.put_nowait({"type": frame_type, "line": line})

    def push_event(self, **payload) -> None:
        for q in list(self.subscribers):
            q.put_nowait({"type": "event", **payload})


class ProcessManager:
    """Tracks all pipeline runs started from the dashboard."""

    def __init__(self) -> None:
        self._runs: dict[str, PipelineRun] = {}

    def get(self, run_id: str) -> PipelineRun | None:
        return self._runs.get(run_id)

    async def start_run(self, config_path: Path) -> PipelineRun:
        run_id = uuid.uuid4().hex[:12]
        run = PipelineRun(run_id=run_id, config_path=config_path)
        self._runs[run_id] = run

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"  # avoid buffering stalls in the log stream

        run.push_line(
            f"$ {sys.executable} run_pipeline.py --config {config_path}"
        )
        run.status = "running"

        loop = asyncio.get_running_loop()
        # See module docstring: plain subprocess.Popen in a worker thread,
        # not asyncio.create_subprocess_exec, to avoid the Windows
        # SelectorEventLoop "NotImplementedError" trap.
        run._reader_task = loop.run_in_executor(
            None, self._run_blocking, run, config_path, env, loop
        )
        return run

    def _run_blocking(
        self,
        run: PipelineRun,
        config_path: Path,
        env: dict[str, str],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Runs entirely on a worker thread. Do not touch asyncio objects
        (Queues, etc.) directly here -- only via ``loop.call_soon_threadsafe``.
        """
        try:
            _extra: dict = {}
            if sys.platform == "win32":
                # Detach from the uvicorn console process group so that any
                # Ctrl+C in the server terminal does NOT propagate into the
                # pipeline (and its onnx2tf grandchild) as a KeyboardInterrupt.
                _extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                [sys.executable, "run_pipeline.py", "--config", str(config_path)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=0,  # unbuffered: we do our own \r/\n framing below
                **_extra,
            )

        except OSError as exc:
            loop.call_soon_threadsafe(self._on_spawn_failed, run, exc)
            return

        run.process = process
        assert process.stdout is not None
        fd = process.stdout.fileno()
        pending = b""
        try:
            # Read raw bytes as they arrive rather than process.stdout.readline(),
            # which only ever splits on "\n". Ultralytics/tqdm progress bars
            # redraw their line with a bare "\r" and never emit "\n" until the
            # bar completes, so readline() blocks holding a full bar's worth of
            # updates in the OS pipe buffer -- nothing appears in the UI until
            # the process exits and the last chunk is finally flushed. os.read()
            # returns as soon as any bytes are available, so a "\r" mid-buffer
            # is visible immediately.
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                pending += chunk
                while True:
                    split = _next_terminal_chunk(pending)
                    if split is None:
                        break
                    raw_line, is_progress, pending = split
                    loop.call_soon_threadsafe(
                        run.push_line, _clean_line(raw_line), is_progress
                    )
            if pending:
                loop.call_soon_threadsafe(run.push_line, _clean_line(pending))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("log pump crashed for run %s", run.run_id)
            loop.call_soon_threadsafe(
                run.push_line, f"[dashboard] log stream error: {exc}"
            )
        finally:
            returncode = process.wait()
            loop.call_soon_threadsafe(self._on_process_done, run, returncode)

    def _on_spawn_failed(self, run: PipelineRun, exc: Exception) -> None:
        run.status = "failed"
        run.ended_at = time.time()
        run.push_line(f"[dashboard] failed to launch pipeline: {exc}")
        run.push_event(event="done", status=run.status, returncode=None)

    def _on_process_done(self, run: PipelineRun, returncode: int) -> None:
        run.returncode = returncode
        run.ended_at = time.time()
        if run.status != "stopped":
            run.status = "success" if returncode == 0 else "failed"
        run.push_line(
            f"[dashboard] pipeline exited with code {returncode} "
            f"({run.status})"
        )
        run.push_event(event="done", status=run.status, returncode=returncode)

    async def stop_run(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.process is None:
            return False
        if run.process.poll() is not None:
            return False
        run.status = "stopped"
        run.push_line("[dashboard] stop requested; terminating process...")
        run.process.terminate()
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, run.process.wait), timeout=5
            )
        except asyncio.TimeoutError:
            run.push_line("[dashboard] process did not exit in time; killing...")
            run.process.kill()
        return True

    async def subscribe(self, run_id: str) -> AsyncIterator[dict]:
        """Yield buffered lines first, then live updates, as dict frames."""
        run = self._runs.get(run_id)
        if run is None:
            yield {"type": "event", "event": "error", "message": "unknown run_id"}
            return

        queue: asyncio.Queue = asyncio.Queue()
        run.subscribers.append(queue)
        try:
            for line in list(run.buffer):
                yield {"type": "line", "line": line}
            if run.process is not None and run.process.returncode is not None:
                yield {
                    "type": "event",
                    "event": "done",
                    "status": run.status,
                    "returncode": run.returncode,
                }
                return
            while True:
                item = await queue.get()
                yield item
                if item.get("type") == "event" and item.get("event") == "done":
                    break
        finally:
            with contextlib.suppress(ValueError):
                run.subscribers.remove(queue)


manager = ProcessManager()
