"""
comms.py — IPC between tictactoe.py, move.py (robot arm), and ui.py (GTK display)
Fixed:
  - Dead code after `break` in CLOSED handler removed
  - win / turn messages forwarded to UI correctly
  - Queue flushing is thread-safe
"""
from __future__ import annotations

import subprocess
import threading
from collections import deque

# ---------------------------------------------------------------------------
# Global handles
# ---------------------------------------------------------------------------
p   = None   # move.py subprocess
app = None   # ui.py  subprocess

_ui_queue_lock     = threading.Lock()
ui_command_queue   = deque()


# ---------------------------------------------------------------------------
# Robot arm (move.py) communication
# ---------------------------------------------------------------------------

def init_move_process() -> None:
    global p
    try:
        p = subprocess.Popen(
            ["python3", "src/board/move.py"],
            stdin=subprocess.PIPE,
            stdout=None,
            stderr=None,
            text=True,
        )
        print("[COMMS] move.py started")
    except Exception as e:
        print(f"[COMMS] Failed to start move.py: {e}")


def send(cmd: str) -> None:
    """Send a command to the robot arm (move.py)."""
    global p
    if p and p.stdin:
        try:
            print(f"[COMMS→ROBOT] {cmd}")
            p.stdin.write(cmd + "\n")
            p.stdin.flush()
        except Exception as e:
            print(f"[COMMS] Robot send error: {e}")
    else:
        print(f"[COMMS] Robot not ready, dropping: {cmd}")


# ---------------------------------------------------------------------------
# UI (ui.py) communication
# ---------------------------------------------------------------------------

def send_ui_update(cmd: str) -> None:
    """Send a real-time update to the UI subprocess (thread-safe)."""
    global app
    try:
        if app and hasattr(app, "stdin") and app.stdin:
            print(f"[COMMS→UI] {cmd}")
            app.stdin.write(cmd + "\n")
            app.stdin.flush()
        else:
            with _ui_queue_lock:
                ui_command_queue.append(cmd)
            print(f"[COMMS] UI not ready, queued: {cmd}")
    except Exception as e:
        print(f"[COMMS] UI send error: {e}")


def flush_ui_queue() -> None:
    """Drain the pending UI command queue once the subprocess is ready."""
    global app
    try:
        with _ui_queue_lock:
            pending = list(ui_command_queue)
            ui_command_queue.clear()
        if app and hasattr(app, "stdin") and app.stdin:
            for cmd in pending:
                print(f"[COMMS→UI queued] {cmd}")
                app.stdin.write(cmd + "\n")
                app.stdin.flush()
    except Exception as e:
        print(f"[COMMS] Flush error: {e}")


# ---------------------------------------------------------------------------
# UI feedback listener  (runs in a daemon thread)
# ---------------------------------------------------------------------------

def listen_feedback(app_instance=None) -> None:
    """
    Read lines from ui.py stdout and translate them into game actions.
    Runs as a daemon thread; exits when the UI process closes.
    """
    global app
    print("[COMMS] UI feedback listener started")

    while True:
        try:
            if app is None or app.stdout is None:
                import time
                time.sleep(0.1)
                continue

            line = app.stdout.readline()
            if not line:
                print("[COMMS] UI subprocess closed")
                break

            line = line.strip()
            if not line:
                continue

            print(f"[COMMS←UI] {line!r}")

            if line == "RESTARTED":
                _handle_restart(app_instance)

            elif line == "START_GAME":
                _handle_start_game(app_instance)

            elif line == "CLOSED":
                print("[COMMS] UI closed — signalling exit")
                if app_instance:
                    app_instance.should_exit = True
                break   # ← nothing below this; dead code removed

            elif line == "STOP":
                send("STOP")

            elif line == "RESUME":
                send("RESUME")

            elif line == "CALIBRATE":
                print("[COMMS] Calibrate requested from UI")
                if app_instance:
                    app_instance.mode = 2  # grid calibration mode

            else:
                print(f"[COMMS] Unhandled UI message: {line!r}")

        except Exception as e:
            print(f"[COMMS] Listener error: {e}")
            break

    print("[COMMS] UI feedback listener exiting")


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------

def _handle_restart(inst) -> None:
    if inst is None:
        return
    if getattr(inst, "board_needs_clearing", False):
        print("[COMMS] Board not clear — restart blocked")
        send_ui_update("BOARD_NOT_CLEAR")
    else:
        inst._reset_game()
        print("[COMMS] Game reset via UI")


def _handle_start_game(inst) -> None:
    if inst is None:
        return
    if getattr(inst, "board_needs_clearing", False):
        print("[COMMS] Board not clear — start blocked")
        send_ui_update("BOARD_NOT_CLEAR")
    else:
        inst._reset_game()
        print("[COMMS] Game started via UI")
