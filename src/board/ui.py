#!/usr/bin/env python3
"""
ui.py — GTK3 live-frame display for the TicTacToe demo.

Protocol (stdin, binary mode):
  Text commands  : UTF-8 lines terminated by \\n
                   move <r> <c> <X|O>
                   reset
                   CALIBRATION_DONE
                   CHEAT_DETECTED
                   BOARD_NOT_CLEAR
                   close
  Binary frames  : b'FRAME <size>\\n' followed by <size> raw JPEG bytes

The parent process (tictactoe.py) opens this subprocess WITHOUT an
encoding kwarg, so stdin/stdout are raw byte streams.
"""

import gi
import os
import sys
import threading
import time

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gdk, GdkPixbuf, Gtk

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
NAVY          = "#001F54"
RED           = "#FF4545"
YELLOW        = "#FFD600"
GREEN_WIN     = "#3CB371"
RED_LOSS      = "#B22222"
GRAY_TIE      = "#5A5A5A"
PANEL_BG      = "#0D1B2A"
ACCENT        = "#00D4FF"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
GLOBAL_CSS = b"""
* { font-family: 'Inter', 'Roboto', 'DejaVu Sans', sans-serif; }

window {
    background-color: #0D1B2A;
}

/* ---- info panel ---- */
#info-panel {
    background-color: #131F30;
    border-left: 2px solid #1E3050;
    padding: 20px 16px;
}

/* ---- title ---- */
#title-label {
    color: #00D4FF;
    font-size: 28px;
    font-weight: bold;
    letter-spacing: 2px;
}

/* ---- status card ---- */
#status-card {
    background-color: #1A2840;
    border-radius: 12px;
    border: 1px solid #243555;
    padding: 14px 18px;
    margin: 8px 0;
}

/* ---- turn label ---- */
#turn-label {
    font-size: 22px;
    font-weight: bold;
    color: #E0F4FF;
}

/* ---- status label ---- */
#status-label {
    font-size: 18px;
    color: #90BCDC;
}

/* ---- move label ---- */
#move-label {
    font-size: 16px;
    color: #6A8FAF;
}

/* ---- separator ---- */
.hr {
    background-color: #1E3050;
    min-height: 1px;
}

/* ---- action button (Calibrate / Start / Restart) ---- */
button.action-btn {
    background: linear-gradient(135deg, #0079D3 0%, #00B4FF 100%);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 14px;
    border: none;
    padding: 14px 0;
    margin: 6px 0;
    box-shadow: 0 4px 18px #0079D355;
    transition: background 0.18s, box-shadow 0.18s;
}
button.action-btn:hover {
    background: linear-gradient(135deg, #00B4FF 0%, #0079D3 100%);
    box-shadow: 0 6px 22px #00B4FF66;
}
button.action-btn:disabled {
    background: #243555;
    color: #4A6080;
    box-shadow: none;
}

/* ---- stop/resume button ---- */
button.stop-btn {
    background: linear-gradient(135deg, #8B0000 0%, #D32F2F 100%);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 14px;
    border: none;
    padding: 14px 0;
    margin: 6px 0;
    box-shadow: 0 4px 18px #D32F2F44;
    transition: background 0.18s;
}
button.stop-btn:hover {
    background: linear-gradient(135deg, #D32F2F 0%, #F44336 100%);
}
button.stop-btn:disabled {
    background: #243555;
    color: #4A6080;
    box-shadow: none;
}

/* ---- close button ---- */
button.close-btn {
    background: transparent;
    color: #6A8FAF;
    font-size: 22px;
    border: none;
    border-radius: 8px;
    padding: 4px 10px;
}
button.close-btn:hover {
    background: #B71C1C22;
    color: #EF5350;
}

/* ---- live-feed placeholder ---- */
#feed-placeholder {
    background-color: #060E1A;
    color: #1E3050;
    font-size: 18px;
}

/* ---- overlay popup ---- */
#popup-label {
    font-weight: bold;
}
"""


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class TicTacToeUI(Gtk.Window):
    def __init__(self):
        super().__init__()
        self.set_decorated(False)
        self.fullscreen()
        self.set_keep_above(True)
        self.present()

        # --- Apply global CSS ---
        css_prov = Gtk.CssProvider()
        css_prov.load_from_data(GLOBAL_CSS)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_prov,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # --- App state ---
        self.calibration_done = False
        self.game_active      = False
        self.current_player   = "X"
        self.move_count       = 0
        self.board            = [[None]*3 for _ in range(3)]

        # --- Build layout ---
        self._build_layout()
        self.show_all()

        # --- Background threads ---
        threading.Thread(target=self._keep_raised,  daemon=True).start()
        threading.Thread(target=self._stdin_listener, daemon=True).start()

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    def _build_layout(self):
        """
        ┌──────────────────────────────┬─────────────────┐
        │  LIVE CAMERA FEED (Gtk.Image)│  Info / Controls│
        └──────────────────────────────┴─────────────────┘
        """
        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.add(root)

        # ---- Left: live feed ----
        feed_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        feed_box.set_hexpand(True)
        feed_box.set_vexpand(True)
        root.pack_start(feed_box, True, True, 0)

        self.feed_image = Gtk.Image()
        self.feed_image.set_name("feed-placeholder")
        self.feed_image.set_hexpand(True)
        self.feed_image.set_vexpand(True)
        # Show a placeholder until the first frame arrives
        self._show_placeholder()
        feed_box.pack_start(self.feed_image, True, True, 0)

        # ---- Right: info panel ----
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        panel.set_name("info-panel")
        panel.set_size_request(360, -1)
        root.pack_start(panel, False, False, 0)

        # Title bar row
        title_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        title_lbl = Gtk.Label(label="TIC·TAC·TOE")
        title_lbl.set_name("title-label")
        title_lbl.set_halign(Gtk.Align.START)
        title_lbl.set_hexpand(True)
        close_btn = Gtk.Button(label="✕")
        close_btn.set_name("close-btn")
        close_btn.get_style_context().add_class("close-btn")
        close_btn.connect("clicked", self._close)
        title_row.pack_start(title_lbl, True, True, 0)
        title_row.pack_end(close_btn, False, False, 0)
        panel.pack_start(title_row, False, False, 12)

        # Separator
        sep = Gtk.Box()
        sep.get_style_context().add_class("hr")
        sep.set_size_request(-1, 1)
        panel.pack_start(sep, False, False, 0)

        # Status card
        status_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        status_card.set_name("status-card")

        self.turn_label = Gtk.Label()
        self.turn_label.set_name("turn-label")
        self.turn_label.set_halign(Gtk.Align.START)
        self.turn_label.set_markup(self._turn_markup())
        status_card.pack_start(self.turn_label, False, False, 0)

        self.status_label = Gtk.Label(label="Not calibrated")
        self.status_label.set_name("status-label")
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_line_wrap(True)
        status_card.pack_start(self.status_label, False, False, 0)

        self.move_label = Gtk.Label()
        self.move_label.set_name("move-label")
        self.move_label.set_halign(Gtk.Align.START)
        self.move_label.set_markup(self._move_markup())
        status_card.pack_start(self.move_label, False, False, 0)

        panel.pack_start(status_card, False, False, 14)

        # Spacer
        panel.pack_start(Gtk.Box(), True, True, 0)

        # Stop / Resume button
        self.stop_btn = Gtk.Button(label="⏹  Stop")
        self.stop_btn.get_style_context().add_class("stop-btn")
        self.stop_btn.set_size_request(320, 58)
        self.stop_btn.set_sensitive(False)
        self.stop_btn.connect("clicked", self._on_stop)
        panel.pack_start(self.stop_btn, False, False, 4)

        # Action button (Calibrate → Start Game → Restart)
        self.action_btn = Gtk.Button(label="⚙  Calibrate")
        self.action_btn.get_style_context().add_class("action-btn")
        self.action_btn.set_size_request(320, 58)
        self.action_btn.connect("clicked", self._on_action)
        panel.pack_start(self.action_btn, False, False, 4)

        # Bottom padding
        panel.pack_start(Gtk.Box(), False, False, 16)

    # -----------------------------------------------------------------------
    # Placeholder shown before first frame
    # -----------------------------------------------------------------------

    def _show_placeholder(self):
        # 1×1 dark blue pixel as placeholder pixbuf
        pb = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, 1, 1)
        pb.fill(0x060E1AFF)
        self.feed_image.set_from_pixbuf(pb)

    # -----------------------------------------------------------------------
    # Frame rendering
    # -----------------------------------------------------------------------

    def _render_jpeg(self, jpeg_bytes: bytes):
        """Load JPEG bytes into a GdkPixbuf and update the Gtk.Image widget."""
        try:
            loader = GdkPixbuf.PixbufLoader.new_with_type("jpeg")
            loader.write(jpeg_bytes)
            loader.close()
            pixbuf = loader.get_pixbuf()
            if pixbuf is None:
                return

            # Scale to fit the allocated area while keeping aspect ratio
            alloc = self.feed_image.get_allocation()
            w, h = alloc.width, alloc.height
            if w > 4 and h > 4:
                src_w = pixbuf.get_width()
                src_h = pixbuf.get_height()
                scale = min(w / src_w, h / src_h)
                nw = max(1, int(src_w * scale))
                nh = max(1, int(src_h * scale))
                pixbuf = pixbuf.scale_simple(nw, nh, GdkPixbuf.InterpType.BILINEAR)

            GLib.idle_add(self.feed_image.set_from_pixbuf, pixbuf)
        except Exception as exc:
            print(f"[UI] Frame render error: {exc}", flush=True)

    # -----------------------------------------------------------------------
    # stdin listener — runs in a daemon thread
    # -----------------------------------------------------------------------

    def _stdin_listener(self):
        """
        Read from binary stdin.  Two packet types:
          1. Text command: a UTF-8 line (does NOT start with b'FRAME ')
          2. Binary frame: line  b'FRAME <size>\\n'  then  <size> raw bytes
        """
        # Python reopens sys.stdin as a text stream; use the underlying buffer.
        stdin_bin = sys.stdin.buffer

        while True:
            try:
                header = stdin_bin.readline()
                if not header:
                    break  # EOF — parent closed the pipe

                if header.startswith(b"FRAME "):
                    # Binary frame packet
                    try:
                        size = int(header[6:].strip())
                    except ValueError:
                        continue
                    data = b""
                    remaining = size
                    while remaining > 0:
                        chunk = stdin_bin.read(remaining)
                        if not chunk:
                            break
                        data += chunk
                        remaining -= len(chunk)
                    if len(data) == size:
                        self._render_jpeg(data)
                else:
                    # Text command
                    cmd = header.decode("utf-8", errors="replace").strip()
                    if not cmd:
                        continue
                    GLib.idle_add(self._handle_command, cmd)

            except Exception as exc:
                print(f"[UI] stdin error: {exc}", flush=True)
                break

    # -----------------------------------------------------------------------
    # Command dispatch
    # -----------------------------------------------------------------------

    def _handle_command(self, cmd: str):
        if cmd.startswith("move "):
            parts = cmd.split()
            if len(parts) == 4:
                try:
                    r, c = int(parts[1]), int(parts[2])
                    player = parts[3].upper()
                    if player in ("X", "O") and 0 <= r < 3 and 0 <= c < 3:
                        self._record_move(r, c, player)
                except Exception:
                    pass

        elif cmd == "reset":
            self._reset_state()

        elif cmd == "CALIBRATION_DONE":
            self._on_calibration_done()

        elif cmd == "CHEAT_DETECTED":
            self._on_cheat()

        elif cmd == "BOARD_NOT_CLEAR":
            self.status_label.set_markup(
                '<span foreground="#EF5350">Clear the board first!</span>')

        elif cmd.startswith("turn "):
            p = cmd[5:].strip()
            if p in ("X", "O"):
                self.current_player = p
                self.turn_label.set_markup(self._turn_markup())

        elif cmd == "close":
            self._close()

        return False  # required for GLib.idle_add

    # -----------------------------------------------------------------------
    # Game state helpers
    # -----------------------------------------------------------------------

    def _record_move(self, r: int, c: int, player: str):
        self.board[r][c] = player
        self.move_count += 1
        self.move_label.set_markup(self._move_markup())
        # Check win / draw
        if self._check_winner(player):
            who = "Human (X)" if player == "X" else "Robot (O)"
            self.status_label.set_markup(
                f'<span foreground="{GREEN_WIN if player=="X" else RED_LOSS}">'
                f'<b>{who} wins!</b></span>')
            self.game_active = False
            color = GREEN_WIN if player == "X" else RED_LOSS
            msg   = "You Win! 🎉" if player == "X" else "Robot Wins!"
            self._flash_popup(msg, color)
        elif self.move_count >= 9:
            self.status_label.set_markup(
                f'<span foreground="{GRAY_TIE}"><b>It\'s a draw!</b></span>')
            self.game_active = False
            self._flash_popup("Draw! 🤝", GRAY_TIE)

    def _reset_state(self):
        self.board        = [[None]*3 for _ in range(3)]
        self.current_player = "X"
        self.move_count   = 0
        self.game_active  = True
        self.turn_label.set_markup(self._turn_markup())
        self.move_label.set_markup(self._move_markup())
        self.status_label.set_markup(
            '<span foreground="#90BCDC">Game in progress…</span>')
        self.action_btn.set_label("↺  Restart")
        self.stop_btn.set_sensitive(True)
        self.stop_btn.set_label("⏹  Stop")

    def _on_calibration_done(self):
        self.calibration_done = True
        self.action_btn.set_label("▶  Start Game")
        self.action_btn.set_sensitive(True)
        self.status_label.set_markup(
            '<span foreground="#00D4FF">Calibration done! Press Start.</span>')

    def _on_cheat(self):
        self.game_active = False
        self.status_label.set_markup(
            '<span foreground="#EF5350" font_weight="bold">INVALID MOVE!</span>')
        self._flash_popup("Invalid move!", RED_LOSS)

    def _check_winner(self, p: str) -> bool:
        b = self.board
        for r in range(3):
            if b[r][0] == b[r][1] == b[r][2] == p: return True
        for c in range(3):
            if b[0][c] == b[1][c] == b[2][c] == p: return True
        if b[0][0] == b[1][1] == b[2][2] == p: return True
        if b[0][2] == b[1][1] == b[2][0] == p: return True
        return False

    # -----------------------------------------------------------------------
    # Button handlers
    # -----------------------------------------------------------------------

    def _on_action(self, _btn=None):
        label = self.action_btn.get_label()
        if "Calibrate" in label:
            self.status_label.set_markup(
                '<span foreground="#00D4FF">Calibrating…</span>')
            self.action_btn.set_label("Calibrating…")
            self.action_btn.set_sensitive(False)
            print("CALIBRATE", flush=True)

        elif "Start" in label:
            self._reset_state()
            print("START_GAME", flush=True)

        else:  # Restart
            self._reset_state()
            print("RESTARTED", flush=True)

    def _on_stop(self, _btn=None):
        if "Stop" in self.stop_btn.get_label():
            self.stop_btn.set_label("▶  Resume")
            print("STOP", flush=True)
        else:
            self.stop_btn.set_label("⏹  Stop")
            print("RESUME", flush=True)

    def _close(self, *_):
        print("CLOSED", flush=True)
        Gtk.main_quit()

    # -----------------------------------------------------------------------
    # Markup helpers
    # -----------------------------------------------------------------------

    def _turn_markup(self) -> str:
        if self.current_player == "X":
            return f'<span foreground="{RED}">Turn: X — Human</span>'
        return f'<span foreground="{YELLOW}">Turn: O — Robot</span>'

    def _move_markup(self) -> str:
        return f'<span foreground="#6A8FAF">Moves: {self.move_count} / 9</span>'

    # -----------------------------------------------------------------------
    # Overlay flash popup
    # -----------------------------------------------------------------------

    def _flash_popup(self, message: str, color: str):
        popup = Gtk.Window(type=Gtk.WindowType.POPUP)
        popup.set_transient_for(self)
        popup.set_decorated(False)
        popup.set_modal(True)
        popup.set_keep_above(True)
        popup.set_position(Gtk.WindowPosition.CENTER_ALWAYS)

        visual = self.get_screen().get_rgba_visual()
        if visual:
            popup.set_visual(visual)
        popup.set_app_paintable(True)

        lbl = Gtk.Label()
        lbl.set_name("popup-label")
        lbl.set_markup(
            f'<span foreground="{color}" font="72" weight="bold">{message}</span>')
        lbl.set_justify(Gtk.Justification.CENTER)

        eb = Gtk.EventBox()
        rgba = Gdk.RGBA(0, 0, 0, 0.72)
        eb.override_background_color(Gtk.StateFlags.NORMAL, rgba)
        eb.add(lbl)
        popup.add(eb)
        popup.set_size_request(780, 260)
        popup.show_all()

        # Centre on parent
        px, py = self.get_position()
        pw, ph = self.get_size()
        popup.move(px + (pw - 780) // 2, py + (ph - 260) // 2)

        # Flash 3 times then destroy
        flashes = [1.0, 0.05, 1.0, 0.05, 1.0]
        delay   = int(4800 // len(flashes))

        def _step(i=[0]):
            lbl.set_opacity(flashes[i[0] % len(flashes)])
            i[0] += 1
            if i[0] < len(flashes):
                GLib.timeout_add(delay, _step)
                return False
            popup.hide()
            popup.destroy()
            return False

        _step()

    # -----------------------------------------------------------------------
    # Keep raised
    # -----------------------------------------------------------------------

    def _keep_raised(self):
        while True:
            time.sleep(0.5)
            GLib.idle_add(self.present)
            GLib.idle_add(self.set_keep_above, True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    win = TicTacToeUI()
    win.connect("destroy", Gtk.main_quit)
    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\n[UI] Closed by user.", flush=True)


if __name__ == "__main__":
    main()
