import gi
import threading
import time
import sys
import subprocess
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

NAVY = "#001F54"
RED = "#FF4545"
YELLOW = "#FFD600"
EMPTY_BTN_BG = "#FFFFFF"
LIGHT_RED_BG = "#E89090"
LIGHT_YELLOW_BG = "#FFE68C"
GREEN_WIN = "#3CB371"
RED_LOSS = "#B22222"
GRAY_TIE = "#5A5A5A"


RESTART_CSS = b"""
button.restart-modern {
    background: linear-gradient(90deg, #f8f8f8 70%, #e6e9ef 100%);
    color: #001F54;
    font-size: 26px;
    font-weight: bold;
    border-radius: 16px;
    border: 2px solid #e0e7ef;
    box-shadow: 0 4px 12px #e0e0ee55;
    margin-top: 20px;
    margin-bottom: 10px;
    letter-spacing: 1px;
    padding: 12px 0;
    transition: background 0.2s;
}
button.restart-modern:hover {
    background: linear-gradient(90deg, #e1f5fe 60%, #b3e5fc 100%);
    color: #01579b;
    border: 2px solid #90caf9;
}
"""

class TicTacToeGame(Gtk.Window):
    def __init__(self):
        super().__init__()
        self.set_decorated(False)
        self.set_focus_on_map(True)
        self.set_accept_focus(True)
        self.fullscreen()
        self.set_keep_above(True)
        self.present()
        self.set_type_hint(Gdk.WindowTypeHint.SPLASHSCREEN)
        self.grab_focus()
        threading.Thread(target=self.keep_raising, daemon=True).start()

        self.current_player = "X"
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.game_active = True
        self.move_count = 0

        self.restart_css_provider = Gtk.CssProvider()
        self.restart_css_provider.load_from_data(RESTART_CSS)
        self.bg_css_provider = Gtk.CssProvider()

        self.calibration_done = False # Game starts only after calibration


        main_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(main_container)

        title_bar = Gtk.EventBox()
        titlebar_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        def drag_begin(widget, event):
            self.begin_move_drag(event.button, int(event.x_root), int(event.y_root), event.time)
        title_bar.connect('button-press-event', drag_begin)
        self.title_label = Gtk.Label()
        self.title_label.set_markup(
            f'<span foreground="{NAVY}" font_weight="bold" font="32">TicTacToe Demo</span>')
        self.title_label.set_halign(Gtk.Align.CENTER)
        titlebar_hbox.pack_start(self.title_label, True, True, 0)
        close_btn = Gtk.Button()
        close_btn.set_tooltip_text("Close")
        close_btn.set_size_request(57, 54)
        close_icon = Gtk.Label()
        close_icon.set_markup(f'<span foreground="{NAVY}" font="30" weight="bold">X</span>')
        close_btn.add(close_icon)
        close_btn.set_relief(Gtk.ReliefStyle.NONE)
        close_btn.connect("clicked", self.close_gui)
        titlebar_hbox.pack_end(close_btn, False, False, 12)
        titlebar_hbox.set_size_request(-1, 54)
        title_bar.add(titlebar_hbox)
        main_container.pack_start(title_bar, False, False, 0)

        content_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=32)
        main_container.pack_start(content_hbox, True, True, 0)

        game_container = Gtk.Frame()
        game_container.set_shadow_type(Gtk.ShadowType.ETCHED_OUT)
        content_hbox.pack_start(game_container, True, True, 0)

        inner_game_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        inner_game_box.set_margin_top(12)
        inner_game_box.set_margin_bottom(12)
        inner_game_box.set_margin_start(12)
        inner_game_box.set_margin_end(12)
        game_container.add(inner_game_box)

        self.game_grid = Gtk.Grid()
        self.game_grid.set_row_spacing(7)
        self.game_grid.set_column_spacing(7)
        inner_game_box.pack_start(self.game_grid, True, True, 0)

        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                button = Gtk.Button(label="")
                button.set_hexpand(True)
                button.set_vexpand(True)
                button.set_size_request(120, 120)
                self.game_grid.attach(button, j, i, 1, 1)
                row.append(button)
            self.buttons.append(row)

        info_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        info_container.set_size_request(500, -1)
        content_hbox.pack_start(info_container, False, False, 0)
        info_container.set_margin_top(8)

        self.subtitle_label = Gtk.Label()
        self.subtitle_label.set_markup(
            f'<span foreground="{NAVY}" font="22">TicTacToe</span>'
        )
        info_container.pack_start(self.subtitle_label, False, False, 0)

        self.turn_label = Gtk.Label()
        self.turn_label.set_justify(Gtk.Justification.CENTER)
        self.turn_label.set_markup(self.get_turn_label_markup())
        info_container.pack_start(self.turn_label, False, False, 0)

        self.status_label = Gtk.Label()
        self.status_label.set_markup(
            f'<span foreground="{NAVY}" font="28">Not Calibrated</span>')
        self.status_label.set_justify(Gtk.Justification.CENTER)

        self.status_label.set_line_wrap(True)
        self.status_label.set_line_wrap_mode(Gtk.WrapMode.WORD)
        self.status_label.set_size_request(500, 72)
        info_container.pack_start(self.status_label, False, False, 6)

        self.move_label = Gtk.Label()
        self.move_label.set_markup(self.get_move_label_markup())
        info_container.pack_start(self.move_label, False, False, 0)

        badge_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        info_container.pack_start(badge_hbox, False, False, 0)

        self.player_robot_label = Gtk.Label()
        badge_hbox.pack_start(self.player_robot_label, True, True, 0)
        self.player_user_label = Gtk.Label()
        badge_hbox.pack_start(self.player_user_label, True, True, 0)
        self.update_player_badges()

        # Spacer to push buttons to the bottom of the container
        spacer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        info_container.pack_start(spacer, True, True, 0)

        # Stop/Resume Button Row (Placed above Restart)
        stop_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.set_size_request(500, 68)
        self.stop_button.get_style_context().add_class("restart-modern") 
        self.stop_button.set_sensitive(False)
        self.stop_button.connect("clicked", self.on_stop_clicked)
        
        # Style for stop button
        self.stop_css_provider = Gtk.CssProvider()
        STOP_CSS = b"""
        button.stop-btn {
            background: #FFCDD2;
            color: #B71C1C;
            font-size: 26px;
            font-weight: bold;
            border-radius: 16px;
            border: 2px solid #EF9A9A;
            box-shadow: 0 4px 12px #e0e0ee55;
            padding: 12px 0;
        }
        button.stop-btn:hover {
            background: #EF9A9A;
            color: #B71C1C;
        }
        """
        self.stop_css_provider.load_from_data(STOP_CSS)
        self.stop_button.get_style_context().add_class("stop-btn")
        Gtk.StyleContext.add_provider(
            self.stop_button.get_style_context(),
            self.stop_css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        stop_hbox.pack_start(self.stop_button, True, True, 0)
        info_container.pack_start(stop_hbox, False, False, 10)

        # Restart/Calibrate Button Row (Bottom)
        restart_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.action_button = Gtk.Button(label="Calibrate")
        self.action_button.set_size_request(500, 68)
        self.action_button.get_style_context().add_class("restart-modern")
        self.action_button.set_can_focus(False)
        self.action_button.connect("clicked", self.on_action_clicked)
        Gtk.StyleContext.add_provider(
            self.action_button.get_style_context(),
            self.restart_css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        restart_hbox.pack_start(self.action_button, True, True, 0)
        info_container.pack_start(restart_hbox, False, False, 16)



        self.update_board_visual()
        threading.Thread(target=self.terminal_info_loop, daemon=True).start()
        self.update_app_bg()
        threading.Thread(target=self.stdin_listener, daemon=True).start()

    def keep_raising(self):
        while True:
            time.sleep(0.5)
            GLib.idle_add(self.present)
            GLib.idle_add(self.grab_focus)
            GLib.idle_add(self.set_keep_above, True)

    def on_resize(self, widget, allocation):
        side = min(allocation.width // 2, allocation.height - 48)
        btn_size = int(0.84 * (side // 3))
        btn_font = int(btn_size * 0.75)  # Increased from 0.55 to 0.75 for bigger markers
        for i, row in enumerate(self.buttons):
            for j, btn in enumerate(row):
                btn.set_size_request(btn_size, btn_size)
                style = Gtk.CssProvider()
                if self.board[i][j] == "X":
                    style.load_from_data(f"button {{background: {RED}; color: {NAVY}; font-size: {btn_font}px; font-weight: bold; border: 3px solid {NAVY}; border-radius: 13px;}}".encode())
                elif self.board[i][j] == "O":
                    style.load_from_data(f"button {{background: {YELLOW}; color: {NAVY}; font-size: {btn_font}px; font-weight: bold; border: 3px solid {NAVY}; border-radius: 13px;}}".encode())
                else:
                    style.load_from_data(f"button {{background: {EMPTY_BTN_BG}; color: {NAVY}; font-size: {btn_font}px; font-weight: bold; border: 3px solid {NAVY}; border-radius: 13px;}}".encode())
                Gtk.StyleContext.add_provider(
                    btn.get_style_context(), style, Gtk.STYLE_PROVIDER_PRIORITY_USER)

    def stdin_listener(self):
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip()
            if cmd.startswith("move "):
                parts = cmd.split()
                if len(parts) == 4:
                    try:
                        x = int(parts[1])
                        y = int(parts[2])
                        player = parts[3].upper()
                        if player in ("X", "O"):
                            GLib.idle_add(self.remote_move, x, y, player)
                    except:
                        pass
            elif cmd == "reset":
                GLib.idle_add(self.restart_game, None)
            elif cmd == "CALIBRATION_DONE":
                GLib.idle_add(self.on_calibration_done)
            elif cmd == "CHEAT_DETECTED":
                GLib.idle_add(self.on_cheat_detected)
            elif cmd == "BOARD_NOT_CLEAR":
                GLib.idle_add(self.on_board_not_clear)
            elif cmd == "close":
                GLib.idle_add(self.close_gui)
                break


    def remote_move(self, x, y, player):
        if (
            self.game_active and 0 <= x < 3 and 0 <= y < 3 and
            self.board[x][y] is None):
            self.current_player = player
            self.on_cell_clicked(x, y)

    def update_app_bg(self):
        bgcol = LIGHT_RED_BG if self.current_player == "X" else LIGHT_YELLOW_BG
        self.bg_css_provider.load_from_data(
            f"window {{ background-color: {bgcol}; }}".encode())
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            self.bg_css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def get_turn_label_markup(self):
        if self.current_player == "X":
            return (
                f'<span foreground="{NAVY}" font="28"><b>Turn: X - Human</b></span>'
            )
        else:
            return (
                f'<span foreground="{NAVY}" font="28"><b>Turn: O - Robots</b></span>'
            )

    def get_move_label_markup(self):
        return f'<span foreground="{NAVY}" font="22">Moves: {self.move_count} / 9</span>'

    def update_player_badges(self):
        self.player_user_label.set_markup(
            f'<span foreground="{NAVY}" font="18" weight="bold">X - Human</span>'
        )
        self.player_robot_label.set_markup(
            f'<span foreground="{NAVY}" font="18" weight="bold">O - Robot</span>'
        )

    def show_overlay_popup(self, message, color="#3CB371", font_size=80):
        popup = Gtk.Window(type=Gtk.WindowType.POPUP)
        popup.set_transient_for(self)
        popup.set_decorated(False)
        popup.set_resizable(False)
        popup.set_skip_taskbar_hint(True)
        popup.set_accept_focus(False)
        popup.set_modal(True)
        popup.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
        visual = self.get_screen().get_rgba_visual()
        if visual:
            popup.set_visual(visual)
        label = Gtk.Label()
        label.set_markup(
            f'<span foreground="{color}" font="{font_size}" weight="bold">{message}</span>'
        )
        label.set_justify(Gtk.Justification.CENTER)
        label.set_halign(Gtk.Align.CENTER)
        label.set_valign(Gtk.Align.CENTER)
        overlay_box = Gtk.EventBox()
        rgba = Gdk.RGBA(0,0,0,0)  # fully transparent
        overlay_box.override_background_color(Gtk.StateFlags.NORMAL, rgba)
        overlay_box.add(label)
        popup.add(overlay_box)
        popup.set_size_request(850, 310)
        popup.set_app_paintable(True)
        popup.set_keep_above(True)
        popup.set_gravity(Gdk.Gravity.CENTER)
        popup.show_all()
        parent_x, parent_y = self.get_position()
        parent_w, parent_h = self.get_size()
        popup.move(
            parent_x + (parent_w - 850) // 2,
            parent_y + (parent_h - 310) // 2
        )
        flashes = [True, False, True, False, True]
        flash_delay = int(5000 / len(flashes))
        def flash_step(i=[0]):
            show = flashes[i[0] % len(flashes)]
            label.set_opacity(1.0 if show else 0.04)
            i[0] += 1
            if i[0] < len(flashes):
                GLib.timeout_add(flash_delay, flash_step)
                return False
            popup.hide()
            popup.destroy()
            return False
        flash_step()

    def on_cell_clicked(self, x, y):
        if not self.calibration_done:
            return
        if self.game_active and self.board[x][y] is None:

            self.board[x][y] = self.current_player
            self.buttons[x][y].set_label(self.current_player)
            self.move_count += 1
            self.update_board_visual()
            if self.check_winner(self.current_player):
                user_won = self.current_player == "X"
                win_msg = "User (X) Wins!" if user_won else "Robotic Arm (O) Wins!"
                self.status_label.set_markup(
                    f'<span foreground="{NAVY}" font="28" weight="bold">{win_msg}</span>')
                self.game_active = False
                if user_won:
                    self.show_overlay_popup("You Win!", color=GREEN_WIN, font_size=88)
                else:
                    self.show_overlay_popup("You Lost", color=RED_LOSS, font_size=88)
            elif self.move_count == 9:
                self.status_label.set_markup(
                    f'<span foreground="{NAVY}" font="24"><b>It\'s a DRAW!</b></span>')
                self.game_active = False
                self.show_overlay_popup("Tie!", color=GRAY_TIE, font_size=85)
            else:
                self.current_player = "O" if self.current_player == "X" else "X"
                self.turn_label.set_markup(self.get_turn_label_markup())
                self.move_label.set_markup(self.get_move_label_markup())
                self.update_app_bg()

    def check_winner(self, player):
        for row in self.board:
            if row[0] == row[1] == row[2] == player: return True
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] == player: return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == player: return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] == player: return True
        return False

    def on_action_clicked(self, button=None):
        if not self.calibration_done:
            # State: Not Calibrated -> Calibrating
            self.status_label.set_markup(f'<span foreground="{NAVY}" font="28">Calibrating...</span>')
            self.action_button.set_label("Calibrating...")
            self.action_button.set_sensitive(False) # Disable while calibrating
            print("CALIBRATE", flush=True)
        elif self.action_button.get_label() == "Start Game":
            # State: Calibrated -> Start Game
            self.restart_game()
            self.action_button.set_label("Restart Game")
            self.stop_button.set_sensitive(True)
            self.status_label.set_markup(f'<span foreground="{NAVY}" font="28">Game Started!</span>')
            print("START_GAME", flush=True)

        else:

            # State: Game Active -> Restart
            self.restart_game()
            print("RESTARTED", flush=True)

    def on_calibration_done(self):
        self.calibration_done = True
        self.action_button.set_sensitive(True)
        self.action_button.set_label("Start Game")
        self.status_label.set_markup(f'<span foreground="{NAVY}" font="24">Calibration Done.</span>')
        # Do not restart automatically. User must clear board and click Start.

    def on_cheat_detected(self):
        self.game_active = False
        self.status_label.set_markup(f'<span foreground="{RED_LOSS}" font="32" weight="bold">INVALID MOVE!</span>')
        self.show_overlay_popup("Invalid Move!", color=RED_LOSS, font_size=88)

    def on_board_not_clear(self):
        self.status_label.set_markup(f'<span foreground="{RED_LOSS}" font="28" weight="bold">Clear Board First!</span>')
        print("Board not clear - please remove all balls")

    def close_gui(self, *args):
        print("CLOSED", flush=True)
        Gtk.main_quit()

    def on_stop_clicked(self, button):
        if self.stop_button.get_label() == "Stop":
            self.stop_button.set_label("Resume")
            print("STOP", flush=True)
        else:
            self.stop_button.set_label("Stop")
            print("RESUME", flush=True)

    def restart_game(self, button=None):
        self.action_button.set_label("Restart Game")
        self.stop_button.set_label("Stop")
        self.stop_button.set_sensitive(True)
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.game_active = True
        self.move_count = 0
        for row in self.buttons:
            for btn in row:
                btn.set_label("")
        self.turn_label.set_markup(self.get_turn_label_markup())
        self.status_label.set_markup(
            f'<span foreground="{NAVY}" font="19">Game in Progress...</span>')
        self.move_label.set_markup(self.get_move_label_markup())
        self.update_board_visual()
        self.update_app_bg()

    def update_board_visual(self):
        for i, row in enumerate(self.buttons):
            for j, btn in enumerate(row):
                style = Gtk.CssProvider()
                if self.board[i][j] == "X":
                    btn.set_label("X")
                    style.load_from_data(f"button {{background: {RED}; color: {NAVY}; font-size: 72px; font-weight: bold; border: 3px solid {NAVY}; border-radius: 13px; }}".encode())
                elif self.board[i][j] == "O":
                    btn.set_label("O")
                    style.load_from_data(f"button {{background: {YELLOW}; color: {NAVY}; font-size: 72px; font-weight: bold; border: 3px solid {NAVY}; border-radius: 13px; }}".encode())
                else:
                    btn.set_label("")
                    style.load_from_data(f"button {{background: {EMPTY_BTN_BG}; color: {NAVY}; font-weight: bold; border: 3px solid {NAVY}; border-radius: 13px;}}".encode())
                Gtk.StyleContext.add_provider(
                    btn.get_style_context(), style, Gtk.STYLE_PROVIDER_PRIORITY_USER)

    def terminal_info_loop(self):
        while True:
            if hasattr(self, 'game_active'):
                if self.game_active:
                    turn = "X - User" if self.current_player == "X" else "O - Robotic Arm"
                    print(f"Turn: {turn} | Moves: {self.move_count}/9")
                    print("Current Board:")
                    for i, row in enumerate(self.board):
                        row_display = []
                        for j, cell in enumerate(row):
                            if cell is None:
                                row_display.append(f"[{i*3+j+1}]")
                            else:
                                row_display.append(f" {cell} ")
                        print("  " + " | ".join(row_display))
                        if i < 2:
                            print("  " + "-" * 15)
                    print("-" * 30)
                time.sleep(3)
            else:
                time.sleep(1)

def main():
    app = TicTacToeGame()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\nGame closed by user.")

if __name__ == "__main__":
    main()


