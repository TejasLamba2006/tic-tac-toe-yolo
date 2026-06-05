from __future__ import annotations

import threading
import tkinter as tk

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:  # Pillow not installed
    _PIL_AVAILABLE = False

from ..ai.yolo_inference import YoloInference
from ..main import AppConfig, FrameAnalysis, analyze_frame
from ..vision.board_detector import BoardDetector
from ..vision.board_state import BoardStateEstimator, format_board
from ..vision.camera import configure_capture, open_camera, parse_camera_source
from ..vision.perspective import canonical_cell_polygon, warp_points
from ..vision.stability import BoardGeometryTracker


FONT_FAMILY = "Segoe UI"


class TkinterVisionApp:
    """Tkinter dashboard that shows board predictions and the recommended move."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.root = tk.Tk()
        self.root.title("Tic-Tac-Toe Vision")
        self.root.configure(bg="#f3f4f6")
        self.root.geometry("1540x820")
        self.root.minsize(1300, 720)

        self._stop_event = threading.Event()
        self._analysis_lock = threading.Lock()
        self._settings_lock = threading.Lock()
        self._latest_analysis: FrameAnalysis | None = None
        self._worker_error: str | None = None
        self._minimum_confidence = config.board_min_confidence
        self._rotation = 0  # 0=0°, 1=90°CW, 2=180°, 3=270°CW

        self.status_text = tk.StringVar(value="Starting camera...")
        self.move_text = tk.StringVar(
            value="Best move: waiting for prediction")
        self.details_text = tk.StringVar(value="")
        self.confidence_text = tk.StringVar(
            value=f"{self._minimum_confidence:.2f}")
        self.rotation_text = tk.StringVar(value="0°")
        self._show_warped = tk.BooleanVar(value=False)
        self._warped_photo = None   # strong reference to prevent GC
        self._camera_photo = None   # strong reference for live camera feed

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(60, self._refresh_ui)

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def run(self) -> int:
        try:
            self.root.mainloop()
        finally:
            self._stop_event.set()
            if self._worker.is_alive():
                self._worker.join(timeout=1.5)
        return 0

    def close(self) -> None:
        self._stop_event.set()
        if self.root.winfo_exists():
            self.root.destroy()

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#f3f4f6")
        header.pack(fill="x", padx=18, pady=(16, 8))

        title = tk.Label(
            header,
            text="Tic-Tac-Toe Vision Dashboard",
            bg="#f3f4f6",
            fg="#111827",
            font=(FONT_FAMILY, 18, "bold"),
        )
        title.pack(side="left")

        control_card = tk.Frame(
            self.root, bg="#ffffff", highlightbackground="#d1d5db", highlightthickness=1)
        control_card.pack(fill="x", padx=18, pady=(0, 12))

        control_left = tk.Frame(control_card, bg="#ffffff")
        control_left.pack(side="left", fill="x", expand=True, padx=16, pady=14)

        tk.Label(
            control_left,
            text="Matrix minimum probability",
            bg="#ffffff",
            fg="#111827",
            font=(FONT_FAMILY, 10, "bold"),
        ).pack(anchor="w")

        scale_row = tk.Frame(control_left, bg="#ffffff")
        scale_row.pack(fill="x", pady=(6, 0))

        self.confidence_scale = tk.Scale(
            scale_row,
            from_=0.05,
            to=0.95,
            resolution=0.01,
            orient="horizontal",
            length=320,
            showvalue=False,
            bg="#ffffff",
            troughcolor="#e5e7eb",
            highlightthickness=0,
            command=self._on_confidence_change,
        )
        self.confidence_scale.set(self._minimum_confidence)
        self.confidence_scale.pack(side="left")

        tk.Label(
            scale_row,
            textvariable=self.confidence_text,
            bg="#ffffff",
            fg="#1d4ed8",
            font=(FONT_FAMILY, 11, "bold"),
            width=6,
        ).pack(side="left", padx=(12, 0))

        tk.Label(
            control_left,
            text="Balls with confidence above this value are kept in the matrix, even if the board is busy.",
            bg="#ffffff",
            fg="#4b5563",
            font=(FONT_FAMILY, 9),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        # ── Rotation controls ────────────────────────────────────────────
        control_right = tk.Frame(control_card, bg="#ffffff")
        control_right.pack(side="right", padx=16, pady=14)

        tk.Label(
            control_right,
            text="Rotate Board",
            bg="#ffffff",
            fg="#111827",
            font=(FONT_FAMILY, 10, "bold"),
        ).pack(anchor="center")

        rot_row = tk.Frame(control_right, bg="#ffffff")
        rot_row.pack(pady=(6, 0))

        btn_style = dict(
            bg="#1e40af",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            font=(FONT_FAMILY, 14, "bold"),
            width=3,
            cursor="hand2",
        )
        tk.Button(
            rot_row,
            text="↺",
            command=self._rotate_left,
            **btn_style,
        ).pack(side="left", padx=(0, 6))

        tk.Label(
            rot_row,
            textvariable=self.rotation_text,
            bg="#ffffff",
            fg="#1d4ed8",
            font=(FONT_FAMILY, 11, "bold"),
            width=4,
        ).pack(side="left")

        tk.Button(
            rot_row,
            text="↻",
            command=self._rotate_right,
            **btn_style,
        ).pack(side="left", padx=(6, 0))

        # Show Warped Frame toggle
        tk.Checkbutton(
            control_right,
            text="Show Warped Frame",
            variable=self._show_warped,
            command=self._toggle_warped_frame,
            bg="#ffffff",
            fg="#111827",
            activebackground="#ffffff",
            selectcolor="#dbeafe",
            font=(FONT_FAMILY, 9, "bold"),
            cursor="hand2",
        ).pack(anchor="center", pady=(8, 0))
        # ────────────────────────────────────────────────────────────────

        tk.Label(
            control_card,
            textvariable=self.status_text,
            bg="#111827",
            fg="#f9fafb",
            font=(FONT_FAMILY, 10, "bold"),
            padx=14,
            pady=10,
        ).pack(side="right", fill="y")

        body = tk.Frame(self.root, bg="#f3f4f6")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        body.columnconfigure(0, weight=3)  # live camera feed
        body.columnconfigure(1, weight=2)  # prediction board
        body.columnconfigure(2, weight=2)  # move details
        body.rowconfigure(0, weight=1)

        # ── Live camera feed (col 0) ─────────────────────────────────
        camera_panel = tk.Frame(
            body, bg="#ffffff", highlightbackground="#d1d5db", highlightthickness=1)
        camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        camera_panel.rowconfigure(1, weight=1)
        camera_panel.columnconfigure(0, weight=1)

        tk.Label(
            camera_panel,
            text="Live Camera Feed",
            bg="#ffffff",
            fg="#111827",
            font=(FONT_FAMILY, 14, "bold"),
            padx=16,
            pady=14,
        ).grid(row=0, column=0, sticky="ew")

        self.camera_canvas = tk.Canvas(
            camera_panel,
            bg="#0f172a",
            highlightthickness=0,
        )
        self.camera_canvas.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        # ────────────────────────────────────────────────────────────────

        # ── Prediction board (col 1) ────────────────────────────────
        board_panel = tk.Frame(
            body, bg="#ffffff", highlightbackground="#d1d5db", highlightthickness=1)
        board_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        board_panel.rowconfigure(1, weight=1)
        board_panel.columnconfigure(0, weight=1)

        tk.Label(
            board_panel,
            text="Prediction Board",
            bg="#ffffff",
            fg="#111827",
            font=(FONT_FAMILY, 14, "bold"),
            padx=16,
            pady=14,
        ).grid(row=0, column=0, sticky="ew")

        self.board_canvas = tk.Canvas(
            board_panel,
            width=620,
            height=620,
            bg="#0f172a",
            highlightthickness=0,
        )
        self.board_canvas.grid(
            row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

        self.board_canvas.bind("<Configure>", self._redraw_current_board)

        self.info_panel = tk.Frame(
            body, bg="#ffffff", highlightbackground="#d1d5db", highlightthickness=1)
        self.info_panel.grid(row=0, column=2, sticky="nsew")
        self.info_panel.rowconfigure(2, weight=1)
        self.info_panel.columnconfigure(0, weight=1)
        info_panel = self.info_panel  # local alias for the rest of _build_layout

        tk.Label(
            info_panel,
            text="Move and Matrix Details",
            bg="#ffffff",
            fg="#111827",
            font=(FONT_FAMILY, 14, "bold"),
            padx=16,
            pady=14,
        ).grid(row=0, column=0, sticky="ew")

        tk.Label(
            info_panel,
            textvariable=self.move_text,
            bg="#ffffff",
            fg="#b45309",
            font=(FONT_FAMILY, 11, "bold"),
            wraplength=360,
            justify="left",
            padx=16,
        ).grid(row=1, column=0, sticky="ew")

        self.details_box = tk.Text(
            info_panel,
            width=42,
            height=24,
            bg="#111827",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            padx=12,
            pady=12,
            font=("Consolas", 10),
            wrap="none",
        )
        self.details_box.grid(
            row=2, column=0, sticky="nsew", padx=16, pady=(10, 12))
        self.details_box.configure(state="disabled")

        # Warped-frame canvas — hidden until toggle is switched on
        self.warped_canvas = tk.Canvas(
            info_panel,
            bg="#0f172a",
            highlightthickness=0,
            height=220,
        )
        # Not gridded yet; shown by _toggle_warped_frame

    def _on_confidence_change(self, value: str) -> None:
        probability = float(value)
        with self._settings_lock:
            self._minimum_confidence = probability
        self.confidence_text.set(f"{probability:.2f}")

    # ── Rotation helpers ─────────────────────────────────────────────────

    def _rotate_left(self) -> None:
        self._rotation = (self._rotation - 1) % 4
        self._update_rotation_label()
        self._redraw_current_board()

    def _rotate_right(self) -> None:
        self._rotation = (self._rotation + 1) % 4
        self._update_rotation_label()
        self._redraw_current_board()

    def _update_rotation_label(self) -> None:
        self.rotation_text.set(f"{self._rotation * 90}°")

    def _apply_rotation(self, board_3x3: list) -> list:
        """Return a new 3×3 grid rotated by self._rotation quarter-turns CW."""
        grid = board_3x3
        for _ in range(self._rotation):
            # One 90° CW step: new[col][2-row] = old[row][col]
            grid = [[grid[2 - c][r] for c in range(3)] for r in range(3)]
        return grid

    def _rotate_move(self, row: int, col: int) -> tuple[int, int]:
        """Map an (row, col) from the original board into the rotated view."""
        r, c = row, col
        for _ in range(self._rotation):
            r, c = c, 2 - r
        return r, c

    # ─────────────────────────────────────────────────────────────────────

    # ── Warped-frame helpers ──────────────────────────────────────────────

    def _toggle_warped_frame(self) -> None:
        """Show or hide the warped-frame canvas when the checkbox is toggled."""
        if self._show_warped.get():
            self.warped_canvas.grid(
                row=3, column=0, sticky="ew", padx=16, pady=(0, 12))
        else:
            self.warped_canvas.grid_remove()

    def _update_warped_canvas(self, analysis: FrameAnalysis) -> None:
        """Render analysis.warped_frame (with grid + detection overlay) onto warped_canvas."""
        if not self._show_warped.get():
            return

        w = max(1, self.warped_canvas.winfo_width())
        h = max(1, self.warped_canvas.winfo_height())

        # Build overlay image (grid lines + detection boxes)
        overlay: np.ndarray = analysis.warped_frame.copy()
        fh, fw = overlay.shape[:2]
        for i in (1, 2):
            cv2.line(overlay, (i * fw // 3, 0), (i * fw // 3, fh), (80, 80, 80), 1, cv2.LINE_AA)
            cv2.line(overlay, (0, i * fh // 3), (fw, i * fh // 3), (80, 80, 80), 1, cv2.LINE_AA)

        _DET_COLORS: dict[str, tuple[int, int, int]] = {
            "red_ball": (60, 80, 255),
            "yellow_ball": (30, 210, 255),
            "empty": (160, 160, 160),
        }
        for det in analysis.detections:
            x1, y1, x2, y2 = (int(v) for v in det.xyxy)
            color = _DET_COLORS.get(det.label, (200, 200, 200))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"{det.label} {det.confidence:.2f}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
            )

        if not _PIL_AVAILABLE:
            self.warped_canvas.delete("all")
            self.warped_canvas.create_text(
                w // 2, h // 2,
                text="Pillow not installed.\npip install Pillow",
                fill="#ef4444",
                font=(FONT_FAMILY, 10, "bold"),
                justify="center",
            )
            return

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
        self._warped_photo = ImageTk.PhotoImage(pil_img)

        self.warped_canvas.delete("all")
        self.warped_canvas.create_image(0, 0, anchor="nw", image=self._warped_photo)

        # Title bar overlay
        is_fallback = analysis.board_result.fallback
        method_text = (
            f"⚠ FALLBACK | {analysis.board_result.method}"
            if is_fallback
            else f"✓ {analysis.board_result.method}"
        )
        badge_color = "#ef4444" if is_fallback else "#22c55e"
        self.warped_canvas.create_rectangle(0, 0, w, 22, fill="#0f172a", outline="")
        self.warped_canvas.create_text(
            w // 2, 11,
            text=method_text,
            fill=badge_color,
            font=(FONT_FAMILY, 9, "bold"),
        )

    # ─────────────────────────────────────────────────────────────────────

    # ── Live camera feed helpers ───────────────────────────────────────────

    _DET_COLORS: dict[str, tuple[int, int, int]] = {
        "red_ball": (60, 80, 255),
        "yellow_ball": (30, 210, 255),
        "empty": (160, 160, 160),
    }

    def _build_annotated_frame(self, analysis: FrameAnalysis) -> np.ndarray:
        """Return the original camera frame annotated with board quad, detection boxes
        (projected from warped space back to original frame), and AI move cell."""
        rendered = analysis.frame.copy()

        # Board quad outline (skip when fallback — corners are the whole frame)
        if not analysis.board_result.fallback:
            corners = analysis.board_result.corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(rendered, [corners], True, (0, 160, 255), 3, cv2.LINE_AA)

        # Detection boxes: project from warped-board space back to original frame
        for det in analysis.detections:
            x1, y1, x2, y2 = det.xyxy
            pts_warped = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            pts_orig = warp_points(
                pts_warped, analysis.transform.inverse_matrix).astype(np.int32)
            color = self._DET_COLORS.get(det.label, (200, 200, 200))
            cv2.polylines(rendered, [pts_orig.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
            top = tuple(pts_orig[0])
            cv2.putText(
                rendered,
                f"{det.label} {det.confidence:.2f}",
                (top[0], max(14, top[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
            )

        # AI move cell projected back to the original frame
        if analysis.decision.recommendation is not None:
            move = analysis.decision.recommendation
            polygon = canonical_cell_polygon(move.row, move.col, analysis.transform.size[0])
            proj = warp_points(polygon, analysis.transform.inverse_matrix)
            cv2.polylines(
                rendered, [proj.astype(np.int32).reshape(-1, 1, 2)],
                True, (0, 255, 255), 3, cv2.LINE_AA,
            )

        return rendered

    def _update_camera_canvas(self, analysis: FrameAnalysis) -> None:
        """Convert the annotated camera frame to a Tkinter PhotoImage and paint it."""
        if not _PIL_AVAILABLE:
            w = max(1, self.camera_canvas.winfo_width())
            h = max(1, self.camera_canvas.winfo_height())
            self.camera_canvas.delete("all")
            self.camera_canvas.create_text(
                w // 2, h // 2,
                text="Pillow not installed.\npip install Pillow",
                fill="#ef4444",
                font=(FONT_FAMILY, 10, "bold"),
                justify="center",
            )
            return

        annotated = self._build_annotated_frame(analysis)
        w = max(1, self.camera_canvas.winfo_width())
        h = max(1, self.camera_canvas.winfo_height())

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
        self._camera_photo = ImageTk.PhotoImage(pil_img)

        self.camera_canvas.delete("all")
        self.camera_canvas.create_image(0, 0, anchor="nw", image=self._camera_photo)

        # Status badge
        is_fallback = analysis.board_result.fallback
        badge_text = (
            f"⚠ FALLBACK | {analysis.board_result.method}"
            if is_fallback
            else f"✓ {analysis.board_result.method}  |  FPS {analysis.fps:.1f}"
        )
        badge_color = "#ef4444" if is_fallback else "#22c55e"
        self.camera_canvas.create_rectangle(0, 0, w, 22, fill="#0f172a", outline="")
        self.camera_canvas.create_text(
            w // 2, 11, text=badge_text, fill=badge_color,
            font=(FONT_FAMILY, 9, "bold"),
        )

    # ─────────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        session = None
        try:
            camera_source = parse_camera_source(self.config.camera)
            session = open_camera(
                camera_source, auto_max_index=self.config.auto_max_index)
            configure_capture(session.capture, self.config.frame_width,
                              self.config.frame_height, self.config.fps)

            detector = YoloInference(
                weights_path=self.config.weights,
                confidence_threshold=self.config.confidence_threshold,
                iou_threshold=self.config.iou_threshold,
                image_size=self.config.image_size,
                device=self.config.device,
            )
            board_detector = BoardDetector()
            board_estimator = BoardStateEstimator(
                minimum_confidence=self.config.board_min_confidence,
                smoothing_window=self.config.smoothing_window,
            )
            geometry_tracker = BoardGeometryTracker()

            self._worker_error = None
            self._set_status("Camera connected. Running prediction loop...")

            while not self._stop_event.is_set():
                ok, frame = session.read()
                if not ok or frame is None:
                    self._set_worker_error(
                        "Failed to read a frame from the camera.")
                    break

                with self._settings_lock:
                    board_estimator.minimum_confidence = self._minimum_confidence

                analysis = analyze_frame(
                    frame=frame,
                    detector=detector,
                    board_detector=board_detector,
                    board_estimator=board_estimator,
                    ai_color=self.config.ai_color,
                    board_size=self.config.board_size,
                    geometry_tracker=geometry_tracker,
                )

                with self._analysis_lock:
                    self._latest_analysis = analysis
                    self._worker_error = None

        except Exception as exc:  # pragma: no cover - depends on hardware/runtime setup
            self._set_worker_error(str(exc))
        finally:
            if session is not None:
                session.release()

    def _set_worker_error(self, message: str) -> None:
        with self._analysis_lock:
            self._worker_error = message
        self._set_status(f"Error: {message}")

    def _set_status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_text.set(message))

    def _refresh_ui(self) -> None:
        with self._analysis_lock:
            analysis = self._latest_analysis
            error = self._worker_error

        if error is not None:
            self.status_text.set(f"Error: {error}")
        elif analysis is None:
            self.status_text.set("Waiting for the first prediction...")
        else:
            self.status_text.set(
                f"FPS {analysis.fps:.1f} | inference {analysis.inference_ms:.1f} ms | detections {len(analysis.detections)}"
            )
            self.move_text.set(self._format_move_text(analysis))
            self._update_details_box(analysis)
            self._redraw_board(analysis)
            self._update_warped_canvas(analysis)
            self._update_camera_canvas(analysis)

        self.root.after(60, self._refresh_ui)

    def _format_move_text(self, analysis: FrameAnalysis) -> str:
        if analysis.decision.recommendation is None:
            return analysis.decision.message

        move = analysis.decision.recommendation
        return f"Best move: row={move.row}, col={move.col}. {analysis.decision.message}"

    def _update_details_box(self, analysis: FrameAnalysis) -> None:
        rotated_board = self._apply_rotation(analysis.observation.board)
        rotated_conf = self._apply_rotation(analysis.observation.cell_confidences)
        rotated_sources = self._apply_rotation(analysis.observation.cell_sources)

        lines: list[str] = []
        lines.append(f"Board matrix (rotated {self._rotation * 90}°):")
        lines.append(format_board(rotated_board))
        lines.append("")
        lines.append("Cell confidences:")
        for row in rotated_conf:
            lines.append(" | ".join(f"{confidence:.2f}" for confidence in row))
        lines.append("")
        lines.append("Cell sources:")
        for row in rotated_sources:
            lines.append(" | ".join(label or "-" for label in row))
        lines.append("")
        lines.append(f"Board detector: {analysis.board_result.method}")
        lines.append(f"Board fallback: {analysis.board_result.fallback}")
        lines.append(
            f"Current minimum probability: {self._minimum_confidence:.2f}")
        lines.append("")
        lines.append("Detections:")
        if analysis.detections:
            for detection in analysis.detections:
                x1, y1, x2, y2 = detection.xyxy
                lines.append(
                    f"{detection.label:<12} conf={detection.confidence:.2f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
                )
        else:
            lines.append("No detections.")

        self.details_box.configure(state="normal")
        self.details_box.delete("1.0", tk.END)
        self.details_box.insert("1.0", "\n".join(lines))
        self.details_box.configure(state="disabled")

    def _redraw_current_board(self, event: tk.Event | None = None) -> None:
        with self._analysis_lock:
            analysis = self._latest_analysis

        self.board_canvas.delete("all")
        if analysis is None:
            self._draw_placeholder_board()
            return

        self._redraw_board(analysis)

    def _draw_placeholder_board(self) -> None:
        width = max(1, int(self.board_canvas.winfo_width()))
        height = max(1, int(self.board_canvas.winfo_height()))
        size = min(width, height) - 48
        cell = size / 3.0
        origin_x = (width - size) / 2.0
        origin_y = (height - size) / 2.0
        for row in range(3):
            for col in range(3):
                x1 = origin_x + col * cell
                y1 = origin_y + row * cell
                x2 = x1 + cell
                y2 = y1 + cell
                self.board_canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#475569", width=2, fill="#0f172a")
        self.board_canvas.create_text(
            width / 2.0,
            24,
            text="Waiting for the first analysis...",
            fill="#cbd5e1",
            font=("Segoe UI", 14, "bold"),
        )

    def _redraw_board(self, analysis: FrameAnalysis) -> None:
        self.board_canvas.delete("all")
        width = max(1, int(self.board_canvas.winfo_width()))
        height = max(1, int(self.board_canvas.winfo_height()))
        size = min(width, height) - 48
        cell = size / 3.0
        origin_x = (width - size) / 2.0
        origin_y = (height - size) / 2.0

        # Apply rotation to board data
        rot_board = self._apply_rotation(analysis.observation.board)
        rot_conf = self._apply_rotation(analysis.observation.cell_confidences)

        rot_label = f" ({self._rotation * 90}° CW)" if self._rotation else ""
        self.board_canvas.create_text(
            width / 2.0,
            22,
            text=analysis.decision.message + rot_label,
            fill="#e2e8f0",
            font=(FONT_FAMILY, 11, "bold"),
        )

        move = analysis.decision.recommendation
        rotated_move: tuple[int, int] | None = None
        if move is not None:
            rotated_move = self._rotate_move(move.row, move.col)

        for row in range(3):
            for col in range(3):
                x1 = origin_x + col * cell
                y1 = origin_y + row * cell
                x2 = x1 + cell
                y2 = y1 + cell

                symbol = rot_board[row][col]
                confidence = rot_conf[row][col]
                fill = self._cell_fill(symbol)
                outline = "#334155"
                line_width = 2
                if rotated_move is not None and rotated_move == (row, col):
                    outline = "#f59e0b"
                    line_width = 5

                self.board_canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=fill,
                    outline=outline,
                    width=line_width,
                )
                self.board_canvas.create_text(
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0 - 10,
                    text=symbol,
                    fill=self._symbol_color(symbol),
                    font=(FONT_FAMILY, 34, "bold"),
                )
                self.board_canvas.create_text(
                    (x1 + x2) / 2.0,
                    y2 - 18,
                    text=f"{confidence:.2f}",
                    fill="#cbd5e1",
                    font=(FONT_FAMILY, 10, "bold"),
                )

        legend_y = origin_y + size + 12
        self.board_canvas.create_text(
            width / 2.0,
            legend_y,
            text="E = empty   R = red ball   Y = yellow ball",
            fill="#94a3b8",
            font=(FONT_FAMILY, 10),
        )

        if rotated_move is not None:
            rm_row, rm_col = rotated_move
            move_x = origin_x + rm_col * cell + cell / 2.0
            move_y = origin_y + rm_row * cell + 24
            self.board_canvas.create_text(
                move_x,
                move_y,
                text="AI MOVE",
                fill="#f59e0b",
                font=(FONT_FAMILY, 10, "bold"),
            )

    def _cell_fill(self, symbol: str) -> str:
        if symbol == "R":
            return "#fef2f2"
        if symbol == "Y":
            return "#fffbeb"
        return "#e0f2fe"

    def _symbol_color(self, symbol: str) -> str:
        if symbol == "R":
            return "#b91c1c"
        if symbol == "Y":
            return "#b45309"
        return "#0f172a"


def run_tkinter_app(config: AppConfig) -> int:
    app = TkinterVisionApp(config)
    return app.run()
