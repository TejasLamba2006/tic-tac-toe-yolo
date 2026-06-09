#!/usr/bin/env python3
"""
STM32MP25 DCMIPP-Aware Tic-Tac-Toe Game with Color Detection
Real-time HSV-based color detection for interactive tic-tac-toe gameplay
Player: Red balls (X), Computer: Yellow circles (0)
Enhanced with default trackbar values and improved terminal output
"""
import cv2
import numpy as np
import argparse
import time
import os
import sys
import subprocess
import json
import random
import threading
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from collections import deque


def listen_feedback(dcmipp_instance=None):
    print("[Controller] Starting feedback listener...")
    while True:
        try:
            # Listen to app (ui.py) instead of p (move.py)
            line = app.stdout.readline()
            if not line:
                print("[Controller] No more lines from UI subprocess")
                break
            line = line.strip()
            print(f"[Controller] Received from UI: '{line}'")  # Debug output
            if line == "RESTARTED":
                print("[Controller] UI was restarted by user click.")
                if dcmipp_instance:
                    # Same reset logic as 'r' key press
                    dcmipp_instance.game.reset_game()
                    dcmipp_instance.detection_stable_count = 0
                    dcmipp_instance.waiting_for_computer = False
                    print("🔄 Game Reset from UI signal!")
            elif line == "CLOSED":
                print("[Controller] UI was closed by user click.")
                if dcmipp_instance:
                    # Signal the main game loop to exit
                    dcmipp_instance.should_exit = True
                    print("🚪 Closing game from UI signal!")
                break
            else:
                print(f"[Controller] UI Output: {line}")
        except Exception as e:
            print(f"[Controller] Error in UI listener: {e}")
            break

def send(cmd):
    print(f"Sending to move.py: {cmd}")
    p.stdin.write(cmd + '\n')
    p.stdin.flush()

def send_ui_update(cmd):
    """Send real-time updates to the UI"""
    try:
        # Check if app exists and is available
        if 'app' in globals() and app and hasattr(app, 'stdin') and app.stdin:
            print(f"Sending to UI: {cmd}")
            app.stdin.write(cmd + '\n')
            app.stdin.flush()
        else:
            print(f"UI not ready, queuing command: {cmd}")
            ui_command_queue.append(cmd)
    except Exception as e:
        print(f"Error sending UI update: {e}")

def flush_ui_queue():
    """Send all queued UI commands"""
    try:
        if 'app' in globals() and app and hasattr(app, 'stdin') and app.stdin:
            while ui_command_queue:
                cmd = ui_command_queue.popleft()
                print(f"Sending queued to UI: {cmd}")
                app.stdin.write(cmd + '\n')
                app.stdin.flush()
    except Exception as e:
        print(f"Error flushing UI queue: {e}")

p = subprocess.Popen(['python3', 'move.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
iu=0
ui_command_queue = deque()  # Queue for UI commands before app is ready
class CameraController:
    """Handle camera zoom and pan functionality"""
    def __init__(self, width: int, height: int):
        self.original_width = width
        self.original_height = height
        self.zoom_factor = 2.7  # Default zoom to 1.9x (trackbar value 19)
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        self.zoom_step = 0.1
        
        # Pan offsets (default values from requirements)
        self.pan_x = 323  # Default pan x
        self.pan_y = 238   # Default pan y
        self.pan_step = 20
        
        # Flip flag
        self.flip_upside_down = False
        
        # Current viewport
        self.viewport_x = 0
        self.viewport_y = 0
        self.viewport_width = width
        self.viewport_height = height
        
        # Trackbar limits for pan (will be updated based on zoom)
        self.max_pan_x = 0
        self.max_pan_y = 0
        
        self.update_viewport()
    
    def update_viewport(self):
        """Update viewport based on zoom and pan"""
        # Calculate zoomed dimensions
        zoomed_width = int(self.original_width / self.zoom_factor)
        zoomed_height = int(self.original_height / self.zoom_factor)
        
        # Update max pan limits
        self.max_pan_x = max(0, self.original_width - zoomed_width)
        self.max_pan_y = max(0, self.original_height - zoomed_height)
        
        # Apply pan with bounds checking
        self.pan_x = max(0, min(int(self.pan_x), self.max_pan_x))
        self.pan_y = max(0, min(int(self.pan_y), self.max_pan_y))
        
        self.viewport_x = int(self.pan_x)
        self.viewport_y = int(self.pan_y)
        self.viewport_width = min(zoomed_width, self.original_width - self.viewport_x)
        self.viewport_height = min(zoomed_height, self.original_height - self.viewport_y)
    
    def set_pan_x(self, value: int):
        """Set pan X from trackbar"""
        self.pan_x = max(0, min(value, self.max_pan_x))
        self.update_viewport()
    
    def set_pan_y(self, value: int):
        """Set pan Y from trackbar"""
        self.pan_y = max(0, min(value, self.max_pan_y))
        self.update_viewport()
    
    def toggle_flip(self):
        """Toggle upside down flip"""
        self.flip_upside_down = not self.flip_upside_down
        return self.flip_upside_down
    
    def zoom_in(self):
        """Zoom in"""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor = min(self.max_zoom, self.zoom_factor + self.zoom_step)
            self.update_viewport()
            return True
        return False
    
    def zoom_out(self):
        """Zoom out"""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor = max(self.min_zoom, self.zoom_factor - self.zoom_step)
            self.update_viewport()
            return True
        return False
    
    def pan_left(self):
        """Pan left"""
        self.pan_x = max(0, self.pan_x - self.pan_step)
        self.update_viewport()
    
    def pan_right(self):
        """Pan right"""
        self.pan_x = min(self.max_pan_x, self.pan_x + self.pan_step)
        self.update_viewport()
    
    def pan_up(self):
        """Pan up"""
        self.pan_y = max(0, self.pan_y - self.pan_step)
        self.update_viewport()
    
    def pan_down(self):
        """Pan down"""
        self.pan_y = min(self.max_pan_y, self.pan_y + self.pan_step)
        self.update_viewport()
    
    def reset_view(self):
        """Reset zoom and pan to default values"""
        self.zoom_factor = 2.7  # Reset to default zoom
        self.pan_x = 323       # Reset to default pan x
        self.pan_y = 238        # Reset to default pan y
        self.flip_upside_down = False
        self.update_viewport()
    
    def apply_transform(self, frame: np.ndarray) -> np.ndarray:
        """Apply zoom, pan, and flip to frame"""
        if frame is None:
            return None
        
        try:
            # Ensure all coordinates are integers
            start_y = int(self.viewport_y)
            end_y = int(self.viewport_y + self.viewport_height)
            start_x = int(self.viewport_x)
            end_x = int(self.viewport_x + self.viewport_width)
            
            # Bounds checking
            start_y = max(0, min(start_y, frame.shape[0]))
            end_y = max(start_y, min(end_y, frame.shape[0]))
            start_x = max(0, min(start_x, frame.shape[1]))
            end_x = max(start_x, min(end_x, frame.shape[1]))
            
            # Extract viewport region
            cropped = frame[start_y:end_y, start_x:end_x]
            
            # Resize to original dimensions
            if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                result = cv2.resize(cropped, (self.original_width, self.original_height))
                
                # Apply flip if enabled
                if self.flip_upside_down:
                    result = cv2.rotate(result, cv2.ROTATE_180)
                
                return result
            else:
                return frame
                
        except Exception as e:
            print(f"Transform error: {e}")
            return frame


class TicTacToeGame:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.game_over = False
        self.winner = None
        self.current_player = 'X'  # Player starts first
        self.game_just_ended = False  # Flag for win message display
        self.win_message_time = 0
        
        # For conditional printing
        self.last_printed_board = [[' ' for _ in range(3)] for _ in range(3)]
        self.last_printed_status = ""
        
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if move is valid"""
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == ' '
    
    def make_move(self, row: int, col: int, player: str) -> bool:
        """Make a move if valid"""
        if self.is_valid_move(row, col):
            if player == "0":
                print(f'{player}')
                i = None # Initialize i

                if row == 0 and col == 0:
                    i = 9
                elif row == 0 and col == 1:
                    i = 8
                elif row == 0 and col == 2:
                    i = 7
                elif row == 1 and col == 0:
                    i = 6
                elif row == 1 and col == 1:
                    i = 5
                elif row == 1 and col == 2:
                    i = 4
                elif row == 2 and col == 0:
                    i = 3
                elif row == 2 and col == 1:
                    i = 2
                elif row == 2 and col == 2:
                    i = 1
                else:
    # Handle cases where the input is outside the expected range
                    i = "Invalid coordinates. Please use row/col values between 0 and 2."
                p.stdin.write(f'{i}\n')
                p.stdin.flush()
                # command = f'python3 move.py {i}'	    
                # os.system(command)
            self.board[row][col] = player
            
            # Send UI update for move made
            # Convert '0' to 'O' for UI compatibility
            ui_player = 'O' if player == '0' else player
            move_cmd = f"move {row} {col} {ui_player}"
            
            # For computer moves (0), delay the UI update by 5 seconds
            if player == '0':
                # Don't send immediately, schedule for later
                pass  # UI update will be handled in process_computer_move
            else:
                # For player moves (X), send immediately
                send_ui_update(move_cmd)
            
            return True
        return False
    
    def check_winner(self) -> Optional[str]:
        """Check for winner"""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None
    
    def is_board_full(self) -> bool:
        """Check if board is full"""
        for row in self.board:
            for cell in row:
                if cell == ' ':
                    return False
        return True
    
    def get_computer_move(self) -> Tuple[int, int]:
        """Simple AI for computer move"""
        # Try to win
        for row in range(3):
            for col in range(3):
                if self.is_valid_move(row, col):
                    self.board[row][col] = '0'
                    if self.check_winner() == '0':
                        self.board[row][col] = ' '  # Undo
                        return row, col
                    self.board[row][col] = ' '  # Undo
        
        # Try to block player
        for row in range(3):
            for col in range(3):
                if self.is_valid_move(row, col):
                    self.board[row][col] = 'X'
                    if self.check_winner() == 'X':
                        self.board[row][col] = ' '  # Undo
                        return row, col
                    self.board[row][col] = ' '  # Undo
        
        # Take center if available
        if self.is_valid_move(1, 1):
            return 1, 1
        
        # Take corners
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        random.shuffle(corners)
        for row, col in corners:
            if self.is_valid_move(row, col):
                return row, col
        
        # Take any available spot
        for row in range(3):
            for col in range(3):
                if self.is_valid_move(row, col):
                    return row, col
        
        return -1, -1  # No moves available
    
    def reset_game(self):
        """Reset the game"""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.game_over = False
        self.winner = None
        self.current_player = 'X'
        self.game_just_ended = False
        self.win_message_time = 0
        self.last_printed_board = [[' ' for _ in range(3)] for _ in range(3)]
        self.last_printed_status = ""
        
        # Send UI update for game reset
        send_ui_update("reset")
    
    def matrices_equal(self, matrix1: List[List[str]], matrix2: List[List[str]]) -> bool:
        """Compare two matrices for equality"""
        for i in range(3):
            for j in range(3):
                if matrix1[i][j] != matrix2[i][j]:
                    return False
        return True
    
    def should_print_status(self) -> bool:
        """Check if game status should be printed (only when board changes)"""
        if not self.matrices_equal(self.board, self.last_printed_board):
            self.last_printed_board = [row[:] for row in self.board]
            return True
        return False
    
    def print_game_status(self):
        """Print current game status to terminal (only when board changes)"""
        if not self.should_print_status():
            return
        
        print("\n" + "="*40)
        print("        TIC-TAC-TOE GAME STATUS")
        print("="*40)
        print("Current Board:")
        
        # Print game board with computer moves
        for i, row in enumerate(self.board):
            display_row = []
            for j, cell in enumerate(row):
                if cell == ' ':
                    display_row.append(' ')
                elif cell == 'X':
                    display_row.append('X')
                elif cell == '0':
                    display_row.append('0')
            print(f" {display_row[0]} | {display_row[1]} | {display_row[2]} ")
            if i < 2:
                print("-----------")
        
        print("="*40)
        
        if self.game_over:
            if self.winner:
                winner_name = "PLAYER" if self.winner == 'X' else "COMPUTER"
                print(f"🎉 GAME OVER! {winner_name} WINS!")
            else:
                print("🤝 GAME OVER! IT'S A TIE!")
        else:
            current_name = "PLAYER" if self.current_player == 'X' else "COMPUTER"
            print(f"Current Turn: {current_name} ({self.current_player})")
            
            if self.current_player == 'X':
                print("🔴 Place your red ball to make a move!")
        
        print("="*40)


class DCMIPPTicTacToe:
    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_type = "unknown"
        self.camera_device = None
        
        # Camera controller for zoom/pan/flip
        self.camera_controller = CameraController(width, height)
        
        # Trackbar window
        self.control_window = "Camera Controls"
        self.trackbars_created = False
        
        # Win message display
        self.win_message_duration = 5.0  # seconds to show win message
        
        # DCMIPP specific
        self.dcmipp_media = "platform:48030000.dcmipp"
        self.dcmipp_sensor = "NOTFOUND"
        
        # V4L2 timeout handling
        self.frame_timeout = 5.0  # seconds
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        self.use_threading = True
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_thread = None
        self.stop_thread = False
        
        # Color detection settings
        self.color_ranges = {
            'red1': {
                'name': 'Red (Lower)',
                'lower': np.array([0, 50, 50]),
                'upper': np.array([10, 255, 255]),
                'color': (0, 0, 255),
                'enabled': False
            },
            'red2': {
                'name': 'Red (Upper)', 
                'lower': np.array([170, 50, 50]),
                'upper': np.array([180, 255, 255]),
                'color': (0, 0, 255),
                'enabled': True
            }
        }
        
        self.min_contour_area = 500
        self.max_contour_area = 50000
        
        # Game state
        self.game = TicTacToeGame()
        self.last_detection_matrix = [[' ' for _ in range(3)] for _ in range(3)]
        self.detection_stable_count = 0
        self.required_stable_frames = 15  # Frames to wait for stable detection
        self.waiting_for_computer = False
        self.computer_move_delay = 2.0  # Seconds to wait before computer move
        self.computer_move_time = 0
        
        # User interaction
        self.manual_mode = False
        self.last_user_confirmation = False
        self.should_exit = False  # Flag to signal game exit from UI
        
        # UI update delay for computer moves
        self.pending_ui_move = None  # Store pending computer move for UI
        self.ui_move_time = 0  # Time when to send UI update

    def create_trackbars(self):
        """Create trackbars for camera control with default values"""
        if not self.trackbars_created:
            cv2.namedWindow(self.control_window, cv2.WINDOW_AUTOSIZE)
            
            # Pan X trackbar with default value
            cv2.createTrackbar('Pan X', self.control_window, 
                             min(int(self.camera_controller.pan_x), self.camera_controller.max_pan_x), 
                             max(1, self.camera_controller.max_pan_x), 
                             self.on_pan_x_change)
            
            # Pan Y trackbar with default value
            cv2.createTrackbar('Pan Y', self.control_window, 
                             min(int(self.camera_controller.pan_y), self.camera_controller.max_pan_y), 
                             max(1, self.camera_controller.max_pan_y), 
                             self.on_pan_y_change)
            
            # Zoom trackbar with default value (19 = 1.9x zoom)
            zoom_val = int(self.camera_controller.zoom_factor * 10)
            max_zoom_val = int(self.camera_controller.max_zoom * 10)
            
            cv2.createTrackbar('Zoom x10', self.control_window, zoom_val, 
                             max_zoom_val, self.on_zoom_change)
            
            # Flip upside down toggle
            cv2.createTrackbar('Flip Upside Down', self.control_window, 
                             1 if self.camera_controller.flip_upside_down else 0, 
                             1, self.on_flip_toggle)
            
            # Color detection trackbars
            cv2.createTrackbar('Red1 Enable', self.control_window, 
                             1 if self.color_ranges['red1']['enabled'] else 0, 
                             1, self.on_red1_toggle)
            
            cv2.createTrackbar('Red2 Enable', self.control_window, 
                             1 if self.color_ranges['red2']['enabled'] else 0, 
                             1, self.on_red2_toggle)
            
            cv2.createTrackbar('Min Area', self.control_window, 
                             self.min_contour_area, 5000, self.on_min_area_change)
            
            cv2.createTrackbar('Max Area', self.control_window, 
                             self.max_contour_area, 100000, self.on_max_area_change)
            
            self.trackbars_created = True
            print("✓ Trackbars created with default values:")
            print(f"  Zoom: {self.camera_controller.zoom_factor:.1f}x")
            print(f"  Pan X: {self.camera_controller.pan_x}")
            print(f"  Pan Y: {self.camera_controller.pan_y}")
    
    def update_trackbar_ranges_only(self):
        """Update only trackbar ranges when zoom changes, without setting positions"""
        if self.trackbars_created:
            try:
                # Only update ranges, not positions
                cv2.setTrackbarMax('Pan X', self.control_window, 
                                 max(1, self.camera_controller.max_pan_x))
                cv2.setTrackbarMax('Pan Y', self.control_window, 
                                 max(1, self.camera_controller.max_pan_y))
            except:
                pass  # Ignore trackbar update errors
    
    def sync_trackbars_from_keyboard(self):
        """Sync trackbar positions only when using keyboard controls"""
        if self.trackbars_created:
            try:
                cv2.setTrackbarPos('Pan X', self.control_window, 
                                 int(self.camera_controller.pan_x))
                cv2.setTrackbarPos('Pan Y', self.control_window, 
                                 int(self.camera_controller.pan_y))
                cv2.setTrackbarPos('Zoom x10', self.control_window, 
                                 int(self.camera_controller.zoom_factor * 10))
                cv2.setTrackbarPos('Flip Upside Down', self.control_window, 
                                 1 if self.camera_controller.flip_upside_down else 0)
            except:
                pass
    
    def on_pan_x_change(self, val):
        """Handle pan X trackbar change"""
        self.camera_controller.set_pan_x(val)
    
    def on_pan_y_change(self, val):
        """Handle pan Y trackbar change"""
        self.camera_controller.set_pan_y(val)
    
    def on_zoom_change(self, val):
        """Handle zoom trackbar change"""
        new_zoom = val / 10.0
        new_zoom = max(self.camera_controller.min_zoom, 
                      min(self.camera_controller.max_zoom, new_zoom))
        self.camera_controller.zoom_factor = new_zoom
        self.camera_controller.update_viewport()
        self.update_trackbar_ranges_only()  # Only update ranges, not positions
    
    def on_flip_toggle(self, val):
        """Handle flip toggle"""
        self.camera_controller.flip_upside_down = bool(val)
        flip_status = "ON" if val else "OFF"
        print(f"Flip upside down: {flip_status}")
    
    def on_red1_toggle(self, val):
        """Handle Red1 detection toggle"""
        self.color_ranges['red1']['enabled'] = bool(val)
        print(f"Red1 detection: {'ON' if val else 'OFF'}")
    
    def on_red2_toggle(self, val):
        """Handle Red2 detection toggle"""
        self.color_ranges['red2']['enabled'] = bool(val)
        print(f"Red2 detection: {'ON' if val else 'OFF'}")
    
    def on_min_area_change(self, val):
        """Handle minimum contour area change"""
        self.min_contour_area = max(1, val)
    
    def on_max_area_change(self, val):
        """Handle maximum contour area change"""
        self.max_contour_area = max(self.min_contour_area + 1, val)

    def frame_capture_thread(self):
        """Background thread for frame capture to handle V4L2 timeouts"""
        while not self.stop_thread and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    if self.consecutive_failures > self.max_consecutive_failures:
                        print("Too many consecutive frame capture failures")
                        break
                
                time.sleep(1.0 / (self.fps * 2))  # Limit thread FPS
                
            except Exception as e:
                print(f"Frame capture thread error: {e}")
                self.consecutive_failures += 1
                time.sleep(0.1)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame with timeout handling"""
        if self.use_threading:
            with self.frame_lock:
                if self.latest_frame is not None:
                    return True, self.latest_frame.copy()
                else:
                    return False, None
        else:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.consecutive_failures = 0
                    return True, frame
                else:
                    self.consecutive_failures += 1
                    return False, None
            except Exception as e:
                print(f"Frame capture error: {e}")
                self.consecutive_failures += 1
                return False, None

    def configure_v4l2_device(self, device_path: str) -> bool:
        """Configure V4L2 device to reduce timeouts"""
        try:
            # Set buffer size to reduce latency
            cmd = ['v4l2-ctl', '-d', device_path, '--set-ctrl', 'buffers=1']
            subprocess.run(cmd, capture_output=True, timeout=5)
            
            # Set frame rate
            cmd = ['v4l2-ctl', '-d', device_path, '--set-parm', f'{int(self.fps)}']
            subprocess.run(cmd, capture_output=True, timeout=5)
            
            print(f"✓ V4L2 device {device_path} configured")
            return True
            
        except Exception as e:
            print(f"V4L2 configuration warning: {e}")
            return False

    def check_dcmipp_camera(self) -> bool:
        """Check if DCMIPP camera system is available"""
        try:
            # Check if media-ctl can access DCMIPP
            result = subprocess.run(['media-ctl', '-d', self.dcmipp_media, '-p'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                print("DCMIPP media controller not found")
                return False
            
            print("✓ DCMIPP media controller found")
            
            # Look for dcmipp_main_capture device
            video_dirs = subprocess.run(['find', '/sys/class/video4linux', '-name', 'video*', '-type', 'l'],
                                      capture_output=True, text=True).stdout.strip().split('\n')
            
            for video_dir in video_dirs:
                if not video_dir:
                    continue
                    
                name_file = os.path.join(video_dir, 'name')
                if os.path.exists(name_file):
                    with open(name_file, 'r') as f:
                        device_name = f.read().strip()
                    
                    if device_name == "dcmipp_main_capture":
                        video_num = os.path.basename(video_dir)
                        self.camera_device = f"/dev/{video_num}"
                        print(f"✓ Found DCMIPP capture device: {self.camera_device}")
                        return True
            
            return False
            
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            print(f"Error checking DCMIPP: {e}")
            return False
    
    def check_webcam(self) -> bool:
        """Check for USB webcam devices"""
        try:
            video_dirs = subprocess.run(['find', '/sys/class/video4linux', '-name', 'video*', '-type', 'l'],
                                      capture_output=True, text=True).stdout.strip().split('\n')
            
            for video_dir in video_dirs:
                if not video_dir:
                    continue
                    
                name_file = os.path.join(video_dir, 'name')
                if os.path.exists(name_file):
                    with open(name_file, 'r') as f:
                        device_name = f.read().strip()
                    
                    # Skip DCMIPP and STM32 internal devices
                    if 'dcmi' not in device_name.lower() and 'stm' not in device_name.lower():
                        video_num = os.path.basename(video_dir)
                        self.camera_device = f"/dev/{video_num}"
                        print(f"✓ Found webcam device: {self.camera_device} ({device_name})")
                        return True
            
            return False
            
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            print(f"Error checking webcam: {e}")
            return False

    def setup_camera(self) -> bool:
        """Setup camera based on available hardware"""
        print("Detecting camera hardware...")
        
        # Check DCMIPP first
        if self.check_dcmipp_camera():
            self.camera_type = "dcmipp"
            print("Using DCMIPP camera system")
        # Check for webcam if DCMIPP not available
        elif self.check_webcam():
            self.camera_type = "webcam"
            print("Using USB webcam")
        else:
            print("ERROR: No camera hardware detected!")
            return False
        
        # Configure V4L2 device
        self.configure_v4l2_device(self.camera_device)
        
        # Try to open camera with V4L2
        try:
            print(f"Opening camera: {self.camera_device}")
            self.cap = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                # Configure camera with timeout handling
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                
                # Set timeout for V4L2
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                print("✓ V4L2 camera opened successfully")
                
                # Start frame capture thread if needed
                if self.use_threading:
                    self.stop_thread = False
                    self.frame_thread = threading.Thread(target=self.frame_capture_thread)
                    self.frame_thread.daemon = True
                    self.frame_thread.start()
                    print("✓ Frame capture thread started")
                
                return True
            else:
                print("Failed to open V4L2 camera")
                return False
                
        except Exception as e:
            print(f"V4L2 error: {e}")
            return False

    def detect_colors(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Detect red colors in frame"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            detections = {}
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for color_key, color_info in self.color_ranges.items():
                if not color_info['enabled']:
                    continue
                    
                mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
                
                # Morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if self.min_contour_area <= area <= self.max_contour_area:
                        valid_contours.append(contour)
                
                detections[color_key] = {
                    'name': color_info['name'],
                    'mask': mask,
                    'contours': valid_contours,
                    'color': color_info['color'],
                    'count': len(valid_contours),
                    'total_area': sum(cv2.contourArea(c) for c in valid_contours)
                }
                
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            return detections, combined_mask
            
        except Exception as e:
            print(f"Error in color detection: {e}")
            return {}, np.zeros(frame.shape[:2], dtype=np.uint8)

    def detect_red_positions(self, detections: Dict) -> List[List[str]]:
        """Detect red ball positions in 3x3 matrix"""
        matrix = [[' ' for _ in range(3)] for _ in range(3)]
        
        # Collect red contours
        red_contours = []
        if 'red1' in detections and self.color_ranges['red1']['enabled']:
            red_contours.extend(detections['red1']['contours'])
        if 'red2' in detections and self.color_ranges['red2']['enabled']:
            red_contours.extend(detections['red2']['contours'])
        
        # Process each red contour
        for contour in red_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Map to 3x3 grid
                col = int(cx / (self.width / 3))
                row = int(cy / (self.height / 3))
                
                # Clamp to valid grid indices
                col = min(max(col, 0), 2)
                row = min(max(row, 0), 2)
                
                matrix[row][col] = 'X'
        
        return matrix

    def matrices_equal(self, matrix1: List[List[str]], matrix2: List[List[str]]) -> bool:
        """Compare two matrices for equality"""
        for i in range(3):
            for j in range(3):
                if matrix1[i][j] != matrix2[i][j]:
                    return False
        return True

    def process_player_move(self, detected_matrix: List[List[str]]) -> bool:
        """Process player's move from detected matrix"""
        if self.game.current_player != 'X' or self.game.game_over:
            return False
        
        # Find new red ball positions
        new_moves = []
        for i in range(3):
            for j in range(3):
                if (detected_matrix[i][j] == 'X' and 
                    self.game.board[i][j] == ' '):
                    new_moves.append((i, j))
        
        if len(new_moves) == 1:
            row, col = new_moves[0]
            if self.game.make_move(row, col, 'X'):
                print(f"✅ Player move: Row {row+1}, Col {col+1}")
                self.game.current_player = '0'
                self.waiting_for_computer = True
                self.computer_move_time = time.time() + self.computer_move_delay
                
                # Send UI update for turn change
                send_ui_update("turn O")
                
                return True
            else:
                print("❌ Invalid move detected!")
                return False
        elif len(new_moves) > 1:
            print("❌ Multiple moves detected! Please place only one red ball.")
            return False
        
        return False

    def process_computer_move(self):
        """Process computer's move"""
        if (self.game.current_player != '0' or 
            self.game.game_over or 
            not self.waiting_for_computer or 
            time.time() < self.computer_move_time):
            return
        
        row, col = self.game.get_computer_move()
        if row != -1 and col != -1:
            if self.game.make_move(row, col, '0'):
                print(f"🤖 Computer move: Row {row+1}, Col {col+1}")
                self.game.current_player = 'X'
                self.waiting_for_computer = False
                
                # Schedule UI update for computer move with 5-second delay
                self.pending_ui_move = f"move {row} {col} O"
                self.ui_move_time = time.time() + 5.0  # 5 second delay
                
                # Send UI update for turn change immediately
                send_ui_update("turn X")
        else:
            print("No valid moves for computer")

    def process_pending_ui_updates(self):
        """Process pending UI updates with delays"""
        if (self.pending_ui_move and 
            time.time() >= self.ui_move_time):
            send_ui_update(self.pending_ui_move)
            print(f"📱 Delayed UI update sent: {self.pending_ui_move}")
            self.pending_ui_move = None
            self.ui_move_time = 0

    def check_game_end(self):
        """Check if game has ended"""
        if not self.game.game_over:
            winner = self.game.check_winner()
            if winner:
                self.game.winner = winner
                self.game.game_over = True
                self.game.game_just_ended = True
                self.game.win_message_time = time.time()
                
                # Send UI update for game win
                # Convert '0' to 'O' for UI compatibility
                ui_winner = 'O' if winner == '0' else winner
                send_ui_update(f"win {ui_winner}")
                
            elif self.game.is_board_full():
                self.game.game_over = True
                self.game.game_just_ended = True
                self.game.win_message_time = time.time()
                
                # Send UI update for game tie
                send_ui_update("tie")

    def draw_game_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw game overlay on frame"""
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw grid lines
        cv2.line(overlay_frame, (w//3, 0), (w//3, h), (255, 255, 255), 2)
        cv2.line(overlay_frame, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 2)
        cv2.line(overlay_frame, (0, h//3), (w, h//3), (255, 255, 255), 2)
        cv2.line(overlay_frame, (0, 2*h//3), (w, 2*h//3), (255, 255, 255), 2)
        
        # Draw game pieces
        for row in range(3):
            for col in range(3):
                cx = col * w//3 + w//6
                cy = row * h//3 + h//6
                
                if self.game.board[row][col] == 'X':
                    # Draw bigger red X
                    cv2.line(overlay_frame, (cx-40, cy-40), (cx+40, cy+40), (0, 0, 255), 8)
                    cv2.line(overlay_frame, (cx+40, cy-40), (cx-40, cy+40), (0, 0, 255), 8)
                elif self.game.board[row][col] == '0':
                    # Draw bigger yellow circle for computer
                    cv2.circle(overlay_frame, (cx, cy), 50, (0, 255, 255), 10)  # Much bigger and thicker
        
        return overlay_frame

    def draw_win_message(self, frame: np.ndarray) -> np.ndarray:
        """Draw win message popup on frame"""
        if not self.game.game_just_ended:
            return frame
        
        # Check if message should still be displayed
        elapsed_time = time.time() - self.game.win_message_time
        if elapsed_time > self.win_message_duration:
            self.game.game_just_ended = False
            return frame
        
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Create semi-transparent background
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Determine win message
        if self.game.winner == 'X':
            main_text = "🎉 PLAYER WINS! 🎉"
            text_color = (0, 255, 0)  # Green
            emoji_text = "🏆 VICTORY! 🏆"
        elif self.game.winner == '0':
            main_text = "🤖 COMPUTER WINS! 🤖"
            text_color = (0, 0, 255)  # Red
            emoji_text = "🎯 AI VICTORY! 🎯"
        else:
            main_text = "🤝 IT'S A TIE! 🤝"
            text_color = (0, 255, 255)  # Yellow
            emoji_text = "⚖️ DRAW GAME! ⚖️"
        
        # Calculate text position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        # Main message
        (text_w, text_h), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = (h // 2) - 20
        
        # Draw text with outline
        cv2.putText(frame, main_text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, main_text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Emoji message
        font_scale_emoji = 0.8
        (emoji_w, emoji_h), _ = cv2.getTextSize(emoji_text, font, font_scale_emoji, 2)
        emoji_x = (w - emoji_w) // 2
        emoji_y = text_y + 60
        
        cv2.putText(frame, emoji_text, (emoji_x + 1, emoji_y + 1), font, font_scale_emoji, (0, 0, 0), 3)
        cv2.putText(frame, emoji_text, (emoji_x, emoji_y), font, font_scale_emoji, (255, 255, 255), 2)
        
        # Instructions
        instruction_text = "Press 'r' to restart"
        font_scale_inst = 0.6
        (inst_w, inst_h), _ = cv2.getTextSize(instruction_text, font, font_scale_inst, 2)
        inst_x = (w - inst_w) // 2
        inst_y = emoji_y + 50
        
        cv2.putText(frame, instruction_text, (inst_x + 1, inst_y + 1), font, font_scale_inst, (0, 0, 0), 3)
        cv2.putText(frame, instruction_text, (inst_x, inst_y), font, font_scale_inst, (200, 200, 200), 2)
        
        # Animated border effect
        border_color = text_color
        border_thickness = int(5 + 3 * abs(np.sin(elapsed_time * 3)))
        cv2.rectangle(frame, (20, 20), (w-20, h-20), border_color, border_thickness)
        
        return frame

    def draw_zoom_pan_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw zoom, pan, and flip information on frame"""
        info_frame = frame.copy()
        
        # Zoom info
        zoom_text = f"Zoom: {self.camera_controller.zoom_factor:.1f}x"
        cv2.putText(info_frame, zoom_text, (10, self.height - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Pan info
        pan_text = f"Pan: ({int(self.camera_controller.pan_x)}, {int(self.camera_controller.pan_y)})"
        cv2.putText(info_frame, pan_text, (10, self.height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Flip info
        flip_text = f"Flip: {'ON' if self.camera_controller.flip_upside_down else 'OFF'}"
        cv2.putText(info_frame, flip_text, (10, self.height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return info_frame

    def initialize(self) -> bool:
        command = f'python3 move.py reset'	    
        os.system(command)

        """Initialize the tic-tac-toe system"""
        print("=" * 50)
        print("STM32MP25 DCMIPP TIC-TAC-TOE GAME")
        print("=" * 50)
        print("You are X (Red balls)")
        print("Computer is 0 (Yellow circles)")
        print("=" * 50)
        
        if not self.setup_camera():
            return False
        
        # Create trackbars after camera setup
        self.create_trackbars()
        
        return True

    def run(self):
        """Main game loop"""
        if not self.cap or not self.cap.isOpened():
            print("ERROR: Camera not initialized!")
            return
        

        
        main_window = "STM32MP25 Tic-Tac-Toe Game"
        cv2.namedWindow(main_window, cv2.WINDOW_AUTOSIZE)
        
        print(f"\nStarting Tic-Tac-Toe game...")
        print("Controls:")
        print("  'q' or ESC - Quit")
        print("  'r' - Reset game (also available in terminal)")
        print("  'm' - Toggle manual confirmation mode")
        print("  'SPACE' - Confirm move (in manual mode)")
        print("  '1' - Toggle Red1 detection")
        print("  '2' - Toggle Red2 detection")
        print("  'f' - Toggle flip upside down")
        print("  Keyboard:")
        print("    '+'/'-' - Zoom in/out")
        print("    Arrow keys - Pan camera")
        print("    'c' - Reset camera view to defaults")
        print("  Trackbars:")
        print("    Pan X/Y - Precise camera positioning")
        print("    Zoom x10 - Precise zoom control")
        print("    Flip Upside Down - Rotate feed 180 degrees")
        print("    Detection controls - Color detection tuning")
        print("\nTerminal Commands:")
        print("  Type 'reset' or 'r' and press Enter to reset game")
        print("  Type 'quit' or 'q' and press Enter to quit")
        
        frame_count = 0
        start_time = time.time()
        actual_fps = self.fps
        
        # Initial game status print
        self.game.print_game_status()
        
        # Send initial UI state
        send_ui_update("start")
        send_ui_update("turn X")
        
        try:
            while True:
                # Check if UI requested exit
                if self.should_exit:
                    print("🚪 Exiting game loop due to UI close signal")
                    break
                    
                ret, frame = self.get_frame()
                
                if not ret or frame is None:
                    if self.consecutive_failures > self.max_consecutive_failures:
                        print("Too many consecutive frame failures, exiting...")
                        break
                    time.sleep(0.1)
                    continue
                
                # Apply zoom, pan, and flip
                transformed_frame = self.camera_controller.apply_transform(frame)
                if transformed_frame is None:
                    continue
                
                # Color detection
                detections, combined_mask = self.detect_colors(transformed_frame)
                detected_matrix = self.detect_red_positions(detections)
                
                # Stability checking for detected moves
                if self.matrices_equal(detected_matrix, self.last_detection_matrix):
                    self.detection_stable_count += 1
                else:
                    self.detection_stable_count = 0
                    self.last_detection_matrix = [row[:] for row in detected_matrix]
                
                # Process moves
                if not self.manual_mode:
                    # Automatic mode - process when detection is stable
                    if self.detection_stable_count >= self.required_stable_frames:
                        self.process_player_move(detected_matrix)
                        self.detection_stable_count = 0
                
                # Process computer move
                self.process_computer_move()
                
                # Process pending UI updates
                self.process_pending_ui_updates()
                
                # Check game end
                self.check_game_end()
                
                # Print game status only when board changes
                self.game.print_game_status()
                
                # Draw overlays
                display_frame = self.draw_game_overlay(transformed_frame)
                display_frame = self.draw_zoom_pan_info(display_frame)
                
                # Draw detection overlay
                for color_key, detection in detections.items():
                    cv2.drawContours(display_frame, detection['contours'], -1, 
                                   detection['color'], 2)
                    
                    for contour in detection['contours']:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), 
                                    detection['color'], 2)
                        
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(display_frame, (cx, cy), 5, detection['color'], -1)
                
                # Add status text
                cv2.putText(display_frame, f"FPS: {actual_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                stability_text = f"Stability: {self.detection_stable_count}/{self.required_stable_frames}"
                cv2.putText(display_frame, stability_text, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                mode_text = f"Mode: {'Manual' if self.manual_mode else 'Auto'}"
                cv2.putText(display_frame, mode_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.game.game_over and not self.game.game_just_ended:
                    cv2.putText(display_frame, "GAME OVER! Press 'r' to restart", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw win message popup (this should be last to overlay everything)
                display_frame = self.draw_win_message(display_frame)
                
                cv2.imshow(main_window, display_frame)
                global iu
                if iu == 0:
                    global app
                    app = subprocess.Popen(
    ['python3', 'ui.py'],      # Use your actual GUI script name
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    encoding='utf-8'
)

                    iu=1            
                    listener_thread = threading.Thread(target=listen_feedback, args=(self,), daemon=True)
                    listener_thread.start()
                    print("✓ UI feedback listener started")
                    
                    # Flush any queued UI commands
                    flush_ui_queue()
                    print("✓ UI command queue flushed") 
               # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        actual_fps = 30 / elapsed
                    start_time = time.time()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # ESC
                    break
                elif key == ord('r'):
                    self.game.reset_game()
                    self.detection_stable_count = 0
                    self.waiting_for_computer = False
                    print("🔄 Game Reset!")
                elif key == ord('m'):
                    self.manual_mode = not self.manual_mode
                    print(f"Manual mode: {'ON' if self.manual_mode else 'OFF'}")
                elif key == ord(' ') and self.manual_mode:
                    # Manual confirmation
                    if self.detection_stable_count >= 5:  # Lower threshold for manual
                        self.process_player_move(detected_matrix)
                        self.detection_stable_count = 0
                elif key == ord('1'):
                    self.color_ranges['red1']['enabled'] = not self.color_ranges['red1']['enabled']
                    cv2.setTrackbarPos('Red1 Enable', self.control_window, 
                                     1 if self.color_ranges['red1']['enabled'] else 0)
                    print(f"Red1 detection: {'ON' if self.color_ranges['red1']['enabled'] else 'OFF'}")
                elif key == ord('2'):
                    self.color_ranges['red2']['enabled'] = not self.color_ranges['red2']['enabled']
                    cv2.setTrackbarPos('Red2 Enable', self.control_window, 
                                     1 if self.color_ranges['red2']['enabled'] else 0)
                    print(f"Red2 detection: {'ON' if self.color_ranges['red2']['enabled'] else 'OFF'}")
                elif key == ord('f'):
                    # Toggle flip upside down
                    flip_state = self.camera_controller.toggle_flip()
                    cv2.setTrackbarPos('Flip Upside Down', self.control_window, 
                                     1 if flip_state else 0)
                    print(f"Flip upside down: {'ON' if flip_state else 'OFF'}")
                
                # Camera control keys
                elif key == ord('+') or key == ord('='):
                    if self.camera_controller.zoom_in():
                        self.sync_trackbars_from_keyboard()
                elif key == ord('2'):
                    self.color_ranges['red2']['enabled'] = not self.color_ranges['red2']['enabled']
                    cv2.setTrackbarPos('Red2 Enable', self.control_window, 
                                     1 if self.color_ranges['red2']['enabled'] else 0)
                    print(f"Red2 detection: {'ON' if self.color_ranges['red2']['enabled'] else 'OFF'}")
                elif key == ord('f'):
                    # Toggle flip upside down
                    flip_state = self.camera_controller.toggle_flip()
                    cv2.setTrackbarPos('Flip Upside Down', self.control_window, 
                                     1 if flip_state else 0)
                    print(f"Flip upside down: {'ON' if flip_state else 'OFF'}")
                
                # Camera control keys
                elif key == ord('+') or key == ord('='):
                    if self.camera_controller.zoom_in():
                        self.sync_trackbars_from_keyboard()
                        print(f"Zoom: {self.camera_controller.zoom_factor:.1f}x")
                elif key == ord('-'):
                    if self.camera_controller.zoom_out():
                        self.sync_trackbars_from_keyboard()
                        print(f"Zoom: {self.camera_controller.zoom_factor:.1f}x")
                elif key == ord('c'):
                    self.camera_controller.reset_view()
                    self.sync_trackbars_from_keyboard()
                    print("Camera view reset to defaults")
                
                # Arrow keys for panning
                elif key == 82:  # Up arrow
                    self.camera_controller.pan_up()
                    self.sync_trackbars_from_keyboard()
                elif key == 84:  # Down arrow
                    self.camera_controller.pan_down()
                    self.sync_trackbars_from_keyboard()
                elif key == 81:  # Left arrow
                    self.camera_controller.pan_left()
                    self.sync_trackbars_from_keyboard()
                elif key == 83:  # Right arrow
                    self.camera_controller.pan_right()
                    self.sync_trackbars_from_keyboard()
                
        except KeyboardInterrupt:
            print("\nGame stopped by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        
        print("Tic-Tac-Toe game ended.")

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop frame capture thread
            if self.frame_thread:
                self.stop_thread = True
                self.frame_thread.join(timeout=2.0)
                print("Frame capture thread stopped")
            
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            print("Resources cleaned up.")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="STM32MP25 DCMIPP Tic-Tac-Toe Game")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30)")
    parser.add_argument("--no-threading", action="store_true", help="Disable threaded frame capture")
    
    args = parser.parse_args()
    
    game = DCMIPPTicTacToe(args.width, args.height, args.fps)
    
    if args.no_threading:
        game.use_threading = False
        print("Threading disabled")
    
    try:
        if game.initialize():
            game.run()
        else:
            print("Failed to initialize DCMIPP Tic-Tac-Toe!")
            sys.exit(1)
    finally:
        game.cleanup()


if __name__ == "__main__":
    main()


