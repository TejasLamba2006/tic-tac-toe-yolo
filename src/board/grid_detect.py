#!/usr/bin/env python3
"""
OpenCV Camera Preview - STM32MP25 Equivalent with Tic-Tac-Toe
Supports multiple camera backends and provides similar functionality to the original shell scripts.
Includes Tic-Tac-Toe game mode using computer vision.
"""

import cv2
import numpy as np
import argparse
import time
import os
import sys
import random
from typing import List, Optional, Tuple, Dict

class TicTacToeLogic:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_turn = 'X' # User is always X
        self.game_over = False
        self.winner = None
        self.winning_line = []

    def reset(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_turn = 'X'
        self.game_over = False
        self.winner = None
        self.winning_line = []

    def make_move(self, row, col, player):
        if self.board[row][col] == ' ' and not self.game_over:
            self.board[row][col] = player
            if self.check_win(player):
                self.game_over = True
                self.winner = player
            elif self.check_draw():
                self.game_over = True
                self.winner = 'Draw'
            else:
                self.current_turn = 'O' if player == 'X' else 'X'
            return True
        return False

    def check_win(self, player):
        # Rows
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)):
                self.winning_line = [(i, 0), (i, 1), (i, 2)]
                return True
        # Cols
        for j in range(3):
            if all(self.board[i][j] == player for i in range(3)):
                self.winning_line = [(0, j), (1, j), (2, j)]
                return True
        # Diagonals
        if all(self.board[i][i] == player for i in range(3)):
            self.winning_line = [(0, 0), (1, 1), (2, 2)]
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            self.winning_line = [(0, 2), (1, 1), (2, 0)]
            return True
        return False

    def check_draw(self):
        return all(self.board[i][j] != ' ' for i in range(3) for j in range(3))

    def ai_move(self):
        if self.game_over or self.current_turn != 'O':
            return

        # Simple AI: 1. Win, 2. Block, 3. Center, 4. Random
        empty_cells = [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == ' ']
        
        if not empty_cells:
            return

        # Try to win
        for r, c in empty_cells:
            self.board[r][c] = 'O'
            if self.check_win('O'):
                self.game_over = True
                self.winner = 'O'
                return
            self.board[r][c] = ' ' # Undo

        # Try to block
        for r, c in empty_cells:
            self.board[r][c] = 'X'
            if self.check_win('X'):
                self.board[r][c] = 'O' # Block!
                self.current_turn = 'X'
                return
            self.board[r][c] = ' ' # Undo

        # Take center
        if (1, 1) in empty_cells:
            self.make_move(1, 1, 'O')
            return

        # Random
        r, c = random.choice(empty_cells)
        self.make_move(r, c, 'O')

class CameraPreview:
    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_id = 0
        self.backend_name = "Unknown"
        
        # Game State
        self.mode = 0 # 0: Preview, 1: Calibration, 2: Game
        self.game = TicTacToeLogic()
        self.grid_centers = {} # (row, col) -> (x, y)
        self.grid_radius = 0
        
        # Move Validation
        self.pending_move = None
        self.pending_move_frames = 0
        
        # Vision Parameters (Red Color)
        # HSV ranges for red (it wraps around 0/180)
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

    def detect_available_cameras(self) -> List[int]:
        """Detect available cameras"""
        available_cameras = []
        print("Scanning for available cameras...")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"  Found camera at /dev/video{i}")
            cap.release()
        return available_cameras
    
    def setup_camera(self) -> bool:
        """Setup camera with best available backend"""
        cameras = self.detect_available_cameras()
        if not cameras:
            print("ERROR: No cameras detected!")
            return False
        self.camera_id = cameras[0]
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_GSTREAMER, "GStreamer"),
            (cv2.CAP_ANY, "Default")
        ]
        for backend_id, backend_name in backends:
            print(f"Trying {backend_name} backend...")
            self.cap = cv2.VideoCapture(self.camera_id, backend_id)
            if self.cap.isOpened():
                self.backend_name = backend_name
                print(f"Successfully opened camera {self.camera_id} with {backend_name}")
                return True
        print("ERROR: Failed to open camera with any backend!")
        return False
    
    def configure_camera_properties(self):
        """Configure camera properties"""
        if not self.cap or not self.cap.isOpened():
            return
        print("Configuring camera properties...")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        except Exception as e:
            print(f"Warning: Some camera properties not supported: {e}")
    
    def get_red_centroids(self, frame) -> List[Tuple[int, int]]:
        """Find centroids of red objects in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 + mask2
        
        # Noise reduction
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500: # Minimum area filter
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
        return centroids

    def calibrate_grid(self, centroids: List[Tuple[int, int]]) -> bool:
        """
        Attempt to form a 3x3 grid from 5 points (Corners + Center).
        Expected points: (0,0), (0,2), (1,1), (2,0), (2,2)
        """
        if len(centroids) != 5:
            return False

        # Sort by Y (top to bottom)
        centroids.sort(key=lambda p: p[1])
        
        # Top row (2 points)
        top = sorted(centroids[:2], key=lambda p: p[0])
        # Middle (1 point)
        mid = centroids[2]
        # Bottom row (2 points)
        bottom = sorted(centroids[3:], key=lambda p: p[0])

        # Assign to grid positions
        # TL, TR
        p00, p02 = top[0], top[1]
        # C
        p11 = mid
        # BL, BR
        p20, p22 = bottom[0], bottom[1]

        # Interpolate missing points
        # (0,1) between (0,0) and (0,2)
        p01 = ((p00[0] + p02[0]) // 2, (p00[1] + p02[1]) // 2)
        # (1,0) between (0,0) and (2,0)
        p10 = ((p00[0] + p20[0]) // 2, (p00[1] + p20[1]) // 2)
        # (1,2) between (0,2) and (2,2)
        p12 = ((p02[0] + p22[0]) // 2, (p02[1] + p22[1]) // 2)
        # (2,1) between (2,0) and (2,2)
        p21 = ((p20[0] + p22[0]) // 2, (p20[1] + p22[1]) // 2)

        self.grid_centers = {
            (0,0): p00, (0,1): p01, (0,2): p02,
            (1,0): p10, (1,1): p11, (1,2): p12,
            (2,0): p20, (2,1): p21, (2,2): p22
        }

        # Calculate average radius for detection zones (half distance to nearest neighbor)
        dist_00_01 = np.linalg.norm(np.array(p00) - np.array(p01))
        self.grid_radius = int(dist_00_01 * 0.4) # 40% of spacing

        return True

    def draw_grid(self, frame):
        """Draw the grid lines and cell centers"""
        if not self.grid_centers:
            return

        # Draw lines
        # Horizontal
        for r in range(3):
            start = self.grid_centers[(r, 0)]
            end = self.grid_centers[(r, 2)]
            cv2.line(frame, start, end, (255, 255, 0), 2)
        # Vertical
        for c in range(3):
            start = self.grid_centers[(0, c)]
            end = self.grid_centers[(2, c)]
            cv2.line(frame, start, end, (255, 255, 0), 2)

        # Draw cells
        for (r, c), (x, y) in self.grid_centers.items():
            color = (0, 255, 0)
            if self.game.board[r][c] == 'X':
                color = (0, 0, 255) # Red for X
                cv2.putText(frame, "X", (x-20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            elif self.game.board[r][c] == 'O':
                color = (255, 0, 0) # Blue for O
                cv2.putText(frame, "O", (x-20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            else:
                # Empty cell marker
                cv2.circle(frame, (x, y), 5, (200, 200, 200), -1)
                cv2.circle(frame, (x, y), self.grid_radius, (100, 100, 100), 1)

        # Draw winning line
        if self.game.winner and self.game.winning_line:
            start_cell = self.game.winning_line[0]
            end_cell = self.game.winning_line[-1]
            pt1 = self.grid_centers[start_cell]
            pt2 = self.grid_centers[end_cell]
            cv2.line(frame, pt1, pt2, (0, 255, 255), 5)

        # Draw pending move indicator
        if self.pending_move:
            r, c = self.pending_move
            if (r, c) in self.grid_centers:
                x, y = self.grid_centers[(r, c)]
                # Draw a filling circle or progress bar
                progress = min(1.0, self.pending_move_frames / 20.0)
                radius = int(self.grid_radius * progress)
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
                cv2.putText(frame, "HOLD...", (x-30, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def update_game_state(self, centroids):
        """Update game board based on detected red objects with stability check"""
        if self.game.game_over or self.game.current_turn != 'X':
            return

        detected_move = None
        
        for (cx, cy) in centroids:
            # Check which cell this centroid belongs to
            for (r, c), (gx, gy) in self.grid_centers.items():
                if self.game.board[r][c] == ' ': # Only check empty cells
                    dist = np.sqrt((cx - gx)**2 + (cy - gy)**2)
                    if dist < self.grid_radius:
                        detected_move = (r, c)
                        break
            if detected_move:
                break
        
        # Stability Check / Dwell Time
        if detected_move:
            if detected_move == self.pending_move:
                self.pending_move_frames += 1
            else:
                self.pending_move = detected_move
                self.pending_move_frames = 1
        else:
            self.pending_move = None
            self.pending_move_frames = 0

        # Confirm move if stable for enough frames (e.g., 20 frames ~= 0.6s at 30fps)
        REQUIRED_FRAMES = 20
        
        if self.pending_move and self.pending_move_frames >= REQUIRED_FRAMES:
            r, c = self.pending_move
            print(f"Confirmed Move: {r}, {c}")
            self.game.make_move(r, c, 'X')
            
            # Reset pending state
            self.pending_move = None
            self.pending_move_frames = 0
            
            # Trigger AI response
            if not self.game.game_over:
                self.game.ai_move()

    def initialize(self) -> bool:
        print("=" * 50)
        print("OpenCV Camera Preview - Tic-Tac-Toe Mode")
        print("=" * 50)
        
        if not self.setup_camera():
            return False
        self.configure_camera_properties()
        return True
    
    def run(self):
        if not self.cap or not self.cap.isOpened():
            print("ERROR: Camera not initialized!")
            return
            
        window_name = "Tic-Tac-Toe Vision"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        print(f"\nStarting...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Start Calibration (Place 5 red balls: Corners + Center)")
        print("  'SPACE' - Confirm Calibration / Start Game")
        print("  'r' - Reset Game")
        
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            
            # Flip frame for mirror effect (easier for user)
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            centroids = self.get_red_centroids(frame)
            
            # Draw all detected red blobs for debug
            for (cx, cy) in centroids:
                cv2.circle(display_frame, (cx, cy), 10, (0, 0, 255), 2)

            if self.mode == 0: # Preview
                cv2.putText(display_frame, "Mode: PREVIEW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to Calibrate", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif self.mode == 1: # Calibration
                cv2.putText(display_frame, "Mode: CALIBRATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Detected Points: {len(centroids)}/5", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.calibrate_grid(centroids):
                    self.draw_grid(display_frame)
                    cv2.putText(display_frame, "Grid Found! Press SPACE to Confirm", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Place red balls at 4 corners + center", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif self.mode == 2: # Game
                cv2.putText(display_frame, "Mode: GAME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                # Update Game
                self.update_game_state(centroids)
                
                # Draw Board
                self.draw_grid(display_frame)
                
                if self.game.game_over:
                    msg = f"Winner: {self.game.winner}"
                    cv2.putText(display_frame, msg, (int(self.width/2)-100, int(self.height/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 215, 255), 4)
                    cv2.putText(display_frame, "Press 'r' to Restart", (int(self.width/2)-80, int(self.height/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.mode = 1
                print("Entering Calibration Mode...")
            elif key == ord(' '):
                if self.mode == 1 and self.grid_centers:
                    self.mode = 2
                    self.game.reset()
                    print("Game Started!")
            elif key == ord('r'):
                if self.mode == 2:
                    self.game.reset()
                    print("Game Reset!")

        self.cleanup()

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up.")

def main():
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    
    preview = CameraPreview(args.width, args.height, args.fps)
    if preview.initialize():
        preview.run()
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
