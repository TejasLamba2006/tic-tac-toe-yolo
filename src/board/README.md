# Tic-Tac-Toe Robot System

This directory contains the modularized Python code for the STM32MP25 DCMIPP-Aware Tic-Tac-Toe Game. The system integrates computer vision, game logic, and robotic control to play a physical game of Tic-Tac-Toe against a human opponent.

## 📂 Module Descriptions

| Module | Filename | Description |
|--------|----------|-------------|
| **Startup Script** | `start_demo.sh` | **Recomended Entry Point**. Runs a camera preview test, cleans up resources, and then launches the main game (`tictactoe.py`). |
| **App Controller** | `tictactoe.py` | The main Python entry point. Orchestrates the application lifecycle, main event loop, user input handling, and integrates all subsystems. |
| **Vision System** | `vision.py` | Handles camera hardware interaction (V4L2/DCMIPP), frame capturing, color detection algorithms, and grid calibration logic. |
| **Game Logic** | `game_logic.py` | Encapsulates the Tic-Tac-Toe rules, board state management, and the Minimax AI algorithm with alpha-beta pruning. |
| **Camera Controller** | `camera_controller.py` | Manages the virtual camera viewport, providing functionality for digital zoom, panning, and flipping the feed. |
| **Renderer** | `renderer.py` | Responsible for all visual output. Draws game overlays, grids, status text, and win messages onto the video frame. |
| **Communications** | `comms.py` | Handles Inter-Process Communication (IPC). Manages connections to the robot control script (`move.py`) and the GUI overlay (`ui.py`). |

## 🏗 System Architecture

The application is structured into distinct layers to separate concerns:

| Layer | Component | Responsibilities |
|-------|-----------|------------------|
| **Presentation** | `renderer.py`, `ui.py` | Visual feedback, game board overlay, user prompts, and status messages. |
| **Application** | `tictactoe.py` | Main loop, state machine (Preview -> Calibrating -> Game), input routing. |
| **Domain Logic** | `game_logic.py` | Core game rules, win validation, AI decision making. |
| **Perception** | `vision.py` | Image acquisition, color segmentation (HSV), consensus/stability checks. |
| **control** | `camera_controller.py` | Viewport transformations (Affine transforms, cropping). |
| **Infrastructure** | `comms.py` | Subprocess management, IPC (pipes) for hardware abstraction. |
| **Hardware** | `move.py`, `servocontrol.py` | Physical robot arm movement and servo control. |

## 🔄 Game Flow & Logic

The system operates in a continuous loop triggered by camera frames.

```mermaid
graph TD
    A[Start tictactoe.py] --> B[Initialize Subsystems]
    B --> C{Main Loop}
    C --> D[Capture Frame (vision.py)]
    D --> E[Apply Viewport Transform (camera_controller.py)]
    E --> F[Detect Colors & Grid (vision.py)]
    
    F --> G{Current Mode?}
    
    G -- Mode 0: Preview --> H[Show Camera Feed & Instructions]
    G -- Mode 1: Calibration --> I[Detect 5 Red Points]
    I --> J{Valid Grid?}
    J -- Yes --> K[Calibrate Grid & Auto-Start]
    J -- No --> I
    
    G -- Mode 2: Game --> L[Analyze Board State]
    L --> M{Cheat Detection}
    M -- Detected --> N[End Game / Warn]
    M -- Operations Normal --> O{Whose Turn?}
    
    O -- Human (X) --> P[Wait for Stability]
    P --> Q[Process Move]
    Q --> R[Update Game State]
    
    O -- Robot (O) --> S[Calculate AI Move (Minimax)]
    S --> T[Send Command (comms.py -> move.py)]
    T --> U[Wait for Visual Confirmation]
    U --> R
    
    R --> V[Render Overlays (renderer.py)]
    V --> W[Update UI Process]
    W --> C
```

## 🚀 Quick Start

1.  **Run the Game:**
    The recommended entry point is the shell script, which initializes the camera preview test before starting the game logic.
    ```bash
    ./start_demo.sh
    ```

2.  **Controls:**
    *   `c` - Enter Calibration Mode (Place 5 red balls: 4 corners + center)
    *   `SPACE` - Confirm Calibration / Start Game
    *   `r` - Reset Game
    *   `m` - Toggle Manual Confirmation Mode
    *   `q` - Quit
    *   `+/-` - Zoom In/Out
    *   `Arrows` - Pan Camera

## 🔧 Dependencies
*   Python 3
*   OpenCV (`cv2`)
*   Numpy
*   V4L2 utilities (for camera config)
