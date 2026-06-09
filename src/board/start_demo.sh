#!/bin/bash

# 1. Start the camera application in the background
echo "Starting Camera Preview..."
/usr/local/demo/application/camera/bin/launch_camera.sh &
CAMERA_PID=$!

# 2. Wait for 5 seconds
sleep 5

# 3. Stop the camera application
echo "Stopping Camera Preview..."
kill $CAMERA_PID 2>/dev/null
# Force kill the specific camera binary if known (often 'camera_app' or similar)
pkill -9 -f "launch_camera.sh"
pkill -9 -f "gst-launch-1.0"
pkill -9 -f "weston-simple-im"
pkill -9 -f "camera" 

# Wait a bit longer for resources to free up
sleep 2

# 4. Start TicTacToe
echo "Starting TicTacToe..."
cd /home/root/demoupdate/run_arm
python3 tictactoe.py
