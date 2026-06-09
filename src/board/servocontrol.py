#!/usr/bin/env python3
"""
Control script for multiple HTS-20H bus servo motors in daisy chain using 
Waveshare Serial Bus Servo Driver Board connected via USB-C port on Linux.
"""

import serial
import time
import struct
import argparse
import sys

class BusServoController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        """
        Initialize the Bus Servo Controller.
        
        Args:
            port (str): Serial port device, typically /dev/ttyACM0 or similar on Linux
            baudrate (int): Communication baudrate, default is 115200
        """
        try:
            self.serial = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(0.1)  # Give some time for the serial connection to establish
        except serial.SerialException as e:
            print(f"Error opening serial port {port}: {e}")
            sys.exit(1)
        
    def __del__(self):
        """Close the serial connection when the object is destroyed."""
        if hasattr(self, 'serial') and self.serial.is_open:
            self.serial.close()
            
    def calculate_checksum(self, data):
        """Calculate the checksum for the command packet."""
        return (~sum(data) & 0xFF)
    
    def send_command(self, servo_id, cmd, params=None):
        """
        Send a command to the servo.
        
        Args:
            servo_id (int): ID of the servo (1-253)
            cmd (int): Command code
            params (list): Parameters for the command (bytes)
        """
        if params is None:
            params = []
            
        # Build packet: header(2) + id(1) + length(1) + cmd(1) + params(n) + checksum(1)
        length = 3 + len(params)  # ID + CMD + PARAMS + CHECKSUM
        packet = [0x55, 0x55, servo_id, length, cmd] + params
        checksum = self.calculate_checksum(packet[2:])
        packet.append(checksum)
        
        # Send the packet
        self.serial.write(bytes(packet))
        time.sleep(0.01)  # Small delay to ensure command is processed
        
        # Read response if available
        response = self.serial.read(self.serial.in_waiting)
        return response
    
    def broadcast_command(self, cmd, params=None):
        """
        Send a command to all servos (broadcast).
        
        Args:
            cmd (int): Command code
            params (list): Parameters for the command (bytes)
        """
        # Use ID 254 for broadcast
        return self.send_command(254, cmd, params)
    
    def set_position(self, servo_id, position, time_ms=0):
        """
        Set the position of the servo.
        
        Args:
            servo_id (int): ID of the servo
            position (int): Target position (0-1000 for HTS-20H)
            time_ms (int): Time to reach the position in milliseconds
        """
        # Convert position and time to bytes
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        time_low = time_ms & 0xFF
        time_high = (time_ms >> 8) & 0xFF
        
        # Send the position command (typically CMD=1 for position control)
        self.send_command(servo_id, 1, [pos_low, pos_high, time_low, time_high])
    
    def set_positions(self, positions_dict, time_ms=0):
        """
        Set positions of multiple servos.
        
        Args:
            positions_dict (dict): Dictionary of {servo_id: position}
            time_ms (int): Time to reach the position in milliseconds
        """
        for servo_id, position in positions_dict.items():
            self.set_position(servo_id, position, time_ms)
            time.sleep(0.02)  # Small delay between commands
    
    def read_position(self, servo_id):
        """
        Read the current position of the servo.
        
        Args:
            servo_id (int): ID of the servo
            
        Returns:
            int: Current position of the servo
        """
        # Command to read position (typically CMD=2)
        response = self.send_command(servo_id, 2)
        
        # Process the response if available
        if response and len(response) >= 8:
            # The position is typically at bytes 5-6
            position = response[5] + (response[6] << 8)
            return position
        return None
    
    def read_all_positions(self, servo_ids):
        """
        Read positions of multiple servos.
        
        Args:
            servo_ids (list): List of servo IDs to read
            
        Returns:
            dict: Dictionary of {servo_id: position}
        """
        positions = {}
        for servo_id in servo_ids:
            position = self.read_position(servo_id)
            if position is not None:
                positions[servo_id] = position
            time.sleep(0.02)  # Small delay between reads
        return positions
    
    def set_id(self, old_id, new_id):
        """
        Change the ID of a servo.
        
        Args:
            old_id (int): Current ID of the servo
            new_id (int): New ID to assign
        """
        if old_id == 254:
            print("Error: Cannot use broadcast ID (254) as old_id when setting a new ID")
            return
        if new_id < 1 or new_id > 253:
            print("Error: New ID must be between 1 and 253")
            return
        self.send_command(old_id, 13, [new_id])
        print(f"Changed servo ID from {old_id} to {new_id}")
        time.sleep(0.5)  # Give the servo time to process the ID change
    
    def detect_servos(self, id_range=(1, 20)):
        """
        Detect which servo IDs are responding in the given range.
        
        Args:
            id_range (tuple): Range of IDs to check (start, end)
            
        Returns:
            list: List of responding servo IDs
        """
        responding_ids = []
        print(f"Scanning for servos in ID range {id_range[0]}-{id_range[1]}...")
        
        for servo_id in range(id_range[0], id_range[1] + 1):
            # Send a simple read position command
            response = self.send_command(servo_id, 2)
            if response and len(response) >= 8:
                position = self.read_position(servo_id)
                responding_ids.append(servo_id)
                print(f"Found servo ID {servo_id} at position {position}")
            time.sleep(0.01)
            
        return responding_ids
    
    def set_mode(self, servo_id, mode, speed=0):
        """
        Set operating mode for the servo.
        
        Args:
            servo_id (int): ID of the servo
            mode (int): 0 for position control, 1 for continuous rotation
            speed (int): Speed for continuous rotation mode
        """
        if mode == 0:  # Position control mode
            self.send_command(servo_id, 29, [0, 0, 0, 0])
        elif mode == 1:  # Continuous rotation mode
            speed_low = speed & 0xFF
            speed_high = (speed >> 8) & 0xFF
            self.send_command(servo_id, 29, [1, 0, speed_low, speed_high])
    
    def stop(self, servo_id):
        """
        Stop the servo motion.
        
        Args:
            servo_id (int): ID of the servo
        """
        # Command to stop motion (typically CMD=12)
        self.send_command(servo_id, 12)
    
    def stop_all(self, servo_ids):
        """
        Stop all specified servos.
        
        Args:
            servo_ids (list): List of servo IDs to stop
        """
        for servo_id in servo_ids:
            self.stop(servo_id)
            time.sleep(0.01)


def setup_new_servos(controller):
    """
    Setup function to assign new IDs to servos in a daisy chain.
    Assumes only one servo is connected initially for safety.
    """
    print("=== SERVO ID SETUP MODE ===")
    print("This process will help you assign unique IDs to up to 6 servos.")
    print("IMPORTANT: Start with only ONE servo connected!")
    print("\nDetecting initial servo...")
    
    # First, detect any connected servos
    initial_servos = controller.detect_servos()
    
    if not initial_servos:
        print("No servos detected. Please check connections and try again.")
        return False
    
    if len(initial_servos) > 1:
        print("Multiple servos detected. Please start with only one servo connected.")
        print(f"Detected servos: {initial_servos}")
        return False
    
    current_id = initial_servos[0]
    print(f"Found initial servo with ID {current_id}")
    
    # Now we'll assign IDs 1-6 to all servos
    assigned_ids = []
    for new_id in range(1, 7):
        if new_id == current_id:
            print(f"Servo already has ID {new_id}, skipping...")
            assigned_ids.append(new_id)
            continue
            
        print(f"\nSetting servo ID from {current_id} to {new_id}...")
        controller.set_id(current_id, new_id)
        assigned_ids.append(new_id)
        
        if new_id < 6:  # Don't prompt after the last servo
            input(f"Now connect the next servo and press Enter (or Ctrl+C to stop if no more servos)...")
            # Detect newly connected servo
            time.sleep(1)  # Give some time for the new servo to initialize
            new_servos = controller.detect_servos()
            new_servos = [s for s in new_servos if s not in assigned_ids]
            
            if not new_servos:
                print("No new servo detected. Please check connections.")
                return False
                
            if len(new_servos) > 1:
                print(f"Multiple new servos detected: {new_servos}. Please connect only one new servo at a time.")
                return False
                
            current_id = new_servos[0]
            print(f"Detected new servo with ID {current_id}")
    
    print("\nID assignment complete! All servos have been assigned IDs 1-6.")
    return True


def run_demo_sequence(controller, servo_ids):
    """Run a demo sequence on all connected servos."""
    print("\n=== RUNNING DEMO SEQUENCE ===")
    
    # Move all servos to center position
    print("Moving all servos to center position...")
    for servo_id in servo_ids:
        controller.set_position(servo_id, 500, 1000)
        time.sleep(0.1)
    time.sleep(1)
    
    # Move each servo in sequence
    print("Moving each servo in sequence...")
    for servo_id in servo_ids:
        print(f"Moving servo {servo_id}...")
        controller.set_position(servo_id, 300, 500)
        time.sleep(0.6)
        controller.set_position(servo_id, 700, 500)
        time.sleep(0.6)
        controller.set_position(servo_id, 500, 500)
        time.sleep(0.6)
    
    # Move all servos together in a wave pattern
    print("Moving all servos in a wave pattern...")
    positions = [300, 400, 500, 600, 700, 600, 500, 400]
    for pos in positions:
        for servo_id in servo_ids:
            controller.set_position(servo_id, pos, 300)
        time.sleep(0.35)
    
    # Return all to center
    print("Returning all servos to center...")
    for servo_id in servo_ids:
        controller.set_position(servo_id, 500, 1000)
    
    print("Demo complete!")


def main():
    parser = argparse.ArgumentParser(description='Control multiple HTS-20H bus servos in daisy chain')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--setup', action='store_true', help='Run setup to assign IDs to new servos')
    parser.add_argument('--scan', action='store_true', help='Scan for connected servos')
    parser.add_argument('--demo', action='store_true', help='Run demo sequence on all connected servos')
    parser.add_argument('--id', type=int, help='Servo ID to control')
    parser.add_argument('--position', type=int, help='Target position (0-1000)')
    parser.add_argument('--time', type=int, default=1000, help='Time to reach position (ms)')
    parser.add_argument('--read', action='store_true', help='Read current position')
    parser.add_argument('--read-all', action='store_true', help='Read positions of all connected servos')
    parser.add_argument('--mode', type=int, choices=[0, 1], help='Set mode (0=position, 1=continuous)')
    parser.add_argument('--speed', type=int, default=0, help='Speed for continuous mode')
    parser.add_argument('--old-id', type=int, help='Current ID of a servo')
    parser.add_argument('--new-id', type=int, help='New ID to set')
    parser.add_argument('--multi-positions', type=str, help='Set multiple positions: "id1:pos1,id2:pos2,..."')
    
    args = parser.parse_args()
    
    try:
        controller = BusServoController(port=args.port)
        print(f"Connected to servo controller on {args.port}")
        
        # Setup mode to assign IDs to new servos
        if args.setup:
            setup_new_servos(controller)
            return
            
        # Scan for connected servos
        if args.scan:
            detected_servos = controller.detect_servos()
            if detected_servos:
                print(f"Found {len(detected_servos)} servos with IDs: {detected_servos}")
            else:
                print("No servos detected. Please check connections.")
            return
            
        # Run demo sequence
        if args.demo:
            detected_servos = controller.detect_servos()
            if detected_servos:
                run_demo_sequence(controller, detected_servos)
            else:
                print("No servos detected. Please check connections.")
            return
            
        # Read all servo positions
        if args.read_all:
            detected_servos = controller.detect_servos()
            if detected_servos:
                positions = controller.read_all_positions(detected_servos)
                print("Current servo positions:")
                for servo_id, position in positions.items():
                    print(f"Servo {servo_id}: Position {position}")
            else:
                print("No servos detected. Please check connections.")
            return
            
        # Set multiple positions at once
        if args.multi_positions:
            positions_dict = {}
            try:
                # Parse input format "id1:pos1,id2:pos2,..."
                for pair in args.multi_positions.split(','):
                    servo_id, position = map(int, pair.split(':'))
                    positions_dict[servo_id] = position
                    
                print(f"Setting multiple positions: {positions_dict}")
                controller.set_positions(positions_dict, args.time)
            except ValueError:
                print("Error: Invalid format for multi-positions. Use 'id1:pos1,id2:pos2,...'")
            return
            
        # Individual servo commands
        if args.id:
            # Read position
            if args.read:
                position = controller.read_position(args.id)
                print(f"Current position of servo {args.id}: {position}")
                
            # Set position
            if args.position is not None:
                print(f"Setting servo {args.id} to position {args.position} over {args.time}ms")
                controller.set_position(args.id, args.position, args.time)
                
            # Set mode
            if args.mode is not None:
                mode_name = "position control" if args.mode == 0 else "continuous rotation"
                print(f"Setting servo {args.id} to {mode_name} mode")
                controller.set_mode(args.id, args.mode, args.speed)
                
        # Change servo ID
        if args.old_id is not None and args.new_id is not None:
            print(f"Changing servo ID from {args.old_id} to {args.new_id}")
            controller.set_id(args.old_id, args.new_id)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Tips for troubleshooting:")
        print("1. Make sure the board is connected via USB-C")
        print("2. Check if you have permission to access the serial port")
        print("3. The port might be different, try: ls /dev/tty*")
        print("4. You might need to add your user to the 'dialout' group: sudo usermod -a -G dialout $USER")


if __name__ == "__main__":
    main()
