import json
import time
import sys
import copy
import threading
import queue
from xarmapi import LSCServoController

SAVE_FILE = "servo_configs.json"
RESET_NAME = "reset"
GRIPPER_NAME = "gripper"
PICK_NAME = "pick"
MAT_PREFIX = "mat"
DROP_SUFFIX = "drop"
MATS = 9
SEQ_RESET   = [5, 6, 4, 3, 2, 1]
SEQ_GRIPPER = [1, 3, 4, 2, 6, 5]
SEQ_PICK    = [1, 2, 3, 4, 5, 6]
SEQ_DROP    = [2, 3, 4, 6, 5, 1]
ANGLE_PER_SECOND = 500  # degrees per second
MIN_TIME_MS = 150
SERVO_RESET = 450  # Default servo position

# Global State
PAUSED = False
CMD_QUEUE = queue.Queue()
EXIT_FLAG = False

def load_json():
    try:
        with open(SAVE_FILE) as f:
            return json.load(f)
    except Exception:
        print(f"Failed to load {SAVE_FILE}.")
        return []



def execute_move(ctrl, targets_dict):
    """
    Executes a move and handles waiting/pausing.
    Replaces ctrl.move_servos + sleep.
    """
    global PAUSED, EXIT_FLAG, ANGLE_PER_SECOND, MIN_TIME_MS
    
    # 1. Calculate duration
    # We need start positions. 
    # Optimization: We could track last known positions globally to avoid reads?
    # For now, let's trust the servo read? It costs ~20ms.
    try:
        current_positions = ctrl.read_servo_positions(list(targets_dict.keys()))
    except:
        print("Read failed, proceeding with default delay")
        current_positions = {}
        
    moves = []
    max_duration = 0
    
    for sid, target in targets_dict.items():
        start_pos = current_positions.get(sid, 1500) # Default if read fails
        dist = abs(target - start_pos)
        duration = max(int((dist / ANGLE_PER_SECOND) * 1000), MIN_TIME_MS)
        max_duration = max(max_duration, duration)
        moves.append((sid, target))
    
    if not moves: return

    # 2. Start Move
    ctrl.move_servos(moves, max_duration)
    
    # 3. Wait with Pause support
    start_time = time.time()
    end_time = start_time + (max_duration / 1000.0) + 0.1
    
    while time.time() < end_time and not EXIT_FLAG:
        if PAUSED:
            # STOP HARDWARE
            print("!!! PAUSING HARDWARE !!!")
            try:
                # Read current to hold
                hold_pos = ctrl.read_servo_positions(list(targets_dict.keys()))
                stop_moves = [(sid, pos) for sid, pos in hold_pos.items()]
                if stop_moves:
                    ctrl.move_servos(stop_moves, 200) # Stop in 200ms
            except Exception as e:
                print(f"Pause error: {e}")
            
            # Wait while paused
            while PAUSED and not EXIT_FLAG:
                time.sleep(0.1)
                
            if EXIT_FLAG: break
            
            # RESUME HARDWARE
            print("... RESUMING HARDWARE ...")
            try:
                # Recalculate path from WHERE WE STOPPED
                curr_pos = ctrl.read_servo_positions(list(targets_dict.keys()))
                resume_moves = []
                new_max_duration = 0
                for sid, target in targets_dict.items():
                    c_pos = curr_pos.get(sid, target)
                    dist = abs(target - c_pos)
                    # Use same speed
                    dur = max(int((dist / ANGLE_PER_SECOND) * 1000), MIN_TIME_MS)
                    new_max_duration = max(new_max_duration, dur)
                    resume_moves.append((sid, target))
                
                if resume_moves:
                    ctrl.move_servos(resume_moves, new_max_duration)
                    # Reset wait timer
                    end_time = time.time() + (new_max_duration / 1000.0) + 0.1
            except Exception as e:
                print(f"Resume error: {e}")
                
        time.sleep(0.05)


def move_servo(ctrl, sid, target_angle, last_position):
    # Wrapper to use execute_move
    execute_move(ctrl, {sid: target_angle})
    return target_angle


def move_row(ctrl, row_data, seq, last_positions, pick_only_id=None):
    global PAUSED, EXIT_FLAG
    
    # Check pause before starting row
    while PAUSED and not EXIT_FLAG: time.sleep(0.1)
    if EXIT_FLAG: return

    print(f"\nApplying '{row_data['name']}' sequence")

    filtered_seq = [s for s in seq if pick_only_id is None or s == pick_only_id]
    if not filtered_seq: return

    first = filtered_seq[0]
    last = filtered_seq[-1]
    middle = filtered_seq[1:-1] if len(filtered_seq) > 2 else []

    # Move first
    if first and str(first) in row_data['servos']:
        target = row_data['servos'][str(first)]
        execute_move(ctrl, {first: target})
        last_positions[first] = target

    # Move middle
    middle_targets = {}
    for sid in middle:
        if str(sid) in row_data['servos']:
            middle_targets[sid] = row_data['servos'][str(sid)]
    
    if middle_targets:
        print(f"Moving middle servos: {list(middle_targets.keys())}")
        execute_move(ctrl, middle_targets)
        for sid, target in middle_targets.items():
            last_positions[sid] = target

    # Move last
    if last and last != first and str(last) in row_data['servos']:
        target = row_data['servos'][str(last)]
        execute_move(ctrl, {last: target})
        last_positions[last] = target


def adjust_angles(row_data, adjustment, servos_to_adjust=None):
    """
    Returns a copy of row_data adjusting specified servos angles by adjustment.
    """
    adjusted = copy.deepcopy(row_data)
    for sid_str, angle in adjusted['servos'].items():
        sid = int(sid_str)
        if servos_to_adjust is None or sid in servos_to_adjust:
            adjusted['servos'][sid_str] = angle + adjustment
    return adjusted


def run_batch(ctrl, row_map, last_positions, mat_num):
    mat_name = f"{MAT_PREFIX}{mat_num}{DROP_SUFFIX}".lower()

    if mat_name not in row_map:
        print(f"Mat configuration '{mat_name}' missing")
        return

    print(f"\nRunning batch for {mat_name.upper()}")

    # Initial reset all servos
    move_row(ctrl, row_map['reset'], SEQ_RESET, last_positions)

    # Update last_positions as reset was done
    for sid in SEQ_RESET:
        if str(sid) in row_map['reset']['servos']:
            last_positions[sid] = row_map['reset']['servos'][str(sid)]

    # Gripper movement regular
    move_row(ctrl, row_map['gripper'], SEQ_GRIPPER, last_positions)

    # Pick step: only servo 1 moves
    move_row(ctrl, row_map['pick'], SEQ_PICK, last_positions, pick_only_id=1)

    # Partial reset after pickup: only servo 5 moves BUT angle adjusted by -50 degrees
    reset_adjusted = adjust_angles(row_map['reset'], -50, servos_to_adjust=[5])
    move_row(ctrl, reset_adjusted, [5], last_positions)

    # Mat drop step
    move_row(ctrl, row_map[mat_name], SEQ_DROP, last_positions)


def stdin_reader():
    global PAUSED, EXIT_FLAG, CMD_QUEUE
    while not EXIT_FLAG:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip()
            if not cmd: continue
            
            if cmd == "STOP":
                PAUSED = True
                print("!!! PAUSED !!!", flush=True)
            elif cmd == "RESUME":
                PAUSED = False
                print("... RESUMING ...", flush=True)
            elif cmd == "0" or cmd.lower() == "exit":
                EXIT_FLAG = True
                CMD_QUEUE.put("EXIT")
            else:
                CMD_QUEUE.put(cmd)
        except Exception as e:
            print(f"Stdin error: {e}")
            break

def main():
    global EXIT_FLAG, CMD_QUEUE
    
    rows = load_json()
    if not rows:
        print("Config file missing or empty, exiting.")
        return

    row_map = {row['name'].lower(): row for row in rows}

    for name in ['reset', 'gripper', 'pick']:
        if name not in row_map:
            print(f"Missing mandatory config '{name}', exiting.")
            return

    # If command line arg provided, run once and exit (legacy mode support if needed)
    # But usually this script is run as a daemon for tictactoe.
    # We'll support args still but generally we expect interactive/piped input.
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Just queue it and run logic below
        CMD_QUEUE.put(arg)
        # Assuming one-shot if arg provided? 
        # The original code did return after arg processing.
        # But for robustness we can just let it run.
        # Actually, if arg is present, original main() returned.
        # Let's keep it simple: if arg, put in queue, set EXIT_FLAG after?
        # No, let's just stick to the loop behavior which matches 'move.py' usage in daemon.
    
    # Start stdin reader
    t = threading.Thread(target=stdin_reader, daemon=True)
    t.start()

    with LSCServoController("/dev/ttyUSB0") as ctrl:
        detected_ids = ctrl.detect_servos()
        try:
            positions = ctrl.read_servo_positions(detected_ids)
            last_positions = {sid: positions.get(str(sid), SERVO_RESET) for sid in detected_ids}
        except Exception:
            print("Could not read servo positions; assuming default.")
            last_positions = {sid: SERVO_RESET for sid in detected_ids}

        print("Servo Controller Ready. Listening for commands...")

        while not EXIT_FLAG:
            try:
                # Non-blocking get or wait short time
                try:
                    cmd = CMD_QUEUE.get(timeout=0.1)
                except queue.Empty:
                    continue

                if cmd == "EXIT":
                    break
                
                # Check Pause before processing command
                while PAUSED and not EXIT_FLAG: time.sleep(0.1)

                if cmd.lower() == 'reset':
                    print("Running reset sequence")
                    move_row(ctrl, row_map['reset'], SEQ_RESET, last_positions)
                    print("Reset done.", flush=True)
                else:
                    try:
                        mat_num = int(cmd)
                        if 1 <= mat_num <= MATS:
                            run_batch(ctrl, row_map, last_positions, mat_num)
                            print("Final reset after batch")
                            move_row(ctrl, row_map['reset'], SEQ_RESET, last_positions)
                            print(f"Batch for mat {mat_num} done.", flush=True)
                        else:
                            print(f"Invalid mat number: {mat_num}")
                    except ValueError:
                        print(f"Ignored unknown command: {cmd}")
                        
            except KeyboardInterrupt:
                EXIT_FLAG = True
                break
            except Exception as e:
                print(f"Error in main loop: {e}")

if __name__ == '__main__':
    main()
