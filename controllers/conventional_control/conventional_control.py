from controller import Robot
import math
import numpy as np
import socket
import json
import time

# ============================================================
# Adjustable motion parameters
# ============================================================
FORWARD_SPEED = 3.0     # rad/s
MAX_TURN_SPEED = 3.5    # rad/s maximum turning speed
AVOID_THRESHOLD = 0.45  # m – threshold for trigger
TURN_GAIN = 5.0         # More aggressive turning
MIN_TURN_SPEED = 0.5    # Higher minimum turn speed to ensure responsiveness
SAFE_DISTANCE = 0.8     # Distance where robot feels completely safe
MAX_SPEED = 6.4         # RotationalMotor's max speed

# ============================================================
# Webots Robot Setup
# ============================================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motors
wheels = [
    robot.getDevice("front left wheel"),
    robot.getDevice("back left wheel"),
    robot.getDevice("front right wheel"),
    robot.getDevice("back right wheel")
]

for w in wheels:
    w.setPosition(float('inf'))
    w.setVelocity(0)

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def set_speed(left, right):
    left = clamp(left, -MAX_SPEED, MAX_SPEED)
    right = clamp(right, -MAX_SPEED, MAX_SPEED)
    wheels[0].setVelocity(left)
    wheels[1].setVelocity(left)
    wheels[2].setVelocity(right)
    wheels[3].setVelocity(right)

# Lidar
lidar = robot.getDevice("Sick LMS 291")
lidar.enable(timestep)

# Socket communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
plotter_address = ('localhost', 8888)

def send_lidar_data(ranges, left_speed, right_speed, state):
    """Send LIDAR data and robot state via socket"""
    data = {
        'ranges': ranges,
        'left_speed': left_speed,
        'right_speed': right_speed,
        'state': state,
        'timestamp': time.time(),
        'avoid_threshold': AVOID_THRESHOLD,
        'safe_distance': SAFE_DISTANCE
    }
    
    try:
        message = json.dumps(data).encode('utf-8')
        sock.sendto(message, plotter_address)
    except Exception as e:
        print(f"Socket error: {e}")

def get_obstacle_distances(ranges):
    """Get minimum distances in different sectors with more granularity"""
    num_points = len(ranges)
    
    # Define sectors (more granular for better decision making)
    sectors = {}
    sectors['far_left'] = np.min(ranges[0:30])  # 0-60 degrees
    sectors['left'] = np.min(ranges[30:60])     # 60-120 degrees  
    sectors['front_left'] = np.min(ranges[60:90]) # 120-180 degrees
    sectors['front'] = np.min(ranges[90:110])   # 180-220 degrees (narrow front)
    sectors['front_right'] = np.min(ranges[110:140]) # 220-300 degrees
    sectors['right'] = np.min(ranges[140:170])  # 300-360 degrees
    sectors['far_right'] = np.min(ranges[170:]) # 360+ degrees
    
    return sectors

def compute_dynamic_turn_speeds(distances):
    """Simple turning logic: turn towards the side with more space"""
    
    # Calculate average distances for left and right sides
    left_avg = (distances['far_left'] + distances['left'] + distances['front_left']) / 3.0
    right_avg = (distances['far_right'] + distances['right'] + distances['front_right']) / 3.0
    
    # Front clearance check
    front_clearance = distances['front']
    
    # Emergency front obstacle - need to turn decisively
    if front_clearance < AVOID_THRESHOLD * 0.6:
        if right_avg > left_avg:
            # More space on right, turn right
            turn_strength = min(1.0, (AVOID_THRESHOLD - front_clearance) / AVOID_THRESHOLD)
            left_speed = FORWARD_SPEED + MAX_TURN_SPEED * turn_strength
            right_speed = FORWARD_SPEED - MAX_TURN_SPEED * turn_strength
            return left_speed, right_speed, "EMERGENCY_RIGHT"
        else:
            # More space on left, turn left
            turn_strength = min(1.0, (AVOID_THRESHOLD - front_clearance) / AVOID_THRESHOLD)
            left_speed = FORWARD_SPEED - MAX_TURN_SPEED * turn_strength
            right_speed = FORWARD_SPEED + MAX_TURN_SPEED * turn_strength
            return left_speed, right_speed, "EMERGENCY_LEFT"
    
    # Normal operation - turn towards the side with more space
    space_difference = right_avg - left_avg
    
    # Calculate turn strength based on how much clearer one side is
    normalized_difference = space_difference / max(left_avg, right_avg, 0.1)  # Avoid division by zero
    
    if abs(normalized_difference) < 0.1:  # Very similar distances
        # Move straight with minimal turning
        return FORWARD_SPEED, FORWARD_SPEED, "CLEAR_STRAIGHT"
    
    else:
        # Calculate turn strength (more aggressive when difference is larger)
        turn_strength = min(MAX_TURN_SPEED, abs(normalized_difference) * TURN_GAIN)
        turn_strength = max(MIN_TURN_SPEED, turn_strength)  # Ensure minimum turning
        
        if space_difference > 0:  # Right side has more space
            left_speed = FORWARD_SPEED + turn_strength
            right_speed = FORWARD_SPEED - turn_strength
            return left_speed, right_speed, "TURN_RIGHT"
        else:  # Left side has more space
            left_speed = FORWARD_SPEED - turn_strength
            right_speed = FORWARD_SPEED + turn_strength
            return left_speed, right_speed, "TURN_LEFT"

def should_stop(distances):
    """Only stop if we're definitely going to crash"""
    # Check if we're heading directly into an obstacle with no escape
    front_critical = distances['front'] < AVOID_THRESHOLD * 0.5
    left_critical = distances['front_left'] < AVOID_THRESHOLD * 0.5  
    right_critical = distances['front_right'] < AVOID_THRESHOLD * 0.5
    
    # Only stop if all forward paths are critically blocked
    if front_critical and left_critical and right_critical:
        return True
    
    return False

print("Controller started. Connecting to plotter...")

try:
    while robot.step(timestep) != -1:
        # Get LIDAR data
        ranges = lidar.getRangeImage()
        
        # Calculate obstacle distances
        distances = get_obstacle_distances(ranges)
        
        # Check if we need to stop (only in emergency)
        if should_stop(distances):
            set_speed(0, 0)
            send_lidar_data(ranges, 0, 0, "EMERGENCY_STOP")
            continue
        
        # Compute dynamic turn speeds
        left_speed, right_speed, state = compute_dynamic_turn_speeds(distances)
        
        # Set motor speeds
        set_speed(left_speed, right_speed)
        
        # Send data to plotter
        send_lidar_data(ranges, left_speed, right_speed, state)
        
except KeyboardInterrupt:
    print("Controller stopped")
finally:
    sock.close()
    set_speed(0, 0)  # Stop motors when exiting