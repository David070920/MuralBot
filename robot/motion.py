"""
Motion control module for MuralBot.

This module provides functions for controlling robot movement.
"""

import time
import math
from .hardware import send_command

def move_to_position(connection, target_x, target_y, current_x, current_y, speed=100):
    """
    Move the robot to a target position.
    
    Args:
        connection: Serial connection object
        target_x: Target X-coordinate
        target_y: Target Y-coordinate
        current_x: Current X-coordinate
        current_y: Current Y-coordinate
        speed: Movement speed (1-100)
    
    Returns:
        True if movement successful, False otherwise
    """
    try:
        # Calculate distance to move
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # If no movement needed, return early
        if distance < 0.1:
            return True
            
        # Send move command to robot
        command = f"MOVE X{target_x:.2f} Y{target_y:.2f} S{speed}"
        response = send_command(connection, command)
        
        # Check response
        if response == "OK" or response.startswith("MOVED"):
            return True
        else:
            print(f"Error moving to position: {response}")
            return False
            
    except Exception as e:
        print(f"Error in move_to_position: {e}")
        return False

def set_spray_state(connection, spray_active):
    """
    Turn spray on or off.
    
    Args:
        connection: Serial connection object
        spray_active: True to activate spray, False to deactivate
        
    Returns:
        True if state change successful, False otherwise
    """
    try:
        # Send spray command
        command = f"SPRAY {'ON' if spray_active else 'OFF'}"
        response = send_command(connection, command)
        
        # Check response
        if response == "OK" or response.startswith("SPRAY"):
            return True
        else:
            print(f"Error setting spray state: {response}")
            return False
            
    except Exception as e:
        print(f"Error in set_spray_state: {e}")
        return False

def calculate_path_kinematics(path_points, max_speed, acceleration):
    """
    Calculate velocity and acceleration profiles for a path.
    
    Args:
        path_points: List of (x, y) points defining the path
        max_speed: Maximum movement speed
        acceleration: Acceleration/deceleration rate
        
    Returns:
        List of (point, speed) tuples for path execution
    """
    if len(path_points) < 2:
        return []
        
    # Calculate path segments and distances
    segments = []
    total_distance = 0
    
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]
        
        # Calculate segment length
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        segments.append({
            'start': p1,
            'end': p2,
            'distance': distance,
            'direction': (dx/distance if distance > 0 else 0, dy/distance if distance > 0 else 0)
        })
        
        total_distance += distance
    
    # Calculate velocity profile
    kinematics = []
    current_distance = 0
    
    for segment in segments:
        # Calculate appropriate speed for this segment
        # Trapezoidal speed profile - accelerate, maintain, decelerate
        distance_from_start = current_distance
        distance_from_end = total_distance - current_distance - segment['distance']
        
        # Calculate maximum allowed speed for this segment
        segment_max_speed = min(
            max_speed,
            math.sqrt(2 * acceleration * min(distance_from_start, distance_from_end))
        )
        
        # Add start and end points with appropriate speeds
        kinematics.append((segment['start'], segment_max_speed))
        kinematics.append((segment['end'], segment_max_speed))
        
        # Update current distance
        current_distance += segment['distance']
    
    return kinematics

def execute_path_with_kinematics(connection, kinematic_points):
    """
    Execute a path with speed control based on kinematic calculations.
    
    Args:
        connection: Serial connection object
        kinematic_points: List of (point, speed) tuples
        
    Returns:
        True if execution successful, False otherwise
    """
    # Execute each point in the path
    for point, speed in kinematic_points:
        x, y = point
        
        # Send move command with calculated speed
        result = move_to_position(connection, x, y, None, None, int(speed))
        
        # If any movement fails, return failure
        if not result:
            return False
    
    return True
