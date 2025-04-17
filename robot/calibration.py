"""
Robot calibration module for MuralBot.

This module provides functions for calibrating the robot positioning system.
"""

import time
from .hardware import send_command

def calibrate_robot(connection):
    """
    Perform robot calibration routine.
    
    Args:
        connection: Serial connection object
        
    Returns:
        True if calibration successful, False otherwise
    """
    try:
        print("Beginning robot calibration sequence...")
        
        # Step 1: Home the axes to find their limits
        print("Homing axes...")
        response = send_command(connection, "HOME")
        if not (response == "OK" or response.startswith("HOMED")):
            print(f"Error during homing: {response}")
            return False
        
        # Step 2: Calibrate spray subsystem
        print("Calibrating spray subsystem...")
        response = send_command(connection, "CALIBRATE SPRAY")
        if not (response == "OK" or response.startswith("CALIBRATED")):
            print(f"Error during spray calibration: {response}")
            return False
        
        # Step 3: Calibrate color selection system
        print("Calibrating color selection system...")
        response = send_command(connection, "CALIBRATE COLOR")
        if not (response == "OK" or response.startswith("CALIBRATED")):
            print(f"Error during color calibration: {response}")
            return False
        
        # Step 4: Grid movement test
        print("Performing grid movement test...")
        
        # Define a small grid of test points
        test_points = [
            (0, 0),
            (100, 0),
            (100, 100),
            (0, 100),
            (0, 0),
            (50, 50),
            (0, 0)
        ]
        
        # Move to each test point
        for x, y in test_points:
            response = send_command(connection, f"MOVE X{x:.2f} Y{y:.2f} S50")
            if not (response == "OK" or response.startswith("MOVED")):
                print(f"Error during test movement: {response}")
                return False
                
            # Short delay between movements
            time.sleep(0.5)
        
        # Step 5: Test spray system
        print("Testing spray system...")
        
        # Turn spray on
        response = send_command(connection, "SPRAY ON")
        if not (response == "OK" or response.startswith("SPRAY")):
            print(f"Error turning spray on: {response}")
            return False
            
        # Short delay
        time.sleep(1.0)
        
        # Turn spray off
        response = send_command(connection, "SPRAY OFF")
        if not (response == "OK" or response.startswith("SPRAY")):
            print(f"Error turning spray off: {response}")
            return False
        
        print("Calibration completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error during calibration: {e}")
        return False

def measure_workspace_boundaries(connection):
    """
    Measure and detect the physical boundaries of the robot workspace.
    
    Args:
        connection: Serial connection object
        
    Returns:
        Dictionary with workspace dimensions if successful, None otherwise
    """
    try:
        print("Measuring workspace boundaries...")
        
        # Request boundary measurement
        response = send_command(connection, "MEASURE WORKSPACE")
        
        # Parse response
        if response.startswith("WORKSPACE"):
            parts = response.split()
            width = float(parts[1].split('=')[1])
            height = float(parts[2].split('=')[1])
            
            return {
                "width": width,
                "height": height,
                "origin_x": 0,
                "origin_y": 0
            }
        else:
            print(f"Error measuring workspace: {response}")
            return None
            
    except Exception as e:
        print(f"Error during workspace measurement: {e}")
        return None
