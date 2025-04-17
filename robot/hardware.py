"""
Robot hardware interface for MuralBot.

This module provides functions to interact with the robot hardware.
"""

import serial
import time

def connect_hardware(port, baud_rate):
    """
    Connect to the robot hardware over a serial connection.
    
    Args:
        port: Serial port to connect to
        baud_rate: Baud rate for serial connection
        
    Returns:
        Serial connection object if successful, None otherwise
    """
    try:
        # Try to connect to the specified serial port
        connection = serial.Serial(port, baud_rate, timeout=2)
        
        # Brief pause to let the connection stabilize
        time.sleep(2)
        
        # Test the connection with a simple command
        connection.write(b"PING\n")
        response = connection.readline().decode('utf-8').strip()
        
        if response == "OK" or response == "PONG":
            return connection
        else:
            print(f"Connected to port, but got unexpected response: {response}")
            connection.close()
            return None
            
    except Exception as e:
        print(f"Error connecting to hardware: {e}")
        return None

def disconnect_hardware(connection):
    """
    Disconnect from the robot hardware.
    
    Args:
        connection: Serial connection object
        
    Returns:
        True if disconnection successful, False otherwise
    """
    try:
        # Send a disconnect command if the hardware supports it
        connection.write(b"DISCONNECT\n")
        time.sleep(0.5)
        
        # Close the serial connection
        connection.close()
        return True
        
    except Exception as e:
        print(f"Error disconnecting from hardware: {e}")
        return False

def send_command(connection, command):
    """
    Send a command to the robot and wait for confirmation.
    
    Args:
        connection: Serial connection object
        command: Command string to send
        
    Returns:
        Response from the robot
    """
    try:
        # Clear any pending data
        connection.reset_input_buffer()
        
        # Send the command
        connection.write((command + "\n").encode('utf-8'))
        
        # Wait for response
        response = connection.readline().decode('utf-8').strip()
        
        return response
        
    except Exception as e:
        print(f"Error sending command: {e}")
        return None
