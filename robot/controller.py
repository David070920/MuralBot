"""
Main robot controller module for MuralBot.

This module contains the core MuralRobotController class which manages
the robot hardware and movement execution.
"""

import json
import time
import os
from tqdm import tqdm

from .hardware import connect_hardware, disconnect_hardware
from .motion import move_to_position, set_spray_state
from .calibration import calibrate_robot

class MuralRobotController:
    """
    Main controller class for the mural painting robot.
    
    This class handles communication with the robot hardware, 
    movement planning, and execution of painting instructions.
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize the MuralRobotController.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config_path = config_path
        self.config = self.load_config(config_path)
        
        # Initialize state
        self.connected = False
        self.calibrated = False
        self.current_x = 0
        self.current_y = 0
        self.spray_active = False
        self.current_color_index = 0
        self.loaded_colors = []
        self.home_position = (0, 0)
        self.color_change_position = self.config.get('hardware', {}).get('color_change_position', [0, 0])
        
        # Hardware connection parameters
        self.port = None
        self.baud_rate = None
        self.connection = None
        
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config file: {e}")
            return {"hardware": {}}

    def connect(self, port=None, baud_rate=None):
        """
        Connect to the robot hardware.
        
        Args:
            port: Serial port to connect to
            baud_rate: Baud rate for serial connection
            
        Returns:
            True if connection successful, False otherwise
        """
        # Use specified parameters or fall back to config
        self.port = port or self.config.get('hardware', {}).get('port')
        self.baud_rate = baud_rate or self.config.get('hardware', {}).get('baud_rate', 115200)
        
        # Validate parameters
        if not self.port:
            print("Error: No serial port specified. Please specify port.")
            return False
            
        # Connect to hardware
        self.connection = connect_hardware(self.port, self.baud_rate)
        
        if self.connection:
            self.connected = True
            print(f"Connected to robot on port {self.port}")
            return True
        else:
            print(f"Failed to connect to robot on port {self.port}")
            return False
            
    def disconnect(self):
        """Disconnect from the robot hardware."""
        if self.connected:
            disconnect_hardware(self.connection)
            self.connected = False
            print("Disconnected from robot")
            return True
        
        print("Not connected to robot")
        return False
        
    def home(self):
        """Return robot to home position."""
        if not self.connected:
            print("Robot not connected. Please connect first.")
            return False
            
        print("Moving to home position...")
        move_to_position(self.connection, self.home_position[0], self.home_position[1], self.current_x, self.current_y)
        
        self.current_x = self.home_position[0]
        self.current_y = self.home_position[1]
        
        return True
        
    def move_to(self, x, y, spray=False):
        """
        Move the robot to the specified coordinates.
        
        Args:
            x: X-coordinate to move to
            y: Y-coordinate to move to
            spray: Whether to spray while moving
        """
        if not self.connected:
            print("Robot not connected. Please connect first.")
            return False
            
        # If spray state needs to change
        if spray != self.spray_active:
            set_spray_state(self.connection, spray)
            self.spray_active = spray
        
        # Move to position
        move_to_position(self.connection, x, y, self.current_x, self.current_y)
        
        # Update position
        self.current_x = x
        self.current_y = y
        
        return True
        
    def set_spray(self, state):
        """
        Set spray on or off.
        
        Args:
            state: True to turn spray on, False to turn it off
        """
        if not self.connected:
            print("Robot not connected. Please connect first.")
            return False
            
        set_spray_state(self.connection, state)
        self.spray_active = state
        
        return True
        
    def set_color(self, color_idx):
        """
        Change to the specified color.
        
        Args:
            color_idx: Index of the color to change to
        """
        if not self.connected:
            print("Robot not connected. Please connect first.")
            return False
            
        if color_idx not in self.loaded_colors:
            print(f"Color {color_idx} not loaded. Please load colors first.")
            return False
            
        print(f"Changing to color {color_idx}")
        # TODO: Implement hardware control to change color
        
        self.current_color_index = color_idx
        return True
        
    def load_colors(self, colors):
        """
        Load colors onto the robot.
        
        Args:
            colors: List of color dictionaries with 'index' and 'rgb' keys
        """
        if not self.connected:
            print("Robot not connected. Please connect first.")
            return False
            
        print(f"Loading {len(colors)} colors onto robot")
        
        # Extract color indices
        self.loaded_colors = [color['index'] for color in colors]
        
        # Set the first color as current
        if self.loaded_colors:
            self.current_color_index = self.loaded_colors[0]
            
        return True
        
    def calibrate(self):
        """Calibrate the robot positioning system."""
        if not self.connected:
            print("Robot not connected. Please connect first.")
            return False
            
        print("Starting robot calibration...")
        result = calibrate_robot(self.connection)
        
        if result:
            self.calibrated = True
            print("Calibration completed successfully")
            return True
        else:
            print("Calibration failed")
            return False

    def execute_instructions(self, instructions_file):
        """
        Execute a set of painting instructions from a JSON file.
        
        Args:
            instructions_file: Path to the JSON file with painting instructions
        """
        if not self.connected:
            print("Robot not connected. Please connect first.")
            return False
            
        try:
            # Load instructions
            with open(instructions_file, 'r') as f:
                data = json.load(f)
                
            instructions = data.get('instructions', [])
            colors = data.get('colors', [])
            
            if not instructions:
                print("No instructions found in file.")
                return False
                
            print(f"Loaded {len(instructions)} instructions.")
            
            # Confirm start
            input("\nPress Enter to start painting...")
            
            # Execute each instruction
            for i, instruction in enumerate(tqdm(instructions, desc="Executing instructions")):
                instruction_type = instruction.get('type')
                message = instruction.get('message', '')
                
                # Log what we're doing
                print(f"\nInstruction {i+1}/{len(instructions)}: {message}")
                
                # Execute based on instruction type
                if instruction_type == 'home':
                    self.home()
                    
                elif instruction_type == 'move':
                    x = instruction.get('x', 0)
                    y = instruction.get('y', 0)
                    spray = instruction.get('spray', False)
                    self.move_to(x, y, spray)
                    
                elif instruction_type == 'spray':
                    state = instruction.get('state', False)
                    self.set_spray(state)
                    
                elif instruction_type == 'color':
                    color_idx = instruction.get('index')
                    self.set_color(color_idx)
                    
                elif instruction_type == 'load_colors':
                    colors_to_load = instruction.get('colors', [])
                    self.load_colors(colors_to_load)
                    
                else:
                    print(f"Unknown instruction type: {instruction_type}")
                
                # Small pause between instructions for stability
                time.sleep(0.1)
            
            print("\nPainting complete!")
            return True
            
        except Exception as e:
            print(f"Error executing instructions: {e}")
            return False
    
    def simulate_execution(self, instructions_file, output_file=None):
        """
        Simulate execution of painting instructions without physical robot.
        
        Args:
            instructions_file: Path to the JSON file with painting instructions
            output_file: Optional path to save the simulation log
            
        Returns:
            True if simulation completed successfully, False otherwise
        """
        try:
            # Load instructions
            with open(instructions_file, 'r') as f:
                data = json.load(f)
                
            instructions = data.get('instructions', [])
            colors = data.get('colors', [])
            
            if not instructions:
                print("No instructions found in file.")
                return False
                
            print(f"Loaded {len(instructions)} instructions for simulation.")
            
            # Initialize simulation state
            self.current_x = 0
            self.current_y = 0
            self.spray_active = False
            self.current_color_index = 0
            self.loaded_colors = []
            self.home_position = (0, 0)
            
            # Execution log
            log = []
            
            # Simulate each instruction
            print("Simulating execution...")
            for i, instruction in enumerate(tqdm(instructions, desc="Simulating")):
                instruction_type = instruction.get('type')
                message = instruction.get('message', '')
                
                # Add to log
                entry = {
                    "instruction_index": i,
                    "type": instruction_type,
                    "details": instruction,
                    "position": {"x": self.current_x, "y": self.current_y},
                    "spray_active": self.spray_active,
                    "color_index": self.current_color_index
                }
                
                # Simulate instruction
                if instruction_type == 'home':
                    self.current_x = self.home_position[0]
                    self.current_y = self.home_position[1]
                    
                elif instruction_type == 'move':
                    self.current_x = instruction.get('x', 0)
                    self.current_y = instruction.get('y', 0)
                    self.spray_active = instruction.get('spray', False)
                    
                elif instruction_type == 'spray':
                    self.spray_active = instruction.get('state', False)
                    
                elif instruction_type == 'color':
                    color_idx = instruction.get('index')
                    if color_idx in self.loaded_colors:
                        self.current_color_index = color_idx
                    else:
                        print(f"Warning: Color {color_idx} not loaded in simulation")
                    
                elif instruction_type == 'load_colors':
                    colors_to_load = instruction.get('colors', [])
                    self.loaded_colors = [c['index'] for c in colors_to_load]
                
                # Update position in log entry
                entry["end_position"] = {"x": self.current_x, "y": self.current_y}
                entry["end_spray_active"] = self.spray_active
                entry["end_color_index"] = self.current_color_index
                
                log.append(entry)
                
                # Simulate time passing (for illustration)
                time.sleep(0.01)
            
            # Save simulation log if output file is specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump({"simulation_log": log}, f, indent=2)
                print(f"Simulation log saved to {output_file}")
            
            print("Simulation complete.")
            return log
            
        except Exception as e:
            print(f"Error simulating instructions: {e}")
            return False
