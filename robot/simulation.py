"""
Robot simulation module for MuralBot.

This module provides functions for simulating robot behavior without physical hardware.
"""

import time
import numpy as np
import cv2
import math
import json
import os
from tqdm import tqdm

class RobotSimulator:
    """
    Simulates the behavior of the mural painting robot.
    
    This class is used to test painting instructions without requiring
    physical hardware, and can generate visualization of the robot's movement.
    """
    
    def __init__(self, wall_width=2000, wall_height=1500, resolution=1):
        """
        Initialize the robot simulator.
        
        Args:
            wall_width: Width of the wall/canvas in mm
            wall_height: Height of the wall/canvas in mm
            resolution: Resolution for simulation in pixels per mm
        """
        self.wall_width = wall_width
        self.wall_height = wall_height
        self.resolution = resolution
        
        # Create simulation canvas
        self.canvas_width = int(wall_width * resolution)
        self.canvas_height = int(wall_height * resolution)
        self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        
        # Robot state
        self.current_x = 0
        self.current_y = 0
        self.spray_active = False
        self.current_color = (0, 0, 0)  # Default color: black
        self.current_color_index = 0
        self.loaded_colors = []
        self.home_position = (0, 0)
        
        # Spray parameters
        self.spray_radius = 5  # mm
        self.spray_fade = 2    # mm
        
        # Movement history for path tracking
        self.movement_history = []
        self.frame_buffer = []
        
    def reset(self):
        """Reset the simulator to initial state."""
        self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        self.current_x = 0
        self.current_y = 0
        self.spray_active = False
        self.current_color = (0, 0, 0)
        self.current_color_index = 0
        self.loaded_colors = []
        self.movement_history = []
        self.frame_buffer = []
        
    def move_to(self, x, y, spray=False):
        """
        Move the simulated robot to the specified coordinates.
        
        Args:
            x: X-coordinate to move to (mm)
            y: Y-coordinate to move to (mm)
            spray: Whether to spray while moving
            
        Returns:
            True if movement successful
        """
        # Convert to canvas coordinates
        x_pixel = int(x * self.resolution)
        y_pixel = int(y * self.resolution)
        
        # Current position in pixel coordinates
        curr_x_pixel = int(self.current_x * self.resolution)
        curr_y_pixel = int(self.current_y * self.resolution)
        
        # Draw path
        if spray and self.spray_active:
            # Draw line with current color
            color_bgr = (self.current_color[2], self.current_color[1], self.current_color[0])
            cv2.line(self.canvas, 
                     (curr_x_pixel, curr_y_pixel),
                     (x_pixel, y_pixel),
                     color_bgr,
                     thickness=int(self.spray_radius * self.resolution))
        
        # Update position
        self.current_x = x
        self.current_y = y
        
        # Add to movement history
        self.movement_history.append({
            'x': x,
            'y': y,
            'spray': spray and self.spray_active,
            'color': self.current_color,
            'color_index': self.current_color_index
        })
        
        return True
    
    def set_spray(self, state):
        """
        Set spray state in the simulation.
        
        Args:
            state: True to turn spray on, False to turn it off
            
        Returns:
            True
        """
        self.spray_active = state
        return True
    
    def set_color(self, color_idx, rgb=None):
        """
        Set the current color in the simulation.
        
        Args:
            color_idx: Index of the color
            rgb: Optional RGB tuple for the color
            
        Returns:
            True if color was set
        """
        if rgb is not None:
            self.current_color = rgb
            self.current_color_index = color_idx
            return True
            
        # Check if color is loaded
        if color_idx not in self.loaded_colors:
            print(f"Warning: Color {color_idx} not loaded in simulation")
            return False
        
        # In a real system we'd look up the RGB values from the loaded colors
        # Here we just use the index to generate a color for visualization
        self.current_color_index = color_idx
        return True
    
    def load_colors(self, colors):
        """
        Load colors in the simulation.
        
        Args:
            colors: List of color dictionaries with 'index' and 'rgb' keys
            
        Returns:
            True
        """
        self.loaded_colors = []
        
        for color in colors:
            idx = color.get('index')
            rgb = color.get('rgb')
            
            if idx is not None:
                self.loaded_colors.append(idx)
                
                # If this is the first color, set it as current
                if not self.loaded_colors:
                    self.current_color_index = idx
                    if rgb:
                        self.current_color = tuple(rgb)
        
        return True
    
    def home(self):
        """
        Return to home position in the simulation.
        
        Returns:
            True
        """
        self.move_to(self.home_position[0], self.home_position[1], False)
        return True
    
    def simulate_instructions(self, instructions, save_frames=True, frame_interval=10):
        """
        Run a simulation of the painting process using the provided instructions.
        
        Args:
            instructions: List of painting instructions
            save_frames: Whether to save frames for animation
            frame_interval: How many instructions to process per animation frame
            
        Returns:
            Simulation log
        """
        # Reset simulator
        self.reset()
        
        # Simulation log
        log = []
        
        # Process instructions
        print("Simulating painting process...")
        for i, instruction in enumerate(tqdm(instructions)):
            instruction_type = instruction.get('type')
            
            # Log entry
            entry = {
                'instruction_index': i,
                'type': instruction_type,
                'details': instruction,
                'before': {
                    'position': (self.current_x, self.current_y),
                    'spray_active': self.spray_active,
                    'color_index': self.current_color_index
                }
            }
            
            # Execute instruction
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
                rgb = instruction.get('rgb')
                self.set_color(color_idx, rgb)
                
            elif instruction_type == 'load_colors':
                colors = instruction.get('colors', [])
                self.load_colors(colors)
            
            # Update log entry with after state
            entry['after'] = {
                'position': (self.current_x, self.current_y),
                'spray_active': self.spray_active,
                'color_index': self.current_color_index
            }
            
            log.append(entry)
            
            # Save frame periodically if requested
            if save_frames and i % frame_interval == 0:
                frame = self.canvas.copy()
                
                # Draw robot indicator
                x_pixel = int(self.current_x * self.resolution)
                y_pixel = int(self.current_y * self.resolution)
                robot_radius = max(5, int(10 * self.resolution))
                
                # Robot body
                cv2.circle(frame, (x_pixel, y_pixel), robot_radius, (50, 50, 50), -1)
                cv2.circle(frame, (x_pixel, y_pixel), robot_radius, (0, 0, 0), 1)
                
                # Spray indicator if active
                if self.spray_active:
                    spray_radius = int(robot_radius * 1.5)
                    overlay = frame.copy()
                    color_bgr = (self.current_color[2], self.current_color[1], self.current_color[0])
                    cv2.circle(overlay, (x_pixel, y_pixel), spray_radius, color_bgr, -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
                    # Spray direction indicator
                    indicator_length = robot_radius * 2
                    cv2.line(frame,
                             (x_pixel, y_pixel),
                             (x_pixel, y_pixel + indicator_length),
                             color_bgr,
                             thickness=2)
                
                self.frame_buffer.append(frame)
        
        return log
    
    def save_simulation_video(self, output_path, fps=30):
        """
        Save the simulation as a video file.
        
        Args:
            output_path: Path to save the video
            fps: Frames per second
            
        Returns:
            True if successful, False otherwise
        """
        if not self.frame_buffer:
            print("No frames to save.")
            return False
            
        try:
            # Create video writer
            h, w = self.frame_buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # Write frames
            for frame in self.frame_buffer:
                out.write(frame)
                
            # Release the writer
            out.release()
            print(f"Simulation video saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving simulation video: {e}")
            return False
    
    def save_simulation_image(self, output_path):
        """
        Save the final state of the simulation as an image.
        
        Args:
            output_path: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cv2.imwrite(output_path, self.canvas)
            print(f"Simulation image saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving simulation image: {e}")
            return False
