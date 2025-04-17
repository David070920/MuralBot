"""
Main visualization module for MuralBot.

This module contains the core visualization class for rendering and animating
mural painting instructions.
"""

import cv2
import numpy as np
import json
import argparse
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from .animation import save_animation_as_video, create_frames
from .path_display import visualize_robot_paths
from .interactive import create_interactive_visualization

class MuralVisualizer:
    """
    Visualize the mural painting process based on the generated instructions.
    
    This class provides:
    1. Static preview of the quantized image
    2. Animation of the painting process
    3. Path visualization for robot movement
    4. Video export of the painting simulation
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize the MuralVisualizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
        
        # Get wall dimensions
        self.wall_width = self.config.get("image_processing", {}).get("wall_width", 2000)
        self.wall_height = self.config.get("image_processing", {}).get("wall_height", 1500)
        
        # Set resolution scale for visualization (pixels per mm)
        self.resolution_scale = self.config.get("visualization", {}).get("resolution_scale", 0.5)
        
        # Animation settings
        self.output_video = self.config.get("visualization", {}).get("output_video", "mural_animation.mp4")
        self.save_animation = self.config.get("visualization", {}).get("save_animation", True)
        
        # For interactive visualization
        self.frame_buffer = []
        self.canvas = None
        self.prev_pos = None
    
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config file: {e}")
            return {
                "image_processing": {
                    "wall_width": 2000,
                    "wall_height": 1500
                },
                "visualization": {
                    "resolution_scale": 0.5,
                    "save_animation": True,
                    "output_video": "mural_animation.mp4"
                }
            }

    def create_preview_image(self, instructions_file):
        """
        Visualize painting instructions by drawing paths on a canvas.
        
        Args:
            instructions_file: Path to the JSON file containing painting instructions
            
        Returns:
            Canvas with painted paths
        """
        try:
            # Ensure instructions file path exists
            instructions_file = os.path.normpath(instructions_file)
            if not os.path.exists(instructions_file):
                print(f"Error: Instructions file '{instructions_file}' does not exist.")
                return None
                
            # Load instructions
            with open(instructions_file, 'r') as f:
                data = json.load(f)
            
            wall_width = data.get('wall_width', self.wall_width)
            wall_height = data.get('wall_height', self.wall_height)
            instructions = data.get('instructions', [])
            
            # Create a blank canvas with white background
            canvas_width = int(wall_width * self.resolution_scale)
            canvas_height = int(wall_height * self.resolution_scale)
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Variables to track painting state
            spray_active = False
            current_color = (0, 0, 0)  # Default to black
            current_pos = (0, 0)
            
            # Process each instruction
            print("Generating preview image...")
            for i, instruction in enumerate(tqdm(instructions)):
                instruction_type = instruction.get('type')
                
                if instruction_type == 'color':
                    rgb = instruction.get('rgb', [0, 0, 0])
                    current_color = tuple(rgb)
                    
                elif instruction_type == 'move':
                    x = int(instruction.get('x', 0) * self.resolution_scale)
                    y = int(instruction.get('y', 0) * self.resolution_scale)
                    spray = instruction.get('spray', False)
                    
                    if spray and spray_active:
                        # Draw a line of the current color
                        cv2.line(canvas, 
                                 (int(current_pos[0]), int(current_pos[1])),
                                 (x, y),
                                 current_color,
                                 thickness=max(1, int(3 * self.resolution_scale)))
                        
                    # Update position
                    current_pos = (x, y)
                    
                elif instruction_type == 'spray':
                    spray_active = instruction.get('state', False)
            
            # Save the preview image
            cv2.imwrite("mural_preview.jpg", canvas)
            print("Preview image saved as 'mural_preview.jpg'")
            
            # Keep canvas for additional visualizations
            self.canvas = canvas
            
            return canvas
            
        except Exception as e:
            print(f"Error creating preview image: {e}")
            return None

    def animate_painting_process(self, instructions_file, output_file=None):
        """
        Create an animation of the painting process.
        
        Args:
            instructions_file: Path to instructions JSON file
            output_file: Path to save animation video (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load and validate instructions file
            instructions_file = os.path.normpath(instructions_file)
            if not os.path.exists(instructions_file):
                print(f"Error: Instructions file '{instructions_file}' does not exist.")
                return False
                
            # Load instructions
            with open(instructions_file, 'r') as f:
                data = json.load(f)
                
            wall_width = data.get('wall_width', self.wall_width)
            wall_height = data.get('wall_height', self.wall_height)
            instructions = data.get('instructions', [])
            
            # Create blank canvas with white background
            canvas_width = int(wall_width * self.resolution_scale)
            canvas_height = int(wall_height * self.resolution_scale)
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Create frame buffer
            total_frames = min(500, len(instructions))  # Limit max frames for efficiency
            
            # Generate animation frames
            self.frame_buffer = create_frames(
                instructions, 
                canvas, 
                total_frames, 
                self.resolution_scale
            )

            print(f"\nGenerated {len(self.frame_buffer)} animation frames")
            
            # Save as video if requested
            if output_file or self.output_video:
                out_path = output_file if output_file else self.output_video
                save_animation_as_video(self.frame_buffer, out_path)
            
            return True
            
        except Exception as e:
            print(f"Error animating painting process: {e}")
            return False

    def visualize_instructions(self, instructions_file, output_file=None, show_preview=True, show_progress=True):
        """
        Visualize painting instructions by drawing paths on a canvas.
        
        Args:
            instructions_file: Path to the JSON file containing painting instructions
            output_file: Path to save the visualization image (if None, will save to painting folder)
            show_preview: Whether to display the preview window
            show_progress: Whether to show progress during visualization
        """
        # Create the path display visualization
        return visualize_robot_paths(
            instructions_file, 
            self.resolution_scale,
            output_file,
            show_preview,
            show_progress
        )
    
    def create_interactive_visualization(self, instructions_file):
        """
        Create an interactive visualization of the painting process.
        Uses matplotlib for scrubbing through the painting process.
        
        Args:
            instructions_file: Path to the instructions JSON file
        """
        if not self.frame_buffer:
            print("No animation frames available. Run animate_painting_process first.")
            return False
            
        return create_interactive_visualization(self.frame_buffer)
