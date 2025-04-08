#!/usr/bin/env python
"""
MuralBot - Configuration Manager

This module handles loading and managing configuration for the MuralBot system.
"""

import os
import json
from pathlib import Path

class ConfigManager:
    """
    Configuration manager for the MuralBot system.
    Handles loading settings from config files and providing defaults when needed.
    """
    
    def __init__(self, config_path="config.json"):
        """Initialize the configuration manager with the given config file."""
        self.config_path = config_path
        self.config = {}
        self.img_config = {}
        self.vis_config = {}
        self.hardware_config = {}
        self.load_config()
        
    def load_config(self):
        """Load configuration from the JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            # Extract main configuration sections
            self.img_config = self.config.get('image_processing', {})
            self.vis_config = self.config.get('visualization', {})
            self.hardware_config = self.config.get('hardware', {})
                
            print(f"Configuration loaded from {self.config_path}")
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default settings")
            self.config = {
                "image_processing": {},
                "visualization": {},
                "hardware": {}
            }
            return False
    
    def get_image_processing_settings(self):
        """Get image processing settings with defaults."""
        return {
            "input_image": self.img_config.get('input_image', 'mural.jpg'),
            "wall_width": self.img_config.get('wall_width', 2000),
            "wall_height": self.img_config.get('wall_height', 1500),
            "resolution_mm": self.img_config.get('resolution_mm', 5.0),
            "output_instructions": self.img_config.get('output_instructions', 'painting_instructions.json'),
            "quantization_method": self.img_config.get('quantization_method', 'kmeans'),
            "dithering": self.img_config.get('dithering', False),
            "max_colors": self.img_config.get('max_colors', 12),
            "robot_capacity": self.img_config.get('robot_capacity', 6),
            "color_selection": self.img_config.get('color_selection', 'auto'),
            "available_colors": self.img_config.get('available_colors', []),
        }
    
    def get_visualization_settings(self):
        """Get visualization settings with defaults."""
        return {
            "instructions_file": self.vis_config.get('instructions_file', 'painting_instructions.json'),
            "output_video": self.vis_config.get('output_video', 'painting_simulation.mp4'),
            "fps": self.vis_config.get('fps', 30),
            "video_duration": self.vis_config.get('video_duration', 60),
            "resolution_scale": self.vis_config.get('resolution_scale', 0.5),
            "video_quality": self.vis_config.get('video_quality', 80),
        }
    
    def get_hardware_settings(self):
        """Get hardware settings with defaults."""
        settings = self.hardware_config.copy()
        # Set some defaults for critical settings
        if 'color_change_position' not in settings:
            settings['color_change_position'] = [0, 2000]
        return settings
    
    def get_painting_folder(self):
        """Get the painting output folder path, creating it if it doesn't exist."""
        painting_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "painting")
        os.makedirs(painting_folder, exist_ok=True)
        return painting_folder
    
    def save_config(self, new_config=None):
        """Save current configuration back to the config file."""
        if new_config:
            self.config = new_config
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False