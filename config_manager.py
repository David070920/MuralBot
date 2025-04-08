#!/usr/bin/env python
"""
MuralBot - Configuration Manager

This module handles loading and managing configuration for the MuralBot system.
"""

import os
import json
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path='config.json'):
        """Initialize the configuration manager."""
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
            return self.get_default_config()
    
    def get_config(self):
        """Get the current configuration."""
        return self.config
    
    def get_image_processing_settings(self):
        """Get image processing settings from configuration."""
        return self.config.get('image_processing', {})
    
    def get_visualization_settings(self):
        """Get visualization settings from configuration."""
        return self.config.get('visualization', {})
        
    def get_hardware_settings(self):
        """Get hardware settings from configuration."""
        return self.config.get('hardware', {})
    
    def save_config(self, config=None):
        """Save configuration to JSON file."""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def update_image_processing_settings(self, settings):
        """Update image processing settings in the configuration."""
        self.config['image_processing'].update(settings)
        return self.save_config()
    
    def update_visualization_settings(self, settings):
        """Update visualization settings in the configuration."""
        self.config['visualization'].update(settings)
        return self.save_config()
        
    def update_hardware_settings(self, settings):
        """Update hardware settings in the configuration."""
        self.config['hardware'].update(settings)
        return self.save_config()
    
    def get_default_config(self):
        """Return default configuration."""
        return {
            "image_processing": {
                "input_image": "mural.jpg",
                "wall_width": 2000,
                "wall_height": 1500,
                "resolution_mm": 5.0,
                "output_instructions": "painting_instructions.json",
                "quantization_method": "kmeans",
                "dithering": "none",
                "dithering_strength": 1.0,
                "fill_pattern": "zigzag",
                "fill_angle": 45,
                "max_colors": 12,
                "robot_capacity": 6,
                "color_selection": "auto"
            },
            "visualization": {
                "instructions_file": "painting_instructions.json",
                "output_video": "painting_simulation.mp4",
                "fps": 30,
                "video_duration": 60,
                "resolution_scale": 0.5,
                "video_quality": 80
            },
            "hardware": {
                "left_stepper_port": "COM3",
                "right_stepper_port": "COM4",
                "spray_servo_port": "COM5",
                "wall_mount_height": 2000,
                "wall_mount_width": 3000,
                "steps_per_mm": 10,
                "max_speed": 1000,
                "home_position": [0, 0],
                "color_change_position": [0, 2000]
            }
        }