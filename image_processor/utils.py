"""
Utility functions for image processing in MuralBot.

This module contains common utility functions used across the image processing pipeline.
"""

import cv2
import numpy as np
import os
import json

def load_mtn94_colors():
    """Load colors from the MTN94 color database."""
    color_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mtn94_colors.json")
    
    try:
        with open(color_db_path, 'r') as f:
            color_db = json.load(f)
            color_data = color_db["colors"]
            print(f"Loaded {len(color_data)} colors from MTN94 database")
            
            # Extract RGB values from the database
            available_colors_rgb = []
            for color in color_data:
                # Ensure each RGB value is properly converted to a tuple of integers
                rgb = tuple(int(c) for c in color["rgb"])
                available_colors_rgb.append(rgb)
                
            return available_colors_rgb
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading color database: {e}")
        print("Using default colors instead")
        # Return basic default colors as fallback
        return [
            (0, 0, 0),       # Black
            (255, 255, 255),  # White
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255)    # Cyan
        ]

def load_image(image_path, wall_width, wall_height):
    """Load and resize image to fit the wall dimensions."""
    import os
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
        
    # Resize to fit wall dimensions while maintaining aspect ratio
    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height
    
    if wall_width / wall_height > aspect_ratio:
        # Wall is wider relative to height
        new_width = int(wall_height * aspect_ratio)
        new_height = wall_height
    else:
        # Wall is taller relative to width
        new_width = wall_width
        new_height = int(wall_width / aspect_ratio)
        
    img = cv2.resize(img, (new_width, new_height))
    
    # Center the image on the wall
    x_offset = (wall_width - new_width) // 2
    y_offset = (wall_height - new_height) // 2
    
    # Create a canvas of wall size with background color (white)
    canvas = np.ones((wall_height, wall_width, 3), dtype=np.uint8) * 255
    
    # Place the image on the canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img
    
    return canvas
