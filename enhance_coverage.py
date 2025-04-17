"""
Script to enhance the MuralBot's paint coverage on the canvas.
This script adds improved painting algorithms to ensure the robot
covers the entire canvas with paint properly.
"""

import cv2
import numpy as np
import os
import sys
import json
import random
from tqdm import tqdm

class EnhancedPaintEffect:
    def __init__(self, resolution_scale=0.5):
        self.resolution_scale = resolution_scale
        # Configure for better coverage
        self.spray_coverage = 2.0   # Increase coverage significantly
        self.spray_overlap = 0.75   # Higher overlap between strokes
        
    def apply_paint(self, canvas, position, color, intensity=1.0, spray_size=None):
        """
        Apply a realistic paint spray effect with enhanced coverage.
        
        Args:
            canvas: The canvas to paint on
            position: (x,y) position of the spray
            color: RGB color tuple of the paint
            intensity: Strength of the spray effect (0.0 to 1.0)
            spray_size: Optional size override for the spray
        """
        x, y = position
        
        # Much larger default spray size for better coverage
        if spray_size is None:
            spray_size = max(15, int(40 * self.resolution_scale * self.spray_coverage))
        
        # Create a mask for the spray pattern
        mask = np.zeros((spray_size*2+1, spray_size*2+1), dtype=np.float32)
        
        # Create a circular gradient with improved coverage
        center = (spray_size, spray_size)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # Calculate distance from center
                distance = np.sqrt((i-center[0])**2 + (j-center[1])**2)
                # Apply improved radial falloff with better coverage at edges
                if distance < spray_size:
                    # Use a gentler falloff curve for better edge coverage (1.5 power instead of linear)
                    value = max(0, 1.0 - (distance/spray_size)**1.5) * intensity
                    # Less randomness for more consistent coverage
                    value *= (0.7 + random.random() * 0.3)
                    mask[i, j] = value
        
        # Apply the spray to the canvas
        # Calculate boundaries
        y_start = max(0, y-spray_size)
        y_end = min(canvas.shape[0], y+spray_size+1)
        x_start = max(0, x-spray_size)
        x_end = min(canvas.shape[1], x+spray_size+1)
        
        # Calculate corresponding mask coordinates
        mask_y_start = max(0, -(y-spray_size))
        mask_y_end = mask.shape[0] - max(0, (y+spray_size+1) - canvas.shape[0])
        mask_x_start = max(0, -(x-spray_size))
        mask_x_end = mask.shape[1] - max(0, (x+spray_size+1) - canvas.shape[1])
        
        # Check for valid regions
        if (y_end <= y_start) or (x_end <= x_start) or (mask_y_end <= mask_y_start) or (mask_x_end <= mask_x_start):
            return  # Skip if invalid region
            
        # Extract the region to modify
        roi = canvas[y_start:y_end, x_start:x_end].copy()
        
        # Apply paint to each channel with alpha blending based on mask
        application_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        # Reshape for broadcasting
        application_mask = application_mask.reshape(application_mask.shape[0], application_mask.shape[1], 1)
        
        # Blend paint color with existing canvas
        try:
            color_array = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
            roi = roi * (1 - application_mask) + color_array * application_mask
            # Update the canvas region
            canvas[y_start:y_end, x_start:x_end] = roi.astype(np.uint8)
        except Exception as e:
            print(f"Error in paint application: {e}")
    
    def apply_paint_path(self, canvas, start_pos, end_pos, color, intensity=1.0):
        """
        Apply paint along a path between two points with improved coverage.
        
        Args:
            canvas: The canvas to paint on
            start_pos: Starting (x,y) position
            end_pos: Ending (x,y) position
            color: RGB color tuple
            intensity: Base intensity to apply
        """
        # Extract positions
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Calculate distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Use more interpolation points for better coverage
        # Lower density_factor means more points
        density_factor = 0.5  # More dense = better coverage
        steps = max(5, int(distance * density_factor))
        
        # Apply paint at points along the path
        for i in range(steps + 1):
            # Calculate position along path
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            # Vary intensity slightly for natural look
            point_intensity = intensity * (0.8 + random.random() * 0.4)
            
            # Apply with normal spray size
            self.apply_paint(canvas, (x, y), color, point_intensity)
            
        # Add extra paint at the endpoint for better finish
        self.apply_paint(canvas, end_pos, color, intensity * 1.2)

def process_instructions(instructions_file, output_file="enhanced_mural.jpg", show_preview=False):
    """
    Process painting instructions with enhanced coverage.
    
    Args:
        instructions_file: Path to the instructions JSON file
        output_file: Path to save the output image
        show_preview: Whether to display a preview window
    """
    print(f"Processing instructions from {instructions_file} with enhanced coverage...")
    
    # Load instructions
    with open(instructions_file, 'r') as f:
        data = json.load(f)
    
    # Get canvas dimensions
    wall_width = data.get('wall_width', 2000)
    wall_height = data.get('wall_height', 1500)
    instructions = data.get('instructions', [])
    
    # Create blank canvas (white background)
    canvas = np.ones((wall_height, wall_width, 3), dtype=np.uint8) * 255
    
    # Variables to track painting state
    spray_active = False
    current_color = (0, 0, 0)  # Default to black
    current_pos = (0, 0)
    
    # Create enhanced paint effect handler
    painter = EnhancedPaintEffect(resolution_scale=0.5)
    
    # Process each instruction
    print("Applying enhanced paint coverage...")
    for i, instruction in enumerate(tqdm(instructions)):
        instruction_type = instruction.get('type')
        
        if instruction_type == 'color':
            rgb = instruction.get('rgb', [0, 0, 0])
            current_color = tuple(int(c) for c in rgb)
            
        elif instruction_type == 'move':
            x = int(instruction.get('x', 0))
            y = int(instruction.get('y', 0))
            spray = instruction.get('spray', False)
            
            if spray and spray_active:
                # Use enhanced paint effect between points
                painter.apply_paint_path(
                    canvas, 
                    current_pos, 
                    (x, y), 
                    current_color
                )
                
            # Update position
            current_pos = (x, y)
            
        elif instruction_type == 'spray':
            spray_active = instruction.get('state', False)
            
        # Show progress preview occasionally
        if show_preview and i % 100 == 0:
            # Resize for display
            preview = cv2.resize(canvas, (800, 600))
            cv2.imshow('Enhanced Paint Coverage', preview)
            cv2.waitKey(1)
    
    # Save the enhanced coverage result
    print(f"Saving enhanced mural to {output_file}...")
    cv2.imwrite(output_file, canvas)
    
    # Show final result
    if show_preview:
        preview = cv2.resize(canvas, (800, 600))
        cv2.imshow('Final Enhanced Paint Coverage', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("Enhanced coverage processing complete!")
    return canvas

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance paint coverage for MuralBot")
    parser.add_argument('--instructions', '-i', required=True, help='Path to instructions JSON file')
    parser.add_argument('--output', '-o', default="enhanced_mural.jpg", help='Output image path')
    parser.add_argument('--preview', '-p', action='store_true', help='Show preview window')
    
    args = parser.parse_args()
    
    process_instructions(args.instructions, args.output, args.preview)
