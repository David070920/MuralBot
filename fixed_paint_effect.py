"""
This module contains the enhanced paint effect implementation to fix canvas coverage issues.
This provides a more realistic paint coverage to ensure the robot properly paints the entire canvas.
"""

import cv2
import numpy as np
import random

def apply_enhanced_paint_effect(canvas, position, color, intensity=1.0, resolution_scale=0.5, 
                                spray_size=None, coverage_factor=1.5):
    """
    Enhanced version of the paint effect that ensures better coverage of the canvas.
    
    Args:
        canvas: The canvas to paint on
        position: (x,y) position of the spray
        color: RGB color tuple of the paint
        intensity: Strength of the spray effect (0.0 to 1.0)
        resolution_scale: Scale factor for rendering resolution
        spray_size: Optional size override for the spray
        coverage_factor: Factor to increase spray coverage (higher = more coverage)
    """
    x, y = position
    
    # Calculate spray size with increased coverage
    if spray_size is None:
        # Significantly larger default spray size
        spray_size = max(15, int(40 * resolution_scale * coverage_factor))
    
    # Create a mask for the spray pattern
    mask = np.zeros((spray_size*2+1, spray_size*2+1), dtype=np.float32)
    
    # Create a circular gradient with better coverage
    center = (spray_size, spray_size)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Calculate distance from center
            distance = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            # Apply improved radial falloff with better coverage
            if distance < spray_size:
                # Adjust the falloff curve for better coverage at edges
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
    
    # Extract the region to modify
    roi = canvas[y_start:y_end, x_start:x_end].copy()
    
    # Apply paint to each channel with alpha blending based on mask
    for c in range(3):
        # Scale mask for this application
        application_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        # Reshape for broadcasting
        application_mask = application_mask.reshape(application_mask.shape[0], application_mask.shape[1], 1)
        # Blend paint color with existing canvas
        roi = roi * (1 - application_mask) + np.array(color).reshape(1, 1, 3) * application_mask
    
    # Update the canvas region
    canvas[y_start:y_end, x_start:x_end] = roi.astype(np.uint8)

def apply_paint_along_path(canvas, start_pos, end_pos, color, resolution_scale=0.5):
    """
    Apply paint effect along a path with improved density for better coverage.
    
    Args:
        canvas: The canvas to paint on
        start_pos: (x,y) starting position
        end_pos: (x,y) ending position
        color: RGB color tuple of the paint
        resolution_scale: Scale factor for rendering resolution
    """
    x1, y1 = start_pos
    x2, y2 = end_pos
    
    # Calculate distance
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Determine how densely to place paint spots
    # More dense placement for better coverage
    density_factor = 0.7  # Lower = more dense coverage
    steps = max(5, int(distance * density_factor))
    
    # Apply paint at multiple points along the path
    for i in range(steps + 1):
        # Calculate position along the path
        t = i / steps
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        
        # Vary intensity slightly for natural look
        intensity = 0.8 + (random.random() * 0.4)
        
        # Vary coverage for more realistic spray pattern
        coverage = 1.3 + (random.random() * 0.4)
        
        # Apply paint at this point with enhanced coverage
        apply_enhanced_paint_effect(
            canvas, 
            (x, y), 
            color, 
            intensity=intensity,
            resolution_scale=resolution_scale,
            coverage_factor=coverage
        )
        
    # Add an extra coat at the endpoints for better coverage
    apply_enhanced_paint_effect(
        canvas, 
        end_pos, 
        color, 
        intensity=1.2,
        resolution_scale=resolution_scale,
        coverage_factor=1.8
    )
