"""
Paint usage calculation for MuralBot.

This module contains functions for estimating paint usage for different colors.
"""

import numpy as np

def calculate_paint_usage(color_indices, available_colors, wall_width, wall_height):
    """
    Calculate estimated paint usage for each color.
    
    Args:
        color_indices: 2D array of color indices
        available_colors: List of available RGB colors as tuples
        wall_width: Width of the wall in mm
        wall_height: Height of the wall in mm
        
    Returns:
        Dictionary mapping color indices to usage metrics
    """
    # Count pixels for each color
    total_pixels = color_indices.size
    color_counts = {}
    
    for color_idx in range(len(available_colors)):
        pixel_count = np.sum(color_indices == color_idx)
        percentage = (pixel_count / total_pixels) * 100
        
        # Calculate estimated area in square mm
        pixel_area_mm2 = (wall_width / color_indices.shape[1]) * (wall_height / color_indices.shape[0])
        area_mm2 = pixel_count * pixel_area_mm2
        
        # Estimate paint volume (ml) - assuming 1ml covers approximately 2500mm²
        # This is a rough estimate and should be calibrated based on actual spray can coverage
        paint_ml = area_mm2 / 2500
        
        color_counts[color_idx] = {
            'pixels': int(pixel_count),
            'percentage': percentage,
            'area_mm2': area_mm2,
            'paint_ml': paint_ml
        }
    
    return color_counts
