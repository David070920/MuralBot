"""
Path planning algorithms for MuralBot.

This module contains functions for generating and optimizing painting paths.
"""

import cv2
import numpy as np
from tqdm import tqdm
import math
from algorithms.fill_patterns import generate_fill_pattern
from algorithms.path_optimization import optimize_path_sequence

def generate_paths(color_regions, resolution_mm, fill_pattern, fill_angle):
    """
    Generate painting paths for each color region.
    
    Args:
        color_regions: Dictionary mapping color indices to regions
        resolution_mm: Resolution in mm for path planning
        fill_pattern: Pattern type for filling regions
        fill_angle: Angle for directional fill patterns
        
    Returns:
        Dictionary mapping color indices to paths
    """
    all_paths = {}
    total_regions = sum(len(regions) for regions in color_regions.values())
    
    # Create progress bar for path generation
    with tqdm(total=total_regions, desc="Generating paths") as pbar:
        for color_idx, regions in color_regions.items():
            color_paths = []
            
            for region_data in regions:
                # Unpack region data (mask, y_start, x_start, y_end, x_end)
                region_mask, y_start, x_start, y_end, x_end = region_data
                
                # Find contours in this region
                contours, _ = cv2.findContours(
                    region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    # Check if contour is large enough to paint
                    area = cv2.contourArea(contour)
                    if area < 2:  # Lower threshold to include smaller regions
                        continue
                    
                    # Adjust contour coordinates to account for region position
                    adjusted_contour = contour.copy()
                    adjusted_contour[:, :, 0] += x_start  # Add x offset
                    adjusted_contour[:, :, 1] += y_start  # Add y offset
                    
                    # Simplify contour to reduce number of points
                    epsilon = 0.01 * cv2.arcLength(adjusted_contour, True)
                    approx = cv2.approxPolyDP(adjusted_contour, epsilon, True)
                    
                    # Get contour points
                    points = [tuple(map(int, point[0])) for point in approx]
                    
                    # If contour is closed, make sure the first and last points connect
                    if len(points) > 2:
                        points.append(points[0])
                        
                    # For larger areas, add fill pattern
                    if area > 500:
                        fill_paths = generate_fill_pattern(
                            adjusted_contour, 
                            max(1, resolution_mm),  # Tighter spacing for better coverage
                            pattern_type=fill_pattern,
                            angle=fill_angle
                        )
                        color_paths.extend(fill_paths)
                    
                    color_paths.append(points)
                
                pbar.update(1)
            
            if color_paths:
                all_paths[color_idx] = color_paths
    
    # Optimize path sequence to minimize travel distance
    print("Optimizing path sequence to reduce travel distance...")
    all_paths = optimize_path_sequence(all_paths, start_point=(0, 0))
    
    return all_paths

def optimize_paths(all_paths):
    """
    Optimize paths to minimize travel distance.
    
    Args:
        all_paths: Dictionary mapping color indices to paths
        
    Returns:
        Optimized dictionary mapping color indices to paths
    """
    optimized_paths = {}
    total_paths = sum(len(paths) for paths in all_paths.values())
    
    # Create progress bar for path optimization
    with tqdm(total=total_paths, desc="Optimizing paths") as pbar:
        for color_idx, paths in all_paths.items():
            # Sort paths to minimize travel distance
            ordered_paths = []
            remaining_paths = paths.copy()
            
            if not remaining_paths:
                continue
                
            # Start from (0, 0) or upper-left corner
            current_pos = (0, 0)
            
            while remaining_paths:
                # Find the closest path
                min_dist = float('inf')
                closest_path_idx = 0
                
                for i, path in enumerate(remaining_paths):
                    start_point = path[0]
                    dist = math.sqrt((current_pos[0] - start_point[0])**2 + 
                                     (current_pos[1] - start_point[1])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_path_idx = i
                
                # Add closest path to ordered paths
                closest_path = remaining_paths.pop(closest_path_idx)
                ordered_paths.append(closest_path)
                
                # Update current position to the end of the path
                current_pos = closest_path[-1]
                
                # Update progress bar
                pbar.update(1)
            
            optimized_paths[color_idx] = ordered_paths
            
    return optimized_paths

def calculate_color_proximity(color_indices, color_list):
    """
    Calculate spatial proximity between colors in the image.
    Returns a dictionary with color pairs as keys and proximity scores as values.
    
    Args:
        color_indices: 2D array of color indices
        color_list: List of color indices to consider
        
    Returns:
        Dictionary mapping color pairs to proximity scores
    """
    # Sample the image to speed up calculation
    h, w = color_indices.shape
    sample_rate = max(1, min(h, w) // 200)  # Sample at most 200 points in each dimension
    
    # Create a proximity matrix (symmetric, so only store one half)
    proximity = {}
    
    print("Calculating color proximity metrics...")
    # For each pixel in the sample
    for y in range(0, h, sample_rate):
        for x in range(0, w, sample_rate):
            color = color_indices[y, x]
            
            # Check only colors we're interested in
            if color not in color_list:
                continue
                
            # Check neighboring pixels
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                        
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        neighbor = color_indices[ny, nx]
                        if neighbor != color and neighbor in color_list:
                            # Store pairs with smaller index first to avoid duplicates
                            pair = (min(color, neighbor), max(color, neighbor))
                            proximity[pair] = proximity.get(pair, 0) + 1
    
    return proximity

def generate_dynamic_color_instructions(all_paths, available_colors, color_change_position):
    """
    Generate painting instructions allowing dynamic color switching.
    
    Args:
        all_paths: Dictionary mapping color indices to paths
        available_colors: List of available RGB colors as tuples
        color_change_position: Position for color changes [x, y]
        
    Returns:
        List of painting instructions
    """
    instructions = []
    instructions.append({
        "type": "home",
        "message": "Moving to home position"
    })
    
    # Collect all paths with color info
    path_entries = []
    for color_idx, paths in all_paths.items():
        for path in paths:
            if path:
                start_point = path[0]
                path_entries.append({
                    "color_idx": color_idx,
                    "path": path,
                    "start": start_point
                })
                
    # Sort paths by starting point proximity (simple y,x sort)
    path_entries.sort(key=lambda e: (e['start'][1], e['start'][0]))
    
    current_color = None
    for entry in path_entries:
        color_idx = entry['color_idx']
        path = entry['path']
        
        # Switch color if needed
        if color_idx != current_color:
            instructions.append({
                "type": "move",
                "x": color_change_position[0],
                "y": color_change_position[1],
                "spray": False,
                "message": "Moving to color change position"
            })
            instructions.append({
                "type": "color",
                "index": color_idx,
                "rgb": available_colors[color_idx],
                "message": f"Changing to color {color_idx} - RGB{available_colors[color_idx]}"
            })
            current_color = color_idx
            
        # Move to start point with spray off
        first_point = path[0]
        instructions.append({
            "type": "move",
            "x": first_point[0],
            "y": first_point[1],
            "spray": False,
            "message": f"Moving to ({first_point[0]}, {first_point[1]})"
        })
        
        # Start spraying
        instructions.append({
            "type": "spray",
            "state": True,
            "message": "Starting to spray"
        })
        
        # Follow path
        for point in path[1:]:
            instructions.append({
                "type": "move",
                "x": point[0],
                "y": point[1],
                "spray": True,
                "message": f"Moving to ({point[0]}, {point[1]}) while spraying"
            })
            
        # Stop spraying
        instructions.append({
            "type": "spray",
            "state": False,
            "message": "Stopping spray"
        })
        
    instructions.append({
        "type": "home",
        "message": "Returning to home position"
    })
    
    return instructions
