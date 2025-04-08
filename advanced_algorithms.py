"""
Advanced algorithms for MuralBot.

This module contains improved dithering and path generation algorithms
for the MuralBot mural painting system.
"""

import numpy as np
from tqdm import tqdm
import cv2

def apply_dithering(original_image, available_colors, color_indices, method='floyd-steinberg', strength=1.0):
    """
    Apply dithering to improve visual appearance.
    
    Args:
        original_image: Original RGB image
        available_colors: List of available RGB colors
        color_indices: Array of color indices mapping to available_colors
        method: Dithering method ('floyd-steinberg', 'jarvis', or 'stucki')
        strength: Dithering strength factor (0.0-1.0)
    
    Returns:
        Tuple of (dithered_image, dithered_indices)
    """
    # Make a copy of the original image in float format for error diffusion
    h, w = original_image.shape[:2]
    dither_image = original_image.astype(np.float32)
    dithered_indices = color_indices.copy()
    
    # Available colors as numpy array
    colors_array = np.array(available_colors)
    
    # Clamp strength parameter
    strength = max(0.0, min(1.0, strength))
    
    # Error diffusion matrices for different algorithms
    if method == 'floyd-steinberg':
        # Floyd-Steinberg dithering
        # 1/16 * [    X 7 ]
        #        [ 3 5 1 ]
        diffusion_pattern = [
            (0, 1, 7/16), 
            (1, -1, 3/16), 
            (1, 0, 5/16), 
            (1, 1, 1/16)
        ]
    elif method == 'jarvis':
        # Jarvis-Judice-Ninke dithering
        # 1/48 * [      X 7 5 ]
        #        [ 3 5 7 5 3 ]
        #        [ 1 3 5 3 1 ]
        diffusion_pattern = [
            (0, 1, 7/48), (0, 2, 5/48),
            (1, -2, 3/48), (1, -1, 5/48), (1, 0, 7/48), (1, 1, 5/48), (1, 2, 3/48),
            (2, -2, 1/48), (2, -1, 3/48), (2, 0, 5/48), (2, 1, 3/48), (2, 2, 1/48)
        ]
    elif method == 'stucki':
        # Stucki dithering
        # 1/42 * [      X 8 4 ]
        #        [ 2 4 8 4 2 ]
        #        [ 1 2 4 2 1 ]
        diffusion_pattern = [
            (0, 1, 8/42), (0, 2, 4/42),
            (1, -2, 2/42), (1, -1, 4/42), (1, 0, 8/42), (1, 1, 4/42), (1, 2, 2/42),
            (2, -2, 1/42), (2, -1, 2/42), (2, 0, 4/42), (2, 1, 2/42), (2, 2, 1/42)
        ]
    else:
        # Default to Floyd-Steinberg
        diffusion_pattern = [
            (0, 1, 7/16), 
            (1, -1, 3/16), 
            (1, 0, 5/16), 
            (1, 1, 1/16)
        ]
    
    # Process the image pixel by pixel
    with tqdm(total=h, desc=f"Applying {method} dithering") as pbar:
        for y in range(h):
            for x in range(w):
                # Get old pixel value
                old_pixel = dither_image[y, x].copy()
                
                # Find nearest color
                distances = np.sqrt(np.sum((old_pixel - colors_array)**2, axis=1))
                nearest_idx = np.argmin(distances)
                nearest_color = colors_array[nearest_idx]
                
                # Update image with nearest color
                dithered_indices[y, x] = nearest_idx
                
                # Compute error
                error = (old_pixel - nearest_color) * strength
                
                # Distribute error according to diffusion pattern
                for dy, dx, factor in diffusion_pattern:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        dither_image[ny, nx] += error * factor
            
            pbar.update(1)
    
    # Convert dithered image back to RGB
    dithered_image = np.array([available_colors[idx] for idx in dithered_indices.flatten()])
    dithered_image = dithered_image.reshape(original_image.shape)
    
    return dithered_image, dithered_indices

def generate_fill_pattern(contour, spacing, pattern_type='zigzag', angle=0):
    """
    Generate optimized fill pattern inside a contour.
    
    Args:
        contour: OpenCV contour
        spacing: Spacing between lines in mm
        pattern_type: Type of fill pattern ('zigzag', 'concentric', 'spiral', or 'dots')
        angle: Angle for directional patterns in degrees
    
    Returns:
        List of point paths for filling
    """
    # Get bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Create a mask for the contour
    mask = np.zeros((y+h+10, x+w+10), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 1, -1, offset=(5, 5))
    
    # Offset contour for mask operations
    contour_offset = contour.copy()
    contour_offset[:, :, 0] += 5
    contour_offset[:, :, 1] += 5
    
    fill_paths = []
    
    if pattern_type == 'zigzag':
        # ZigZag pattern (improved)
        # Apply rotation if needed
        rotation_matrix = None
        if angle != 0:
            center = ((x + w//2), (y + h//2))
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
        
        # Get rotated bounding rectangle if needed
        if rotation_matrix is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return fill_paths
            x, y, w, h = cv2.boundingRect(contours[0])
        
        # Generate horizontal lines with spacing
        for y_pos in range(y, y+h, spacing):
            points = []
            
            # Find intersections with the contour mask
            line_mask = mask[y_pos, x:x+w]
            transitions = np.where(np.diff(line_mask) != 0)[0]
            
            if len(transitions) < 2:
                continue
            
            # Create segments from transition pairs
            for i in range(0, len(transitions), 2):
                if i+1 >= len(transitions):
                    break
                    
                start_x = x + transitions[i] + 1
                end_x = x + transitions[i+1] + 1
                
                # Skip tiny segments
                if end_x - start_x < 3:
                    continue
                
                # Create a zigzag line
                if len(points) == 0:
                    points.append((start_x, y_pos))
                    points.append((end_x, y_pos))
                elif (y_pos // spacing) % 2 == 0:
                    points.append((start_x, y_pos))
                    points.append((end_x, y_pos))
                else:
                    points.append((end_x, y_pos))
                    points.append((start_x, y_pos))
            
            if points:
                fill_paths.append(points)
                
    elif pattern_type == 'concentric':
        # Concentric pattern
        # Erode the mask to create concentric rings
        struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        current_mask = mask.copy()
        
        for i in range(0, max(w, h), spacing):
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 5:  # Need at least 5 points for ellipse
                    continue
                    
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                # Get points
                points = [tuple(map(int, point[0])) for point in approx]
                if len(points) > 2:
                    points.append(points[0])  # Close the loop
                    fill_paths.append(points)
            
            # Erode for next iteration
            current_mask = cv2.erode(current_mask, struct_element, iterations=spacing)
            if cv2.countNonZero(current_mask) == 0:
                break
                
    elif pattern_type == 'spiral':
        # Spiral pattern
        # Find contour center
        M = cv2.moments(contour_offset)
        if M['m00'] == 0:
            return fill_paths
            
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        
        # Create spiral path
        max_radius = max(w, h) // 2
        theta = np.linspace(0, 15*2*np.pi, 1000)
        radius = np.linspace(0, max_radius, 1000)
        
        points = []
        for t, r in zip(theta, radius):
            x = int(center_x + r * np.cos(t))
            y = int(center_y + r * np.sin(t))
            
            # Check if point is inside mask
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                if len(points) == 0 or (x, y) != points[-1]:
                    points.append((x, y))
            
        if points:
            fill_paths.append(points)
            
    elif pattern_type == 'dots':
        # Dot pattern
        spacing = max(spacing, 2)  # Ensure minimum spacing
        
        for y_pos in range(y, y+h, spacing):
            for x_pos in range(x, x+w, spacing):
                if y_pos < mask.shape[0] and x_pos < mask.shape[1] and mask[y_pos, x_pos]:
                    fill_paths.append([(x_pos, y_pos)])
    
    return fill_paths

def optimize_path_sequence(all_paths, start_point=(0, 0)):
    """
    Optimize path sequence to reduce travel distance.
    
    Args:
        all_paths: Dictionary of all paths, keyed by color index
        start_point: Starting point coordinates
    
    Returns:
        Optimized dictionary of paths
    """
    optimized_paths = {}
    
    for color_idx, paths in all_paths.items():
        # Skip if no paths for this color
        if not paths:
            continue
        
        optimized_color_paths = []
        remaining_paths = paths.copy()
        
        # Start from the closest path to the start point
        current_point = start_point
        
        while remaining_paths:
            closest_dist = float('inf')
            closest_path_idx = 0
            closest_path_reverse = False
            
            # Find the path with closest start/end point to current location
            for i, path in enumerate(remaining_paths):
                if not path:
                    continue
                    
                # Check distance to path start
                start_dist = np.sqrt((path[0][0] - current_point[0])**2 + 
                                     (path[0][1] - current_point[1])**2)
                
                # Check distance to path end
                end_dist = np.sqrt((path[-1][0] - current_point[0])**2 + 
                                   (path[-1][1] - current_point[1])**2)
                
                if start_dist < closest_dist:
                    closest_dist = start_dist
                    closest_path_idx = i
                    closest_path_reverse = False
                
                if end_dist < closest_dist:
                    closest_dist = end_dist
                    closest_path_idx = i
                    closest_path_reverse = True
            
            # Get the closest path
            closest_path = remaining_paths.pop(closest_path_idx)
            
            # Reverse path if end is closer
            if closest_path_reverse and len(closest_path) > 1:
                closest_path = closest_path[::-1]
            
            # Add to optimized paths
            optimized_color_paths.append(closest_path)
            
            # Update current point
            if closest_path:
                current_point = closest_path[-1]
        
        optimized_paths[color_idx] = optimized_color_paths
    
    return optimized_paths
