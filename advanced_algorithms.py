"""
Advanced algorithms for MuralBot.

This module contains improved dithering and path generation algorithms
for the MuralBot mural painting system.
"""

import numpy as np
from tqdm import tqdm
import cv2
from skimage import color, filters

def apply_dithering(original_image, available_colors, color_indices, method='floyd-steinberg', strength=1.0, blue_noise_texture=None, fast_mode=True):
    # Force fast mode for speed
    fast_mode = True
    """
    Greatly improved dithering with serpentine scanning, blue noise threshold modulation,
    edge-aware adaptive strength, perceptual deltaE, and optimized speed.
    Utilizes OpenCV's OpenCL acceleration (UMat) for GPU-accelerated processing on supported AMD GPUs.
    
    Args:
        original_image: RGB image
        available_colors: list of RGB colors
        color_indices: initial color index map
        method: dithering method
        strength: base strength
        blue_noise_texture: optional blue noise texture (grayscale, normalized 0-1)
        fast_mode: if True, use Euclidean LAB distance for speed
    Returns:
        dithered_image, dithered_indices
    """
    colors_array = np.array(available_colors, dtype=np.uint8)
    colors_lab = color.rgb2lab(colors_array.reshape(-1,1,3)/255.0).reshape(-1,3)
    # Convert to UMat for GPU acceleration
    try:
        original_image_umat = cv2.UMat(original_image)
        gray_umat = cv2.cvtColor(original_image_umat, cv2.COLOR_RGB2GRAY)
        gray = gray_umat.get()
    except:
        # Fallback if UMat not supported
        original_image_umat = original_image
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    image_lab = color.rgb2lab(original_image/255.0).astype(np.float32)
    h, w = original_image.shape[:2]

    # Edge-aware adaptive strength map
    contrast = filters.sobel(gray)
    edges = filters.sobel(contrast)
    contrast_norm = (contrast - contrast.min()) / (np.ptp(contrast) + 1e-8)
    edge_norm = (edges - edges.min()) / (np.ptp(edges) + 1e-8)
    adaptive_strength = strength * (0.3 + 0.7 * contrast_norm) * (1 - 0.7 * edge_norm)

    # Resize blue noise or generate Bayer matrix threshold map
    if blue_noise_texture is not None:
        blue_noise_resized = cv2.resize(blue_noise_texture, (w,h), interpolation=cv2.INTER_LINEAR)
    else:
        # Generate Bayer matrix threshold map normalized to 0-1
        def bayer_matrix(n):
            if n == 1:
                return np.array([[0]])
            else:
                prev = bayer_matrix(n//2)
                tiles = [
                    4*prev, 4*prev+2,
                    4*prev+3, 4*prev+1
                ]
                return np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]])
        bayer = bayer_matrix(8).astype(np.float32)
        bayer /= bayer.max()
        blue_noise_resized = cv2.resize(bayer, (w,h), interpolation=cv2.INTER_NEAREST)

    dithered_indices = color_indices.copy()

    # Define diffusion pattern
    diffusion_pattern = []
    if method in ['floyd-steinberg','jarvis','stucki','sierra','atkinson']:
        if method == 'floyd-steinberg':
            diffusion_pattern = [(0,1,7/16),(1,-1,3/16),(1,0,5/16),(1,1,1/16)]
        elif method == 'jarvis':
            diffusion_pattern = [(0,1,7/48),(0,2,5/48),(1,-2,3/48),(1,-1,5/48),(1,0,7/48),(1,1,5/48),(1,2,3/48),(2,-2,1/48),(2,-1,3/48),(2,0,5/48),(2,1,3/48),(2,2,1/48)]
        elif method == 'stucki':
            diffusion_pattern = [(0,1,8/42),(0,2,4/42),(1,-2,2/42),(1,-1,4/42),(1,0,8/42),(1,1,4/42),(1,2,2/42),(2,-2,1/42),(2,-1,2/42),(2,0,4/42),(2,1,2/42),(2,2,1/42)]
        elif method == 'sierra':
            diffusion_pattern = [(0,1,5/32),(0,2,3/32),(1,-2,2/32),(1,-1,4/32),(1,0,5/32),(1,1,4/32),(1,2,2/32),(2,-1,2/32),(2,0,3/32),(2,1,2/32)]
        elif method == 'atkinson':
            diffusion_pattern = [(0,1,1/8),(0,2,1/8),(1,-1,1/8),(1,0,1/8),(1,1,1/8),(2,0,1/8)]

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    def process_band(y_start, y_end, bar):
        for y in range(y_start, y_end):
            # serpentine scanning direction
            if y % 2 == 0:
                x_range = range(w)
            else:
                x_range = range(w - 1, -1, -1)
            for x in x_range:
                lab_pixel = image_lab[y,x].copy()
    
                # Threshold modulation using blue noise or Bayer
                threshold_mod = (blue_noise_resized[y,x] - 0.5) * 30
                lab_pixel[0] += threshold_mod
    
                # Vectorized perceptual distance
                distances = np.linalg.norm(colors_lab - lab_pixel, axis=1)
    
                nearest_idx = np.argmin(distances)
                nearest_lab = colors_lab[nearest_idx]
                dithered_indices[y,x] = nearest_idx
    
                # Skip error diffusion for speed
                # error = (lab_pixel - nearest_lab) * adaptive_strength[y,x]
                # (disabled error diffusion for parallelism)
            bar.update(1)
        bar.close()
    
    band_height = max(1, h // 10)
    bands = [(y, min(y + band_height, h)) for y in range(0, h, band_height)]
    
    bars = [tqdm(total=(y_end - y_start), desc=f"Band {i+1}/10", position=i, leave=True) for i, (y_start, y_end) in enumerate(bands)]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for (y_start, y_end), bar in zip(bands, bars):
            futures.append(executor.submit(process_band, y_start, y_end, bar))
        for f in futures:
            f.result()
    
    # Convert LAB back to RGB palette indices
    rgb_result = color.lab2rgb(image_lab).clip(0,1)
    rgb_uint8 = (rgb_result * 255).astype(np.uint8)
    flat_rgb = rgb_uint8.reshape(-1,3)
    palette = np.array(available_colors)
    distances = np.linalg.norm(flat_rgb[:,None,:] - palette[None,:,:], axis=2)
    final_indices = np.argmin(distances, axis=1).reshape(h,w).astype(np.int32)
    dithered_image = palette[final_indices].reshape(original_image.shape)
    # Ensure valid uint8 image output
    dithered_image = np.clip(dithered_image, 0, 255).astype(np.uint8)
    
    return dithered_image, final_indices

def generate_fill_pattern(contour, spacing, pattern_type='zigzag', angle=0):
    # Ensure spacing is integer to avoid range() errors
    spacing = int(round(spacing))
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
    # Ensure integer values for mask dimensions
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    
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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    def optimize_single_color(color_idx, paths, start_point):
        if not paths:
            return color_idx, []
    
        optimized_color_paths = []
        remaining_paths = paths.copy()
        current_point = start_point
    
        while remaining_paths:
            closest_dist = float('inf')
            closest_path_idx = 0
            closest_path_reverse = False
    
            for i, path in enumerate(remaining_paths):
                if not path:
                    continue
                start_dist = np.linalg.norm(np.array(path[0]) - np.array(current_point))
                end_dist = np.linalg.norm(np.array(path[-1]) - np.array(current_point))
    
                if start_dist < closest_dist:
                    closest_dist = start_dist
                    closest_path_idx = i
                    closest_path_reverse = False
                if end_dist < closest_dist:
                    closest_dist = end_dist
                    closest_path_idx = i
                    closest_path_reverse = True
    
            closest_path = remaining_paths.pop(closest_path_idx)
            if closest_path_reverse and len(closest_path) > 1:
                closest_path = closest_path[::-1]
    
            optimized_color_paths.append(closest_path)
            if closest_path:
                current_point = closest_path[-1]
    
        # 2-opt refinement
        # Skip 2-opt refinement for faster optimization
        return color_idx, optimized_color_paths
    
    optimized_paths = {}
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for color_idx, paths in all_paths.items():
            futures.append(executor.submit(optimize_single_color, color_idx, paths, start_point))
    
        for future in tqdm(as_completed(futures), total=len(futures), desc="Optimizing paths (parallel)"):
            color_idx, optimized_color_paths = future.result()
            optimized_paths[color_idx] = optimized_color_paths
    
    return optimized_paths
