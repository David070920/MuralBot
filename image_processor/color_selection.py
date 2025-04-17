"""
Color selection algorithms for MuralBot.

This module contains functions for selecting optimal colors based on image analysis.
"""

import numpy as np
import json
import os
from sklearn.cluster import KMeans

def select_colors_from_image(image_rgb, max_colors):
    """
    Analyze the image and select the best colors from the MTN94 spray paint database.
    
    Args:
        image_rgb: RGB image as numpy array
        max_colors: Maximum number of colors to select
        
    Returns:
        List of RGB color tuples
    """
    # Load the MTN94 color database
    color_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mtn94_colors.json")
    
    try:
        with open(color_db_path, 'r') as f:
            color_db = json.load(f)
            color_data = color_db["colors"]
            print(f"Loaded {len(color_data)} colors from MTN94 database for color selection")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading color database: {e}")
        print("Using basic colors instead")
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
        ][:max_colors]
        
    # Use available colors from the database
    available_colors_rgb = []
    for color in color_data:
        # Ensure each RGB value is properly converted to a tuple of integers
        rgb = tuple(int(c) for c in color["rgb"])
        available_colors_rgb.append(rgb)
    
    # Reshape image to a list of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Sample pixels for faster processing
    sample_size = min(100000, len(pixels))
    pixel_sample = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
    
    # Find optimal colors using K-means clustering
    print("Analyzing image colors using K-means clustering...")
    # Use max_colors directly instead of capping at 30
    num_clusters = max_colors
    
    # Run K-means to find dominant colors
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(pixel_sample)
    centroids = kmeans.cluster_centers_
    
    # Get the pixel count for each cluster
    labels = kmeans.predict(pixel_sample)
    cluster_sizes = np.bincount(labels)
    
    # Sort clusters by size (highest first)
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    sorted_centroids = centroids[sorted_indices]
    
    # Match each centroid to the closest MTN94 color
    selected_colors = []
    selected_color_indices = []
    
    print("Matching dominant colors to available spray paints...")
    for centroid in sorted_centroids:
        # Convert centroid to array with proper shape for distance calculation
        centroid_array = np.array(centroid, dtype=np.float32)
        colors_array = np.array(available_colors_rgb, dtype=np.float32)
        
        # Find closest match in the MTN94 palette using Euclidean distance
        distances = np.sqrt(np.sum((colors_array - centroid_array)**2, axis=1))
        closest_idx = np.argmin(distances)
        
        # Avoid duplicate colors
        if closest_idx not in selected_color_indices:
            selected_color_indices.append(closest_idx)
            selected_colors.append(available_colors_rgb[closest_idx])
            
            # Print the selected color and its name
            color_name = color_data[closest_idx]["name"]
            code = color_data[closest_idx]["code"]
            rgb = available_colors_rgb[closest_idx]
            print(f"Selected: {color_name} ({code}) - RGB{rgb}")
    
    # Ensure we always include black and white for better results
    black_idx = None
    white_idx = None
    
    # Find indices for black and white
    for i, color in enumerate(available_colors_rgb):
        if color[0] < 20 and color[1] < 20 and color[2] < 20:
            black_idx = i
            break
    
    for i, color in enumerate(available_colors_rgb):
        if color[0] > 240 and color[1] > 240 and color[2] > 240:
            white_idx = i
            break
    
    # Add black and white if they weren't already selected
    if black_idx is not None and black_idx not in selected_color_indices:
        selected_color_indices.append(black_idx)
        selected_colors.append(available_colors_rgb[black_idx])
        print(f"Added black: {color_data[black_idx]['name']} - RGB{available_colors_rgb[black_idx]}")
    
    if white_idx is not None and white_idx not in selected_color_indices:
        selected_color_indices.append(white_idx)
        selected_colors.append(available_colors_rgb[white_idx])
        print(f"Added white: {color_data[white_idx]['name']} - RGB{available_colors_rgb[white_idx]}")
        
    # Limit to max_colors
    selected_colors = selected_colors[:max_colors]
    
    print(f"Final selection: {len(selected_colors)} colors")
    return selected_colors
