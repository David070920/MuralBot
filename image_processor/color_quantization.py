"""
Color quantization algorithms for MuralBot.

This module contains different methods for quantizing an image to use a limited set of colors.
"""

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

def quantize_euclidean(image_rgb, available_colors):
    """
    Original Euclidean distance color quantization method.
    
    Args:
        image_rgb: RGB image as numpy array
        available_colors: List of available RGB colors as tuples
        
    Returns:
        Tuple of (quantized_image, color_indices)
    """
    # Reshape image to a list of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Convert available colors to numpy array
    colors_array = np.array(available_colors)
    
    # Process in batches to avoid memory issues with very large images
    batch_size = 100000  # Adjust based on available RAM
    num_batches = (pixels.shape[0] + batch_size - 1) // batch_size
    color_indices = np.zeros(len(pixels), dtype=int)
    
    # Create progress bar
    with tqdm(total=num_batches, desc="Color matching") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, pixels.shape[0])
            batch = pixels[start_idx:end_idx]
            
            # Compute distances from each pixel to each color
            # Using broadcasting to avoid loops
            distances = np.sqrt(np.sum((batch[:, np.newaxis, :] - colors_array[np.newaxis, :, :])**2, axis=2))
            
            # Find index of minimum distance for each pixel
            color_indices[start_idx:end_idx] = np.argmin(distances, axis=1)
            
            # Update progress bar
            pbar.update(1)
    
    # Map each pixel to its closest available color
    quantized = np.array([available_colors[idx] for idx in color_indices])
    
    # Reshape back to image dimensions
    quantized_image = quantized.reshape(image_rgb.shape)
    color_indices = color_indices.reshape(image_rgb.shape[:2])
    
    return quantized_image, color_indices

def quantize_kmeans(image_rgb, available_colors):
    """
    K-means based color quantization using available colors as initial centers.
    
    Args:
        image_rgb: RGB image as numpy array
        available_colors: List of available RGB colors as tuples
        
    Returns:
        Tuple of (quantized_image, color_indices)
    """
    # Reshape image to a list of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Convert available colors to numpy array
    colors_array = np.array(available_colors)
    num_colors = len(colors_array)
    
    # Ensure we have at least one color
    if num_colors == 0:
        print("Warning: No available colors specified. Using default colors.")
        colors_array = np.array([
            (0, 0, 0),       # Black
            (255, 255, 255),  # White
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
        ])
        num_colors = len(colors_array)
    
    print("  Performing K-means clustering for improved color mapping...")
    
    # Create color palette using K-means with available colors as initial centers
    kmeans_model = KMeans(
        n_clusters=num_colors,
        init=colors_array,
        n_init=1,
        max_iter=20,
        random_state=42
    )
    
    # Fit K-means on a sample of pixels for efficiency
    sample_size = min(100000, len(pixels))
    indices = np.random.choice(len(pixels), size=sample_size, replace=False)
    kmeans_model.fit(pixels[indices])
    
    # Get optimized centers
    centers = kmeans_model.cluster_centers_
    
    # Assign each center to closest available color
    center_assignments = []
    for center in centers:
        distances = np.sqrt(np.sum((center - colors_array)**2, axis=1))
        closest_idx = np.argmin(distances)
        center_assignments.append(closest_idx)
        
    # Now quantize the entire image using the trained model
    with tqdm(total=1, desc="Applying K-means quantization") as pbar:
        # Use the model to predict clusters for all pixels
        clusters = kmeans_model.predict(pixels)
        
        # Map clusters to actual available colors
        color_indices = np.array([center_assignments[c] for c in clusters])
        pbar.update(1)
    
    # Map each pixel to its assigned color
    quantized = np.array([available_colors[idx] for idx in color_indices])
    
    # Reshape back to image dimensions
    quantized_image = quantized.reshape(image_rgb.shape)
    color_indices = color_indices.reshape(image_rgb.shape[:2])
    
    return quantized_image, color_indices

def quantize_color_palette(image_rgb, available_colors):
    """
    Color quantization using optimized color palette generation.
    
    Args:
        image_rgb: RGB image as numpy array
        available_colors: List of available RGB colors as tuples
        
    Returns:
        Tuple of (quantized_image, color_indices)
    """
    # Reshape image to a list of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Convert available colors to numpy array
    colors_array = np.array(available_colors)
    
    # First, find dominant colors in the image
    print("  Analyzing dominant colors in image...")
    
    # Sample pixels for faster processing
    sample_size = min(100000, len(pixels))
    pixel_sample = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
    
    # Find optimal color mapping using color histogram
    hist_size = 8
    color_hist = np.zeros((hist_size, hist_size, hist_size), dtype=np.float32)
    
    # Quantize the color space for histogram
    bin_width = 256 // hist_size
    pixel_bins = (pixel_sample // bin_width).astype(int)
    pixel_bins = np.clip(pixel_bins, 0, hist_size-1)
    
    # Build histogram
    for i in range(sample_size):
        r, g, b = pixel_bins[i]
        color_hist[r, g, b] += 1
    
    # Normalize histogram
    color_hist /= sample_size
    
    # Find peak locations in the histogram
    peaks = []
    for r in range(hist_size):
        for g in range(hist_size):
            for b in range(hist_size):
                if color_hist[r, g, b] > 0.001:  # Threshold for significant colors
                    peaks.append((color_hist[r, g, b], (r*bin_width + bin_width//2, 
                                                      g*bin_width + bin_width//2, 
                                                      b*bin_width + bin_width//2)))
    
    # Sort peaks by frequency
    peaks.sort(reverse=True)
    
    # Create weighted color mapping based on dominant colors
    color_weights = np.ones(len(colors_array))
    
    for weight, peak in peaks[:min(10, len(peaks))]:
        distances = np.sqrt(np.sum((colors_array - peak)**2, axis=1))
        # Increase weight of colors that are close to peaks
        boost = 10.0 * weight / (distances + 1.0)
        color_weights += boost
    
    # Process in batches with weighted color mapping
    batch_size = 100000
    num_batches = (pixels.shape[0] + batch_size - 1) // batch_size
    color_indices = np.zeros(len(pixels), dtype=int)
    
    with tqdm(total=num_batches, desc="Optimized color matching") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, pixels.shape[0])
            batch = pixels[start_idx:end_idx]
            
            # Compute weighted distances
            distances = np.sqrt(np.sum((batch[:, np.newaxis, :] - colors_array[np.newaxis, :, :])**2, axis=2))
            distances /= color_weights
            
            # Find index of minimum distance for each pixel
            color_indices[start_idx:end_idx] = np.argmin(distances, axis=1)
            pbar.update(1)
    
    # Map each pixel to its closest available color
    quantized = np.array([available_colors[idx] for idx in color_indices])
    
    # Reshape back to image dimensions
    quantized_image = quantized.reshape(image_rgb.shape)
    color_indices = color_indices.reshape(image_rgb.shape[:2])
    
    return quantized_image, color_indices

def quantize_adaptive_kmeans(image_rgb, available_colors, max_colors):
    """
    Adaptive K-means quantization with automatic cluster count selection.
    
    Args:
        image_rgb: RGB image as numpy array
        available_colors: List of available RGB colors as tuples
        max_colors: Maximum number of colors to use
        
    Returns:
        Tuple of (quantized_image, color_indices)
    """
    # Convert to floating point for LAB conversion
    pixels_rgb = image_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    pixels_lab = rgb2lab(pixels_rgb.reshape(-1,1,3)).reshape(-1,3)

    best_k = 4
    best_inertia = None
    max_k = min(max_colors, 30)
    inertias = []

    # Try different cluster counts
    for k in range(4, max_k+1):
        kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
        kmeans.fit(pixels_lab)
        inertias.append(kmeans.inertia_)
        if best_inertia is None or kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_k = k

        # Early stopping if inertia reduction is small
        if len(inertias) > 1 and (inertias[-2] - inertias[-1]) / inertias[-2] < 0.05:
            break

    # Final KMeans with best_k
    kmeans = KMeans(n_clusters=best_k, n_init=5, max_iter=200, random_state=42)
    kmeans.fit(pixels_lab)
    centers_lab = kmeans.cluster_centers_

    # Convert available colors to LAB
    available_rgb = np.array(available_colors).astype(np.float32) / 255.0
    available_lab = rgb2lab(available_rgb.reshape(-1,1,3)).reshape(-1,3)

    # Map cluster centers to closest available paint color
    assignments = []
    for center in centers_lab:
        distances = np.linalg.norm(available_lab - center, axis=1)
        closest_idx = np.argmin(distances)
        assignments.append(closest_idx)

    # Assign pixels
    labels = kmeans.predict(pixels_lab)
    color_indices = np.array([assignments[label] for label in labels])

    quantized_rgb = np.array([available_colors[idx] for idx in color_indices])
    quantized_image = quantized_rgb.reshape(image_rgb.shape)
    color_indices = color_indices.reshape(image_rgb.shape[:2])

    return quantized_image, color_indices
