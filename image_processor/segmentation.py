"""
Image segmentation algorithms for MuralBot.

This module contains functions for segmenting images into distinct color regions.
"""

import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

def find_color_regions(color_indices, num_colors):
    """
    Find connected regions for each color.
    
    Args:
        color_indices: 2D array of color indices
        num_colors: Number of colors in the palette
        
    Returns:
        Dictionary mapping color indices to regions
    """
    color_regions = {}
    
    # Get image dimensions
    height, width = color_indices.shape
    
    # Create progress bar for color regions
    with tqdm(total=num_colors, desc="Finding regions") as pbar:
        for color_idx in range(num_colors):
            # Skip if this color isn't used
            if not np.any(color_indices == color_idx):
                pbar.update(1)
                continue
            
            # Process large images in chunks to avoid memory issues
            chunk_height = 500  # Process 500 rows at a time
            regions = []
            
            # Create a reusable structure for connected components
            structure = np.ones((3, 3), dtype=np.int32)
            
            # Generate a unique label ID for each chunk to avoid conflicts
            next_label_id = 1
            
            # Process the image in horizontal chunks
            for y_start in range(0, height, chunk_height):
                y_end = min(y_start + chunk_height, height)
                
                # Create binary mask for this color in the current chunk
                chunk_mask = (color_indices[y_start:y_end, :] == color_idx).astype(np.uint8)
                
                # Skip if no pixels of this color in the chunk
                if not np.any(chunk_mask):
                    continue
                    
                # Find connected components in this chunk
                labeled_chunk, num_features = label(chunk_mask, structure)
                
                # Process each feature in this chunk
                for feature_idx in range(1, num_features + 1):
                    feature_mask = np.zeros_like(chunk_mask, dtype=np.uint8)
                    feature_mask[labeled_chunk == feature_idx] = 1
                    
                    # Check if this region is large enough to be worth processing
                    if np.sum(feature_mask) > 5:  # Skip tiny regions with fewer than 5 pixels
                        regions.append((feature_mask, y_start, 0, y_end, width))
            
            # Store regions for this color
            color_regions[color_idx] = regions
            pbar.update(1)
        
    return color_regions

def segment_slic(color_indices, available_colors):
    """
    Segment image using SLIC superpixels and group by dominant color.
    
    Args:
        color_indices: 2D array of color indices
        available_colors: List of available RGB colors as tuples
        
    Returns:
        Dictionary mapping color indices to regions
    """
    from skimage.segmentation import slic
    import numpy as np

    # Reconstruct quantized RGB image from color indices
    h, w = color_indices.shape
    quantized_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(available_colors):
        mask = (color_indices == idx)
        quantized_rgb[mask] = color

    # Run SLIC
    segments = slic(quantized_rgb, n_segments=500, compactness=10, start_label=1, convert2lab=True)

    # Group superpixels by dominant color
    color_regions = {i: [] for i in range(len(available_colors))}
    for seg_val in np.unique(segments):
        mask = (segments == seg_val)
        if np.sum(mask) < 5:
            continue
        # Find dominant color index in this superpixel
        dominant_color = np.bincount(color_indices[mask].flatten()).argmax()
        color_mask = np.zeros_like(mask, dtype=np.uint8)
        color_mask[mask] = 1
        color_regions[dominant_color].append((color_mask, 0, 0, h, w))

    return color_regions
