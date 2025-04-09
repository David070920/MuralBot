import cv2
import numpy as np
import json
import math
import argparse
import os
from collections import defaultdict
from scipy.ndimage import label
from tqdm import tqdm
from scipy.cluster.vq import kmeans, vq
# Set environment variable to avoid joblib warnings about CPU cores detection
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
from sklearn.cluster import KMeans
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import skimage.morphology as morphology
from skimage.graph import route_through_array
from scipy.spatial import Delaunay

from advanced_algorithms import apply_dithering, generate_fill_pattern, optimize_path_sequence

class MuralInstructionGenerator:
    def __init__(self, wall_width, wall_height, available_colors=None, resolution_mm=5,
                 quantization_method="euclidean", dithering="none", dithering_strength=1.0, max_colors=30,
                 robot_capacity=6, color_selection="auto", fill_pattern="zigzag", fill_angle=0, fast_mode=True,
                 segmentation_method="connected_components"):
        """
        Initialize the mural painting instruction generator.
        
        Args:
            wall_width: Width of the wall/canvas in mm
            wall_height: Height of the wall/canvas in mm
            available_colors: List of (R,G,B) tuples representing available spray colors
            resolution_mm: Resolution in mm for path planning
            quantization_method: Method for color quantization ("euclidean", "kmeans", "adaptive_kmeans", or "color_palette")
            dithering: Dithering method ("none", "floyd-steinberg", "jarvis", or "stucki")
            dithering_strength: Strength of dithering effect (0.0-1.0)
            max_colors: Maximum number of colors to use in the mural
            robot_capacity: Number of colors the robot can hold simultaneously
            color_selection: Method for color selection ("auto" or "manual")
            fill_pattern: Pattern for filling regions ("zigzag", "concentric", "spiral", or "dots")
            fill_angle: Angle for directional patterns in degrees
            fast_mode: Use fast approximate dithering (default True)
            segmentation_method: Method for segmenting color regions ("connected_components" or "slic")
        """
        self.wall_width = wall_width
        self.wall_height = wall_height
        self.resolution_mm = resolution_mm
        self.quantization_method = quantization_method
        self.dithering = dithering
        self.dithering_strength = dithering_strength
        self.paint_usage = None  # Will store paint usage estimates per color
        self.max_colors = max_colors  # Maximum colors to use in the mural
        self.robot_capacity = robot_capacity  # Number of colors robot can hold at once
        self.color_selection = color_selection  # Color selection method
        self.color_change_position = [0, 2000]  # Position to change colors (bottom left)
        self.optimized_color_groups = []  # Will store grouped colors for robot capacity
        self.fill_pattern = fill_pattern  # Fill pattern type
        self.fill_angle = fill_angle  # Angle for directional fill patterns
        self.fast_mode = fast_mode  # Use fast approximate dithering
        self.segmentation_method = segmentation_method  # Segmentation method
        
        # Load the MTN94 color database by default
        self.available_colors = self._load_mtn94_colors() if available_colors is None else available_colors
    
    def _load_mtn94_colors(self):
        """Load colors from the MTN94 color database."""
        color_db_path = os.path.join(os.path.dirname(__file__), "data", "mtn94_colors.json")
        
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
    
    def load_image(self, image_path):
        """Load and resize image to fit the wall dimensions."""
        import os
        
        # Load image
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' does not exist.")
            return None, None
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
            
        # Resize to fit wall dimensions while maintaining aspect ratio
        img_height, img_width = img.shape[:2]
        aspect_ratio = img_width / img_height
        
        if self.wall_width / self.wall_height > aspect_ratio:
            # Wall is wider relative to height
            new_width = int(self.wall_height * aspect_ratio)
            new_height = self.wall_height
        else:
            # Wall is taller relative to width
            new_width = self.wall_width
            new_height = int(self.wall_width / aspect_ratio)
            
        img = cv2.resize(img, (new_width, new_height))
        
        # Center the image on the wall
        x_offset = (self.wall_width - new_width) // 2
        y_offset = (self.wall_height - new_height) // 2
        
        # Create a canvas of wall size with background color (white)
        canvas = np.ones((self.wall_height, self.wall_width, 3), dtype=np.uint8) * 255
        
        # Place the image on the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img
        
        return canvas
        
    def quantize_to_available_colors(self, image):
        """Quantize image to only use available spray can colors using selected method."""
        # Convert image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape image to a list of pixels
        pixels = image_rgb.reshape(-1, 3)
        
        print(f"  Processing color quantization using {self.quantization_method} method...")
        
        if self.quantization_method == "euclidean":
            quantized_image, color_indices = self._quantize_euclidean(image_rgb)
        elif self.quantization_method == "kmeans":
            quantized_image, color_indices = self._quantize_kmeans(image_rgb)
        elif self.quantization_method == "adaptive_kmeans":
            quantized_image, color_indices = self._quantize_adaptive_kmeans(image_rgb)
        elif self.quantization_method == "color_palette":
            quantized_image, color_indices = self._quantize_color_palette(image_rgb)
        else:
            print(f"Warning: Unknown quantization method '{self.quantization_method}', falling back to euclidean")
            quantized_image, color_indices = self._quantize_euclidean(image_rgb)
        
        # Apply dithering if enabled
        if self.dithering != "none":
            print(f"  Applying {self.dithering} dithering with strength {self.dithering_strength}...")
            quantized_image, color_indices = apply_dithering(
                image_rgb, 
                self.available_colors, 
                color_indices,
                method=self.dithering,
                strength=self.dithering_strength,
                fast_mode=self.fast_mode
            )
            
        # Calculate estimated paint usage
        self._calculate_paint_usage(color_indices)
        
        return cv2.cvtColor(quantized_image.astype(np.uint8), cv2.COLOR_RGB2BGR), color_indices

    def _quantize_euclidean(self, image_rgb):
        """Original Euclidean distance color quantization method."""
        # Reshape image to a list of pixels
        pixels = image_rgb.reshape(-1, 3)
        
        # Convert available colors to numpy array
        colors_array = np.array(self.available_colors)
        
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
        quantized = np.array([self.available_colors[idx] for idx in color_indices])
        
        # Reshape back to image dimensions
        quantized_image = quantized.reshape(image_rgb.shape)
        color_indices = color_indices.reshape(image_rgb.shape[:2])
        
        return quantized_image, color_indices

    def _quantize_kmeans(self, image_rgb):
        """K-means based color quantization using available colors as initial centers."""
        # Reshape image to a list of pixels
        pixels = image_rgb.reshape(-1, 3)
        
        # Convert available colors to numpy array
        colors_array = np.array(self.available_colors)
        num_colors = len(colors_array)
        
        # Ensure we have at least one color
        if num_colors == 0:
            print("Warning: No available colors specified. Loading colors from MTN94 database.")
            # Load colors from the MTN94 database
            self.available_colors = self._load_mtn94_colors()
            colors_array = np.array(self.available_colors)
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
        quantized = np.array([self.available_colors[idx] for idx in color_indices])
        
        # Reshape back to image dimensions
        quantized_image = quantized.reshape(image_rgb.shape)
        color_indices = color_indices.reshape(image_rgb.shape[:2])
        
        return quantized_image, color_indices

    def _quantize_color_palette(self, image_rgb):
        """Color quantization using optimized color palette generation."""
        # Reshape image to a list of pixels
        pixels = image_rgb.reshape(-1, 3)
        
        # Convert available colors to numpy array
        colors_array = np.array(self.available_colors)
        
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
        quantized = np.array([self.available_colors[idx] for idx in color_indices])
        
        # Reshape back to image dimensions
        quantized_image = quantized.reshape(image_rgb.shape)
        color_indices = color_indices.reshape(image_rgb.shape[:2])
        
        return quantized_image, color_indices
    
    def generate_paths(self, color_regions):
        """Generate painting paths for each color region."""
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
                        if area < 10:  # Skip tiny regions
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
                                self.resolution_mm * 2,  # Spacing between fill lines
                                pattern_type=self.fill_pattern,
                                angle=self.fill_angle
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

    def _calculate_paint_usage(self, color_indices):
        """Calculate estimated paint usage for each color."""
        # Count pixels for each color
        total_pixels = color_indices.size
        color_counts = {}
        
        for color_idx in range(len(self.available_colors)):
            pixel_count = np.sum(color_indices == color_idx)
            percentage = (pixel_count / total_pixels) * 100
            
            # Calculate estimated area in square mm
            pixel_area_mm2 = (self.wall_width / color_indices.shape[1]) * (self.wall_height / color_indices.shape[0])
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
        
        self.paint_usage = color_counts
        return color_counts

    def get_paint_usage_report(self):
        """Generate a human-readable paint usage report."""
        if not self.paint_usage:
            return "Paint usage has not been calculated yet. Process an image first."
        
        report = "Estimated Paint Usage:\n"
        report += "=====================\n"
        
        total_paint_ml = 0
        
        for color_idx, usage in self.paint_usage.items():
            if usage['pixels'] == 0:
                continue
                
            rgb = tuple(self.available_colors[color_idx])
            report += f"Color {color_idx} - RGB{rgb}:\n"
            report += f"  - Pixels: {usage['pixels']:,}\n"
            report += f"  - Coverage: {usage['percentage']:.2f}%\n"
            report += f"  - Area: {usage['area_mm2']/1000:.2f} cm²\n"
            report += f"  - Paint required: {usage['paint_ml']:.2f} ml"
            
            # Add warning if paint usage is high
            if usage['paint_ml'] > 100:
                report += " (High consumption: consider using a larger can for this color)"
            
            report += "\n"
            total_paint_ml += usage['paint_ml']
        
        report += f"\nTotal paint required: {total_paint_ml:.2f} ml\n"
        report += f"Estimated number of standard spray cans: {math.ceil(total_paint_ml / 400)}\n"
        report += "(Based on a standard 400ml spray can)"
        
        return report

    def find_color_regions(self, color_indices):
        """Find connected regions for each color."""
        if hasattr(self, 'segmentation_method') and self.segmentation_method == "slic":
            return self._segment_slic(color_indices)
        
        color_regions = {}
        num_colors = len(self.available_colors)
        
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
    
    def optimize_paths(self, all_paths):
        """Optimize paths to minimize travel distance."""
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
    
    def generate_instructions(self, all_paths):
        """Generate a sequence of painting instructions."""
        instructions = []
        
        # Start with a "home" instruction
        instructions.append({
            "type": "home",
            "message": "Moving to home position"
        })
        
        # For each color
        for color_idx, paths in all_paths.items():
            # Change to this color
            instructions.append({
                "type": "color",
                "index": color_idx,
                "rgb": self.available_colors[color_idx],
                "message": f"Changing to color {color_idx} - RGB{self.available_colors[color_idx]}"
            })
            
            # For each path with this color
            for path in paths:
                if not path:
                    continue
                    
                # Move to the first point with spray off
                first_point = path[0]
                instructions.append({
                    "type": "move",
                    "x": first_point[0],
                    "y": first_point[1],
                    "spray": False,
                    "message": f"Moving to ({first_point[0]}, {first_point[1]})"
                })
                
                # Turn spray on
                instructions.append({
                    "type": "spray",
                    "state": True,
                    "message": "Starting to spray"
                })
                
                # Follow the path with spray on
                for point in path[1:]:
                    instructions.append({
                        "type": "move",
                        "x": point[0],
                        "y": point[1],
                        "spray": True,
                        "message": f"Moving to ({point[0]}, {point[1]}) while spraying"
                    })
                
                # Turn spray off
                instructions.append({
                    "type": "spray",
                    "state": False,
                    "message": "Stopping spray"
                })
        
        # End with a "home" instruction
        instructions.append({
            "type": "home",
            "message": "Returning to home position"
        })
        
        return instructions
    
    def group_colors_by_robot_capacity(self, color_indices):
        """
        Group colors into batches that can be painted simultaneously by the robot.
        Colors are grouped by usage frequency and spatial proximity.
        """
        # Get color usage statistics
        color_stats = {}
        for color_idx in range(len(self.available_colors)):
            pixel_count = np.sum(color_indices == color_idx)
            if pixel_count > 0:
                color_stats[color_idx] = {
                    'count': int(pixel_count),
                    'percentage': (pixel_count / color_indices.size) * 100
                }
        
        # Sort colors by usage (most used first)
        sorted_colors = sorted(color_stats.keys(), key=lambda x: color_stats[x]['count'], reverse=True)
        
        # Calculate color proximity matrix (how close colors tend to be to each other)
        proximity_matrix = self._calculate_color_proximity(color_indices, sorted_colors)
        
        # Group colors into batches that fit robot capacity
        color_groups = []
        remaining_colors = sorted_colors.copy()
        
        while remaining_colors:
            current_group = []
            
            # Start with the most used remaining color
            if remaining_colors:
                most_used = remaining_colors[0]
                current_group.append(most_used)
                remaining_colors.remove(most_used)
            
            # Fill the group with colors that are most proximate to existing group colors
            while len(current_group) < self.robot_capacity and remaining_colors:
                best_score = float('-inf')
                best_color = None
                
                for color in remaining_colors:
                    # Calculate proximity score to current group
                    score = 0
                    for group_color in current_group:
                        score += proximity_matrix.get((min(color, group_color), max(color, group_color)), 0)
                    
                    # Also consider color usage
                    usage_weight = 0.3  # Weight for usage vs. proximity
                    score += usage_weight * color_stats[color]['percentage']
                    
                    if score > best_score:
                        best_score = score
                        best_color = color
                
                if best_color is not None:
                    current_group.append(best_color)
                    remaining_colors.remove(best_color)
                else:
                    break
            
            # Add the group if not empty
            if current_group:
                color_groups.append(current_group)
        
        print(f"Grouped {len(sorted_colors)} colors into {len(color_groups)} batches for the robot")
        for i, group in enumerate(color_groups):
            print(f"  Batch {i+1}: Colors {group}")
        
        # Store the optimized groups
        self.optimized_color_groups = color_groups
        return color_groups
    
    def _calculate_color_proximity(self, color_indices, color_list):
        """
        Calculate spatial proximity between colors in the image.
        Returns a dictionary with color pairs as keys and proximity scores as values.
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
    
    def generate_optimized_instructions(self, all_paths):
        """
        Generate painting instructions optimized for the robot's limited color capacity.
        Takes into account the need to visit the color change position when switching between color groups.
        """
        if not self.optimized_color_groups:
            print("Warning: Color groups not set. Running automatic color grouping.")
            # This shouldn't happen normally, but just in case
            # We'll need some dummy color indices for this
            dummy_indices = np.zeros((100, 100), dtype=int)
            for color_idx in all_paths.keys():
                dummy_indices[color_idx, color_idx] = 1
            self.group_colors_by_robot_capacity(dummy_indices)
        
        instructions = []
        
        # Start with a "home" instruction
        instructions.append({
            "type": "home",
            "message": "Moving to home position"
        })
        
        # For each color group
        for group_idx, color_group in enumerate(self.optimized_color_groups):
            # Move to color change position
            instructions.append({
                "type": "move",
                "x": self.color_change_position[0],
                "y": self.color_change_position[1],
                "spray": False,
                "message": f"Moving to color change position to load color group {group_idx+1}"
            })
            
            # Load colors for this group
            instructions.append({
                "type": "load_colors",
                "colors": [{"index": color_idx, "rgb": self.available_colors[color_idx]} for color_idx in color_group],
                "message": f"Loading colors for group {group_idx+1}: {color_group}"
            })
            
            # Now paint with each color in this group
            for color_idx in color_group:
                # Skip if color has no paths
                if color_idx not in all_paths:
                    continue
                    
                # Change to this color
                instructions.append({
                    "type": "color",
                    "index": color_idx,
                    "rgb": self.available_colors[color_idx],
                    "message": f"Changing to color {color_idx} - RGB{self.available_colors[color_idx]}"
                })
                
                # For each path with this color
                for path in all_paths[color_idx]:
                    if not path:
                        continue
                        
                    # Move to the first point with spray off
                    first_point = path[0]
                    instructions.append({
                        "type": "move",
                        "x": first_point[0],
                        "y": first_point[1],
                        "spray": False,
                        "message": f"Moving to ({first_point[0]}, {first_point[1]})"
                    })
                    
                    # Turn spray on
                    instructions.append({
                        "type": "spray",
                        "state": True,
                        "message": "Starting to spray"
                    })
                    
                    # Follow the path with spray on
                    for point in path[1:]:
                        instructions.append({
                            "type": "move",
                            "x": point[0],
                            "y": point[1],
                            "spray": True,
                            "message": f"Moving to ({point[0]}, {point[1]}) while spraying"
                        })
                    
                    # Turn spray off
                    instructions.append({
                        "type": "spray",
                        "state": False,
                        "message": "Stopping spray"
                    })
        
        # End with a "home" instruction
        instructions.append({
            "type": "home",
            "message": "Returning to home position"
        })
        
        return instructions

    def select_optimal_colors(self, image_rgb):
        """
        Select the optimal colors for the mural based on the image content.
        Returns a list of RGB color tuples.
        """
        print(f"Selecting optimal {self.max_colors} colors for the mural...")
        
        if self.color_selection == "auto":
            # Automatically select colors based on image content
            return self._select_colors_from_image(image_rgb)
        else:
            # Use the default colors provided
            print("Using manually specified colors")
            return self.available_colors

    def _select_colors_from_image(self, image_rgb):
        """
        Analyze the image and select the best colors from the MTN94 spray paint database.
        """
        # Load the MTN94 color database
        color_db_path = os.path.join(os.path.dirname(__file__), "data", "mtn94_colors.json")
        
        try:
            with open(color_db_path, 'r') as f:
                color_db = json.load(f)
                color_data = color_db["colors"]
                print(f"Loaded {len(color_data)} colors from MTN94 database for color selection")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading color database: {e}")
            print("Using already loaded colors instead")
            return self.available_colors[:self.max_colors]  # Return a subset of already loaded colors
            
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
        # Use self.max_colors directly instead of capping at 30
        num_clusters = self.max_colors
        
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
        selected_colors = selected_colors[:self.max_colors]
        
        print(f"Final selection: {len(selected_colors)} colors")
        return selected_colors

    def process_image(self, image_path, output_path=None):
        """Process an image and generate painting instructions."""
        # Ensure painting folder exists
        painting_folder = os.path.join(os.path.dirname(__file__), "painting")
        os.makedirs(painting_folder, exist_ok=True)
        
        # Load and resize image
        print("Step 1: Loading image...")
        image = self.load_image(image_path)
        if image is None or (isinstance(image, tuple) and any(i is None for i in image)):
            raise ValueError(f"Failed to load image from {image_path}. Please check the file path.")
        
        # If using auto color selection, analyze the image and select optimal colors first
        if self.color_selection == "auto":
            print("Step 2: Selecting optimal colors...")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            optimal_colors = self.select_optimal_colors(image_rgb)
            # Important: Update the available colors to ONLY use the optimal selection
            self.available_colors = optimal_colors
        
        # Quantize image to available colors
        print("Step 3: Matching to available colors...")
        quantized_image, color_indices = self.quantize_to_available_colors(image)
        
        # Print paint usage report
        print("\nPaint Usage Report:")
        print(self.get_paint_usage_report())
        print()
        
        # Find color regions
        print("Step 4: Finding color regions...")
        color_regions = self.find_color_regions(color_indices)
        
        # Generate paths
        print("Step 5: Generating paths...")
        paths = self.generate_paths(color_regions)
        
        # Optimize paths
        print("Step 6: Optimizing paths...")
        optimized_paths = self.optimize_paths(paths)
        
        # Group colors by robot capacity
        print("Step 7: Grouping colors by robot capacity...")
        self.group_colors_by_robot_capacity(color_indices)
        
        # Generate optimized instructions
        print("Step 8: Generating optimized instructions...")
        instructions = self.generate_optimized_instructions(optimized_paths)
        
        # Save instructions to JSON if output_path is provided
        if output_path:
            # Always save inside the painting folder, regardless of user input
            painting_folder = os.path.join(os.path.dirname(__file__), "painting")
            os.makedirs(painting_folder, exist_ok=True)

            output_filename = os.path.basename(output_path)
            output_path = os.path.join(painting_folder, output_filename)
            output_path = os.path.normpath(output_path)
            
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                elif isinstance(obj, tuple):
                    return [convert_to_serializable(i) for i in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                else:
                    return obj

            serializable_data = convert_to_serializable({
                "wall_width": self.wall_width,
                "wall_height": self.wall_height,
                "colors": self.available_colors,
                "instructions": instructions,
                "paint_usage": self.paint_usage
            })

            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            print(f"Instructions saved to {output_path}")
        
        # Also save the quantized image for reference
        painting_folder = os.path.join(os.path.dirname(__file__), "painting")
        os.makedirs(painting_folder, exist_ok=True)

        preview_path = os.path.join(painting_folder, "quantized_preview.jpg")
        cv2.imwrite(preview_path, quantized_image)
        return instructions, quantized_image
    def _quantize_adaptive_kmeans(self, image_rgb):
        """Adaptive K-means quantization with automatic cluster count selection."""
        from sklearn.cluster import KMeans
        from skimage.color import rgb2lab
        import numpy as np

        pixels_rgb = image_rgb.reshape(-1, 3).astype(np.float32) / 255.0
        pixels_lab = rgb2lab(pixels_rgb.reshape(-1,1,3)).reshape(-1,3)

        best_k = 4
        best_inertia = None
        max_k = min(self.max_colors, 30)
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
        available_rgb = np.array(self.available_colors).astype(np.float32) / 255.0
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

        quantized_rgb = np.array([self.available_colors[idx] for idx in color_indices])
        quantized_image = quantized_rgb.reshape(image_rgb.shape)
        color_indices = color_indices.reshape(image_rgb.shape[:2])

        return quantized_image, color_indices

    def _segment_slic(self, color_indices):
        """Segment image using SLIC superpixels and group by dominant color."""
        from skimage.segmentation import slic
        from skimage.color import label2rgb
        import numpy as np

        # Reconstruct quantized RGB image from color indices
        h, w = color_indices.shape
        quantized_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, color in enumerate(self.available_colors):
            mask = (color_indices == idx)
            quantized_rgb[mask] = color

        # Run SLIC
        segments = slic(quantized_rgb, n_segments=500, compactness=10, start_label=1, convert2lab=True)

        # Group superpixels by dominant color
        color_regions = {i: [] for i in range(len(self.available_colors))}
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
        print(f"Preview image saved as '{preview_path}'")
        
        # Save paint usage report to a text file
        painting_folder = os.path.join(os.path.dirname(__file__), "painting")
        os.makedirs(painting_folder, exist_ok=True)

        usage_report_path = os.path.join(painting_folder, "paint_usage_report.txt")
        with open(usage_report_path, 'w') as f:
            f.write(self.get_paint_usage_report())
        print(f"Paint usage report saved as '{usage_report_path}'")
        
        return instructions, quantized_image

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert an image to mural painting instructions')
    parser.add_argument('--config', '-c', default='config.json', 
                        help='Path to the configuration file')
    parser.add_argument('--image_path', help='Path to the input image (overrides config file)')
    parser.add_argument('--output', '-o', help='Output JSON file path for instructions (overrides config file)')
    parser.add_argument('--width', '-w', type=int, help='Wall/canvas width in mm (overrides config file)')
    parser.add_argument('--height', '-ht', type=int, help='Wall/canvas height in mm (overrides config file)')
    parser.add_argument('--resolution', '-r', type=float, help='Painting resolution in mm (overrides config file)')
    parser.add_argument('--quantization', '-q', choices=['euclidean', 'kmeans', 'color_palette'],
                       help='Color quantization method (overrides config file)')
    parser.add_argument('--dithering', '-d', choices=['none', 'floyd-steinberg', 'jarvis', 'stucki'],
                       help='Dithering method (overrides config file)')
    parser.add_argument('--dithering_strength', '-ds', type=float, help='Strength of dithering effect (overrides config file)')
    parser.add_argument('--max_colors', '-mc', type=int, help='Maximum number of colors to use (overrides config file)')
    parser.add_argument('--robot_capacity', '-rc', type=int, 
                       help='Number of colors the robot can hold simultaneously (overrides config file)')
    parser.add_argument('--color_selection', '-cs', choices=['auto', 'manual'],
                       help='Color selection method (overrides config file)')
    parser.add_argument('--fill_pattern', '-fp', choices=['zigzag', 'concentric', 'spiral', 'dots'],
                       help='Fill pattern for regions (overrides config file)')
    parser.add_argument('--fill_angle', '-fa', type=float, help='Angle for directional fill patterns (overrides config file)')
    
    args = parser.parse_args()
    
    # Load config file
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        img_config = config['image_processing']
        hardware_config = config.get('hardware', {})
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading config file: {e}")
        img_config = {}
        hardware_config = {}
    
    # Get parameters, prioritizing command line arguments over config file
    image_path = args.image_path or img_config.get('input_image')
    output_path = args.output or img_config.get('output_instructions', 'painting_instructions.json')
    wall_width = args.width or img_config.get('wall_width', 2000)
    wall_height = args.height or img_config.get('wall_height', 1500)
    resolution_mm = args.resolution or img_config.get('resolution_mm', 5.0)
    quantization_method = args.quantization or img_config.get('quantization_method', 'euclidean')
    dithering = args.dithering or img_config.get('dithering', 'none')
    dithering_strength = args.dithering_strength or img_config.get('dithering_strength', 1.0)
    max_colors = args.max_colors or img_config.get('max_colors', 30)
    robot_capacity = args.robot_capacity or img_config.get('robot_capacity', 6)
    color_selection = args.color_selection or img_config.get('color_selection', 'auto')
    fill_pattern = args.fill_pattern or img_config.get('fill_pattern', 'zigzag')
    fill_angle = args.fill_angle or img_config.get('fill_angle', 0)
    
    # Get color change position from hardware config
    color_change_position = hardware_config.get('color_change_position', [0, wall_height])
    
    # Check if image path is provided
    if not image_path:
        print("Error: No input image specified. Please provide it in the config file or as a command line argument.")
        exit(1)
    
    # Create instruction generator
    generator = MuralInstructionGenerator(
        wall_width=wall_width,
        wall_height=wall_height,
        available_colors=None,  # Set to None to use MTN94 colors by default
        resolution_mm=resolution_mm,
        quantization_method=quantization_method,
        dithering=dithering,
        dithering_strength=dithering_strength,
        max_colors=max_colors,
        robot_capacity=robot_capacity,
        color_selection=color_selection,
        fill_pattern=fill_pattern,
        fill_angle=fill_angle
    )
    
    # Set the color change position
    generator.color_change_position = color_change_position
    
    # If using auto color selection, load and analyze the image first to select optimal colors
    if color_selection == "auto":
        print("Loading image for color analysis...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            exit(1)
        
        # Convert to RGB for color analysis
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Select optimal colors
        optimal_colors = generator.select_optimal_colors(image_rgb)
        
        # Update available colors
        generator.available_colors = optimal_colors
    
    # Process image and generate instructions
    instructions, _ = generator.process_image(image_path, output_path)
    
    print(f"Generated {len(instructions)} painting instructions")