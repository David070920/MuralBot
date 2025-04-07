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

class MuralInstructionGenerator:
    def __init__(self, wall_width, wall_height, available_colors, resolution_mm=5, quantization_method="euclidean", dithering=False):
        """
        Initialize the mural painting instruction generator.
        
        Args:
            wall_width: Width of the wall/canvas in mm
            wall_height: Height of the wall/canvas in mm
            available_colors: List of (R,G,B) tuples representing available spray colors
            resolution_mm: Resolution in mm for path planning
            quantization_method: Method for color quantization ("euclidean", "kmeans", or "color_palette")
            dithering: Whether to apply Floyd-Steinberg dithering to improve color appearance
        """
        self.wall_width = wall_width
        self.wall_height = wall_height
        self.available_colors = available_colors
        self.resolution_mm = resolution_mm
        self.quantization_method = quantization_method
        self.dithering = dithering
        self.paint_usage = None  # Will store paint usage estimates per color
        
    def load_image(self, image_path):
        """Load and resize image to fit the wall dimensions."""
        # Load image
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
        elif self.quantization_method == "color_palette":
            quantized_image, color_indices = self._quantize_color_palette(image_rgb)
        else:
            print(f"Warning: Unknown quantization method '{self.quantization_method}', falling back to euclidean")
            quantized_image, color_indices = self._quantize_euclidean(image_rgb)
        
        # Apply dithering if enabled
        if self.dithering:
            print("  Applying Floyd-Steinberg dithering...")
            quantized_image, color_indices = self._apply_dithering(image_rgb, quantized_image, color_indices)
            
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
    
    def _apply_dithering(self, original_image, quantized_image, color_indices):
        """Apply Floyd-Steinberg dithering to improve visual appearance."""
        # Make a copy of the original image in float format for error diffusion
        h, w = original_image.shape[:2]
        dither_image = original_image.astype(np.float32)
        dithered_indices = color_indices.copy()
        
        # Available colors as numpy array
        colors_array = np.array(self.available_colors)
        
        # Process the image pixel by pixel
        with tqdm(total=h, desc="Applying dithering") as pbar:
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
                    error = old_pixel - nearest_color
                    
                    # Distribute error to neighboring pixels (Floyd-Steinberg)
                    if x + 1 < w:
                        dither_image[y, x + 1] += error * 7 / 16
                    if y + 1 < h:
                        if x > 0:
                            dither_image[y + 1, x - 1] += error * 3 / 16
                        dither_image[y + 1, x] += error * 5 / 16
                        if x + 1 < w:
                            dither_image[y + 1, x + 1] += error * 1 / 16
                
                pbar.update(1)
        
        # Convert dithered image back to uint8
        dithered_image = np.array([self.available_colors[idx] for idx in dithered_indices.flatten()])
        dithered_image = dithered_image.reshape(original_image.shape)
        
        return dithered_image, dithered_indices

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
        color_regions = {}
        num_colors = len(self.available_colors)
        
        # Create progress bar for color regions
        with tqdm(total=num_colors, desc="Finding regions") as pbar:
            for color_idx in range(num_colors):
                # Create binary mask for this color
                mask = (color_indices == color_idx).astype(np.uint8)
                
                # Skip if no pixels of this color
                if not np.any(mask):
                    pbar.update(1)
                    continue
                    
                # Find connected components
                structure = np.ones((3, 3), dtype=np.int32)
                labeled, num_features = label(mask, structure)
                
                # Store regions for this color
                regions = []
                for feature_idx in range(1, num_features + 1):
                    feature_mask = (labeled == feature_idx).astype(np.uint8)
                    regions.append(feature_mask)
                    
                color_regions[color_idx] = regions
                pbar.update(1)
            
        return color_regions
    
    def generate_paths(self, color_regions):
        """Generate painting paths for each color region."""
        all_paths = {}
        total_regions = sum(len(regions) for regions in color_regions.values())
        
        # Create progress bar for path generation
        with tqdm(total=total_regions, desc="Generating paths") as pbar:
            for color_idx, regions in color_regions.items():
                color_paths = []
                
                for region_mask in regions:
                    # Find contours
                    contours, _ = cv2.findContours(
                        region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for contour in contours:
                        # Check if contour is large enough to paint
                        area = cv2.contourArea(contour)
                        if area < 100:  # Skip tiny regions
                            continue
                            
                        # Simplify contour to reduce number of points
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Get contour points
                        points = [tuple(map(int, point[0])) for point in approx]
                        
                        # If contour is closed, make sure the first and last points connect
                        if len(points) > 2:
                            points.append(points[0])
                            
                        # For larger areas, add fill pattern
                        if area > 500:
                            fill_paths = self.generate_fill_pattern(contour, self.resolution_mm)
                            color_paths.extend(fill_paths)
                        
                        color_paths.append(points)
                    
                    pbar.update(1)
                
                if color_paths:
                    all_paths[color_idx] = color_paths
                    
            return all_paths
    
    def generate_fill_pattern(self, contour, spacing):
        """Generate a fill pattern inside a contour."""
        # Get contour boundaries
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create a mask for the contour
        mask = np.zeros((y + h + 10, x + w + 10), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1, offset=(5, 5))
        
        # Generate horizontal lines
        fill_paths = []
        for row in range(y + int(spacing / 2), y + h, int(spacing)):
            line_points = []
            inside = False
            start_point = None
            
            for col in range(x, x + w + 1):
                # Check if point is inside contour
                is_inside = mask[row, col] > 0
                
                if is_inside and not inside:
                    # Entering contour
                    inside = True
                    start_point = (col, row)
                elif not is_inside and inside:
                    # Exiting contour
                    inside = False
                    if start_point:
                        line_points = [start_point, (col - 1, row)]
                        fill_paths.append(line_points)
            
            # Handle case where line ends while still inside contour
            if inside and start_point:
                line_points = [start_point, (x + w, row)]
                fill_paths.append(line_points)
        
        return fill_paths
    
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
    
    def process_image(self, image_path, output_path=None):
        """Process an image and generate painting instructions."""
        # Load and resize image
        print("Step 1: Loading image...")
        image = self.load_image(image_path)
        
        # Quantize image to available colors
        print("Step 2: Matching to available colors...")
        quantized_image, color_indices = self.quantize_to_available_colors(image)
        
        # Print paint usage report
        print("\nPaint Usage Report:")
        print(self.get_paint_usage_report())
        print()
        
        # Find color regions
        print("Step 3: Finding color regions...")
        color_regions = self.find_color_regions(color_indices)
        
        # Generate paths
        print("Step 4: Generating paths...")
        paths = self.generate_paths(color_regions)
        
        # Optimize paths
        print("Step 5: Optimizing paths...")
        optimized_paths = self.optimize_paths(paths)
        
        # Generate instructions
        print("Step 6: Generating instructions...")
        instructions = self.generate_instructions(optimized_paths)
        
        # Save instructions to JSON if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump({
                    "wall_width": self.wall_width,
                    "wall_height": self.wall_height,
                    "colors": self.available_colors,
                    "instructions": instructions,
                    "paint_usage": self.paint_usage
                }, f, indent=2)
            print(f"Instructions saved to {output_path}")
        
        # Also save the quantized image for reference
        cv2.imwrite("quantized_preview.jpg", quantized_image)
        print("Preview image saved as 'quantized_preview.jpg'")
        
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
    parser.add_argument('--dithering', '-d', action='store_true',
                       help='Enable dithering for better visual appearance')
    
    args = parser.parse_args()
    
    # Load config file
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        img_config = config['image_processing']
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading config file: {e}")
        img_config = {}
    
    # Get parameters, prioritizing command line arguments over config file
    image_path = args.image_path or img_config.get('input_image')
    output_path = args.output or img_config.get('output_instructions', 'painting_instructions.json')
    wall_width = args.width or img_config.get('wall_width', 2000)
    wall_height = args.height or img_config.get('wall_height', 1500)
    resolution_mm = args.resolution or img_config.get('resolution_mm', 5.0)
    quantization_method = args.quantization or img_config.get('quantization_method', 'euclidean')
    dithering = args.dithering or img_config.get('dithering', False)
    
    # Define available colors from config or defaults
    available_colors = img_config.get('available_colors', [
        (0, 0, 0),       # Black
        (255, 0, 0),     # Red
        (0, 0, 255),     # Blue
        (0, 255, 0),     # Green
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 255, 255)  # White
    ])
    
    # Check if image path is provided
    if not image_path:
        print("Error: No input image specified. Please provide it in the config file or as a command line argument.")
        exit(1)
    
    # Create instruction generator
    generator = MuralInstructionGenerator(
        wall_width=wall_width,
        wall_height=wall_height,
        available_colors=available_colors,
        resolution_mm=resolution_mm,
        quantization_method=quantization_method,
        dithering=dithering
    )
    
    # Process image and generate instructions
    instructions, _ = generator.process_image(image_path, output_path)
    
    print(f"Generated {len(instructions)} painting instructions")