"""
Main instruction generator class for MuralBot.

This module contains the core MuralInstructionGenerator class which coordinates
the entire image processing and instruction generation pipeline.
"""

import cv2
import numpy as np
import json
import math
import os
from tqdm import tqdm
import time

from .color_quantization import quantize_euclidean, quantize_kmeans, quantize_adaptive_kmeans, quantize_color_palette
from .path_planning import generate_paths, optimize_paths, generate_dynamic_color_instructions
from .segmentation import find_color_regions, segment_slic
from .paint_usage import calculate_paint_usage
from .utils import load_mtn94_colors, load_image

class MuralInstructionGenerator:
    """
    Main class for generating mural painting instructions from an image.
    
    This class coordinates the entire pipeline from image loading to instruction generation.
    """
    
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
        self.available_colors = load_mtn94_colors() if available_colors is None else available_colors

    def quantize_to_available_colors(self, image):
        """Quantize image to only use available spray can colors using selected method."""
        # Convert image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"  Processing color quantization using {self.quantization_method} method...")
        
        if self.quantization_method == "euclidean":
            quantized_image, color_indices = quantize_euclidean(image_rgb, self.available_colors)
        elif self.quantization_method == "kmeans":
            quantized_image, color_indices = quantize_kmeans(image_rgb, self.available_colors)
        elif self.quantization_method == "adaptive_kmeans":
            quantized_image, color_indices = quantize_adaptive_kmeans(image_rgb, self.available_colors, self.max_colors)
        elif self.quantization_method == "color_palette":
            quantized_image, color_indices = quantize_color_palette(image_rgb, self.available_colors)
        else:
            print(f"Warning: Unknown quantization method '{self.quantization_method}', falling back to euclidean")
            quantized_image, color_indices = quantize_euclidean(image_rgb, self.available_colors)
        
        # Apply dithering if enabled
        if self.dithering != "none":
            from algorithms.dithering import apply_dithering
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
        self.paint_usage = calculate_paint_usage(color_indices, self.available_colors, self.wall_width, self.wall_height)
        
        return cv2.cvtColor(quantized_image.astype(np.uint8), cv2.COLOR_RGB2BGR), color_indices

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

    def select_optimal_colors(self, image_rgb):
        """
        Select the optimal colors for the mural based on the image content.
        Returns a list of RGB color tuples.
        """
        from .color_selection import select_colors_from_image
        
        print(f"Selecting optimal {self.max_colors} colors for the mural...")
        
        if self.color_selection == "auto":
            # Automatically select colors based on image content
            return select_colors_from_image(image_rgb, self.max_colors)
        else:
            # Use the default colors provided
            print("Using manually specified colors")
            return self.available_colors

    def process_image(self, image_path, output_path=None):
        """Process an image and generate painting instructions."""
        # Ensure painting folder exists
        painting_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "painting")
        os.makedirs(painting_folder, exist_ok=True)
        
        # Load and resize image
        print("Step 1: Loading image...")
        image = load_image(image_path, self.wall_width, self.wall_height)
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
        if self.segmentation_method == "slic":
            color_regions = segment_slic(color_indices, self.available_colors)
        else:
            color_regions = find_color_regions(color_indices, len(self.available_colors))
        
        # Generate paths
        print("Step 5: Generating paths...")
        paths = generate_paths(color_regions, self.resolution_mm, self.fill_pattern, self.fill_angle)
        
        # Optimize paths
        print("Step 6: Optimizing paths...")
        optimized_paths = optimize_paths(paths)
        
        # Generate dynamic color switching instructions
        print("Step 7: Generating dynamic color switching instructions...")
        instructions = generate_dynamic_color_instructions(optimized_paths, self.available_colors, self.color_change_position)
        
        # Save instructions to JSON if output_path is provided
        if output_path:
            # Always save inside the painting folder, regardless of user input
            painting_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "painting")
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
        preview_path = os.path.join(painting_folder, "quantized_preview.jpg")
        cv2.imwrite(preview_path, quantized_image)
        print(f"Preview image saved as '{preview_path}'")
        
        # Save paint usage report to a text file
        painting_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "painting")
        os.makedirs(painting_folder, exist_ok=True)

        usage_report_path = os.path.join(painting_folder, "paint_usage_report.txt")
        with open(usage_report_path, 'w') as f:
            f.write(self.get_paint_usage_report())
        print(f"Paint usage report saved as '{usage_report_path}'")
        
        return instructions, quantized_image
