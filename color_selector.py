"""
MuralBot Color Selector

Advanced color selection algorithms for MuralBot to choose optimal
spray paint colors based on image analysis.
"""

import os
import json
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from skimage import color as skcolor
import colorsys

class ColorSelector:
    """
    Advanced color selection for the MuralBot system.
    Provides multiple strategies for selecting optimal colors from a spray paint database.
    """
    
    def __init__(self, color_db_path=None, max_colors=30):
        """
        Initialize the color selector.
        
        Args:
            color_db_path (str): Path to color database JSON file. If None, uses default MTN94 database.
            max_colors (int): Maximum number of colors to select.
        """
        self.max_colors = max_colors
        self.color_db_path = color_db_path
        
        if color_db_path is None:
            # Use default path relative to this file
            self.color_db_path = os.path.join(os.path.dirname(__file__), "data", "mtn94_colors.json")
        
        # Load color database
        self.color_data = self._load_color_database()
        self.available_colors_rgb = self._extract_rgb_colors()
        self.available_colors_lab = self._convert_to_lab(self.available_colors_rgb)

    def _load_color_database(self):
        """Load colors from the color database file."""
        try:
            with open(self.color_db_path, 'r') as f:
                color_db = json.load(f)
                color_data = color_db["colors"]
                print(f"Loaded {len(color_data)} colors from database")
                return color_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading color database: {e}")
            print("Using default colors instead")
            # Return basic default colors as fallback
            default_colors = [
                {"name": "Black", "code": "BLK", "rgb": [0, 0, 0]},
                {"name": "White", "code": "WHT", "rgb": [255, 255, 255]},
                {"name": "Red", "code": "RED", "rgb": [255, 0, 0]},
                {"name": "Green", "code": "GRN", "rgb": [0, 255, 0]},
                {"name": "Blue", "code": "BLU", "rgb": [0, 0, 255]},
                {"name": "Yellow", "code": "YLW", "rgb": [255, 255, 0]},
                {"name": "Magenta", "code": "MAG", "rgb": [255, 0, 255]},
                {"name": "Cyan", "code": "CYN", "rgb": [0, 255, 255]}
            ]
            return default_colors

    def _extract_rgb_colors(self):
        """Extract RGB values from the color database."""
        available_colors_rgb = []
        for color in self.color_data:
            rgb = tuple(int(c) for c in color["rgb"])
            available_colors_rgb.append(rgb)
        return available_colors_rgb

    def _convert_to_lab(self, rgb_colors):
        """Convert RGB colors to LAB color space for perceptual distance calculations."""
        # Convert to numpy array with correct shape for skimage
        rgb_array = np.array(rgb_colors, dtype=np.float64) / 255.0
        # Convert to LAB color space
        lab_colors = skcolor.rgb2lab(rgb_array.reshape(1, -1, 3)).reshape(-1, 3)
        return lab_colors

    def _rgb_to_lab(self, rgb_color):
        """Convert a single RGB color to LAB."""
        rgb = np.array([rgb_color], dtype=np.float64) / 255.0
        return skcolor.rgb2lab(rgb.reshape(1, 1, 3)).reshape(-1)

    def _rgb_to_hsv(self, rgb_color):
        """Convert a single RGB color to HSV."""
        r, g, b = rgb_color
        return colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    
    def _calculate_color_histogram(self, image_rgb, bins=16):
        """
        Calculate color histogram in HSV space to better represent color distribution.
        Returns histogram and bin edges for each channel.
        """
        # Convert to HSV for better color analysis
        hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Create flattened array of HSV values
        h, w = image_rgb.shape[:2]
        hsv_flat = hsv_img.reshape(-1, 3)
        
        # Calculate histograms for each channel
        h_hist, h_edges = np.histogram(hsv_flat[:, 0], bins=bins, range=(0, 180))
        s_hist, s_edges = np.histogram(hsv_flat[:, 1], bins=bins, range=(0, 256))
        v_hist, v_edges = np.histogram(hsv_flat[:, 2], bins=bins, range=(0, 256))
        
        # Normalize histograms
        h_hist = h_hist.astype(np.float32) / np.sum(h_hist)
        s_hist = s_hist.astype(np.float32) / np.sum(s_hist)
        v_hist = v_hist.astype(np.float32) / np.sum(v_hist)
        
        return (h_hist, s_hist, v_hist), (h_edges, s_edges, v_edges)

    def select_colors_diverse_spectrum(self, image_rgb):
        """
        Select colors ensuring a diverse range of hues, saturations, and values.
        This method focuses on providing a balanced palette across the color spectrum.
        
        Args:
            image_rgb: RGB image as numpy array (height, width, 3)
            
        Returns:
            List of selected RGB color tuples
        """
        print(f"Selecting {self.max_colors} colors using diverse spectrum method...")
        
        # First, analyze the image's color distribution in HSV space
        histograms, bin_edges = self._calculate_color_histogram(image_rgb, bins=24)
        h_hist, s_hist, v_hist = histograms
        
        # Convert available colors to HSV for better hue comparison
        available_colors_hsv = [self._rgb_to_hsv(color) for color in self.available_colors_rgb]
        
        # Calculate importance of each hue range in the image
        h_importance = np.where(h_hist > 0.01, h_hist, 0)  # Ignore very minor hues
        h_importance = h_importance / np.sum(h_importance) if np.sum(h_importance) > 0 else h_importance
        
        # Calculate weights for the saturation and value dimensions
        s_importance = np.where(s_hist > 0.01, s_hist, 0)
        s_importance = s_importance / np.sum(s_importance) if np.sum(s_importance) > 0 else s_importance
        
        v_importance = np.where(v_hist > 0.01, v_hist, 0)
        v_importance = v_importance / np.sum(v_importance) if np.sum(v_importance) > 0 else v_importance
        
        # Target number of colors per hue segment (adaptive based on histogram)
        # We want to make sure we get a good distribution of colors across the hue spectrum
        hue_segments = 8  # Divide the color wheel into 8 segments
        seg_size = 1.0 / hue_segments
        
        # Calculate the importance of each hue segment
        hue_segment_importance = np.zeros(hue_segments)
        for i in range(hue_segments):
            segment_start = i * seg_size
            segment_end = (i + 1) * seg_size
            
            # Find histogram bins that fall in this segment
            bin_width = 1.0 / len(h_hist)
            bin_indices = [j for j in range(len(h_hist)) 
                          if segment_start <= (j * bin_width) < segment_end]
            
            # Sum the importance of these bins
            for j in bin_indices:
                hue_segment_importance[i] += h_hist[j]
        
        # Normalize hue segment importance
        if np.sum(hue_segment_importance) > 0:
            hue_segment_importance = hue_segment_importance / np.sum(hue_segment_importance)
        
        # Calculate target colors per segment (minimum 1 if segment is used in image)
        colors_per_segment = np.zeros(hue_segments, dtype=int)
        remaining_colors = self.max_colors - 2  # Reserve for black and white
        
        # Assign at least one color to segments with importance > threshold
        min_importance_threshold = 0.05
        important_segments = np.where(hue_segment_importance > min_importance_threshold)[0]
        
        # Ensure at least one color from important segments
        for segment in important_segments:
            colors_per_segment[segment] = 1
            remaining_colors -= 1
        
        # Distribute remaining colors proportionally to segment importance
        if remaining_colors > 0 and np.sum(hue_segment_importance) > 0:
            # Additional colors based on importance
            additional_colors = np.round(hue_segment_importance * remaining_colors).astype(int)
            
            # Adjust to match exactly the remaining count
            while np.sum(additional_colors) > remaining_colors:
                # Remove one from the least important segment that has colors
                idx = np.where(additional_colors > 0)[0]
                if len(idx) > 0:
                    least_important = idx[np.argmin(hue_segment_importance[idx])]
                    additional_colors[least_important] -= 1
            
            while np.sum(additional_colors) < remaining_colors:
                # Add one to the most important segment
                most_important = np.argmax(hue_segment_importance)
                additional_colors[most_important] += 1
            
            colors_per_segment += additional_colors
        
        # Select black and white first
        selected_colors = []
        selected_indices = []
        
        black_idx = self._find_black_index()
        white_idx = self._find_white_index()
        
        if black_idx is not None:
            selected_indices.append(black_idx)
            selected_colors.append(self.available_colors_rgb[black_idx])
            print(f"Added black: {self.color_data[black_idx]['name']} - RGB{self.available_colors_rgb[black_idx]}")
        
        if white_idx is not None:
            selected_indices.append(white_idx)
            selected_colors.append(self.available_colors_rgb[white_idx])
            print(f"Added white: {self.color_data[white_idx]['name']} - RGB{self.available_colors_rgb[white_idx]}")
        
        # For each hue segment, select the best colors
        available_indices = [i for i in range(len(self.available_colors_rgb)) 
                            if i not in selected_indices]
        
        print("Selecting diverse color spectrum...")
        for segment in range(hue_segments):
            # Skip if no colors allocated to this segment
            if colors_per_segment[segment] == 0:
                continue
                
            segment_start = segment * seg_size
            segment_end = (segment + 1) * seg_size
            
            # Find available colors in this hue segment
            segment_colors = []
            for i in available_indices:
                h, s, v = available_colors_hsv[i]
                # Check if color's hue falls in this segment
                # Handle the circular nature of hue
                if segment == hue_segments - 1:  # Last segment wraps around
                    if h >= segment_start or h < segment_end:
                        segment_colors.append(i)
                else:
                    if segment_start <= h < segment_end:
                        segment_colors.append(i)
            
            # If not enough colors in this segment, borrow from adjacent segments
            if len(segment_colors) < colors_per_segment[segment]:
                # Expand search to nearby segments
                expanded_colors = []
                expand_range = 0.1  # Look 10% beyond segment boundaries
                
                for i in available_indices:
                    if i not in segment_colors:
                        h, s, v = available_colors_hsv[i]
                        # Check expanded range
                        if ((segment_start - expand_range) % 1.0 <= h < (segment_end + expand_range) % 1.0):
                            expanded_colors.append(i)
                
                # Add expanded colors until we have enough
                segment_colors.extend(expanded_colors)
            
            # If we still don't have enough colors, just use any available colors
            if len(segment_colors) < colors_per_segment[segment]:
                remaining_needed = colors_per_segment[segment] - len(segment_colors)
                for i in available_indices:
                    if i not in segment_colors:
                        segment_colors.append(i)
                        remaining_needed -= 1
                        if remaining_needed <= 0:
                            break
            
            # Score colors by saturation and value distribution
            color_scores = []
            for i in segment_colors:
                h, s, v = available_colors_hsv[i]
                
                # Find which bins this color falls into
                s_bin = min(int(s * len(s_hist)), len(s_hist) - 1)
                v_bin = min(int(v * len(v_hist)), len(v_hist) - 1)
                
                # Score based on histogram importance
                s_score = s_importance[s_bin] if s_bin < len(s_importance) else 0
                v_score = v_importance[v_bin] if v_bin < len(v_importance) else 0
                
                # Combine scores - give more weight to saturation
                combined_score = 0.6 * s_score + 0.4 * v_score
                
                # Boost score for colors with medium to high saturation (more vibrant)
                if 0.3 < s < 0.9:
                    combined_score *= 1.5
                
                color_scores.append((i, combined_score))
            
            # Sort by score and select the best ones
            color_scores.sort(key=lambda x: x[1], reverse=True)
            selected_from_segment = 0
            
            for i, score in color_scores:
                if selected_from_segment >= colors_per_segment[segment]:
                    break
                    
                if i not in selected_indices:
                    selected_indices.append(i)
                    selected_colors.append(self.available_colors_rgb[i])
                    available_indices.remove(i)
                    selected_from_segment += 1
                    
                    # Print the selected color
                    color_name = self.color_data[i]["name"]
                    code = self.color_data[i]["code"] if "code" in self.color_data[i] else ""
                    rgb = self.available_colors_rgb[i]
                    h, s, v = available_colors_hsv[i]
                    print(f"Selected: {color_name} ({code}) - RGB{rgb} - HSV({h:.2f}, {s:.2f}, {v:.2f})")
        
        # If we still don't have enough colors, add the most vibrant remaining colors
        while len(selected_colors) < self.max_colors and available_indices:
            # Find the most vibrant remaining colors (high saturation, medium-high value)
            best_vibrant_idx = -1
            best_vibrant_score = -1
            
            for i in available_indices:
                h, s, v = available_colors_hsv[i]
                # Favor saturated colors with good value
                vibrance_score = s * (0.5 + 0.5 * v)  # Balance saturation with value
                
                if vibrance_score > best_vibrant_score:
                    best_vibrant_score = vibrance_score
                    best_vibrant_idx = i
            
            if best_vibrant_idx != -1:
                selected_indices.append(best_vibrant_idx)
                selected_colors.append(self.available_colors_rgb[best_vibrant_idx])
                available_indices.remove(best_vibrant_idx)
                
                # Print the selected color
                color_name = self.color_data[best_vibrant_idx]["name"]
                code = self.color_data[best_vibrant_idx]["code"] if "code" in self.color_data[best_vibrant_idx] else ""
                rgb = self.available_colors_rgb[best_vibrant_idx]
                print(f"Added vibrant color: {color_name} ({code}) - RGB{rgb}")
            else:
                break
                
        print(f"Final diverse selection: {len(selected_colors)} colors")
        return selected_colors

    def _find_black_index(self):
        """Find index of black color in the database."""
        for i, color in enumerate(self.available_colors_rgb):
            if color[0] < 20 and color[1] < 20 and color[2] < 20:
                return i
        return None

    def _find_white_index(self):
        """Find index of white color in the database."""
        for i, color in enumerate(self.available_colors_rgb):
            if color[0] > 240 and color[1] > 240 and color[2] > 240:
                return i
        return None

    def select_colors(self, image_rgb, strategy="dominant_kmeans"):
        """
        Select optimal colors for painting the image.
        
        Args:
            image_rgb: RGB image as numpy array (height, width, 3)
            strategy (str): Color selection strategy:
                - "dominant_kmeans": Select colors based on dominant colors (k-means)
                - "perceptual_distribution": Maximize perceptual color space coverage
                - "region_based": Analyze image regions and select representative colors
                - "diverse_spectrum": Ensure diverse color range across hue spectrum
                
        Returns:
            List of selected RGB color tuples
        """
        if strategy == "dominant_kmeans":
            return self.select_colors_dominant_kmeans(image_rgb)
        elif strategy == "perceptual_distribution":
            return self.select_colors_perceptual_distribution(image_rgb)
        elif strategy == "region_based":
            return self.select_colors_region_based(image_rgb)
        elif strategy == "diverse_spectrum":
            return self.select_colors_diverse_spectrum(image_rgb)
        else:
            print(f"Unknown strategy '{strategy}', defaulting to diverse_spectrum")
            return self.select_colors_diverse_spectrum(image_rgb)

# Testing function
def main():
    """Test the color selector with a sample image."""
    import sys
    import matplotlib.pyplot as plt
    
    if len(sys.argv) < 2:
        print("Usage: python color_selector.py <image_path> [max_colors] [strategy]")
        return
        
    image_path = sys.argv[1]
    max_colors = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    strategy = sys.argv[3] if len(sys.argv) > 3 else "diverse_spectrum"
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return
    
    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create color selector
    selector = ColorSelector(max_colors=max_colors)
    
    # Select colors
    selected_colors = selector.select_colors(image_rgb, strategy=strategy)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    # Display original image
    plt.subplot(2, 1, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # Display selected colors
    plt.subplot(2, 1, 2)
    for i, color in enumerate(selected_colors):
        plt.fill([i, i+0.9, i+0.9, i], [0, 0, 1, 1], color=[c/255 for c in color])
        plt.text(i+0.45, 0.5, f"{i+1}", ha='center', va='center', 
                 color='white' if sum(color) < 380 else 'black')
    
    plt.xlim(0, len(selected_colors))
    plt.ylim(0, 1)
    plt.title(f"Selected Colors ({strategy})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"selected_colors_{strategy}.png")
    plt.show()

if __name__ == "__main__":
    main()
