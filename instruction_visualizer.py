# filepath: c:\MuralBot\MuralBot\instruction_visualizer.py
import cv2
import numpy as np
import json
import argparse
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import random  # For randomized paint effects

class MuralVisualizer:
    """
    Visualize the mural painting process based on the generated instructions.
    
    This class provides:
    1. Static preview of the quantized image
    2. Animation of the painting process
    3. Path visualization for robot movement
    4. Video export of the painting simulation
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize the visualizer with settings from the config file.
        
        Args:
            config_path: Path to the configuration file
        """
        # Default visualization parameters - ENHANCED for better coverage
        self.spray_coverage = 2.0  # Increased coverage multiplier (was 1.5)
        self.spray_overlap = 0.75   # Higher overlap factor between strokes (was 0.7)
        
        # Load configuration
        self.load_config(config_path)
        
        # Initialize visualization state
        self.canvas = None
        self.frame_buffer = []
        self.loaded_colors = []
        self.current_color_index = None
        self.robot_pos = None
        self.save_animation = False
        
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load visualization configuration
            vis_config = config.get('visualization', {})
            self.output_video = vis_config.get('output_video', 'painting_simulation.mp4')
            self.fps = vis_config.get('fps', 30)
            self.video_duration = vis_config.get('video_duration', 60)
            self.resolution_scale = vis_config.get('resolution_scale', 0.5)
            self.video_quality = vis_config.get('video_quality', 80)
            
            # Enhanced parameters - Use config if specified, otherwise use enhanced defaults
            self.spray_coverage = vis_config.get('spray_coverage', 2.0)  # Increased default coverage
            self.spray_overlap = vis_config.get('spray_overlap', 0.75)   # Increased default overlap
            
            # Load wall/canvas dimensions
            img_config = config.get('image_processing', {})
            self.wall_width = img_config.get('wall_width', 2000)
            self.wall_height = img_config.get('wall_height', 1500)
            
            print(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default settings")
            
            # Set default values
            self.output_video = 'painting_simulation.mp4'
            self.fps = 30
            self.video_duration = 60
            self.resolution_scale = 0.5
            self.video_quality = 80
            self.wall_width = 2000
            self.wall_height = 1500

    def _apply_paint_effect(self, canvas, position, color, intensity=1.0, spray_size=None):
        """
        Apply a realistic paint spray effect with enhanced coverage.
        
        Args:
            canvas: The canvas to paint on
            position: (x,y) position of the spray
            color: RGB color tuple of the paint
            intensity: Strength of the spray effect (0.0 to 1.0)
            spray_size: Optional size override for the spray
        """
        x, y = position
        
        # ENHANCED: Much larger default spray size for better coverage
        if spray_size is None:
            spray_size = max(15, int(40 * self.resolution_scale * self.spray_coverage))
        
        # Create a mask for the spray pattern
        mask = np.zeros((spray_size*2+1, spray_size*2+1), dtype=np.float32)
        
        # Create a circular gradient with improved coverage
        center = (spray_size, spray_size)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # Calculate distance from center
                distance = np.sqrt((i-center[0])**2 + (j-center[1])**2)
                # ENHANCED: Apply improved radial falloff with better edge coverage
                if distance < spray_size:
                    # Use a gentler falloff curve (power of 1.5 instead of linear)
                    value = max(0, 1.0 - (distance/spray_size)**1.5) * intensity
                    # Less randomness for more consistent coverage
                    value *= (0.7 + random.random() * 0.3)
                    mask[i, j] = value
        
        # Apply the spray to the canvas
        # Calculate boundaries
        y_start = max(0, y-spray_size)
        y_end = min(canvas.shape[0], y+spray_size+1)
        x_start = max(0, x-spray_size)
        x_end = min(canvas.shape[1], x+spray_size+1)
        
        # Check if we have valid coordinates - added a safety check
        if y_end <= y_start or x_end <= x_start:
            return
        
        # Calculate corresponding mask coordinates
        mask_y_start = max(0, -(y-spray_size))
        mask_y_end = mask.shape[0] - max(0, (y+spray_size+1) - canvas.shape[0])
        mask_x_start = max(0, -(x-spray_size))
        mask_x_end = mask.shape[1] - max(0, (x+spray_size+1) - canvas.shape[1])
        
        # Additional check for valid mask size
        if mask_y_end <= mask_y_start or mask_x_end <= mask_x_start:
            return
        
        try:
            # Extract the region to modify
            roi = canvas[y_start:y_end, x_start:x_end].copy()
            
            # Apply paint to each channel with alpha blending based on mask
            # Scale mask for this application
            application_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            # Reshape for broadcasting
            application_mask = application_mask.reshape(application_mask.shape[0], application_mask.shape[1], 1)
            # Blend paint color with existing canvas
            roi = roi * (1 - application_mask) + np.array(color).reshape(1, 1, 3) * application_mask
            
            # Update the canvas region
            canvas[y_start:y_end, x_start:x_end] = roi.astype(np.uint8)
        except Exception as e:
            print(f"Error applying paint effect: {e}")

    def _apply_paint_along_path(self, canvas, start_pos, end_pos, color, base_intensity=0.9):
        """
        Apply paint along a path between two points with enhanced coverage.
        
        Args:
            canvas: The canvas to paint on
            start_pos: (x,y) starting position
            end_pos: (x,y) ending position
            color: RGB color tuple of the paint
            base_intensity: Base intensity for the paint effect
        """
        # Extract coordinates
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Calculate distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # ENHANCED: Use more interpolation points for better coverage
        # Lower density_factor means more points
        density_factor = 0.5  # More dense = better coverage
        steps = max(5, int(distance * density_factor))
        
        # Apply paint at multiple points along the path
        for i in range(steps + 1):
            # Calculate position along path
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            # ENHANCED: Vary intensity slightly for natural look
            intensity = base_intensity * (0.8 + random.random() * 0.4)
            
            # ENHANCED: Apply with enhanced spray size
            spray_size = max(15, int(35 * self.resolution_scale * self.spray_coverage * (0.9 + random.random() * 0.2)))
            self._apply_paint_effect(canvas, (x, y), color, intensity, spray_size)
        
        # ENHANCED: Add extra paint at the endpoints for better coverage
        self._apply_paint_effect(
            canvas, 
            end_pos, 
            color, 
            intensity=base_intensity * 1.2,
            spray_size=max(15, int(40 * self.resolution_scale * self.spray_coverage))
        )

    def create_preview_image(self, instructions_file):
        """
        Create a static preview image of how the mural will look.
        
        Args:
            instructions_file: Path to the instructions JSON file
        
        Returns:
            Numpy array representing the preview image
        """
        try:
            # Normalize and check instructions file path
            instructions_file = os.path.normpath(instructions_file)
            if not os.path.exists(instructions_file):
                print(f"Error: Instructions file '{instructions_file}' does not exist.")
                return None
            # Load instructions
            with open(instructions_file, 'r') as f:
                data = json.load(f)
                
            instructions = data.get('instructions', [])
            colors = data.get('colors', [])
            
            if not instructions or not colors:
                print("No instructions or colors found in file.")
                return None
            
            # Calculate canvas dimensions based on resolution scale
            canvas_width = int(self.wall_width * self.resolution_scale)
            canvas_height = int(self.wall_height * self.resolution_scale)
            
            # Create blank canvas (white background)
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Variables to track painting state
            spray_active = False
            current_color = (0, 0, 0)  # Default to black
            current_pos = (0, 0)
            
            # Process each instruction
            print("Generating preview image with enhanced coverage...")
            for i, instruction in enumerate(tqdm(instructions)):
                instruction_type = instruction.get('type')
                
                if instruction_type == 'color':
                    rgb = instruction.get('rgb', [0, 0, 0])
                    current_color = tuple(rgb)
                    
                elif instruction_type == 'move':
                    x = int(instruction.get('x', 0) * self.resolution_scale)
                    y = int(instruction.get('y', 0) * self.resolution_scale)
                    spray = instruction.get('spray', False)
                    
                    if spray and spray_active:
                        # ENHANCED: Use the improved path painting for better coverage
                        self._apply_paint_along_path(
                            canvas,
                            (int(current_pos[0]), int(current_pos[1])),
                            (x, y),
                            current_color
                        )
                    
                    # Update position
                    current_pos = (x, y)
                    
                elif instruction_type == 'spray':
                    spray_active = instruction.get('state', False)
            
            # Save the preview image
            cv2.imwrite("mural_preview.jpg", canvas)
            print("Preview image saved as 'mural_preview.jpg'")
            
            # Keep canvas for additional visualizations
            self.canvas = canvas
            
            return canvas
            
        except Exception as e:
            print(f"Error creating preview image: {e}")
            return None

    def visualize_robot_paths(self, instructions_file):
        """
        Create a visualization of the robot's movement paths.
        Different colors represent different painting colors.
        
        Args:
            instructions_file: Path to instructions JSON file
        """
        try:
            # Normalize and check instructions file path
            instructions_file = os.path.normpath(instructions_file)
            if not os.path.exists(instructions_file):
                print(f"Error: Instructions file '{instructions_file}' does not exist.")
                return False
            # Load instructions
            with open(instructions_file, 'r') as f:
                data = json.load(f)
                
            instructions = data.get('instructions', [])
            colors = data.get('colors', [])
            
            if not instructions:
                print("No instructions found in file.")
                return False
                
            # Create a blank canvas with white background
            canvas_width = int(self.wall_width * self.resolution_scale)
            canvas_height = int(self.wall_height * self.resolution_scale)
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Initialize plot
            plt.figure(figsize=(12, 8))
            
            # Keep track of paths by color
            color_paths = {}
            current_color_idx = None
            current_path = []
            spray_active = False
            
            # Process instructions to gather paths
            for instruction in instructions:
                instruction_type = instruction.get('type')
                
                if instruction_type == 'color':
                    # Start a new path for this color
                    current_color_idx = instruction.get('index')
                    
                elif instruction_type == 'move':
                    x = instruction.get('x', 0) * self.resolution_scale
                    y = instruction.get('y', 0) * self.resolution_scale
                    
                    # Store the point in the current path
                    current_path.append((x, y))
                    
                    # If this is a spray move and we have a current color, track it
                    if instruction.get('spray', False) and current_color_idx is not None:
                        if current_color_idx not in color_paths:
                            color_paths[current_color_idx] = []
                        
                        if len(current_path) >= 2:
                            # Store the segment
                            color_paths[current_color_idx].append(current_path[-2:])
                
                elif instruction_type == 'spray':
                    spray_active = instruction.get('state', False)
                    if not spray_active:
                        # Clear the current path when spray turns off
                        current_path = [current_path[-1]] if current_path else []
            
            # Plot paths by color
            for color_idx, paths in color_paths.items():
                # Get color RGB values
                rgb = None
                for instruction in instructions:
                    if instruction.get('type') == 'color' and instruction.get('index') == color_idx:
                        rgb = instruction.get('rgb', [0, 0, 0])
                        break
                
                if rgb is None:
                    rgb = [0, 0, 0]  # Default to black
                
                # Normalize RGB for matplotlib
                normalized_rgb = [c/255 for c in rgb]
                
                # Plot all path segments for this color
                for path in paths:
                    x_coords, y_coords = zip(*path)
                    plt.plot(x_coords, y_coords, color=normalized_rgb, linewidth=2, alpha=0.8)
            
            # Set plot limits and labels
            plt.xlim(0, canvas_width)
            plt.ylim(canvas_height, 0)  # Invert Y-axis to match image coordinates
            plt.title("Robot Painting Paths")
            plt.xlabel("X Position (pixels)")
            plt.ylabel("Y Position (pixels)")
            plt.grid(alpha=0.3)
            
            # Add color legend
            legend_handles = []
            for color_idx in sorted(color_paths.keys()):
                # Find color RGB
                rgb = [0, 0, 0]  # Default
                for instruction in instructions:
                    if instruction.get('type') == 'color' and instruction.get('index') == color_idx:
                        rgb = instruction.get('rgb', [0, 0, 0])
                        break
                
                # Add to legend
                normalized_rgb = [c/255 for c in rgb]
                legend_handles.append(plt.Line2D([0], [0], color=normalized_rgb, lw=4, 
                                               label=f"Color {color_idx}"))
            
            plt.legend(handles=legend_handles, title="Colors", loc='upper right')
            
            # Save the path visualization
            plt.savefig("robot_paths.png", dpi=150, bbox_inches='tight')
            print("Path visualization saved as 'robot_paths.png'")
            
            # Show the plot
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"Error visualizing robot paths: {e}")
            return False
    
    def animate_painting_process(self, instructions_file, output_file=None):
        """
        Create an animation of the painting process.
        
        Args:
            instructions_file: Path to the instructions JSON file
            output_file: Path to save the animation video (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load instructions
            with open(instructions_file, 'r') as f:
                data = json.load(f)
                
            instructions = data.get('instructions', [])
            colors = data.get('colors', [])
            
            if not instructions:
                print("No instructions found in file.")
                return False
            
            # Calculate canvas dimensions
            canvas_width = int(self.wall_width * self.resolution_scale)
            canvas_height = int(self.wall_height * self.resolution_scale)
            
            # Create blank canvas (white background)
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Calculate total frames based on desired duration and fps
            total_frames = self.fps * self.video_duration
            
            # Determine how many instructions to process per frame
            instructions_per_frame = max(1, len(instructions) // total_frames)
            
            # Variables to track painting state
            spray_active = False
            current_color = (0, 0, 0)  # Default to black
            current_pos = (0, 0)
            self.robot_pos = (0, 0)  # Robot position for visualization
            
            # Prepare frame buffer
            self.frame_buffer = []
            
            # Process instructions sequentially and create frames
            print("Generating animation frames with enhanced coverage...")
            frame_count = 0
            pos = current_pos
            spray = spray_active
            color = current_color

            for idx, instruction in enumerate(tqdm(instructions, desc="Generating animation frames")):
                instruction_type = instruction.get('type')

                if instruction_type == 'color':
                    rgb = instruction.get('rgb', [0, 0, 0])
                    color = tuple(int(round(c)) for c in rgb)

                elif instruction_type == 'move':
                    x = int(instruction.get('x', 0) * self.resolution_scale)
                    y = int(instruction.get('y', 0) * self.resolution_scale)
                    spray_flag = instruction.get('spray', False)

                    if spray_flag and spray:
                        # ENHANCED: Use the improved path painting for better coverage
                        self._apply_paint_along_path(
                            canvas,
                            pos,
                            (x, y),
                            color
                        )

                    pos = (x, y)

                elif instruction_type == 'spray':
                    spray = instruction.get('state', False)

                elif instruction_type == 'load_colors':
                    pass  # skip

                # Save frame every instructions_per_frame steps
                if idx % instructions_per_frame == 0:
                    frame = canvas.copy()
                    self._draw_robot_on_frame(frame, pos, spray, color)
                    self.frame_buffer.append(frame)

            # Save final frame
            frame = canvas.copy()
            self._draw_robot_on_frame(frame, pos, spray, color)
            self.frame_buffer.append(frame)

            print("\nGenerated", len(self.frame_buffer), "animation frames")
            
            # Save as video if requested
            if output_file or self.output_video:
                out_path = output_file if output_file else self.output_video
                self._save_animation_as_video(out_path)
            
            return True
            
        except Exception as e:
            print(f"Error animating painting process: {e}")
            return False
    
    def _draw_robot_on_frame(self, frame, position, spray_active, color):
        """
        Draw the robot and spray indication on a frame.
        
        Args:
            frame: The image frame to draw on
            position: (x,y) position of the robot
            spray_active: Whether the spray is active
            color: Current spray color
        """
        x, y = position
        
        # Draw robot body
        robot_radius = max(5, int(10 * self.resolution_scale))
        cv2.circle(frame, (x, y), robot_radius, (50, 50, 50), -1)  # Dark gray robot
        cv2.circle(frame, (x, y), robot_radius, (0, 0, 0), 1)      # Black outline
        
        # Draw spray indicator if active
        if spray_active:
            # Draw spray direction cone
            spray_length = robot_radius * 3
            spray_width = robot_radius * 1.5
            
            # Create a slightly transparent overlay for spray cone
            overlay = frame.copy()
            
            # Draw a spray cone using a triangle
            pt1 = (x, y)
            pt2 = (x - spray_width, y + spray_length)
            pt3 = (x + spray_width, y + spray_length)
            spray_pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
            
            # Fill with color
            cv2.fillPoly(overlay, [spray_pts], color)
            # Add some spray particles
            for _ in range(10):
                # Random position within spray cone
                rx = x + random.randint(-int(spray_width*0.8), int(spray_width*0.8))
                ry = y + random.randint(0, int(spray_length*0.8))
                # Random size
                r_size = random.randint(1, 3)
                cv2.circle(overlay, (rx, ry), r_size, color, -1)
            
            # Apply overlay with transparency
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Draw spray direction indicator
            indicator_length = robot_radius * 2
            cv2.line(frame, 
                    (x, y),
                    (x, y + indicator_length),
                    color,
                    thickness=2)
    
    def _save_animation_as_video(self, output_file):
        """
        Save the animation frames as a video file.
        
        Args:
            output_file: Path to save the video file
        """
        if not self.frame_buffer:
            print("No frames to save.")
            return False
            
        try:
            # Get video dimensions from first frame
            height, width = self.frame_buffer[0].shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, self.fps, (width, height))
            
            # Write frames
            print(f"Saving animation to {output_file}...")
            for frame in tqdm(self.frame_buffer):
                out.write(frame)
                
            # Release video writer
            out.release()
            
            print(f"Video saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error saving animation as video: {e}")
            return False
    
    def create_interactive_visualization(self, instructions_file):
        """
        Create an interactive visualization of the painting process.
        Uses matplotlib for scrubbing through the painting process.
        
        Args:
            instructions_file: Path to the instructions JSON file
        """
        if not self.frame_buffer:
            print("No animation frames available. Run animate_painting_process first.")
            return False
            
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
            
            # Initialize with first frame
            img_plot = ax.imshow(cv2.cvtColor(self.frame_buffer[0], cv2.COLOR_BGR2RGB))
            
            # Add slider for scrubbing through frames
            from matplotlib.widgets import Slider
            
            # Add axes for slider
            slider_ax = plt.axes([0.25, 0.02, 0.65, 0.03])
            frame_slider = Slider(
                slider_ax, 'Frame', 0, len(self.frame_buffer) - 1,
                valinit=0, valstep=1
            )
            
            # Update function for slider
            def update(val):
                frame_idx = int(frame_slider.val)
                img_plot.set_data(cv2.cvtColor(self.frame_buffer[frame_idx], cv2.COLOR_BGR2RGB))
                progress = frame_idx / (len(self.frame_buffer) - 1) * 100
                ax.set_title(f"Painting Process - {progress:.1f}% Complete")
                fig.canvas.draw_idle()
                
            # Connect slider to update function
            frame_slider.on_changed(update)
            
            # Set initial title
            ax.set_title("Painting Process - 0.0% Complete")
            
            # Add instructions
            fig.text(0.5, 0.96, "Use slider to scrub through the painting process",
                    ha='center', va='center', fontsize=10)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Show the interactive plot
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"Error creating interactive visualization: {e}")
            return False

    def visualize_instructions(self, instructions_file, output_file=None, show_preview=True, show_progress=True):
        """
        Visualize painting instructions by drawing paths on a canvas.
        
        Args:
            instructions_file: Path to the JSON file containing painting instructions
            output_file: Path to save the visualization image (if None, will save to painting folder)
            show_preview: Whether to display the preview window
            show_progress: Whether to show progress during visualization
        """
        # Ensure painting folder exists
        painting_folder = os.path.join(os.path.dirname(__file__), "painting")
        os.makedirs(painting_folder, exist_ok=True)
        
        # Load instructions
        with open(instructions_file, 'r') as f:
            data = json.load(f)
        
        wall_width = data.get('wall_width', 2000)
        wall_height = data.get('wall_height', 1500)
        instructions = data.get('instructions', [])
        
        # Create a blank canvas with white background
        canvas = np.ones((wall_height, wall_width, 3), dtype=np.uint8) * 255
        
        # Track the current color and spray state
        current_color = (0, 0, 0)  # Default to black
        spray_on = False
        
        # For animation
        frames = []
        save_frames = getattr(self, 'save_animation', False)
        
        # For progress tracking
        if show_progress:
            progress_bar = tqdm(total=len(instructions), desc="Visualizing with enhanced coverage")
        
        # Process each instruction
        for idx, instruction in enumerate(instructions):
            instr_type = instruction.get('type')
            
            if instr_type == 'color':
                # Change current color
                rgb = instruction.get('rgb')
                if rgb:
                    # Convert to BGR for OpenCV
                    current_color = (rgb[2], rgb[1], rgb[0])
                # Reset previous position to avoid connecting lines across color changes
                self.prev_pos = None
            
            elif instr_type == 'spray':
                # Update spray state
                spray_on = instruction.get('state', False)
            
            elif instr_type == 'move':
                # Get coordinates
                x = instruction.get('x', 0)
                y = instruction.get('y', 0)
                spray = instruction.get('spray', False)
                
                # Convert to pixel coordinates (origin is top-left in OpenCV)
                x_pixel = int(x)
                y_pixel = int(y)
                
                # Draw if spraying
                if spray and spray_on:
                    # ENHANCED: Use improved paint path for better coverage
                    if hasattr(self, 'prev_pos') and self.prev_pos:
                        # Apply enhanced paint along the path
                        self._apply_paint_along_path(
                            canvas,
                            self.prev_pos,
                            (x_pixel, y_pixel),
                            current_color[::-1]  # BGR to RGB
                        )
                    else:
                        # If first point, just add paint at that location
                        self._apply_paint_effect(canvas, (x_pixel, y_pixel), current_color[::-1])
                
                # Update previous position
                self.prev_pos = (x_pixel, y_pixel)
                
                # Save frame for animation if enabled
                if save_frames and idx % 10 == 0:  # Save every 10th instruction to reduce size
                    frames.append(canvas.copy())
            
            # Update progress bar
            if show_progress:
                progress_bar.update(1)
                
            # Show live preview if requested
            if show_preview and idx % 50 == 0:  # Update every 50 instructions
                cv2.imshow('Mural Visualization', canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break
        
        # Close progress bar
        if show_progress:
            progress_bar.close()
        
        # Determine output file path
        if output_file is None:
            output_file = os.path.join(painting_folder, "mural_visualization.jpg")
        elif not os.path.isabs(output_file):
            # If a relative path is provided, save to the painting folder
            output_file = os.path.join(painting_folder, output_file)
        
        # Save the final visualization
        cv2.imwrite(output_file, canvas)
        print(f"Visualization saved to {output_file}")
        
        # Save the animation if enabled
        if save_frames and frames:
            animation_file = os.path.join(painting_folder, "mural_animation.mp4")
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(animation_file, fourcc, 15, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Animation saved to {animation_file}")
        
        # Show the final result
        if show_preview:
            cv2.imshow('Mural Visualization', canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return canvas

# Example usage when run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize the mural painting process')
    parser.add_argument('--config', '-c', default='config.json', help='Path to configuration file')
    parser.add_argument('--instructions', '-i', required=True, help='Path to instructions JSON file')
    parser.add_argument('--preview', '-p', action='store_true', help='Generate a preview image')
    parser.add_argument('--paths', '-pa', action='store_true', help='Visualize robot paths')
    parser.add_argument('--animate', '-a', action='store_true', help='Create animation')
    parser.add_argument('--interactive', '-int', action='store_true', help='Show interactive visualization')
    parser.add_argument('--output', '-o', help='Output video file path')
    
    args = parser.parse_args()
    
    visualizer = MuralVisualizer(args.config)
    
    if args.preview:
        visualizer.create_preview_image(args.instructions)
        
    if args.paths:
        visualizer.visualize_robot_paths(args.instructions)
        
    if args.animate or args.interactive:
        visualizer.animate_painting_process(args.instructions, args.output)
        
    if args.interactive:
        visualizer.create_interactive_visualization(args.instructions)
