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
        # Load configuration
        self.load_config(config_path)
        
        # Initialize visualization state
        self.canvas = None
        self.frame_buffer = []
        self.loaded_colors = []
        self.current_color_index = None
        self.robot_pos = None
        
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

    def create_preview_image(self, instructions_file):
        """
        Create a static preview image of how the mural will look.
        
        Args:
            instructions_file: Path to the instructions JSON file
        
        Returns:
            Numpy array representing the preview image
        """
        try:
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
            print("Generating preview image...")
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
                        # Draw a line of the current color
                        cv2.line(canvas, 
                                 (int(current_pos[0]), int(current_pos[1])),
                                 (x, y),
                                 current_color,
                                 thickness=max(1, int(3 * self.resolution_scale)))
                        
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
            
            # Process instructions and create frames
            print("Generating animation frames...")
            frame_count = 0
            
            # Process instructions in chunks
            for i in range(0, len(instructions), instructions_per_frame):
                # Process a chunk of instructions
                chunk = instructions[i:i + instructions_per_frame]
                
                for instruction in chunk:
                    instruction_type = instruction.get('type')
                    
                    if instruction_type == 'color':
                        rgb = instruction.get('rgb', [0, 0, 0])
                        current_color = tuple(rgb)
                        
                    elif instruction_type == 'move':
                        x = int(instruction.get('x', 0) * self.resolution_scale)
                        y = int(instruction.get('y', 0) * self.resolution_scale)
                        spray = instruction.get('spray', False)
                        
                        if spray and spray_active:
                            # Draw a line of the current color
                            cv2.line(canvas, 
                                    (int(current_pos[0]), int(current_pos[1])),
                                    (x, y),
                                    current_color,
                                    thickness=max(1, int(3 * self.resolution_scale)))
                        
                        # Update positions
                        current_pos = (x, y)
                        self.robot_pos = (x, y)
                        
                    elif instruction_type == 'spray':
                        spray_active = instruction.get('state', False)
                        
                    elif instruction_type == 'load_colors':
                        # Update loaded colors
                        self.loaded_colors = [c['index'] for c in instruction.get('colors', [])]
                        
                # Create a copy of the canvas for this frame
                frame = canvas.copy()
                
                # Draw the robot position
                self._draw_robot_on_frame(frame, self.robot_pos, spray_active, current_color)
                
                # Add frame to buffer
                self.frame_buffer.append(frame)
                frame_count += 1
                
                # Update progress bar every 10 frames
                if frame_count % 10 == 0:
                    percent = min(100, int((frame_count / total_frames) * 100))
                    print(f"\rGenerating frames: {percent}% complete", end="")
            
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
            spray_radius = int(robot_radius * 1.5)
            # Create a slightly transparent overlay
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), spray_radius, color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
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
            fig, ax = plt.subplots(figsize=(12, 8))
            
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
            plt.tight_layout()
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
        save_frames = self.save_animation
        
        # For progress tracking
        if show_progress:
            progress_bar = tqdm(total=len(instructions), desc="Visualizing")
        
        # Process each instruction
        for idx, instruction in enumerate(instructions):
            instr_type = instruction.get('type')
            
            if instr_type == 'color':
                # Change current color
                rgb = instruction.get('rgb')
                if rgb:
                    # Convert to BGR for OpenCV
                    current_color = (rgb[2], rgb[1], rgb[0])
            
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
                    # Draw a line from the previous position
                    if hasattr(self, 'prev_pos'):
                        cv2.line(canvas, self.prev_pos, (x_pixel, y_pixel), current_color, 2)
                    else:
                        # If first point, draw a circle
                        cv2.circle(canvas, (x_pixel, y_pixel), 1, current_color, -1)
                
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
            self._save_animation(frames, animation_file, fps=30)
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