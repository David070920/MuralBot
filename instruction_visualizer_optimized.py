# filepath: c:\MuralBot\MuralBot\instruction_visualizer_optimized.py
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
import concurrent.futures  # For parallel processing

class MuralVisualizer:
    """
    Visualize the mural painting process based on the generated instructions.
    
    This class provides:
    1. Static preview of the quantized image
    2. Animation of the painting process
    3. Path visualization for robot movement
    4. Video export of the painting simulation (optimized)
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
        
        # Video optimization parameters
        self.frame_skip = 5  # Process only every nth instruction for video frames
        self.keyframe_interval = 20  # Store a keyframe every n processed frames
        self.use_fast_encoding = True  # Use faster encoding settings
        
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
            
            # Optimization parameters
            self.frame_skip = vis_config.get('frame_skip', 5)
            self.keyframe_interval = vis_config.get('keyframe_interval', 20)
            self.use_fast_encoding = vis_config.get('use_fast_encoding', True)
            
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
        """Apply paint effect at a specific position."""
        # Implementation unchanged for this optimization
        pass
        
    def _apply_paint_along_path(self, canvas, start_pos, end_pos, color, base_intensity=0.9):
        """Apply paint effect along a line path."""
        # Implementation unchanged for this optimization
        pass
        
    def create_preview_image(self, instructions_file):
        """Create a static preview image of how the mural will look."""
        # Implementation unchanged for this optimization
        pass

    def visualize_robot_paths(self, instructions_file):
        """Create a visualization of the robot's movement paths."""
        # Implementation unchanged for this optimization
        pass
        
    def _draw_robot_simplified(self, frame, position, spray_active, color):
        """Simplified version of robot drawing for faster rendering."""
        x, y = position
        
        # Draw simple robot body (just a circle)
        robot_radius = max(5, int(8 * self.resolution_scale))
        cv2.circle(frame, (x, y), robot_radius, (50, 50, 50), -1)  # Dark gray robot
        
        # Simple spray indicator if active
        if spray_active:
            # Draw simple spray direction as a line
            indicator_length = robot_radius * 2
            cv2.line(frame, 
                    (x, y),
                    (x, y + indicator_length),
                    color,
                    thickness=2)
                    
    def animate_painting_process(self, instructions_file, output_file=None):
        """
        Create an optimized animation of the painting process.
        
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
            
            # Calculate target number of frames based on duration
            # Using reduced frame rate for processing but keeping display FPS
            processing_fps = max(10, self.fps // 2)  # Reduce processing framerate
            target_frames = processing_fps * self.video_duration
            
            # Increase instructions per frame for optimization
            instructions_per_frame = max(1, len(instructions) // target_frames)
            instructions_per_frame *= self.frame_skip  # Skip more frames for faster processing
            
            # Variables to track painting state
            spray = False
            color = (0, 0, 0)  # Default to black
            pos = (0, 0)
            
            # Prepare frame buffer with reduced size
            self.frame_buffer = []
            
            print("Generating optimized animation frames...")
            frame_count = 0
            processed_frames = 0
            
            # Process instructions in larger chunks
            for idx in tqdm(range(0, len(instructions), self.frame_skip), desc="Optimizing animation"):
                # Process a batch of instructions without creating intermediate frames
                end_idx = min(idx + self.frame_skip, len(instructions))
                
                # Process this chunk of instructions
                for i in range(idx, end_idx):
                    instruction = instructions[i]
                    instruction_type = instruction.get('type')
                    
                    if instruction_type == 'color':
                        rgb = instruction.get('rgb', [0, 0, 0])
                        color = tuple(int(round(c)) for c in rgb)
                    
                    elif instruction_type == 'move':
                        x = int(instruction.get('x', 0) * self.resolution_scale)
                        y = int(instruction.get('y', 0) * self.resolution_scale)
                        spray_flag = instruction.get('spray', False)
                        
                        if spray_flag and spray:
                            # Apply paint along path
                            self._apply_paint_along_path(
                                canvas,
                                pos,
                                (x, y),
                                color
                            )
                        
                        pos = (x, y)
                    
                    elif instruction_type == 'spray':
                        spray = instruction.get('state', False)
                
                # Save frame at keyframe intervals to reduce memory usage
                processed_frames += 1
                if processed_frames % self.keyframe_interval == 0:
                    frame = canvas.copy()
                    # Use simplified robot drawing for better performance
                    self._draw_robot_simplified(frame, pos, spray, color)
                    self.frame_buffer.append(frame)
                    frame_count += 1
            
            # Ensure we have at least one frame
            if not self.frame_buffer:
                frame = canvas.copy()
                self._draw_robot_simplified(frame, pos, spray, color)
                self.frame_buffer.append(frame)
            
            print(f"\nGenerated {len(self.frame_buffer)} optimized frames (reduced from potential {len(instructions)})")
            
            # Save as video if requested
            if output_file or self.output_video:
                out_path = output_file if output_file else self.output_video
                self._save_animation_as_video_optimized(out_path)
            
            return True
            
        except Exception as e:
            print(f"Error animating painting process: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _save_animation_as_video_optimized(self, output_file):
        """
        Save the animation frames as a video file with optimized encoding.
        
        Args:
            output_file: Path to save the video file
        """
        if not self.frame_buffer:
            print("No frames to save.")
            return False
            
        try:
            # Get video dimensions from first frame
            height, width = self.frame_buffer[0].shape[:2]
            
            # Use more efficient encoder if available
            try:
                # Try using H.264 encoder which should be faster on most systems
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                # For even faster encoding at cost of some quality, use 'XVID' or 'MJPG'
                if self.use_fast_encoding:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
            except:
                # Fall back to standard mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Set parameters for optimized encoding
            fps = min(30, self.fps)  # Cap at 30 fps for better performance
            
            print(f"Encoding optimized video with {len(self.frame_buffer)} frames at {fps} fps...")
            start_time = time.time()
            
            # Initialize video writer with optimized parameters
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            # Write frames in batches to reduce overhead
            batch_size = 10
            for i in range(0, len(self.frame_buffer), batch_size):
                batch = self.frame_buffer[i:i + batch_size]
                for frame in batch:
                    out.write(frame)
                
                # Update progress every few batches
                if i % 50 == 0:
                    progress = i / len(self.frame_buffer) * 100
                    print(f"Encoding: {progress:.1f}% complete", end="\r")
            
            # Release video writer
            out.release()
            
            elapsed = time.time() - start_time
            print(f"\nVideo saved to {output_file} in {elapsed:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"Error saving animation as video: {e}")
            return False
    
    # Other methods in the class remain unchanged
