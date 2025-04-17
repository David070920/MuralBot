"""
Animation module for MuralBot.

This module contains functions for generating and saving animations
of the painting process.
"""

import cv2
import numpy as np
from tqdm import tqdm

def create_frames(instructions, initial_canvas, total_frames, resolution_scale):
    """
    Generate animation frames for the painting process.
    
    Args:
        instructions: List of painting instructions
        initial_canvas: Initial canvas to draw on
        total_frames: Target number of frames to generate
        resolution_scale: Resolution scale for visualization (pixels per mm)
        
    Returns:
        List of animation frames
    """
    # Determine how many instructions to process per frame
    instructions_per_frame = max(1, len(instructions) // total_frames)
    
    # Variables to track painting state
    spray_active = False
    current_color = (0, 0, 0)  # Default to black
    current_pos = (0, 0)
    robot_pos = (0, 0)  # Robot position for visualization
    
    # Prepare frame buffer
    frame_buffer = []
    
    # Start with a copy of the initial canvas
    canvas = initial_canvas.copy()
    
    # Process instructions sequentially and create frames
    print("Generating animation frames...")
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
            x = int(instruction.get('x', 0) * resolution_scale)
            y = int(instruction.get('y', 0) * resolution_scale)
            spray_flag = instruction.get('spray', False)

            if spray_flag and spray:
                cv2.line(canvas,
                        (int(pos[0]), int(pos[1])),
                        (x, y),
                        color,
                        thickness=max(1, int(3 * resolution_scale)))

            pos = (x, y)

        elif instruction_type == 'spray':
            spray = instruction.get('state', False)

        elif instruction_type == 'load_colors':
            pass  # skip

        # Save frame every instructions_per_frame steps
        if idx % instructions_per_frame == 0:
            frame = canvas.copy()
            draw_robot_on_frame(frame, pos, spray, color, resolution_scale)
            frame_buffer.append(frame)

    # Save final frame
    frame = canvas.copy()
    draw_robot_on_frame(frame, pos, spray, color, resolution_scale)
    frame_buffer.append(frame)
    
    return frame_buffer

def draw_robot_on_frame(frame, position, spray_active, color, resolution_scale):
    """
    Draw the robot and spray indication on a frame.
    
    Args:
        frame: The image frame to draw on
        position: (x,y) position of the robot
        spray_active: Whether the spray is active
        color: Current spray color
        resolution_scale: Resolution scale for visualization
    """
    x, y = position
    
    # Draw robot body
    robot_radius = max(5, int(10 * resolution_scale))
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

def save_animation_as_video(frame_buffer, output_file, fps=30):
    """
    Save animation frames as video.
    
    Args:
        frame_buffer: List of animation frames
        output_file: Path to save the video
        fps: Frames per second for the video
        
    Returns:
        True if successful, False otherwise
    """
    if not frame_buffer:
        print("No frames to save.")
        return False
    
    try:
        # Ensure parent directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Get frame dimensions
        height, width = frame_buffer[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Write each frame to video
        for frame in frame_buffer:
            out.write(frame)
            
        # Release video writer
        out.release()
        print(f"Animation saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error saving animation as video: {e}")
        return False

# Import at the top but defined here to avoid circular import
import os
