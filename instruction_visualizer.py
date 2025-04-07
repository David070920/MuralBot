import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm

class InstructionVisualizer:
    def __init__(self, instructions_file):
        """Load and visualize mural painting instructions."""
        with open(instructions_file, 'r') as f:
            data = json.load(f)
            
        self.wall_width = data['wall_width']
        self.wall_height = data['wall_height']
        self.colors = data['colors']
        self.instructions = data['instructions']
        
    def create_simulation(self, output_video_path, fps=15):
        """Create a video simulation of the painting process."""
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, 
                               (self.wall_width, self.wall_height))
        
        # Create initial blank canvas
        canvas = np.ones((self.wall_height, self.wall_width, 3), dtype=np.uint8) * 255
        
        # Current robot position and state
        current_x = 0
        current_y = 0
        current_color = 0
        is_spraying = False
        
        print(f"Creating simulation video with {len(self.instructions)} instructions...")
        
        # Process each instruction with a progress bar
        for i, instruction in tqdm(enumerate(self.instructions), total=len(self.instructions), desc="Rendering video"):
            inst_type = instruction['type']
            
            if inst_type == 'home':
                # Move to home position
                current_x = 0
                current_y = 0
                is_spraying = False
                
            elif inst_type == 'color':
                # Change color
                current_color = instruction['index']
                
            elif inst_type == 'spray':
                # Set spray state
                is_spraying = instruction['state']
                
            elif inst_type == 'move':
                # Move to new position
                new_x = instruction['x']
                new_y = instruction['y']
                
                if is_spraying:
                    # Draw a line from current position to new position
                    color_bgr = (self.colors[current_color][2], 
                                 self.colors[current_color][1], 
                                 self.colors[current_color][0])
                    cv2.line(canvas, (current_x, current_y), (new_x, new_y), color_bgr, 5)
                
                # Update position
                current_x = new_x
                current_y = new_y
            
            # Draw robot position
            robot_canvas = canvas.copy()
            cv2.circle(robot_canvas, (current_x, current_y), 10, (0, 0, 0), -1)
            
            # Add instruction text
            cv2.putText(robot_canvas, f"Step {i+1}/{len(self.instructions)}: {instruction.get('message', '')}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add to video
            video.write(robot_canvas)
            
        # Release video
        video.release()
        print(f"Simulation video saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize mural painting instructions')
    parser.add_argument('--config', '-c', default='config.json', 
                        help='Path to the configuration file')
    parser.add_argument('--instructions', '-i', help='Path to the JSON instructions file (overrides config file)')
    parser.add_argument('--output', '-o', help='Output video file path (overrides config file)')
    parser.add_argument('--fps', type=int, help='Frames per second for the simulation video (overrides config file)')
    
    args = parser.parse_args()
    
    # Load config file
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        viz_config = config['visualization']
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading config file: {e}")
        viz_config = {}
    
    # Get parameters, prioritizing command line arguments over config file
    instructions_file = args.instructions or viz_config.get('instructions_file')
    output_video = args.output or viz_config.get('output_video', 'painting_simulation.mp4')
    fps = args.fps or viz_config.get('fps', 15)
    
    # Check if instructions file is provided
    if not instructions_file:
        print("Error: No instructions file specified. Please provide it in the config file or as a command line argument.")
        exit(1)
    
    visualizer = InstructionVisualizer(instructions_file)
    visualizer.create_simulation(output_video, fps)