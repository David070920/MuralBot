import math
import json
import time
import threading
import serial
from tqdm import tqdm

class MuralRobotController:
    """
    Controller for a string-based mural painting robot.
    
    This class handles:
    1. Converting X,Y coordinates to string lengths for the robot's position
    2. Sending commands to the hardware to control stepper motors and spray mechanisms
    3. Optimizing movement paths for efficient painting
    4. Calibration routines
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize the robot controller with settings from the config file.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.load_config(config_path)
        
        # Initialize hardware connections
        self.left_motor = None
        self.right_motor = None
        self.spray_servo = None
        self.connected = False
        
        # Current position and state
        self.current_x = 0
        self.current_y = 0
        self.current_color_index = None
        self.spray_active = False
        
        # Loaded colors in robot
        self.loaded_colors = []
        
        # Cached string lengths to avoid recalculation
        self.cached_string_lengths = {}
        
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load hardware configuration
            hardware_config = config.get('hardware', {})
            self.left_stepper_port = hardware_config.get('left_stepper_port', 'COM3')
            self.right_stepper_port = hardware_config.get('right_stepper_port', 'COM4')
            self.spray_servo_port = hardware_config.get('spray_servo_port', 'COM5')
            self.wall_mount_height = hardware_config.get('wall_mount_height', 2000)
            self.wall_mount_width = hardware_config.get('wall_mount_width', 3000)
            self.steps_per_mm = hardware_config.get('steps_per_mm', 10)
            self.max_speed = hardware_config.get('max_speed', 1000)
            self.home_position = hardware_config.get('home_position', [0, 0])
            self.color_change_position = hardware_config.get('color_change_position', [0, 2000])
            
            # Load wall/canvas dimensions
            image_config = config.get('image_processing', {})
            self.wall_width = image_config.get('wall_width', 2000)
            self.wall_height = image_config.get('wall_height', 1500)
            
            print(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default settings")
            
            # Set default values
            self.left_stepper_port = 'COM3'
            self.right_stepper_port = 'COM4'
            self.spray_servo_port = 'COM5'
            self.wall_mount_height = 2000
            self.wall_mount_width = 3000
            self.steps_per_mm = 10
            self.max_speed = 1000
            self.wall_width = 2000
            self.wall_height = 1500
            self.home_position = [0, 0]
            self.color_change_position = [0, 2000]
    
    def connect(self):
        """Connect to the hardware components."""
        try:
            # Connect to motor controllers and spray mechanism
            # Using different baud rates depending on the controller type
            print("Connecting to left stepper motor...")
            self.left_motor = serial.Serial(self.left_stepper_port, 115200, timeout=1)
            
            print("Connecting to right stepper motor...")
            self.right_motor = serial.Serial(self.right_stepper_port, 115200, timeout=1)
            
            print("Connecting to spray mechanism...")
            self.spray_servo = serial.Serial(self.spray_servo_port, 9600, timeout=1)
            
            # Wait for connections to stabilize
            time.sleep(2)
            
            # Send initialization commands
            self._send_command(self.left_motor, "INIT")
            self._send_command(self.right_motor, "INIT")
            self._send_command(self.spray_servo, "INIT")
            
            self.connected = True
            print("All hardware components connected successfully")
            
            # Home the robot
            self.home()
            
            return True
            
        except Exception as e:
            print(f"Error connecting to hardware: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from all hardware components."""
        if not self.connected:
            return
            
        try:
            if self.left_motor:
                self._send_command(self.left_motor, "STOP")
                self.left_motor.close()
                
            if self.right_motor:
                self._send_command(self.right_motor, "STOP")
                self.right_motor.close()
                
            if self.spray_servo:
                # Ensure spray is off before disconnecting
                self.set_spray(False)
                self.spray_servo.close()
                
            self.connected = False
            print("All hardware components disconnected")
            
        except Exception as e:
            print(f"Error during disconnect: {e}")
    
    def _send_command(self, device, command):
        """
        Send a command to a connected device.
        
        Args:
            device: Serial device to send command to
            command: Command string to send
        """
        if not device:
            return False
            
        try:
            # Add newline termination and encode to bytes
            full_command = f"{command}\n".encode()
            device.write(full_command)
            device.flush()
            
            # Get acknowledgment (depends on your firmware implementation)
            response = device.readline().decode().strip()
            
            if response != "OK":
                print(f"Warning: Unexpected response from device: {response}")
                
            return True
            
        except Exception as e:
            print(f"Error sending command '{command}': {e}")
            return False
    
    def _xy_to_string_lengths(self, x, y):
        """
        Convert X,Y coordinates to left and right string lengths.
        Uses caching for improved performance.
        
        Args:
            x: X-coordinate on the wall (mm)
            y: Y-coordinate on the wall (mm)
            
        Returns:
            Tuple of (left_length, right_length) in mm
        """
        # Check cache first
        cache_key = (x, y)
        if cache_key in self.cached_string_lengths:
            return self.cached_string_lengths[cache_key]
        
        # Calculate the absolute wall coordinates
        # Assuming 0,0 is at top-left corner of the painting area
        # We need to translate to the actual wall mounting points
        
        # X-offset to center the painting area on the wall
        x_offset = (self.wall_mount_width - self.wall_width) // 2
        
        # Y-offset from top of wall
        y_offset = 0  # Assuming the top of painting area is at top of wall
        
        # Translate the coordinates
        wall_x = x + x_offset
        wall_y = y + y_offset
        
        # Calculate string lengths using the Pythagorean theorem
        # Left string: from (0,0) to (wall_x, wall_y)
        left_length = math.sqrt(wall_x**2 + wall_y**2)
        
        # Right string: from (wall_mount_width,0) to (wall_x, wall_y)
        right_length = math.sqrt((self.wall_mount_width - wall_x)**2 + wall_y**2)
        
        # Store in cache
        self.cached_string_lengths[cache_key] = (left_length, right_length)
        
        return left_length, right_length
    
    def _string_lengths_to_steps(self, left_length, right_length):
        """
        Convert string lengths to motor steps.
        
        Args:
            left_length: Left string length in mm
            right_length: Right string length in mm
            
        Returns:
            Tuple of (left_steps, right_steps)
        """
        left_steps = int(left_length * self.steps_per_mm)
        right_steps = int(right_length * self.steps_per_mm)
        return left_steps, right_steps
    
    def _move_stepper(self, motor, steps, speed=None):
        """
        Send stepper command to move a specific number of steps.
        
        Args:
            motor: Serial connection to the stepper motor
            steps: Number of steps to move (positive or negative)
            speed: Speed in steps per second (optional)
        """
        if not motor:
            return False
        
        if speed is None:
            speed = self.max_speed
            
        direction = "FWD" if steps >= 0 else "REV"
        abs_steps = abs(steps)
        
        # Send command to motor
        command = f"MOVE {direction} {abs_steps} {speed}"
        return self._send_command(motor, command)
    
    def _calculate_movement(self, target_x, target_y):
        """
        Calculate the stepper movements needed to move from current position to target.
        
        Args:
            target_x: Target X-coordinate
            target_y: Target Y-coordinate
            
        Returns:
            Tuple of (left_steps, right_steps) to move
        """
        # Calculate current and target string lengths
        current_left, current_right = self._xy_to_string_lengths(self.current_x, self.current_y)
        target_left, target_right = self._xy_to_string_lengths(target_x, target_y)
        
        # Calculate string length differences
        delta_left = target_left - current_left
        delta_right = target_right - current_right
        
        # Convert to steps
        left_steps = int(delta_left * self.steps_per_mm)
        right_steps = int(delta_right * self.steps_per_mm)
        
        return left_steps, right_steps
    
    def move_to(self, x, y, spray_active=False):
        """
        Move the robot to the given X,Y coordinates.
        
        Args:
            x: Target X-coordinate on the wall (mm)
            y: Target Y-coordinate on the wall (mm)
            spray_active: Whether to spray while moving
        
        Returns:
            True if movement successful, False otherwise
        """
        if not self.connected:
            print("Robot not connected.")
            return False
            
        # Ensure coordinates are within the wall boundaries
        x = max(0, min(x, self.wall_width))
        y = max(0, min(y, self.wall_height))
        
        # Calculate required motor movements
        left_steps, right_steps = self._calculate_movement(x, y)
        
        # Determine optimal speed based on distance
        total_steps = max(abs(left_steps), abs(right_steps))
        move_speed = min(self.max_speed, max(100, total_steps // 10))
        
        # Set spray state before moving
        if spray_active != self.spray_active:
            self.set_spray(spray_active)
        
        # Calculate expected travel time
        distance = math.sqrt((x - self.current_x)**2 + (y - self.current_y)**2)
        time_seconds = distance / (move_speed / self.steps_per_mm)
        
        # Start both motors in parallel threads for simultaneous motion
        left_thread = threading.Thread(target=self._move_stepper, 
                                      args=(self.left_motor, left_steps, move_speed))
        right_thread = threading.Thread(target=self._move_stepper, 
                                       args=(self.right_motor, right_steps, move_speed))
        
        left_thread.start()
        right_thread.start()
        
        # Show progress
        with tqdm(total=100, desc=f"Moving to ({x},{y})", unit="%") as pbar:
            start_time = time.time()
            last_update = 0
            while left_thread.is_alive() or right_thread.is_alive():
                elapsed = time.time() - start_time
                progress = min(100, int((elapsed / time_seconds) * 100))
                if progress > last_update:
                    pbar.update(progress - last_update)
                    last_update = progress
                time.sleep(0.1)
            
            # Make sure we show 100% at the end
            if last_update < 100:
                pbar.update(100 - last_update)
        
        # Wait for movement to complete
        left_thread.join()
        right_thread.join()
        
        # Update current position
        self.current_x = x
        self.current_y = y
        
        return True
    
    def home(self):
        """Move the robot to the home position."""
        print("Homing robot...")
        result = self.move_to(self.home_position[0], self.home_position[1])
        
        if result:
            print("Robot homed successfully")
        else:
            print("Failed to home robot")
            
        return result
    
    def set_spray(self, active):
        """
        Turn spray on or off.
        
        Args:
            active: True to activate spray, False to deactivate
        """
        if not self.connected:
            print("Robot not connected.")
            return False
            
        if active == self.spray_active:
            return True  # Already in requested state
            
        command = "SPRAY ON" if active else "SPRAY OFF"
        result = self._send_command(self.spray_servo, command)
        
        if result:
            self.spray_active = active
            print(f"Spray {'activated' if active else 'deactivated'}")
        else:
            print(f"Failed to {'activate' if active else 'deactivate'} spray")
            
        return result
    
    def set_color(self, color_index):
        """
        Change the active spray color.
        
        Args:
            color_index: Index of the color to use
        """
        if not self.connected:
            print("Robot not connected.")
            return False
            
        # Check if color is loaded
        if color_index not in self.loaded_colors:
            print(f"Error: Color {color_index} is not loaded in the robot")
            return False
            
        # Turn off spray before changing color
        if self.spray_active:
            self.set_spray(False)
        
        # Send color change command
        command = f"COLOR {color_index}"
        result = self._send_command(self.spray_servo, command)
        
        if result:
            self.current_color_index = color_index
            print(f"Changed to color {color_index}")
        else:
            print(f"Failed to change to color {color_index}")
            
        return result
    
    def load_colors(self, color_list):
        """
        Load a set of colors into the robot.
        This simulates the process of changing the spray cans.
        
        Args:
            color_list: List of color dictionaries with 'index' and 'rgb' keys
        """
        if not self.connected:
            print("Robot not connected.")
            return False
            
        # First, move to the color change position if not already there
        if (self.current_x, self.current_y) != tuple(self.color_change_position):
            print("Moving to color change position...")
            self.move_to(self.color_change_position[0], self.color_change_position[1])
            
        # Display instructions to the user for loading colors
        print("\nPLEASE LOAD THE FOLLOWING COLORS INTO THE ROBOT:")
        for i, color in enumerate(color_list):
            color_idx = color['index']
            rgb = color['rgb']
            print(f"Port {i+1}: Color {color_idx} - RGB{tuple(rgb)}")
            
        # In a real system, you might implement a sensor or button to confirm loading
        # For simulation, just wait for confirmation
        input("\nPress Enter when colors are loaded...")
        
        # Update the loaded colors
        self.loaded_colors = [color['index'] for color in color_list]
        print(f"Loaded {len(self.loaded_colors)} colors: {self.loaded_colors}")
        
        return True
    
    def calibrate(self):
        """
        Run the calibration procedure for the robot.
        This establishes the baseline string lengths and motor positions.
        """
        if not self.connected:
            print("Robot not connected.")
            return False
            
        print("\nStarting robot calibration procedure...")
        
        # Step 1: Move to home position
        print("Step 1: Moving to home position...")
        self._send_command(self.left_motor, "HOME")
        self._send_command(self.right_motor, "HOME")
        time.sleep(2)  # Wait for homing to complete
        
        # Step 2: Measure and set initial string lengths
        print("Step 2: Setting initial string lengths...")
        left_init, right_init = self._xy_to_string_lengths(self.home_position[0], self.home_position[1])
        self._send_command(self.left_motor, f"SET_LENGTH {left_init:.2f}")
        self._send_command(self.right_motor, f"SET_LENGTH {right_init:.2f}")
        
        # Step 3: Test movements to calibration points
        print("Step 3: Testing calibration movements...")
        
        # Test points: corners and center
        test_points = [
            (0, 0),                     # Top-left
            (self.wall_width, 0),       # Top-right
            (0, self.wall_height),      # Bottom-left
            (self.wall_width, self.wall_height),  # Bottom-right
            (self.wall_width/2, self.wall_height/2)  # Center
        ]
        
        for i, (x, y) in enumerate(test_points):
            print(f"Testing point {i+1}: ({x}, {y})...")
            self.move_to(x, y)
            input("Press Enter to continue to next point...")
        
        # Step 4: Return to home
        print("Step 4: Returning to home position...")
        self.home()
        
        print("\nCalibration complete!")
        return True
    
    def execute_instructions(self, instructions_file):
        """
        Execute a set of painting instructions from a JSON file.
        
        Args:
            instructions_file: Path to the JSON file with painting instructions
        """
        if not self.connected:
            print("Robot not connected. Please connect first.")
            return False
            
        try:
            # Load instructions
            with open(instructions_file, 'r') as f:
                data = json.load(f)
                
            instructions = data.get('instructions', [])
            colors = data.get('colors', [])
            
            if not instructions:
                print("No instructions found in file.")
                return False
                
            print(f"Loaded {len(instructions)} instructions.")
            
            # Confirm start
            input("\nPress Enter to start painting...")
            
            # Execute each instruction
            for i, instruction in enumerate(tqdm(instructions, desc="Executing instructions")):
                instruction_type = instruction.get('type')
                message = instruction.get('message', '')
                
                # Log what we're doing
                print(f"\nInstruction {i+1}/{len(instructions)}: {message}")
                
                # Execute based on instruction type
                if instruction_type == 'home':
                    self.home()
                    
                elif instruction_type == 'move':
                    x = instruction.get('x', 0)
                    y = instruction.get('y', 0)
                    spray = instruction.get('spray', False)
                    self.move_to(x, y, spray)
                    
                elif instruction_type == 'spray':
                    state = instruction.get('state', False)
                    self.set_spray(state)
                    
                elif instruction_type == 'color':
                    color_idx = instruction.get('index')
                    self.set_color(color_idx)
                    
                elif instruction_type == 'load_colors':
                    colors_to_load = instruction.get('colors', [])
                    self.load_colors(colors_to_load)
                    
                else:
                    print(f"Unknown instruction type: {instruction_type}")
                
                # Small pause between instructions for stability
                time.sleep(0.1)
                
            print("\nPainting complete!")
            return True
            
        except Exception as e:
            print(f"Error executing instructions: {e}")
            return False
    
    def simulate_execution(self, instructions_file, output_file=None):
        """
        Simulate the execution of painting instructions without real hardware.
        Useful for testing and previewing.
        
        Args:
            instructions_file: Path to the JSON file with painting instructions
            output_file: Path to output a simulation log (optional)
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
                
            print(f"Loaded {len(instructions)} instructions for simulation.")
            
            # Set up simulation
            self.current_x = 0
            self.current_y = 0
            self.current_color_index = None
            self.spray_active = False
            self.loaded_colors = []
            
            # Simulation log
            log = []
            
            # Execute each instruction in simulation
            for i, instruction in enumerate(tqdm(instructions, desc="Simulating")):
                instruction_type = instruction.get('type')
                message = instruction.get('message', '')
                
                # Add to log
                entry = {
                    "instruction_index": i,
                    "type": instruction_type,
                    "details": instruction,
                    "position": {"x": self.current_x, "y": self.current_y},
                    "spray_active": self.spray_active,
                    "color_index": self.current_color_index
                }
                
                # Simulate instruction
                if instruction_type == 'home':
                    self.current_x = self.home_position[0]
                    self.current_y = self.home_position[1]
                    
                elif instruction_type == 'move':
                    self.current_x = instruction.get('x', 0)
                    self.current_y = instruction.get('y', 0)
                    self.spray_active = instruction.get('spray', False)
                    
                elif instruction_type == 'spray':
                    self.spray_active = instruction.get('state', False)
                    
                elif instruction_type == 'color':
                    color_idx = instruction.get('index')
                    if color_idx in self.loaded_colors:
                        self.current_color_index = color_idx
                    else:
                        print(f"Warning: Color {color_idx} not loaded in simulation")
                    
                elif instruction_type == 'load_colors':
                    colors_to_load = instruction.get('colors', [])
                    self.loaded_colors = [c['index'] for c in colors_to_load]
                
                # Update position in log entry
                entry["end_position"] = {"x": self.current_x, "y": self.current_y}
                entry["end_spray_active"] = self.spray_active
                entry["end_color_index"] = self.current_color_index
                
                log.append(entry)
                
                # Simulate time passing (for illustration)
                time.sleep(0.01)
            
            # Save simulation log if output file is specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump({
                        "simulation_log": log,
                        "wall_width": self.wall_width,
                        "wall_height": self.wall_height,
                        "colors": colors
                    }, f, indent=2)
                print(f"Simulation log saved to {output_file}")
                
            print("\nSimulation complete!")
            
            # Calculate some statistics
            spray_count = sum(1 for entry in log if entry["type"] == "spray" and entry["details"].get("state", False))
            move_distance = 0
            prev_pos = {"x": self.home_position[0], "y": self.home_position[1]}
            
            for entry in log:
                if entry["type"] == "move":
                    x = entry["details"].get("x", 0)
                    y = entry["details"].get("y", 0)
                    move_distance += math.sqrt((x - prev_pos["x"])**2 + (y - prev_pos["y"])**2)
                    prev_pos = {"x": x, "y": y}
            
            print(f"Total spray actions: {spray_count}")
            print(f"Total travel distance: {move_distance:.2f} mm")
            
            return True
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return False

# Example usage when run as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Control a string-based mural painting robot')
    parser.add_argument('--config', '-c', default='config.json', help='Path to configuration file')
    parser.add_argument('--instructions', '-i', help='Path to painting instructions JSON file')
    parser.add_argument('--simulate', '-s', action='store_true', help='Run in simulation mode')
    parser.add_argument('--calibrate', '-cal', action='store_true', help='Run calibration procedure')
    parser.add_argument('--output', '-o', help='Output file for simulation log')
    
    args = parser.parse_args()
    
    controller = MuralRobotController(args.config)
    
    if args.simulate:
        # Run in simulation mode
        if not args.instructions:
            print("Error: Simulation requires --instructions parameter")
            exit(1)
            
        controller.simulate_execution(args.instructions, args.output)
        
    else:
        # Connect to hardware
        if controller.connect():
            # Run calibration if requested
            if args.calibrate:
                controller.calibrate()
                
            # Execute instructions if provided
            if args.instructions:
                controller.execute_instructions(args.instructions)
                
            # Disconnect at the end
            controller.disconnect()