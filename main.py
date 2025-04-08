#!/usr/bin/env python
"""
MuralBot - Main Program

This is the main entry point for the MuralBot mural painting robot system.
Run this file without parameters to start the interactive interface.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Import our modules
from image_to_instructions import MuralInstructionGenerator
from robot_controller import MuralRobotController
from instruction_visualizer import MuralVisualizer

class MuralBotApp:
    """
    Main application for controlling the MuralBot mural painting robot.
    Provides a simple workflow-based interface for the full process.
    """
    
    def __init__(self, config_path="config.json"):
        """Initialize the application with the given config file."""
        self.config_path = config_path
        self.load_config()
        
        # Ensure painting folder exists
        self.painting_folder = os.path.join(os.path.dirname(__file__), "painting")
        os.makedirs(self.painting_folder, exist_ok=True)
        
        # Create components
        self.instruction_generator = MuralInstructionGenerator(
            wall_width=self.wall_width,
            wall_height=self.wall_height,
            available_colors=self.available_colors,
            resolution_mm=self.resolution_mm,
            quantization_method=self.quantization_method,
            dithering=self.dithering,
            max_colors=self.max_colors,
            robot_capacity=self.robot_capacity,
            color_selection=self.color_selection
        )
        
        self.robot_controller = MuralRobotController(self.config_path)
        self.visualizer = MuralVisualizer(self.config_path)
        
    def load_config(self):
        """Load configuration from the JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            # Load image processing settings
            img_config = config.get('image_processing', {})
            self.input_image = img_config.get('input_image', 'mural.jpg')
            self.wall_width = img_config.get('wall_width', 2000)
            self.wall_height = img_config.get('wall_height', 1500)
            self.resolution_mm = img_config.get('resolution_mm', 5.0)
            self.output_instructions = img_config.get('output_instructions', 'painting_instructions.json')
            self.quantization_method = img_config.get('quantization_method', 'kmeans')
            self.dithering = img_config.get('dithering', False)
            self.max_colors = img_config.get('max_colors', 12)
            self.robot_capacity = img_config.get('robot_capacity', 6)
            self.color_selection = img_config.get('color_selection', 'auto')
            self.available_colors = img_config.get('available_colors', [])
            
            # Load visualization settings
            vis_config = config.get('visualization', {})
            self.instructions_file = vis_config.get('instructions_file', 'painting_instructions.json')
            self.output_video = vis_config.get('output_video', 'painting_simulation.mp4')
            
            print(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default settings")
            
            # Set default values for critical parameters
            self.input_image = 'mural.jpg'
            self.wall_width = 2000
            self.wall_height = 1500
            self.resolution_mm = 5.0
            self.output_instructions = 'painting_instructions.json'
            self.quantization_method = 'kmeans'
            self.dithering = False
            self.max_colors = 12
            self.robot_capacity = 6
            self.color_selection = 'auto'
            self.instructions_file = 'painting_instructions.json'
            self.output_video = 'painting_simulation.mp4'
            self.available_colors = []
    
    def run_console(self):
        """Run the application in console mode (non-GUI)."""
        print("\n" + "="*50)
        print("   MURALBOT - Mural Painting Robot Controller")
        print("="*50)
        print("\nWelcome to MuralBot! This wizard will guide you through the process.")
        
        # Step 1: Choose the input image
        print("\nSTEP 1: Select Input Image")
        print(f"Default image: {self.input_image}")
        choice = input("Use default image? (Y/n): ").strip().lower()
        if choice and choice != 'y':
            new_image = input("Enter path to image file: ").strip()
            if os.path.exists(new_image):
                self.input_image = new_image
                print(f"Using image: {self.input_image}")
            else:
                print(f"Image not found, using default: {self.input_image}")
        
        # Step 2: Process the image
        print("\nSTEP 2: Process Image")
        print(f"This will convert the image into painting instructions.")
        print(f"- Wall dimensions: {self.wall_width}mm x {self.wall_height}mm")
        print(f"- Quantization method: {self.quantization_method}")
        print(f"- Color selection: {self.color_selection}")
        print(f"- Maximum colors: {self.max_colors}")
        print(f"- Robot capacity: {self.robot_capacity} colors at once")
        
        choice = input("Process the image now? (Y/n): ").strip().lower()
        if not choice or choice == 'y':
            print("\nProcessing image...")
            output_path = os.path.join(self.painting_folder, self.output_instructions)
            instructions, quantized_image = self.instruction_generator.process_image(
                self.input_image, output_path)
            print(f"Instructions saved to {output_path}")
            
            # Show paint usage report
            print("\nPaint Usage Report:")
            print(self.instruction_generator.get_paint_usage_report())
        else:
            print("Skipping image processing.")
            
        # Step 3: Visualization
        print("\nSTEP 3: Visualization")
        print("You can generate visualizations to see how the mural will look.")
        
        choice = input("Generate a preview image? (Y/n): ").strip().lower()
        if not choice or choice == 'y':
            print("\nGenerating preview image...")
            self.visualizer.create_preview_image(output_path)
            
        choice = input("Generate robot path visualization? (Y/n): ").strip().lower()
        if not choice or choice == 'y':
            print("\nGenerating robot path visualization...")
            self.visualizer.visualize_robot_paths(output_path)
            
        choice = input("Generate painting animation? (Y/n): ").strip().lower()
        if not choice or choice == 'y':
            print("\nGenerating painting animation...")
            animation_path = os.path.join(self.painting_folder, self.output_video)
            self.visualizer.animate_painting_process(
                output_path, animation_path)
            print(f"Animation saved to {animation_path}")
        
        # Step 4: Robot Control
        print("\nSTEP 4: Robot Control")
        print("You can run a simulation or control the actual robot.")
        
        choice = input("Run in simulation mode? (Y/n): ").strip().lower()
        if not choice or choice == 'y':
            print("\nRunning simulation...")
            simulation_log_path = os.path.join(self.painting_folder, "simulation_log.json")
            self.robot_controller.simulate_execution(
                output_path, simulation_log_path)
            
        choice = input("Connect to real robot hardware? (y/N): ").strip().lower()
        if choice == 'y':
            print("\nConnecting to robot hardware...")
            if self.robot_controller.connect():
                print("\nRobot connected successfully.")
                
                choice = input("Run calibration procedure? (y/N): ").strip().lower()
                if choice == 'y':
                    self.robot_controller.calibrate()
                
                choice = input("Execute painting instructions on robot? (y/N): ").strip().lower()
                if choice == 'y':
                    print("\nExecuting painting instructions on robot...")
                    self.robot_controller.execute_instructions(output_path)
                    
                # Disconnect at the end
                self.robot_controller.disconnect()
            else:
                print("Failed to connect to robot. Make sure hardware is connected.")
        
        print("\nMuralBot process complete! Thanks for using MuralBot.")
        
    def run_gui(self):
        """Run the application in GUI mode."""
        root = tk.Tk()
        root.title("MuralBot Controller")
        root.geometry("800x600")
        
        # Set style
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        
        # Create frame for steps
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = ttk.Label(main_frame, text="MuralBot - Mural Painting Robot Controller", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create notebook for different steps
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Step 1: Image Selection
        step1_frame = ttk.Frame(notebook, padding=10)
        notebook.add(step1_frame, text="1. Select Image")
        
        ttk.Label(step1_frame, text="Select Input Image", font=("Arial", 12, "bold")).pack(anchor="w")
        
        image_frame = ttk.Frame(step1_frame)
        image_frame.pack(fill="x", pady=5)
        
        self.image_var = tk.StringVar(value=self.input_image)
        ttk.Entry(image_frame, textvariable=self.image_var, width=50).pack(side="left", padx=5)
        ttk.Button(image_frame, text="Browse...", 
                 command=lambda: self.image_var.set(filedialog.askopenfilename(
                     filetypes=[("Image files", "*.jpg *.jpeg *.png")]))).pack(side="left")
        
        # Step 2: Image Processing
        step2_frame = ttk.Frame(notebook, padding=10)
        notebook.add(step2_frame, text="2. Process Image")
        
        ttk.Label(step2_frame, text="Image Processing Settings", font=("Arial", 12, "bold")).pack(anchor="w")
        
        settings_frame = ttk.Frame(step2_frame)
        settings_frame.pack(fill="x", pady=10)
        
        # Settings grid
        ttk.Label(settings_frame, text="Wall Width (mm):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.width_var = tk.StringVar(value=str(self.wall_width))
        ttk.Entry(settings_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Wall Height (mm):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.height_var = tk.StringVar(value=str(self.wall_height))
        ttk.Entry(settings_frame, textvariable=self.height_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Quantization:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.quant_var = tk.StringVar(value=self.quantization_method)
        ttk.Combobox(settings_frame, textvariable=self.quant_var, values=["euclidean", "kmeans", "color_palette"],
                   width=15).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Maximum Colors:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.max_colors_var = tk.StringVar(value=str(self.max_colors))
        ttk.Entry(settings_frame, textvariable=self.max_colors_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Robot Capacity:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.robot_capacity_var = tk.StringVar(value=str(self.robot_capacity))
        ttk.Entry(settings_frame, textvariable=self.robot_capacity_var, width=10).grid(row=4, column=1, padx=5, pady=2)
        
        self.dither_var = tk.BooleanVar(value=self.dithering)
        ttk.Checkbutton(settings_frame, text="Enable Dithering", variable=self.dither_var).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Process button
        process_button = ttk.Button(step2_frame, text="Process Image", 
                                  command=self.process_image_gui)
        process_button.pack(pady=10)
        
        self.process_status = tk.StringVar(value="")
        ttk.Label(step2_frame, textvariable=self.process_status).pack(pady=5)
        
        # Step 3: Visualization
        step3_frame = ttk.Frame(notebook, padding=10)
        notebook.add(step3_frame, text="3. Visualization")
        
        ttk.Label(step3_frame, text="Visualization Options", font=("Arial", 12, "bold")).pack(anchor="w")
        
        preview_button = ttk.Button(step3_frame, text="Generate Preview Image", 
                                  command=self.generate_preview_gui)
        preview_button.pack(fill="x", pady=5)
        
        paths_button = ttk.Button(step3_frame, text="Show Robot Paths", 
                                command=self.show_paths_gui)
        paths_button.pack(fill="x", pady=5)
        
        animation_button = ttk.Button(step3_frame, text="Generate Animation", 
                                    command=self.generate_animation_gui)
        animation_button.pack(fill="x", pady=5)
        
        self.vis_status = tk.StringVar(value="")
        ttk.Label(step3_frame, textvariable=self.vis_status).pack(pady=5)
        
        # Step 4: Robot Control
        step4_frame = ttk.Frame(notebook, padding=10)
        notebook.add(step4_frame, text="4. Robot Control")
        
        ttk.Label(step4_frame, text="Robot Control", font=("Arial", 12, "bold")).pack(anchor="w")
        
        sim_button = ttk.Button(step4_frame, text="Run Simulation", 
                              command=self.run_simulation_gui)
        sim_button.pack(fill="x", pady=5)
        
        connect_button = ttk.Button(step4_frame, text="Connect to Robot", 
                                  command=self.connect_robot_gui)
        connect_button.pack(fill="x", pady=5)
        
        calibrate_button = ttk.Button(step4_frame, text="Calibrate Robot", 
                                    command=self.calibrate_robot_gui)
        calibrate_button.pack(fill="x", pady=5)
        
        execute_button = ttk.Button(step4_frame, text="Execute Painting", 
                                  command=self.execute_painting_gui)
        execute_button.pack(fill="x", pady=5)
        
        self.robot_status = tk.StringVar(value="Robot Status: Disconnected")
        ttk.Label(step4_frame, textvariable=self.robot_status).pack(pady=5)
        
        # Help info
        help_text = "Start by selecting an image, then follow the numbered steps."
        help_label = ttk.Label(main_frame, text=help_text, font=("Arial", 9, "italic"))
        help_label.pack(pady=5)
        
        # Status bar
        status_frame = ttk.Frame(root, relief="sunken", padding=(2, 2))
        status_frame.pack(side="bottom", fill="x")
        
        self.status_text = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_text, anchor="w")
        status_label.pack(side="left", fill="x", expand=True)
        
        root.mainloop()
    
    def process_image_gui(self):
        """Process image button handler for GUI."""
        try:
            self.status_text.set("Processing image...")
            self.process_status.set("Processing...")
            
            # Update settings from GUI
            self.input_image = self.image_var.get()
            self.wall_width = int(self.width_var.get())
            self.wall_height = int(self.height_var.get())
            self.quantization_method = self.quant_var.get()
            self.max_colors = int(self.max_colors_var.get())
            self.robot_capacity = int(self.robot_capacity_var.get())
            self.dithering = self.dither_var.get()
            
            # Recreate instruction generator with new settings
            self.instruction_generator = MuralInstructionGenerator(
                wall_width=self.wall_width,
                wall_height=self.wall_height,
                available_colors=self.available_colors,
                resolution_mm=self.resolution_mm,
                quantization_method=self.quantization_method,
                dithering=self.dithering,
                max_colors=self.max_colors,
                robot_capacity=self.robot_capacity,
                color_selection=self.color_selection
            )
            
            # Process the image
            output_path = os.path.join(self.painting_folder, self.output_instructions)
            instructions, quantized_image = self.instruction_generator.process_image(
                self.input_image, output_path)
            
            message = f"Processing complete. Instructions saved to {output_path}"
            self.process_status.set(message)
            self.status_text.set(message)
            
            messagebox.showinfo("Processing Complete", 
                              "Image processing complete!\n\nProceed to Visualization tab.")
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            self.process_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Processing Error", error_msg)
    
    def generate_preview_gui(self):
        """Generate preview button handler for GUI."""
        try:
            self.status_text.set("Generating preview...")
            self.vis_status.set("Generating preview image...")
            
            output_path = os.path.join(self.painting_folder, self.output_instructions)
            canvas = self.visualizer.create_preview_image(output_path)
            
            message = "Preview image saved as 'mural_preview.jpg'"
            self.vis_status.set(message)
            self.status_text.set("Preview complete")
            
            # Try to open the image with system viewer
            try:
                import webbrowser
                webbrowser.open('mural_preview.jpg')
            except Exception:
                pass
            
        except Exception as e:
            error_msg = f"Error generating preview: {str(e)}"
            self.vis_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Visualization Error", error_msg)
    
    def show_paths_gui(self):
        """Show paths button handler for GUI."""
        try:
            self.status_text.set("Generating path visualization...")
            self.vis_status.set("Generating robot paths...")
            
            output_path = os.path.join(self.painting_folder, self.output_instructions)
            self.visualizer.visualize_robot_paths(output_path)
            
            message = "Path visualization complete"
            self.vis_status.set(message)
            self.status_text.set(message)
            
        except Exception as e:
            error_msg = f"Error visualizing paths: {str(e)}"
            self.vis_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Visualization Error", error_msg)
    
    def generate_animation_gui(self):
        """Generate animation button handler for GUI."""
        try:
            self.status_text.set("Generating animation...")
            self.vis_status.set("Creating animation...")
            
            output_path = os.path.join(self.painting_folder, self.output_instructions)
            animation_path = os.path.join(self.painting_folder, self.output_video)
            self.visualizer.animate_painting_process(
                output_path, animation_path)
            
            message = f"Animation saved to {animation_path}"
            self.vis_status.set(message)
            self.status_text.set("Animation complete")
            
            # Ask if user wants to see interactive visualization
            if messagebox.askyesno("Animation Complete", 
                                "Animation complete. Open interactive visualizer?"):
                self.visualizer.create_interactive_visualization(output_path)
            
        except Exception as e:
            error_msg = f"Error generating animation: {str(e)}"
            self.vis_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Animation Error", error_msg)
    
    def run_simulation_gui(self):
        """Run simulation button handler for GUI."""
        try:
            self.status_text.set("Running simulation...")
            self.robot_status.set("Simulation in progress...")
            
            output_path = os.path.join(self.painting_folder, self.output_instructions)
            simulation_log_path = os.path.join(self.painting_folder, "simulation_log.json")
            self.robot_controller.simulate_execution(
                output_path, simulation_log_path)
            
            message = f"Simulation complete. Log saved to {simulation_log_path}"
            self.robot_status.set(message)
            self.status_text.set("Simulation complete")
            
            messagebox.showinfo("Simulation Complete", message)
            
        except Exception as e:
            error_msg = f"Simulation error: {str(e)}"
            self.robot_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Simulation Error", error_msg)
    
    def connect_robot_gui(self):
        """Connect robot button handler for GUI."""
        try:
            self.status_text.set("Connecting to robot...")
            self.robot_status.set("Connecting...")
            
            if self.robot_controller.connect():
                message = "Robot connected"
                self.robot_status.set(message)
                self.status_text.set(message)
                messagebox.showinfo("Connection Success", "Robot connected successfully.")
            else:
                message = "Failed to connect to robot"
                self.robot_status.set(message)
                self.status_text.set("Connection failed")
                messagebox.showerror("Connection Failed", 
                                    "Failed to connect to robot. Check connections and ports.")
            
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            self.robot_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Connection Error", error_msg)
    
    def calibrate_robot_gui(self):
        """Calibrate robot button handler for GUI."""
        try:
            self.status_text.set("Calibrating robot...")
            self.robot_status.set("Calibration in progress...")
            
            if not hasattr(self.robot_controller, 'connected') or not self.robot_controller.connected:
                messagebox.showwarning("Robot Disconnected", 
                                     "Robot is not connected. Connect first.")
                self.robot_status.set("Robot not connected")
                self.status_text.set("Calibration aborted")
                return
                
            self.robot_controller.calibrate()
            
            message = "Calibration complete"
            self.robot_status.set(message)
            self.status_text.set(message)
            
        except Exception as e:
            error_msg = f"Calibration error: {str(e)}"
            self.robot_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Calibration Error", error_msg)
    
    def execute_painting_gui(self):
        """Execute painting button handler for GUI."""
        try:
            if not hasattr(self.robot_controller, 'connected') or not self.robot_controller.connected:
                messagebox.showwarning("Robot Disconnected", 
                                     "Robot is not connected. Connect first.")
                self.robot_status.set("Robot not connected")
                self.status_text.set("Execution aborted")
                return
                
            if not messagebox.askyesno("Confirm Execution", 
                                     "This will start the painting process. Continue?"):
                return
                
            self.status_text.set("Executing painting instructions...")
            self.robot_status.set("Painting in progress...")
            
            output_path = os.path.join(self.painting_folder, self.output_instructions)
            self.robot_controller.execute_instructions(output_path)
            
            message = "Painting complete"
            self.robot_status.set(message)
            self.status_text.set(message)
            messagebox.showinfo("Painting Complete", "The painting process has finished.")
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            self.robot_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Execution Error", error_msg)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='MuralBot: Convert image to robot painting instructions')
    parser.add_argument('--config', '-c', default='config.json', help='Path to configuration file')
    parser.add_argument('--image', '-i', help='Path to input image (overrides config file)')
    parser.add_argument('--output', '-o', help='Output path for instructions (overrides config file)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization')
    parser.add_argument('--animate', '-a', action='store_true', help='Create animation of painting process')
    parser.add_argument('--no_preview', '-np', action='store_true', help='Disable preview window')
    
    args = parser.parse_args()
    
    # Ensure painting folder exists
    painting_folder = os.path.join(os.path.dirname(__file__), "painting")
    os.makedirs(painting_folder, exist_ok=True)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default settings")
        config = {
            "image_processing": {},
            "visualization": {}
        }
    
    # Extract needed configuration
    img_config = config.get('image_processing', {})
    
    # Get image path (prioritize command line argument)
    image_path = args.image if args.image else img_config.get('input_image')
    if not image_path:
        print("Error: No input image specified. Please provide it in config file or as a command line argument.")
        return
        
    # Set output path for instructions (prioritize command line argument)
    output_path = args.output if args.output else img_config.get('output_instructions')
    if not output_path:
        output_path = "painting_instructions.json"
        
    # If output_path is not an absolute path, save to the painting folder
    if not os.path.isabs(output_path):
        output_path = os.path.join(painting_folder, output_path)
    
    # Step 1: Process image to generate instructions
    print(f"Processing image: {image_path}")
    generator = MuralInstructionGenerator(
        wall_width=img_config.get('wall_width', 2000),
        wall_height=img_config.get('wall_height', 1500),
        resolution_mm=img_config.get('resolution_mm', 5.0),
        quantization_method=img_config.get('quantization_method', 'euclidean'),
        dithering=img_config.get('dithering', False),
        max_colors=img_config.get('max_colors', 30),
        robot_capacity=img_config.get('robot_capacity', 6),
        color_selection=img_config.get('color_selection', 'auto')
    )
    
    # Process image and generate instructions
    instructions, quantized_image = generator.process_image(image_path, output_path)
    
    print(f"Instructions saved to {output_path}")
    
    # Save quantized preview
    quantized_preview_path = os.path.join(painting_folder, "quantized_preview.jpg")
    print(f"Quantized preview saved to {quantized_preview_path}")
    
    # Step 2: Visualize if requested
    if args.visualize or args.animate:
        print("\nGenerating visualization...")
        visualizer = MuralVisualizer(args.config)
        
        # Create visualization of the painting process
        visualization_path = os.path.join(painting_folder, "mural_visualization.jpg")
        visualizer.visualize_instructions(
            output_path, 
            visualization_path,
            show_preview=not args.no_preview,
            show_progress=True
        )
        
        # Generate animation if requested
        if args.animate:
            print("\nGenerating animation...")
            animation_path = os.path.join(painting_folder, "mural_animation.mp4")
            visualizer.animate_painting_process(output_path, animation_path)
    
    print("\nMuralBot processing complete!")

if __name__ == "__main__":
    main()