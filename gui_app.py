#!/usr/bin/env python
"""
MuralBot - GUI Application

This module contains the GUI interface for the MuralBot system.
"""

import os
import sys
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config_manager import ConfigManager
from image_to_instructions import MuralInstructionGenerator
from robot_controller import MuralRobotController
from instruction_visualizer import MuralVisualizer

class GUIApp:
    """
    GUI application for controlling the MuralBot mural painting robot.
    Provides a simple tab-based interface for the full process.
    """
    
    def __init__(self, config_path="config.json"):
        """Initialize the application with the given config file."""
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config_path = config_path
        
        # Get settings
        img_settings = self.config_manager.get_image_processing_settings()
        self.painting_folder = self.config_manager.get_painting_folder()
        
        # Extract settings
        self.input_image = img_settings["input_image"]
        self.wall_width = img_settings["wall_width"]
        self.wall_height = img_settings["wall_height"]
        self.resolution_mm = img_settings["resolution_mm"]
        self.output_instructions = img_settings["output_instructions"]
        self.quantization_method = img_settings["quantization_method"]
        self.dithering = img_settings["dithering"]
        self.max_colors = img_settings["max_colors"]
        self.robot_capacity = img_settings["robot_capacity"]
        self.color_selection = img_settings["color_selection"]
        self.available_colors = img_settings["available_colors"]
        
        # Get visualization settings
        vis_settings = self.config_manager.get_visualization_settings()
        self.output_video = vis_settings["output_video"]
        
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
        
        # Set any additional hardware settings
        hardware_settings = self.config_manager.get_hardware_settings()
        if 'color_change_position' in hardware_settings:
            self.instruction_generator.color_change_position = hardware_settings['color_change_position']
        
        self.robot_controller = MuralRobotController(config_path)
        self.visualizer = MuralVisualizer(config_path)
        
    def run(self):
        """Run the GUI application."""
        self.root = tk.Tk()
        self.root.title("MuralBot Controller")
        self.root.geometry("800x600")
        
        # Set style
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        
        # Create frame for steps
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = ttk.Label(main_frame, text="MuralBot - Mural Painting Robot Controller", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create notebook for different steps
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create all tabs
        self._create_image_selection_tab()
        self._create_image_processing_tab()
        self._create_visualization_tab()
        self._create_robot_control_tab()
        
        # Help info
        help_text = "Start by selecting an image, then follow the numbered steps."
        help_label = ttk.Label(main_frame, text=help_text, font=("Arial", 9, "italic"))
        help_label.pack(pady=5)
        
        # Status bar
        status_frame = ttk.Frame(self.root, relief="sunken", padding=(2, 2))
        status_frame.pack(side="bottom", fill="x")
        
        self.status_text = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_text, anchor="w")
        status_label.pack(side="left", fill="x", expand=True)
        
        self.root.mainloop()
        
    def _create_image_selection_tab(self):
        """Create the image selection tab."""
        step1_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(step1_frame, text="1. Select Image")
        
        ttk.Label(step1_frame, text="Select Input Image", font=("Arial", 12, "bold")).pack(anchor="w")
        
        image_frame = ttk.Frame(step1_frame)
        image_frame.pack(fill="x", pady=5)
        
        self.image_var = tk.StringVar(value=self.input_image)
        ttk.Entry(image_frame, textvariable=self.image_var, width=50).pack(side="left", padx=5)
        ttk.Button(image_frame, text="Browse...", 
                 command=lambda: self.image_var.set(filedialog.askopenfilename(
                     filetypes=[("Image files", "*.jpg *.jpeg *.png")]))).pack(side="left")
                     
    def _create_image_processing_tab(self):
        """Create the image processing tab."""
        step2_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(step2_frame, text="2. Process Image")
        
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
                                  command=self.process_image)
        process_button.pack(pady=10)
        
        self.process_status = tk.StringVar(value="")
        ttk.Label(step2_frame, textvariable=self.process_status).pack(pady=5)
        
    def _create_visualization_tab(self):
        """Create the visualization tab."""
        step3_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(step3_frame, text="3. Visualization")
        
        ttk.Label(step3_frame, text="Visualization Options", font=("Arial", 12, "bold")).pack(anchor="w")
        
        preview_button = ttk.Button(step3_frame, text="Generate Preview Image", 
                                  command=self.generate_preview)
        preview_button.pack(fill="x", pady=5)
        
        paths_button = ttk.Button(step3_frame, text="Show Robot Paths", 
                                command=self.show_paths)
        paths_button.pack(fill="x", pady=5)
        
        animation_button = ttk.Button(step3_frame, text="Generate Animation", 
                                    command=self.generate_animation)
        animation_button.pack(fill="x", pady=5)
        
        self.vis_status = tk.StringVar(value="")
        ttk.Label(step3_frame, textvariable=self.vis_status).pack(pady=5)
        
    def _create_robot_control_tab(self):
        """Create the robot control tab."""
        step4_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(step4_frame, text="4. Robot Control")
        
        ttk.Label(step4_frame, text="Robot Control", font=("Arial", 12, "bold")).pack(anchor="w")
        
        sim_button = ttk.Button(step4_frame, text="Run Simulation", 
                              command=self.run_simulation)
        sim_button.pack(fill="x", pady=5)
        
        connect_button = ttk.Button(step4_frame, text="Connect to Robot", 
                                  command=self.connect_robot)
        connect_button.pack(fill="x", pady=5)
        
        calibrate_button = ttk.Button(step4_frame, text="Calibrate Robot", 
                                    command=self.calibrate_robot)
        calibrate_button.pack(fill="x", pady=5)
        
        execute_button = ttk.Button(step4_frame, text="Execute Painting", 
                                  command=self.execute_painting)
        execute_button.pack(fill="x", pady=5)
        
        self.robot_status = tk.StringVar(value="Robot Status: Disconnected")
        ttk.Label(step4_frame, textvariable=self.robot_status).pack(pady=5)
        
    def process_image(self):
        """Process image button handler."""
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
            
            # Advanced to next tab
            self.notebook.select(2)  # Switch to visualization tab (0-indexed)
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            self.process_status.set(error_msg)
            self.status_text.set("Error")
            messagebox.showerror("Processing Error", error_msg)
    
    def generate_preview(self):
        """Generate preview button handler."""
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
    
    def show_paths(self):
        """Show paths button handler."""
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
    
    def generate_animation(self):
        """Generate animation button handler."""
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
    
    def run_simulation(self):
        """Run simulation button handler."""
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
    
    def connect_robot(self):
        """Connect robot button handler."""
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
    
    def calibrate_robot(self):
        """Calibrate robot button handler."""
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
    
    def execute_painting(self):
        """Execute painting button handler."""
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