#!/usr/bin/env python
"""
MuralBot - Console Application

This module contains the console interface for the MuralBot system.
"""

import os
import sys
import json
import time

from config_manager import ConfigManager
from image_to_instructions import MuralInstructionGenerator
from robot_controller import MuralRobotController
from instruction_visualizer import MuralVisualizer

class ConsoleApp:
    """
    Console application for the MuralBot mural painting robot.
    Provides a wizard-based interface for running the MuralBot in a terminal.
    """
    
    def __init__(self, config_path="config.json"):
        """Initialize the application with the given config file."""
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        
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
        """Run the console application in wizard mode."""
        print("\n" + "="*50)
        print("   MURALBOT - Mural Painting Robot Controller")
        print("="*50)
        print("\nWelcome to MuralBot! This wizard will guide you through the process.")
        
        # Step 1: Choose the input image
        self._step1_select_image()
        
        # Step 2: Process the image
        output_path = self._step2_process_image()
        
        # Step 3: Visualization
        self._step3_visualization(output_path)
        
        # Step 4: Robot Control
        self._step4_robot_control(output_path)
        
        print("\nMuralBot process complete! Thanks for using MuralBot.")
        
    def _step1_select_image(self):
        """Step 1: Select input image."""
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
                
    def _step2_process_image(self):
        """Step 2: Process the image."""
        print("\nSTEP 2: Process Image")
        print(f"This will convert the image into painting instructions.")
        print(f"- Wall dimensions: {self.wall_width}mm x {self.wall_height}mm")
        print(f"- Quantization method: {self.quantization_method}")
        print(f"- Color selection: {self.color_selection}")
        print(f"- Maximum colors: {self.max_colors}")
        print(f"- Robot capacity: {self.robot_capacity} colors at once")
        
        choice = input("Process the image now? (Y/n): ").strip().lower()
        output_path = os.path.join(self.painting_folder, self.output_instructions)
        
        if not choice or choice == 'y':
            print("\nProcessing image...")
            instructions, quantized_image = self.instruction_generator.process_image(
                self.input_image, output_path)
            print(f"Instructions saved to {output_path}")
            
            # Show paint usage report
            print("\nPaint Usage Report:")
            print(self.instruction_generator.get_paint_usage_report())
        else:
            print("Skipping image processing.")
            
        return output_path
            
    def _step3_visualization(self, output_path):
        """Step 3: Generate visualizations."""
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
    
    def _step4_robot_control(self, output_path):
        """Step 4: Robot control."""
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
    
    def run_batch(self, image_path, output_path=None, visualize=False, animate=False):
        """Run in batch mode with specified parameters."""
        if not output_path:
            output_path = os.path.join(self.painting_folder, self.output_instructions)
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self.painting_folder, output_path)
        
        # Update the input image
        self.input_image = image_path
        
        # Process image
        print(f"Processing image: {image_path}")
        instructions, quantized_image = self.instruction_generator.process_image(
            self.input_image, output_path)
        
        print(f"Instructions saved to {output_path}")
        
        # Create visualizations if requested
        if visualize or animate:
            print("\nGenerating visualization...")
            self.visualizer.visualize_instructions(output_path, output_path.replace('.json', '.jpg'))
            
            if animate:
                print("\nGenerating animation...")
                animation_path = os.path.join(self.painting_folder, self.output_video)
                self.visualizer.animate_painting_process(output_path, animation_path)
                
        return output_path