#!/usr/bin/env python
"""
MuralBot - Main Program

This is the main entry point for the MuralBot mural painting robot system.
Run this file without parameters to start the interactive interface.
"""

import os
import sys
import argparse

from config_manager import ConfigManager
from console_app import ConsoleApp
from gui_app import GUIApp
from image_to_instructions import MuralInstructionGenerator
from instruction_visualizer import MuralVisualizer

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='MuralBot: Convert image to robot painting instructions')
    parser.add_argument('--config', '-c', default='config.json', help='Path to configuration file')
    parser.add_argument('--image', '-i', help='Path to input image (overrides config file)')
    parser.add_argument('--output', '-o', help='Output path for instructions (overrides config file)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization')
    parser.add_argument('--animate', '-a', action='store_true', help='Create animation of painting process')
    parser.add_argument('--no_preview', '-np', action='store_true', help='Disable preview window')
    parser.add_argument('--console', action='store_true', help='Force console mode even if GUI is available')
    parser.add_argument('--gui', action='store_true', help='Force GUI mode')
    
    args = parser.parse_args()
    
    # Load configuration manager
    config_manager = ConfigManager(args.config)
    
    # Determine if we should use GUI or console mode
    use_gui = False
    if args.gui:
        use_gui = True
    elif args.console:
        use_gui = False
    else:
        # Auto-detect: Use GUI if tkinter is available and we're not in a headless environment
        try:
            import tkinter
            use_gui = True
            # Check if we're in a headless environment (no display)
            if "DISPLAY" in os.environ and not os.environ["DISPLAY"]:
                use_gui = False
        except ImportError:
            use_gui = False
    
    # Check if specific commands were provided
    command_mode = args.image is not None
    
    if command_mode:
        # Run in command/batch mode
        console_app = ConsoleApp(args.config)
        output_path = console_app.run_batch(
            args.image, 
            args.output, 
            args.visualize, 
            args.animate
        )
        print("\nMuralBot processing complete!")
    elif use_gui:
        # Run in GUI mode
        print("Starting MuralBot in GUI mode...")
        app = GUIApp(args.config)
        app.run()
    else:
        # Run in console mode
        print("Starting MuralBot in console mode...")
        app = ConsoleApp(args.config)
        app.run()

if __name__ == "__main__":
    main()