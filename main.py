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
from instruction_visualizer_optimized import MuralVisualizer

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='MuralBot: Convert image to robot painting instructions')
    parser.add_argument('--config', '-c', default='config.json', help='Path to configuration file')
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
    else:        # Auto-detect: Use GUI if tkinter is available
        try:
            import tkinter
            # Try creating a test window to ensure GUI works
            test_window = tkinter.Tk()
            test_window.withdraw()  # Hide the window
            test_window.destroy()   # Destroy the window
            use_gui = True
        except (ImportError, tkinter.TclError):
            # Either tkinter is not available or it can't create a window
            use_gui = False
    
    # All additional parameters now come from config file or GUI
    if use_gui:
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