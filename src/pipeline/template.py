"""
Template for creating new processor components in the Foundation Pose pipeline.

This template demonstrates how to create a new processor with proper logging
and error handling integrated.

Usage:
1. Copy this template
2. Replace YourProcessorName with your actual processor name
3. Implement the required methods
4. Add your processor to the pipeline
"""

import os
import logging
import numpy as np
from .processor import BaseProcessor

class YourProcessorName(BaseProcessor):
    """
    A template processor for the pipeline.
    
    Replace this docstring with a description of what your processor does.
    """
    
    def __init__(self, parameter1, parameter2=None, debug_dir=None):
        """
        Initialize your processor.
        
        Args:
            parameter1: Description of parameter1
            parameter2: Description of parameter2 (default: None)
            debug_dir: Directory for debug outputs (default: None)
        """
        # Always call the parent constructor first to set up logging
        super().__init__(debug_dir)
        
        # Store parameters
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        
        # Initialize any other instance variables
        self.initialized = False
        
        # Initialize your component
        try:
            # Try to initialize components, libraries, etc.
            # Import any required libraries here
            
            self.initialized = True
            self.logger.info("✅ YourProcessorName initialized successfully")
        except Exception as e:
            self.log_error(f"Failed to initialize: {str(e)}")
    
    def process(self, data):
        """
        Process the input data.
        
        Args:
            data: PipelineData object
            
        Returns:
            Updated PipelineData object
        """
        # Check if initialization was successful
        if not self.initialized:
            self.log_error("Component not properly initialized", data)
            return data
        
        # Check required inputs
        if data.some_required_property is None:
            self.log_error("Missing required input", data)
            return data
            
        self.logger.info("🚀 Starting YourProcessorName processing")
        
        try:
            # Log the start of a specific step
            self.log_step_start("first step")
            
            # Your processing code here
            # ...
            
            # Log completion of the step
            self.log_step_complete("first step")
            
            # Another processing step
            self.log_step_start("second step")
            
            # More processing code
            # ...
            
            # Save debug visualization if needed
            if self.debug_dir:
                debug_path = os.path.join(self.debug_dir, f"debug_{data.simple_timestamp}.png")
                # Save your debug output here
                self.logger.info(f"✅ Debug output saved to {debug_path}")
            
            # Store results in the data object
            data.your_output = "your result"
            
            # Create visualization for the pipeline
            vis = data.rgb_image.copy()  # Example: create a visualization based on the RGB image
            
            # Store visualization for later use
            data.save_debug_image("your_processor", vis)
            
            self.log_step_complete("second step")
            self.logger.info("✅ YourProcessorName completed successfully")
            return data
            
        except Exception as e:
            self.log_error(f"Error during processing: {str(e)}", data)
            import traceback
            traceback.print_exc()
            return data
    
    def visualize(self, data):
        """
        Visualize the results of processing.
        
        Args:
            data: PipelineData object after processing
        """
        # Check if we have visualization data
        if "your_processor" in data.debug_images and data.debug_images["your_processor"] is not None:
            import cv2
            
            # Get the visualization image
            vis_img = data.debug_images["your_processor"]
            
            # Display the visualization
            window_title = "Your Processor Results"
            cv2.imshow(window_title, vis_img)
            
            # Wait for a key press and then close the window
            key = cv2.waitKey(0)
            cv2.destroyWindow(window_title) 