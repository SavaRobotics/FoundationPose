import cv2
import numpy as np
import os
from ...pipeline.base import Processor

class FileImageLoader(Processor):
    """Loads RGB and depth images from local files."""
    
    def __init__(self, rgb_path, depth_path=None, depth_npy_path=None):
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.depth_npy_path = depth_npy_path
        
    def process(self, data):
        """Load images and update pipeline data."""
        # Load RGB image
        print(f"Loading RGB image from {self.rgb_path}")
        rgb_img = cv2.imread(self.rgb_path)
        if rgb_img is None:
            data.add_error("FileImageLoader", f"Failed to load RGB image from {self.rgb_path}")
            return data
        
        # Load depth image
        depth_img = None
        
        # Try loading from NPY file first (for better precision)
        if self.depth_npy_path and os.path.exists(self.depth_npy_path):
            print(f"Loading depth from NPY file: {self.depth_npy_path}")
            try:
                depth_img = np.load(self.depth_npy_path)
            except Exception as e:
                print(f"Error loading depth from NPY: {str(e)}")
                
        # If NPY loading failed or no NPY file specified, try loading from image file
        if depth_img is None and self.depth_path:
            print(f"Loading depth from image file: {self.depth_path}")
            depth_img = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED)
            
            # If depth is in millimeters, convert to meters
            if depth_img is not None and (np.max(depth_img) > 10) and depth_img.dtype != np.float32:
                print("Converting depth from millimeters to meters...")
                depth_img = depth_img.astype(np.float32) / 1000.0
                
            # Ensure the depth image is 2D
            if depth_img is not None and len(depth_img.shape) > 2:
                print("WARNING: Depth image has more than 2 dimensions, taking first channel...")
                depth_img = depth_img[:,:,0]
                
            # Apply a minimal threshold to filter out noise
            if depth_img is not None:
                depth_img[depth_img < 0.001] = 0
        
        # Create depth visualization
        depth_viz_img = None
        if depth_img is not None:
            depth_viz_img = self._visualize_depth(depth_img)
            
        # Update data
        data.rgb_image = rgb_img
        data.depth_image = depth_img
        data.depth_viz_image = depth_viz_img
        
        return data
    
    def _visualize_depth(self, depth_img):
        """Create a heatmap visualization of depth data for debugging."""
        if depth_img is None:
            return None
            
        # Normalize depth for visualization
        depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Mark invalid regions (0 depth) as black
        if len(depth_img.shape) == 2:  # Make sure depth_img is 2D
            depth_colormap[depth_img <= 0] = [0, 0, 0]
        
        return depth_colormap
        
    def visualize(self, data):
        """Display loaded images for debugging."""
        if data.rgb_image is not None:
            cv2.imshow("RGB Image", data.rgb_image)
            
        if data.depth_viz_image is not None:
            cv2.imshow("Depth Visualization", data.depth_viz_image)
        
        cv2.destroyAllWindows()