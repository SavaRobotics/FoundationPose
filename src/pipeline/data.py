import numpy as np
from datetime import datetime
import copy
import time
import os
import cv2
from typing import List, Dict, Any, Optional

class PipelineData:
    """Container for data passing through the pipeline."""
    
    def __init__(self):
        # Image data
        self.rgb_image = None
        self.depth_image = None
        self.depth_viz_image = None
        
        # Processing results
        self.detection_boxes = None  # [[x1,y1,x2,y2,score], ...]
        self.detection_labels = None
        self.masks = []              # List of masks for multiple objects
        self.mask = None             # Combined mask (for FoundationPose)
        
        # Pose data
        self.camera_intrinsics = None  # 3x3 camera matrix
        self.pose_matrix = None        # 4x4 transformation matrix
        self.pose_6d = None            # [x,y,z,roll,pitch,yaw]
        self.mesh = None               # Trimesh object
        
        # Metadata
        self.timestamp = time.time()
        self.simple_timestamp = str(int(self.timestamp))
        self.base_url = None
        self.errors = []
        self.output_dir = None
        
        # Debug data
        self.debug_images = {}  # Store debugging visualizations
    
    def clone(self):
        """Create a deep copy of this object."""
        return copy.deepcopy(self)
    
    def add_error(self, module: str, message: str) -> None:
        """Add an error from a module to the error log."""
        self.errors.append({
            'module': module,
            'message': message,
            'timestamp': time.time()
        })
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all errors."""
        return self.errors
    
    def save_debug_image(self, key: str, image: np.ndarray) -> None:
        """Save a debug visualization image with a key."""
        self.debug_images[key] = image.copy()
    
    def save_to_disk(self, output_dir: str) -> None:
        """Save all data to disk in the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images
        if self.rgb_image is not None:
            cv2.imwrite(os.path.join(output_dir, f"rgb_{self.simple_timestamp}.png"), self.rgb_image)
        
        if self.depth_image is not None:
            # Save depth as both visualization and numpy array
            depth_vis = self.depth_image.copy()
            if depth_vis.max() > 0:
                depth_vis = depth_vis / depth_vis.max() * 255
            depth_vis = depth_vis.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"depth_{self.simple_timestamp}.png"), depth_vis)
            np.save(os.path.join(output_dir, f"depth_{self.simple_timestamp}.npy"), self.depth_image)
        
        if self.mask is not None:
            mask_vis = self.mask.copy()
            if mask_vis.max() > 0:
                mask_vis = mask_vis / mask_vis.max() * 255
            mask_vis = mask_vis.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"mask_{self.simple_timestamp}.png"), mask_vis)
        
        # Save pose if available
        if self.pose_matrix is not None:
            np.savetxt(os.path.join(output_dir, f"pose_{self.simple_timestamp}.txt"), self.pose_matrix)
        
        # Save debug images
        for key, img in self.debug_images.items():
            cv2.imwrite(os.path.join(output_dir, f"{key}_{self.simple_timestamp}.png"), img)