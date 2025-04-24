import numpy as np
from datetime import datetime
import copy

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
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.simple_timestamp = "000000"  # Simple timestamp for backward compatibility
        self.base_url = None
        self.errors = []
        self.output_dir = "output"
        
        # Debug data
        self.debug_images = {}  # Store debugging visualizations
    
    def clone(self):
        """Create a deep copy of this object."""
        return copy.deepcopy(self)
    
    def add_error(self, module, message):
        """Add an error to the error log.
        
        Args:
            module: Name of the module that produced the error
            message: Error message
        """
        self.errors.append({"module": module, "message": message, "time": datetime.now()})
    
    def save_debug_image(self, name, image):
        """Save an image for debugging.
        
        Args:
            name: Name/key for the image
            image: Image data (numpy array)
        """
        self.debug_images[name] = image.copy() if image is not None else None