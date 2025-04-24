import os
import cv2
import numpy as np
from ...pipeline.base import Processor

class ResultSaver(Processor):
    """Saves processing results to files."""
    
    def __init__(self, output_dir, save_images=True, save_pose=True, save_visualizations=True):
        self.output_dir = output_dir
        self.save_images = save_images
        self.save_pose = save_pose
        self.save_visualizations = save_visualizations
    
    def process(self, data):
        """Save results to files."""
        # Create output directories
        dirs = self._create_output_dirs()
        
        timestamp = data.simple_timestamp
        
        # Save images if requested
        if self.save_images:
            self._save_images(data, timestamp)
        
        # Save pose if requested
        if self.save_pose and data.pose_matrix is not None:
            self._save_pose(data, timestamp)
        
        # Save visualizations if requested
        if self.save_visualizations:
            self._save_visualizations(data, timestamp)
        
        return data
    
    def _create_output_dirs(self):
        """Create all necessary output directories."""
        dirs = [
            self.output_dir,
            os.path.join(self.output_dir, "rgb"),
            os.path.join(self.output_dir, "depth"),
            os.path.join(self.output_dir, "mask"),
            os.path.join(self.output_dir, "debug"),
            os.path.join(self.output_dir, "debug", "track_vis"),
            os.path.join(self.output_dir, "debug", "ob_in_cam"),
            os.path.join(self.output_dir, "debug", "detection"),
            os.path.join(self.output_dir, "debug", "segmentation"),
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            
        return dirs
    
    def _save_images(self, data, timestamp):
        """Save RGB, depth, and mask images."""
        if data.rgb_image is not None:
            rgb_path = os.path.join(self.output_dir, "rgb", f"{timestamp}.png")
            cv2.imwrite(rgb_path, data.rgb_image)
            print(f"💾 Saved RGB image to {rgb_path}")
        
        if data.depth_image is not None:
            # Save as NPY for perfect precision
            depth_npy_path = os.path.join(self.output_dir, "depth", f"{timestamp}.npy")
            np.save(depth_npy_path, data.depth_image)
            
            # Save as PNG for visualization
            depth_path = os.path.join(self.output_dir, "depth", f"{timestamp}.png")
            if data.depth_image.dtype == np.float32 or data.depth_image.dtype == np.float64:
                # Convert meters to mm for storage
                depth_scaled = (data.depth_image * 1000.0).astype(np.uint16)
                cv2.imwrite(depth_path, depth_scaled)
            else:
                cv2.imwrite(depth_path, data.depth_image)
                
            print(f"💾 Saved depth data to {depth_path} and {depth_npy_path}")
        
        if data.mask is not None:
            mask_path = os.path.join(self.output_dir, "mask", f"{timestamp}.png")
            cv2.imwrite(mask_path, data.mask)
            print(f"💾 Saved mask to {mask_path}")
    
    def _save_pose(self, data, timestamp):
        """Save pose matrix and 6D pose."""
        # Save 4x4 pose matrix
        pose_matrix_path = os.path.join(self.output_dir, "debug", "ob_in_cam", f"{timestamp}.txt")
        np.savetxt(pose_matrix_path, data.pose_matrix)
        
        # Save 6D pose if available
        if data.pose_6d is not None:
            pose_6d_path = os.path.join(self.output_dir, "debug", "ob_in_cam", f"{timestamp}_6d.txt")
            with open(pose_6d_path, 'w') as f:
                f.write(",".join(map(str, data.pose_6d)))
                
        print(f"💾 Saved pose data to {pose_matrix_path}")
    
    def _save_visualizations(self, data, timestamp):
        """Save visualization images."""
        # Save detection visualization
        if "detection" in data.debug_images and data.debug_images["detection"] is not None:
            detection_path = os.path.join(self.output_dir, "debug", "detection", f"{timestamp}.png")
            cv2.imwrite(detection_path, data.debug_images["detection"])
            print(f"💾 Saved detection visualization to {detection_path}")
        
        # Save segmentation visualization
        if "segmentation" in data.debug_images and data.debug_images["segmentation"] is not None:
            segmentation_path = os.path.join(self.output_dir, "debug", "segmentation", f"{timestamp}.png")
            cv2.imwrite(segmentation_path, data.debug_images["segmentation"])
            print(f"💾 Saved segmentation visualization to {segmentation_path}")
        
        # Save pose visualization
        if "pose" in data.debug_images and data.debug_images["pose"] is not None:
            pose_path = os.path.join(self.output_dir, "debug", "track_vis", f"{timestamp}.png")
            cv2.imwrite(pose_path, data.debug_images["pose"])
            print(f"💾 Saved pose visualization to {pose_path}")