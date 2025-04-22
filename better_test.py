#!/usr/bin/env python3
"""
FoundationPose - Minimal and Composable Implementation
-----------------------------------------------------
Simplified version of the FoundationPose pipeline with modular components
"""
import cv2
import numpy as np
import os
import argparse
import trimesh
import torch
import sys
import re
from ultralytics import YOLO, SAM

# Import FoundationPose components
try:
    from estimater import *
    from datareader import *
except ImportError:
    print("Error: Cannot import FoundationPose modules.")
    sys.exit(1)

class PoseEstimation:
    def __init__(self, mesh_file, cam_intrinsics_file, output_dir="output_dir", 
                 prompt="a small orange", debug_level=1, camera_section="LEFT_CAM_FHD1200"):
        self.mesh_file = mesh_file
        self.cam_intrinsics_file = cam_intrinsics_file
        self.output_dir = output_dir
        self.prompt = prompt
        self.debug_level = debug_level
        self.camera_section = camera_section
        
        # Create necessary directories
        self._create_dirs()
        
        # Load CAD model
        self.mesh = self._load_mesh()
        
        # Load camera intrinsics
        self.K = self._load_camera_intrinsics()
        
        # Initialize models
        # Using YOLO-World instead of regular YOLO
        self.yolo_model = YOLO("yolov8s-world.pt", task='detect')
        # Using SAM2 instead of SAM
        self.sam_model = SAM("sam_b.pt")
        
        # Initialize FoundationPose
        self._init_foundation_pose()
    
    def _create_dirs(self):
        """Create necessary output directories"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/rgb",
            f"{self.output_dir}/depth",
            f"{self.output_dir}/mask",
            f"{self.output_dir}/debug/track_vis",
            f"{self.output_dir}/debug/ob_in_cam",
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _load_mesh(self):
        """Load the CAD model mesh"""
        try:
            mesh = trimesh.load(self.mesh_file)
            
            # Handle case where a Scene is loaded instead of a Trimesh
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    mesh = list(mesh.geometry.values())[0]
                else:
                    raise ValueError("Model file contains an empty scene with no meshes")
            return mesh
        except Exception as e:
            print(f"Error loading mesh file: {str(e)}")
            sys.exit(1)
    
    def _load_camera_intrinsics(self):
        """Load camera intrinsics from file"""
        try:
            # Check if the file is in the old parameter-based format
            with open(self.cam_intrinsics_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('['):
                    return self._convert_camera_intrinsics()
            
            # Original format (3x3 matrix)
            with open(self.cam_intrinsics_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    K = np.array([
                        [float(val) for val in lines[0].strip().split()],
                        [float(val) for val in lines[1].strip().split()],
                        [float(val) for val in lines[2].strip().split()]
                    ])
                    return K
                else:
                    raise ValueError("Intrinsics file has incorrect format")
        except Exception as e:
            print(f"Error loading camera intrinsics: {str(e)}")
            sys.exit(1)
    
    def _convert_camera_intrinsics(self):
        """
        Convert camera intrinsics from parameter-based format to 3x3 matrix format
        """
        try:
            print(f"Converting camera intrinsics from {self.camera_section} section...")
            with open(self.cam_intrinsics_file, "r") as f:
                content = f.read()

            # Find the section
            section_pattern = r"\[" + self.camera_section + r"\](.*?)(?=\[|$)"
            section_match = re.search(section_pattern, content, re.DOTALL)

            if section_match:
                section_text = section_match.group(1)
                
                # Extract parameters
                fx = float(re.search(r"fx=([\d\.e-]+)", section_text).group(1))
                fy = float(re.search(r"fy=([\d\.e-]+)", section_text).group(1))
                cx = float(re.search(r"cx=([\d\.e-]+)", section_text).group(1))
                cy = float(re.search(r"cy=([\d\.e-]+)", section_text).group(1))
                
                # Create the 3x3 intrinsics matrix
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                
                print(f"Camera intrinsics converted successfully:")
                print(f"   fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                
                return K
            else:
                raise ValueError(f"Camera section {self.camera_section} not found in intrinsics file")
        
        except Exception as e:
            print(f"Error converting camera intrinsics: {str(e)}")
            sys.exit(1)
    
    def _init_foundation_pose(self):
        """Initialize FoundationPose estimator"""
        try:
            to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
            self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
            self.to_origin = to_origin
            
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()
            
            self.estimator = FoundationPose(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=f"{self.output_dir}/debug",
                debug=self.debug_level,
                glctx=glctx
            )
        except Exception as e:
            print(f"Error initializing FoundationPose: {str(e)}")
            sys.exit(1)
    
    def load_images(self, data_dir="data", timestamp="000000"):
        """Load RGB and depth images from data directory"""
        # Load RGB image
        rgb_path = f"{data_dir}/rgb.png"
        self.rgb_img = cv2.imread(rgb_path)
        if self.rgb_img is None:
            print(f"Error: Cannot load RGB image from {rgb_path}")
            sys.exit(1)
        
        # Load depth image from numpy file
        depth_npy_path = f"{data_dir}/depth.npy"
        if os.path.exists(depth_npy_path):
            # Load from numpy file if available (perfect precision)
            self.depth_img = np.load(depth_npy_path)
            print(f"Loaded depth from NPY file: {depth_npy_path}")
        else:
            # Fallback to loading from PNG (may have quantization)
            depth_path = f"{data_dir}/depth.png"
            self.depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            print(f"Loaded depth from PNG file: {depth_path}")
            
            # If we loaded depth in mm, convert to meters
            if np.max(self.depth_img) > 10 and self.depth_img.dtype != np.float32:
                self.depth_img = self.depth_img.astype(np.float32) / 1000.0
        
        # Ensure the depth image is 2D
        if len(self.depth_img.shape) > 2:
            self.depth_img = self.depth_img[:,:,0]
            
        # Apply a minimal threshold to filter out noise
        self.depth_img[self.depth_img < 0.001] = 0
        
        # Save copies to the output directory for consistency
        os.makedirs(f"{self.output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{self.output_dir}/depth", exist_ok=True)
        cv2.imwrite(f"{self.output_dir}/rgb/{timestamp}.png", self.rgb_img)
        
        # Save depth as both png and numpy
        np.save(f"{self.output_dir}/depth/{timestamp}.npy", self.depth_img)
        
        # Save a visualization of the depth image
        depth_vis = self._visualize_depth(self.depth_img)
        cv2.imwrite(f"{self.output_dir}/depth/{timestamp}.png", depth_vis)
        
        return self.rgb_img, self.depth_img
    
    def _visualize_depth(self, depth_img):
        """Create a heatmap visualization of depth data for debugging"""
        # Normalize depth for visualization
        if depth_img.max() > 0:
            depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            depth_norm = depth_norm.astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            
            # Mark invalid regions (0 depth) as black
            depth_colormap[depth_img <= 0] = [0, 0, 0]
        else:
            # Create a black image if no valid depth data
            depth_colormap = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        
        return depth_colormap
    
    def detect_with_yolo(self):
        """Detect objects using YOLO-World model with a text prompt"""
        try:
            # Run YOLO-World prediction with the prompt
            results = self.yolo_model.predict(
                source=self.rgb_img,
                verbose=False,
                conf=0.25,
                classes=[55]  # 55 is the class ID for "orange" in COCO dataset
            )
            
            # Get bounding boxes
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get bounding boxes from results
                self.boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = box.conf[0].cpu().numpy().tolist()
                    self.boxes.append([x1, y1, x2, y2, conf])
                
                # Visualize the detected boxes
                self._visualize_detections()
                
                return self.boxes
            else:
                print("No objects detected by YOLO-World")
                return None
        except Exception as e:
            print(f"Error during YOLO-World detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_detections(self):
        """Visualize detected bounding boxes on the image"""
        vis_img = self.rgb_img.copy()
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{self.prompt}: {score:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw the center point (will be used for SAM)
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(vis_img, (center_x, center_y), 5, (255, 0, 0), -1)
        
        cv2.imshow("YOLO-World Detection", vis_img)
        cv2.waitKey(1)
    
    def segment_with_sam(self):
        """Segment objects using SAM2 with detected boxes"""
        if not hasattr(self, 'boxes') or not self.boxes:
            print("No boxes detected, cannot segment with SAM2")
            return None
        
        try:
            # Create list for masks
            all_masks = []
            
            for idx, box in enumerate(self.boxes):
                x1, y1, x2, y2, _ = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center point of the bounding box
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Use both the bounding box and the center point as prompts for SAM2
                # SAM2 expects bboxes in the format [x1, y1, x2, y2]
                results = self.sam_model.predict(
                    source=self.rgb_img,
                    bboxes=[x1, y1, x2, y2],  # Bounding box
                    points=[center_x, center_y],  # Center point
                    labels=[1],  # 1 indicates a positive point
                    conf=0.5,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                
                # Extract masks from results
                mask = np.zeros(self.rgb_img.shape[:2], dtype=np.uint8)
                
                if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                    masks_tensor = results[0].masks.data
                    for i in range(len(masks_tensor)):
                        mask_np = masks_tensor[i].cpu().numpy()
                        # Resize mask to match image dimensions
                        mask_resized = cv2.resize(
                            mask_np.astype(np.float32), 
                            (self.rgb_img.shape[1], self.rgb_img.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        # Convert to binary mask
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        mask = cv2.bitwise_or(mask, mask_binary)
                    
                    all_masks.append(mask)
                else:
                    print(f"No mask found for object at box {[x1, y1, x2, y2]}")
            
            # Combine all masks
            self.mask_img = self._combine_masks(all_masks)
            
            # Visualize the segmentation
            self._visualize_segmentation()
            
            # Save the mask
            cv2.imwrite(f"{self.output_dir}/mask/000000.png", self.mask_img)
            
            return self.mask_img
        
        except Exception as e:
            print(f"Error during SAM2 segmentation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _combine_masks(self, masks):
        """Combine multiple masks into one"""
        if not masks:
            return np.zeros(self.rgb_img.shape[:2], dtype=np.uint8)
        
        combined_mask = np.zeros(self.rgb_img.shape[:2], dtype=np.uint8)
        for mask in masks:
            if mask is not None:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask
    
    def _visualize_segmentation(self):
        """Visualize the segmentation mask"""
        if not hasattr(self, 'mask_img'):
            return
        
        vis_img = self.rgb_img.copy()
        
        # Create color overlay for mask
        mask_color = np.zeros_like(vis_img)
        mask_color[:,:,0] = 0   # B
        mask_color[:,:,1] = 0   # G
        mask_color[:,:,2] = 255 # R
        
        # Apply mask
        mask_bool = self.mask_img > 0
        mask_overlay = cv2.bitwise_and(mask_color, mask_color, mask=self.mask_img)
        vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.5, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(self.mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (0, 255, 255), 2)
        
        cv2.imshow("SAM2 Segmentation", vis_img)
        cv2.waitKey(1)
    
    def estimate_pose(self, iterations=5):
        """Run pose estimation using FoundationPose"""
        if not hasattr(self, 'mask_img'):
            print("No mask available for pose estimation")
            return None
        
        try:
            # Convert mask to boolean
            mask_bool = self.mask_img.astype(bool)
            
            # Check if mask overlaps with valid depth
            valid_depth = self.depth_img > 0
            overlap = np.logical_and(mask_bool, valid_depth)
            overlap_count = np.count_nonzero(overlap)
            
            if overlap_count == 0:
                print("No valid depth points within the mask region")
                return None
            
            # Run pose estimation
            pose = self.estimator.register(
                K=self.K,
                rgb=self.rgb_img,
                depth=self.depth_img,
                ob_mask=mask_bool,
                iteration=iterations
            )
            
            # Save the pose
            pose_path = f"{self.output_dir}/debug/ob_in_cam/000000.txt"
            np.savetxt(pose_path, pose.reshape(4, 4))
            
            # Visualize the result
            self._visualize_pose(pose)
            
            return pose
        
        except Exception as e:
            print(f"Error during pose estimation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_pose(self, pose):
        """Visualize the estimated pose"""
        center_pose = pose @ np.linalg.inv(self.to_origin)
        
        # Draw 3D bounding box
        vis = draw_posed_3d_box(self.K, img=self.rgb_img, ob_in_cam=center_pose, bbox=self.bbox)
        
        # Draw coordinate axes
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
        
        # Save and show visualization
        vis_path = f"{self.output_dir}/debug/track_vis/000000.png"
        cv2.imwrite(vis_path, vis)
        
        cv2.imshow("Pose Estimation Result", vis)
        cv2.waitKey(1)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FoundationPose with YOLO-World and SAM2 - Minimal Implementation')
    parser.add_argument('--mesh_file', default="data/bent_test_1.obj", 
                        help='Path to the CAD model file (.obj)')
    parser.add_argument('--output_dir', default="output_dir",
                        help='Output directory for results')
    parser.add_argument('--data_dir', default="data",
                        help='Directory containing input data (rgb.png, depth.npy, etc.)')
    parser.add_argument('--cam_intrinsics', default="data/cam_K.txt",
                        help='Path to camera intrinsics file')
    parser.add_argument('--camera_section', default="LEFT_CAM_FHD1200",
                        help='Camera section to use from intrinsics file (default: LEFT_CAM_FHD1200)')
    parser.add_argument('--prompt', default="a small round orange",
                        help='Text prompt for YOLO-World detection')
    parser.add_argument('--debug', type=int, default=1,
                        help='Debug level (0-3)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of pose refinement iterations')
    args = parser.parse_args()
    
    try:
        # Initialize pose estimation
        pose_est = PoseEstimation(
            mesh_file=args.mesh_file,
            cam_intrinsics_file=args.cam_intrinsics,
            output_dir=args.output_dir,
            prompt=args.prompt,
            debug_level=args.debug,
            camera_section=args.camera_section
        )
        
        # Load images from data directory
        pose_est.load_images(data_dir=args.data_dir)
        
        # Detect objects with YOLO-World
        boxes = pose_est.detect_with_yolo()
        if boxes is None or len(boxes) == 0:
            print("No objects detected, exiting.")
            return 1
        
        # Segment objects with SAM2
        mask = pose_est.segment_with_sam()
        if mask is None:
            print("Segmentation failed, exiting.")
            return 1
        
        # Estimate object pose
        pose = pose_est.estimate_pose(iterations=args.iterations)
        if pose is None:
            print("Pose estimation failed, exiting.")
            return 1
        
        print("Pose estimation completed successfully.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
