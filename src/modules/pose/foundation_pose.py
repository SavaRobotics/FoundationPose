import cv2
import numpy as np
import os
import sys
import trimesh
import logging
from ...pipeline.processor import BaseProcessor

# Move wildcard imports to module level
try:
    from ...utils.estimater import *
    from ...utils.datareader import *
    FP_AVAILABLE = True
except ImportError:
    FP_AVAILABLE = False

class FoundationPoseEstimator(BaseProcessor):
    """Estimates 3D pose using FoundationPose."""
    
    def __init__(self, mesh_file, camera_intrinsics_file, camera_section="LEFT_CAM_FHD1200", 
            est_refine_iter=5, debug_level=2, debug_dir=None):
        super().__init__(debug_dir)
        
        # Validate input parameters
        if mesh_file is None:
            raise ValueError("mesh_file cannot be None")
        if camera_intrinsics_file is None:
            raise ValueError("camera_intrinsics_file cannot be None")
        
        # Check if files exist
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        if not os.path.exists(camera_intrinsics_file):
            raise FileNotFoundError(f"Camera intrinsics file not found: {camera_intrinsics_file}")
            
        self.mesh_file = mesh_file
        self.camera_intrinsics_file = camera_intrinsics_file
        self.camera_section = camera_section
        self.est_refine_iter = est_refine_iter
        self.debug_level = debug_level
        self.fp_available = FP_AVAILABLE
        
        if FP_AVAILABLE:
            self.fp_modules = {
                'ScorePredictor': ScorePredictor,
                'PoseRefinePredictor': PoseRefinePredictor,
                'FoundationPose': FoundationPose,
                'draw_posed_3d_box': draw_posed_3d_box,
                'draw_xyz_axis': draw_xyz_axis,
                'depth2xyzmap': depth2xyzmap,
                'toOpen3dCloud': toOpen3dCloud
            }
        else:
            print("❌ Error: Cannot import FoundationPose modules. Please make sure the repository is correctly set up and in your PYTHONPATH.")
            sys.exit(1)
    
    def process(self, data):
        """Estimate 3D pose using FoundationPose."""
        if not self.fp_available:
            self.log_error("FoundationPose modules are not available", data)
            return data
            
        if data.rgb_image is None:
            self.log_error("No RGB image available for pose estimation", data)
            return data
            
        if data.depth_image is None:
            self.log_error("No depth image available for pose estimation", data)
            return data
            
        if data.mask is None:
            self.log_error("No mask available for pose estimation", data)
            return data
            
        self.logger.info("🚀 Running FoundationPose estimation...")
        
        # Create debug directories if specified
        if self.debug_dir:
            os.makedirs(os.path.join(self.debug_dir, "track_vis"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "ob_in_cam"), exist_ok=True)
        
        try:
            # Load camera intrinsics
            self.log_step_start("loading camera intrinsics")
            K = self._load_camera_intrinsics()
            if K is None:
                self.log_error("Failed to load camera intrinsics", data)
                return data
                
            self.log_step_complete("loading camera intrinsics")
            data.camera_intrinsics = K
            
            # Load the CAD model
            self.logger.info(f"📦 Loading CAD model from {self.mesh_file}...")
            mesh = trimesh.load(self.mesh_file)
            
            # Handle case where a Scene is loaded instead of a Trimesh
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    # Extract the first mesh from the scene
                    mesh = list(mesh.geometry.values())[0]
                    self.logger.info("✅ Extracted mesh from Trimesh Scene")
                else:
                    self.log_error("Model file contains an empty scene with no meshes", data)
                    return data
            
            self.logger.info("✅ CAD model loaded successfully")        
            data.mesh = mesh
            
            # Set up FoundationPose
            self.log_step_start("initializing FoundationPose")
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
            
            # Import dynamically from the saved modules
            ScorePredictor = self.fp_modules['ScorePredictor']
            PoseRefinePredictor = self.fp_modules['PoseRefinePredictor']
            FoundationPose = self.fp_modules['FoundationPose']
            
            # Initialize components
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            
            # Initialize differentiable rendering context
            import torch
            try:
                import pytorch3d.renderer.dibr as dr
                glctx = dr.RasterizeCudaContext()
                self.logger.info("Using dibr.RasterizeCudaContext")
            except (ImportError, ModuleNotFoundError):
                self.logger.warning("pytorch3d.renderer.dibr not available - using fallback renderer")
                glctx = None  # Fallback to None context
             
            # Initialize FoundationPose estimator
            est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=self.debug_dir,
                debug=self.debug_level,
                glctx=glctx
            )
            
            self.log_step_complete("initializing FoundationPose")
            
            # Run pose estimation
            self.log_step_start("pose registration")
            mask_bool = data.mask.astype(bool)
            # Before the register call - validate inputs
            self.logger.info("Validating inputs for pose registration...")
            self.logger.info(f"K shape: {K.shape}, type: {K.dtype}")
            self.logger.info(f"RGB shape: {data.rgb_image.shape}, type: {data.rgb_image.dtype}")
            self.logger.info(f"Depth shape: {data.depth_image.shape}, type: {data.depth_image.dtype}")
            self.logger.info(f"Mask non-zero pixels: {np.sum(mask_bool)}")

            # Try with explicit error catching around just the register call
            try:
                pose = est.register(
                    K=K,
                    rgb=data.rgb_image,
                    depth=data.depth_image,
                    ob_mask=mask_bool,
                    iteration=self.est_refine_iter
                )
                
                # Immediately validate the result
                if pose is None:
                    self.logger.error("❌ est.register() returned None")
                    return data
                    
                if not isinstance(pose, np.ndarray):
                    self.logger.error(f"❌ est.register() returned non-array type: {type(pose)}")
                    return data
                
                if pose.size != 16:  # 4x4 matrix has 16 elements
                    self.logger.error(f"❌ est.register() returned array with wrong size: {pose.shape}")
                    return data
                    
                self.logger.info(f"✅ Pose matrix successfully created with shape {pose.shape}")
                self.logger.debug(f"Pose matrix:\n{pose}")
                
            except Exception as e:
                self.logger.error(f"❌ Error during pose registration: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return data
            
            # Save pose result
            if self.debug_dir:
                pose_path = os.path.join(self.debug_dir, "ob_in_cam", f"{data.simple_timestamp}.txt")
                np.savetxt(pose_path, pose.reshape(4, 4))
                self.logger.info(f"✅ Pose saved to {pose_path}")
            
            # Store pose in data
            data.pose_matrix = pose.reshape(4, 4)
            
            # Create visualization for the pipeline
            self.log_step_start("creating visualization")
            draw_posed_3d_box = self.fp_modules['draw_posed_3d_box']
            draw_xyz_axis = self.fp_modules['draw_xyz_axis']
            
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = data.rgb_image.copy()
            vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            
            # Store visualization for later use
            data.save_debug_image("pose", vis)
            
            # Save visualization
            if self.debug_dir:
                vis_path = os.path.join(self.debug_dir, "track_vis", f"{data.simple_timestamp}.png")
                cv2.imwrite(vis_path, vis)
                self.logger.info(f"✅ Pose visualization saved to {vis_path}")
            self.log_step_complete("creating visualization")
            
            # Export model with transformed pose if high debug level
            if self.debug_level >= 3 and self.debug_dir:
                self.log_step_start("exporting 3D model and point cloud")
                depth2xyzmap = self.fp_modules['depth2xyzmap']
                toOpen3dCloud = self.fp_modules['toOpen3dCloud']
                
                m = mesh.copy()
                m.apply_transform(pose)
                model_path = os.path.join(self.debug_dir, "model_tf.obj")
                m.export(model_path)
                self.logger.info(f"✅ Transformed model saved to {model_path}")
                
                import open3d as o3d
                xyz_map = depth2xyzmap(data.depth_image, K)
                valid = data.depth_image >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], data.rgb_image[valid])
                pcd_path = os.path.join(self.debug_dir, "scene_complete.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                self.logger.info(f"✅ Point cloud saved to {pcd_path}")
                self.log_step_complete("exporting 3D model and point cloud")
            
            # Extract 6D pose parameters for easier access
            self.log_step_start("extracting 6D pose parameters")
            try:
                from scipy.spatial.transform import Rotation as R
                # Extract translation
                translation = pose[:3, 3]
                # Extract rotation matrix and convert to Euler angles
                rotation_matrix = pose[:3, :3]
                euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
                
                # Store as [x, y, z, roll, pitch, yaw]
                data.pose_6d = [
                    translation[0], translation[1], translation[2],
                    euler_angles[0], euler_angles[1], euler_angles[2]
                ]
                
                self.logger.info("✅ 6D Pose extracted:")
                self.logger.info(f"  Position: x={translation[0]:.4f}, y={translation[1]:.4f}, z={translation[2]:.4f}")
                self.logger.info(f"  Rotation: roll={euler_angles[0]:.4f}, pitch={euler_angles[1]:.4f}, yaw={euler_angles[2]:.4f}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not extract 6D pose parameters: {str(e)}")
            self.log_step_complete("extracting 6D pose parameters")
            
            self.logger.info("✅ FoundationPose estimation completed successfully")
            return data
            
        except Exception as e:
            self.log_error(f"Error during pose estimation: {str(e)}", data)
            import traceback
            traceback.print_exc()
            return data
    
    def _load_camera_intrinsics(self):
        """Load camera intrinsics from a file."""
        try:
            self.logger.info(f"Loading camera intrinsics from {self.camera_intrinsics_file}")
            
            # First, check if the file exists
            if not os.path.exists(self.camera_intrinsics_file):
                self.logger.error(f"❌ Camera intrinsics file not found: {self.camera_intrinsics_file}")
                return None
                
            # Check if the file is in the parameter-based format
            with open(self.camera_intrinsics_file, 'r') as f:
                first_line = f.readline().strip()
                self.logger.debug(f"First line of intrinsics file: '{first_line}'")
                
                if first_line.startswith('['):
                    self.logger.info(f"Detected parameter-based format, using section: {self.camera_section}")
                    return self._convert_camera_intrinsics()
            
            # Original format (3x3 matrix)
            self.logger.info("Using matrix format for camera intrinsics")
            with open(self.camera_intrinsics_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    K = np.array([
                        [float(val) for val in lines[0].strip().split()],
                        [float(val) for val in lines[1].strip().split()],
                        [float(val) for val in lines[2].strip().split()]
                    ])
                    self.logger.info(f"Read 3x3 camera matrix:\n{K}")
                    return K
                else:
                    self.logger.error(f"❌ Camera intrinsics file has too few lines: {len(lines)}")
                    raise ValueError("Intrinsics file has incorrect format - needs at least 3 lines")
        except Exception as e:
            self.logger.error(f"❌ Error loading camera intrinsics: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def _convert_camera_intrinsics(self):
        """Convert camera intrinsics from parameter-based format to 3x3 matrix format."""
        import re
        
        try:
            self.logger.info(f"🔄 Converting camera intrinsics from {self.camera_section} section...")
            with open(self.camera_intrinsics_file, "r") as f:
                content = f.read()

            # Find the section
            section_pattern = r"\[" + self.camera_section + r"\](.*?)(?=\[|$)"
            section_match = re.search(section_pattern, content, re.DOTALL)

            if section_match:
                section_text = section_match.group(1)
                self.logger.debug(f"Found section text: {section_text}")
                
                # Extract parameters
                try:
                    fx = float(re.search(r"fx=([\d\.e-]+)", section_text).group(1))
                    fy = float(re.search(r"fy=([\d\.e-]+)", section_text).group(1))
                    cx = float(re.search(r"cx=([\d\.e-]+)", section_text).group(1))
                    cy = float(re.search(r"cy=([\d\.e-]+)", section_text).group(1))
                except AttributeError as e:
                    self.logger.error(f"❌ Failed to extract camera parameters: {e}")
                    self.logger.error(f"Section text: {section_text}")
                    raise ValueError(f"Missing parameters in {self.camera_section} section")
                
                # Create the 3x3 intrinsics matrix
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                
                self.logger.info(f"✅ Camera intrinsics converted successfully:")
                self.logger.info(f"   fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                
                return K
            else:
                # List available sections for debugging
                available_sections = re.findall(r"\[(.*?)\]", content)
                self.logger.error(f"❌ Camera section '{self.camera_section}' not found. Available sections: {available_sections}")
                raise ValueError(f"Camera section {self.camera_section} not found in intrinsics file")
        
        except Exception as e:
            self.logger.error(f"❌ Error converting camera intrinsics: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def visualize(self, data):
        """Visualize pose estimation results."""
        if "pose" in data.debug_images and data.debug_images["pose"] is not None:
            # Create a more informative window title
            window_title = "FoundationPose Estimation Results"
            
            # Get the visualization image
            vis_img = data.debug_images["pose"]
            
            # Add overlay text with pose information if available
            if data.pose_6d is not None:
                x, y, z, roll, pitch, yaw = data.pose_6d
                
                # Add text with pose information
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                color = (0, 255, 0)  # Green text
                
                # Add background for text
                text_bg = vis_img.copy()
                cv2.rectangle(text_bg, (10, 10), (400, 130), (0, 0, 0), -1)
                alpha = 0.7
                vis_img = cv2.addWeighted(text_bg, alpha, vis_img, 1 - alpha, 0)
                
                # Add pose information as text
                cv2.putText(vis_img, f"Position (x,y,z): {x:.3f}, {y:.3f}, {z:.3f}", 
                            (20, 40), font, font_scale, color, thickness)
                cv2.putText(vis_img, f"Rotation (roll): {roll:.2f} degrees", 
                            (20, 70), font, font_scale, color, thickness)
                cv2.putText(vis_img, f"Rotation (pitch): {pitch:.2f} degrees", 
                            (20, 100), font, font_scale, color, thickness)
                cv2.putText(vis_img, f"Rotation (yaw): {yaw:.2f} degrees", 
                            (20, 130), font, font_scale, color, thickness)
            
            # Display the visualization
            cv2.imshow(window_title, vis_img)
            
            # Wait for a key press and then close the window
            key = cv2.waitKey(0)
            cv2.destroyWindow(window_title)