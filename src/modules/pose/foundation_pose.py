import cv2
import numpy as np
import os
import trimesh
import logging
from ...pipeline.base import Processor

class FoundationPoseEstimator(Processor):
    """Estimates 3D pose using FoundationPose."""
    
    def __init__(self, mesh_file, camera_intrinsics_file, camera_section="LEFT_CAM_FHD1200", 
                est_refine_iter=5, debug_level=2, debug_dir=None):
        self.mesh_file = mesh_file
        self.camera_intrinsics_file = camera_intrinsics_file
        self.camera_section = camera_section
        self.est_refine_iter = est_refine_iter
        self.debug_level = debug_level
        self.debug_dir = debug_dir
        self.fp_available = False
        
        # Try to import FoundationPose components
        try:
            from ...utils.estimater import (
                ScorePredictor, PoseRefinePredictor, FoundationPose,
                draw_posed_3d_box, draw_xyz_axis, depth2xyzmap, toOpen3dCloud
            )
            from ...utils.datareader import DataReader
            self.fp_available = True
            self.fp_modules = {
                'ScorePredictor': ScorePredictor,
                'PoseRefinePredictor': PoseRefinePredictor,
                'FoundationPose': FoundationPose,
                'draw_posed_3d_box': draw_posed_3d_box,
                'draw_xyz_axis': draw_xyz_axis,
                'depth2xyzmap': depth2xyzmap,
                'toOpen3dCloud': toOpen3dCloud
            }
        except ImportError:
            print("❌ Error: Cannot import FoundationPose modules. Please make sure the repository is correctly set up.")
            self.fp_available = False
    
    def process(self, data):
        """Estimate 3D pose using FoundationPose."""
        if not self.fp_available:
            data.add_error("FoundationPoseEstimator", "FoundationPose modules are not available")
            return data
            
        if data.rgb_image is None:
            data.add_error("FoundationPoseEstimator", "No RGB image available")
            return data
            
        if data.depth_image is None:
            data.add_error("FoundationPoseEstimator", "No depth image available")
            return data
            
        if data.mask is None:
            data.add_error("FoundationPoseEstimator", "No mask available")
            return data
            
        print("🚀 Running FoundationPose estimation...")
        
        # Create debug directories if specified
        if self.debug_dir:
            os.makedirs(os.path.join(self.debug_dir, "track_vis"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "ob_in_cam"), exist_ok=True)
        
        try:
            # Load camera intrinsics
            K = self._load_camera_intrinsics()
            if K is None:
                data.add_error("FoundationPoseEstimator", "Failed to load camera intrinsics")
                return data
                
            data.camera_intrinsics = K
            
            # Load the CAD model
            print(f"📦 Loading CAD model from {self.mesh_file}...")
            mesh = trimesh.load(self.mesh_file)
            
            # Handle case where a Scene is loaded instead of a Trimesh
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    # Extract the first mesh from the scene
                    mesh = list(mesh.geometry.values())[0]
                else:
                    data.add_error("FoundationPoseEstimator", "Model file contains an empty scene with no meshes")
                    return data
                    
            data.mesh = mesh
            
            # Set up FoundationPose
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
            import pytorch3d.renderer.dibr as dr
            glctx = dr.RasterizeCudaContext()
            
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
            
            # Run pose estimation
            mask_bool = data.mask.astype(bool)
            pose = est.register(
                K=K,
                rgb=data.rgb_image,
                depth=data.depth_image,
                ob_mask=mask_bool,
                iteration=self.est_refine_iter
            )
            
            # Save pose result
            if self.debug_dir:
                pose_path = os.path.join(self.debug_dir, "ob_in_cam", f"{data.simple_timestamp}.txt")
                np.savetxt(pose_path, pose.reshape(4, 4))
                print(f"✅ Pose saved to {pose_path}")
            
            # Store pose in data
            data.pose_matrix = pose.reshape(4, 4)
            
            # Visualize result if debug mode is enabled
            if self.debug_level >= 1:
                draw_posed_3d_box = self.fp_modules['draw_posed_3d_box']
                draw_xyz_axis = self.fp_modules['draw_xyz_axis']
                
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=data.rgb_image, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(data.rgb_image, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                
                # Save visualization
                if self.debug_dir:
                    vis_path = os.path.join(self.debug_dir, "track_vis", f"{data.simple_timestamp}.png")
                    cv2.imwrite(vis_path, vis)
                
                # Store visualization
                data.save_debug_image("pose", vis)
                
            # Export model with transformed pose if high debug level
            if self.debug_level >= 3 and self.debug_dir:
                depth2xyzmap = self.fp_modules['depth2xyzmap']
                toOpen3dCloud = self.fp_modules['toOpen3dCloud']
                
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(os.path.join(self.debug_dir, "model_tf.obj"))
                
                import open3d as o3d
                xyz_map = depth2xyzmap(data.depth_image, K)
                valid = data.depth_image >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], data.rgb_image[valid])
                o3d.io.write_point_cloud(os.path.join(self.debug_dir, "scene_complete.ply"), pcd)
            
            print("✅ FoundationPose estimation completed successfully")
            return data
            
        except Exception as e:
            data.add_error("FoundationPoseEstimator", f"Error during pose estimation: {str(e)}")
            import traceback
            traceback.print_exc()
            return data
    
    def _load_camera_intrinsics(self):
        """Load camera intrinsics from a file."""
        try:
            # Check if the file is in the old parameter-based format
            with open(self.camera_intrinsics_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('['):
                    return self._convert_camera_intrinsics()
            
            # Original format (3x3 matrix)
            with open(self.camera_intrinsics_file, 'r') as f:
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
            print(f"❌ Error loading camera intrinsics: {str(e)}")
            return None
    
    def _convert_camera_intrinsics(self):
        """Convert camera intrinsics from parameter-based format to 3x3 matrix format."""
        import re
        
        try:
            print(f"🔄 Converting camera intrinsics from {self.camera_section} section...")
            with open(self.camera_intrinsics_file, "r") as f:
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
                
                print(f"✅ Camera intrinsics converted successfully:")
                print(f"   fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                
                return K
            else:
                raise ValueError(f"Camera section {self.camera_section} not found in intrinsics file")
        
        except Exception as e:
            print(f"❌ Error converting camera intrinsics: {str(e)}")
            return None
    
    def visualize(self, data):
        """Visualize pose estimation results."""
        if "pose" in data.debug_images and data.debug_images["pose"] is not None:
            cv2.imshow("Pose Estimation Result", data.debug_images["pose"])
            print("Pose estimation results shown. Press any key to continue.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()