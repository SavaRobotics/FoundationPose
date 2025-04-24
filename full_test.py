#!/usr/bin/env python3
"""
FoundationPose HTTP Image Processor
-----------------------------------
This script fetches a single RGB and depth image from an HTTP server,
creates a mask, and processes it through the FoundationPose model.
"""
import cv2
import numpy as np
import os
import argparse
import requests
import logging
import trimesh
import json
import sys
import re
import signal
from urllib.parse import urlparse
import math

# Import FoundationPose components
# These imports assume you have the FoundationPose repo properly set up
try:
    from src.foundationpose.utils.estimater import *
    from src.foundationpose.utils.datareader import *
except ImportError:
    print("❌ Error: Cannot import FoundationPose modules. Please make sure the repository is correctly set up and in your PYTHONPATH.")
    sys.exit(1)

# Set up logging
def set_logging_format():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Global variables for mask creation
drawing = False
brush_size = 10
mask = None
image_for_mask = None
exit_requested = False

# Signal handler for graceful exit
def signal_handler(sig, frame):
    global exit_requested
    print("\n⚠️ Ctrl+C detected! Closing windows and exiting...")
    exit_requested = True
    cv2.destroyAllWindows()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def create_output_dirs(output_dir):
    """Create all necessary output directories"""
    dirs = [
        output_dir,
        f"{output_dir}/rgb",
        f"{output_dir}/depth",
        f"{output_dir}/mask",
        f"{output_dir}/debug",
        f"{output_dir}/debug/track_vis",
        f"{output_dir}/debug/ob_in_cam",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def fetch_image(url, timeout=5):
    """Fetch an image from the given URL"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img, None
        else:
            return None, f"Failed to fetch image: HTTP {response.status_code}"
    except requests.RequestException as e:
        return None, f"Error fetching image: {str(e)}"

def fetch_depth(url, timeout=5):
    """Fetch raw depth data from the /depth endpoint"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            depth = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            
            # Debug information
            print(f"Depth image shape: {depth.shape}, dtype: {depth.dtype}")
            valid_count = np.count_nonzero(depth > 0)
            total_pixels = depth.size
            valid_percent = (valid_count / total_pixels) * 100 if total_pixels > 0 else 0
            print(f"Valid depth points: {valid_count}/{total_pixels} ({valid_percent:.2f}%)")
            
            # If depth is all zeros or almost all zeros, there's a problem
            if valid_percent < 1:
                print("⚠️ WARNING: Less than 1% of depth values are valid!")
            
            print(f"Depth range: min={np.min(depth)}, max={np.max(depth)}")
            
            # Ensure values are in the expected range for FoundationPose
            # If depth is in millimeters, convert to meters
            if np.max(depth) > 10 and depth.dtype != np.float32:  # Assuming depth > 10 means it's in mm
                print("Converting depth from millimeters to meters...")
                depth = depth.astype(np.float32) / 1000.0
                print(f"New depth range: min={np.min(depth)}, max={np.max(depth)}")
            
            # Apply a minimal threshold to filter out noise
            depth[depth < 0.001] = 0  # Filter out very close points that might be noise
            
            # Ensure the depth image is 2D
            if len(depth.shape) > 2:
                print("WARNING: Depth image has more than 2 dimensions, taking first channel...")
                depth = depth[:,:,0]
            
            return depth, None
        else:
            return None, f"Failed to fetch depth data: HTTP {response.status_code}"
    except requests.RequestException as e:
        return None, f"Error fetching depth data: {str(e)}"
    
# In your client code, after fetching the depth:
def visualize_depth(depth_img):
    """Create a heatmap visualization of depth data for debugging"""
    # Normalize depth for visualization
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Mark invalid regions (0 depth) as black
    depth_colormap[depth_img <= 0] = [0, 0, 0]
    
    return depth_colormap

def fetch_depth_viz(url, timeout=5):
    """Fetch depth visualization for display purposes"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img, None
        else:
            return None, f"Failed to fetch depth visualization: HTTP {response.status_code}"
    except requests.RequestException as e:
        return None, f"Error fetching depth visualization: {str(e)}"

# Mask creation functions
def paint_mask(event, x, y, flags, param):
    """Mouse callback function for mask painting"""
    global drawing, mask, brush_size
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), brush_size, 255, -1)  # Fill
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), brush_size, 255, -1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def change_brush_size(event, x, y, flags, param):
    """Mouse wheel callback for brush size adjustment"""
    global brush_size
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            brush_size = min(brush_size + 2, 50)  # Max 50
        else:
            brush_size = max(brush_size - 2, 2)   # Min 2
        print(f"🖌 Brush Size: {brush_size}")

def create_mask(rgb_image):
    """Create a mask for the given RGB image using interactive painting"""
    global mask, drawing, brush_size, image_for_mask, exit_requested
    
    image_for_mask = rgb_image.copy()
    mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
    drawing = False
    
    window_name = "Paint Mask (Press 's' to save and continue, ESC to cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, paint_mask)
    
    print("\n🖌 Mask Creation Instructions:")
    print("- Left-click and drag to paint the mask")
    print("- Press 's' key (on the OpenCV window, not terminal) to save the mask and continue")
    print("- Press ESC key (on the OpenCV window, not terminal) to cancel")
    print("- You can also press Ctrl+C in the terminal to exit at any time")
    print("- Note: You must click on the OpenCV window first, then press the keys\n")
    
    while not exit_requested:
        try:
            mask_overlay = cv2.addWeighted(image_for_mask, 0.6, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
            cv2.putText(mask_overlay, f"Brush Size: {brush_size}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(mask_overlay, "Press 's' to save, ESC to cancel", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, mask_overlay)
            
            # Use a short timeout to check for exit_requested flag periodically
            key = cv2.waitKey(100) & 0xFF
            
            if key == 27:  # ESC key to cancel
                cv2.destroyWindow(window_name)
                return None
            elif key == ord('s'):  # 's' key to save
                cv2.destroyWindow(window_name)
                return mask
                
            # Check if exit was requested by signal handler
            if exit_requested:
                cv2.destroyWindow(window_name)
                return None
                
        except KeyboardInterrupt:
            print("\n⚠️ Keyboard interrupt detected! Exiting mask creation...")
            cv2.destroyWindow(window_name)
            return None
            
    # This will be reached if exit_requested is set by signal handler
    cv2.destroyWindow(window_name)
    return None

def load_camera_intrinsics(intrinsics_file, camera_section="LEFT_CAM_FHD1200"):
    """Load camera intrinsics from a file"""
    try:
        # Check if the file is in the old parameter-based format
        with open(intrinsics_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('['):
                return convert_camera_intrinsics(intrinsics_file, camera_section)
        
        # Original format (3x3 matrix)
        with open(intrinsics_file, 'r') as f:
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

def convert_camera_intrinsics(intrinsics_file, camera_section="LEFT_CAM_FHD1200"):
    """
    Convert camera intrinsics from parameter-based format to 3x3 matrix format
    
    Args:
        intrinsics_file: Path to the cam_K.txt file with parameter-based format
        camera_section: Camera section to use (e.g., "LEFT_CAM_FHD1200")
        
    Returns:
        3x3 numpy array with camera intrinsics matrix
    """
    try:
        print(f"🔄 Converting camera intrinsics from {camera_section} section...")
        with open(intrinsics_file, "r") as f:
            content = f.read()

        # Find the section
        section_pattern = r"\[" + camera_section + r"\](.*?)(?=\[|$)"
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
            raise ValueError(f"Camera section {camera_section} not found in intrinsics file")
    
    except Exception as e:
        print(f"❌ Error converting camera intrinsics: {str(e)}")
        return None

def rotation_matrix_to_euler_angles(R):
    """Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in radians"""
    # Check if we're in the singularity known as "Gimbal lock"
    if abs(R[2, 0]) > 0.9999:
        # Special case: pitch is around ±90°
        yaw = 0
        if R[2, 0] < 0:
            pitch = math.pi/2
            roll = math.atan2(R[0, 1], R[1, 1])
        else:
            pitch = -math.pi/2
            roll = -math.atan2(R[0, 1], R[1, 1])
    else:
        # Standard case
        pitch = -math.asin(R[2, 0])
        roll = math.atan2(R[2, 1]/math.cos(pitch), R[2, 2]/math.cos(pitch))
        yaw = math.atan2(R[1, 0]/math.cos(pitch), R[0, 0]/math.cos(pitch))
    
    return roll, pitch, yaw

def convert_pose_matrix_to_6d(pose_matrix):
    """Convert 4x4 pose matrix to 6D pose (x, y, z, roll, pitch, yaw)"""
    # Extract translation (in meters)
    x = pose_matrix[0, 3]
    y = pose_matrix[1, 3]
    z = pose_matrix[2, 3]
    
    # Extract rotation and convert to Euler angles (in radians)
    R = pose_matrix[:3, :3]
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
    
    # Convert radians to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    # Convert meters to inches (1 meter = 39.3701 inches)
    x_inch = x * 39.3701
    y_inch = y * 39.3701
    z_inch = z * 39.3701
    
    return x_inch, y_inch, z_inch, roll_deg, pitch_deg, yaw_deg

def send_pose_to_server(url, pose_matrix):
    """Send the pose matrix to the server via POST request"""
    try:
        # Flatten the pose matrix to a space-separated string
        pose_str = ' '.join(map(str, pose_matrix.flatten()))
        
        # Send the POST request to /pose endpoint
        response = requests.post(f"{url}/pose", data=pose_str)
        
        if response.status_code == 200:
            # Parse the 6D pose received from the server
            pose_6d = response.text
            print(f"✅ Pose sent successfully. Received 6D pose: {pose_6d}")
            return pose_6d
        else:
            print(f"❌ Failed to send pose: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error sending pose: {str(e)}")
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FoundationPose HTTP Image Processor')
    parser.add_argument('--url', default="https://2548-162-218-227-129.ngrok-free.app", 
                        help='Base URL of the HTTP server (e.g., http://localhost:8080)')
    parser.add_argument('--mesh_file', default="data/bent_test_1.obj", 
                        help='Path to the CAD model file (.obj)')
    parser.add_argument('--output_dir', default="output_dir",
                        help='Output directory to save images and results')
    parser.add_argument('--cam_intrinsics', default="data/cam_K.txt",
                        help='Path to camera intrinsics file (cam_K.txt)')
    parser.add_argument('--camera_section', default="LEFT_CAM_FHD1200",
                        help='Camera section to use from intrinsics file (default: LEFT_CAM_FHD1200)')
    parser.add_argument('--est_refine_iter', type=int, default=5,
                        help='Number of estimation refinement iterations')
    parser.add_argument('--debug', type=int, default=2,
                        help='Debug level (0-3)')
    args = parser.parse_args()
    
    try:
        # Set up logging
        set_logging_format()
        set_seed(0)
        
        # Create output directories
        dirs = create_output_dirs(args.output_dir)
        debug_dir = f"{args.output_dir}/debug"
        
        # Ensure the URL doesn't have a trailing slash
        base_url = args.url.rstrip('/')
        
        print(f"🌐 Connecting to {base_url}")
        print(f"   RGB endpoint: {base_url}/rgb")
        print(f"   Depth endpoint: {base_url}/depth")
        
        # Fetch a single RGB image
        print("📸 Fetching RGB image...")
        rgb_img, rgb_error = fetch_image(f"{base_url}/rgb")
        if rgb_img is None:
            print(f"❌ {rgb_error}")
            return 1

        # Fetch raw depth data (2D) for processing
        print("📏 Fetching raw depth data...")
        depth_img, depth_error = fetch_depth(f"{base_url}/depth")
        if depth_img is None:
            print(f"❌ {depth_error}")
            return 1

        # Fetch depth visualization for display
        print("🌈 Fetching depth visualization...")
        depth_viz_img, depth_viz_error = fetch_depth_viz(f"{base_url}/depth_viz")
        if depth_viz_img is None:
            print(f"⚠️ {depth_viz_error}")
            print("Continuing without depth visualization...")
        else:
            # Display the depth visualization instead of raw depth
            cv2.imshow("Depth Visualization", depth_viz_img)

        # Then display raw depth heatmap
        print("🌈 Displaying depth heatmap visualization...")
        depth_visual = visualize_depth(depth_img)
        cv2.imshow("Depth Heatmap", depth_visual)

        # Display the RGB image
        cv2.imshow("RGB Image", rgb_img)
        print("✅ Images fetched successfully. Press any key to continue to mask creation.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Create mask
        print("🎭 Creating mask for the object...")
        mask_img = create_mask(rgb_img)
        if mask_img is None:
            print("❌ Mask creation cancelled.")
            return 1

        
        # Save the RGB, depth and mask images
        timestamp = "000000"  # Using a fixed timestamp for a single image
        rgb_path = f"{args.output_dir}/rgb/{timestamp}.png"
        depth_path = f"{args.output_dir}/depth/{timestamp}.png"
        mask_path = f"{args.output_dir}/mask/{timestamp}.png"
        
        print(f"💾 Saving images to {args.output_dir}...")
        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img)
        cv2.imwrite(mask_path, mask_img)
        
        # After creating the mask but before running pose estimation:
        print(f"Mask stats: {np.count_nonzero(mask_img)}/{mask_img.size} pixels are True ({(np.count_nonzero(mask_img)/mask_img.size)*100:.2f}%)")

        # Check if mask overlaps with valid depth
        valid_depth = depth_img > 0  # Assuming 0 means invalid depth
        overlap = np.logical_and(mask_img.astype(bool), valid_depth)
        overlap_count = np.count_nonzero(overlap)
        print(f"Mask-depth overlap: {overlap_count} valid depth points within mask")

        if overlap_count == 0:
            print("❌ ERROR: No valid depth points within the mask region!")
            print("Suggestions:")
            print("1. Check if the depth camera is capturing data for that object")
            print("2. Try creating a mask over a different area with valid depth")
            print("3. Verify that the depth and RGB images are properly aligned")
            return 1

        # Load camera intrinsics
        K = load_camera_intrinsics(args.cam_intrinsics, args.camera_section)
        if K is None:
            return 1
        
        # Load the CAD model
        try:
            print(f"📦 Loading CAD model from {args.mesh_file}...")
            mesh = trimesh.load(args.mesh_file)
            
            # Handle case where a Scene is loaded instead of a Trimesh
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    # Extract the first mesh from the scene
                    mesh = list(mesh.geometry.values())[0]
                else:
                    print(f"❌ Error: Model file contains an empty scene with no meshes")
                    return 1
        except Exception as e:
            print(f"❌ Error loading mesh file: {str(e)}")
            return 1

        # Set up FoundationPose
        print("🚀 Initializing FoundationPose model...")
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        
        try:
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()
            est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=debug_dir,
                debug=args.debug,
                glctx=glctx
            )
            logging.info("✅ Estimator initialization done")
        except Exception as e:
            print(f"❌ Error initializing FoundationPose: {str(e)}")
            return 1
        
        # Run pose estimation
        try:
            print("🔍 Running pose estimation...")
            mask_bool = mask_img.astype(bool)
            pose = est.register(
                K=K,
                rgb=rgb_img,
                depth=depth_img,
                ob_mask=mask_bool,
                iteration=args.est_refine_iter
            )
            
            # Save pose result
            pose_path = f"{args.output_dir}/debug/ob_in_cam/{timestamp}.txt"
            np.savetxt(pose_path, pose.reshape(4, 4))
            print(f"✅ Pose saved to {pose_path}")
            
            # Send the pose to the server
            print(f"📤 Sending pose to {base_url}/pose...")
            pose_6d = send_pose_to_server(base_url, pose.reshape(4, 4))
            if pose_6d:
                x, y, z, roll, pitch, yaw = map(float, pose_6d.split(','))
                print(f"📏 Object position (inches): x={x:.4f}, y={y:.4f}, z={z:.4f}")
                print(f"🔄 Object rotation (degrees): roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")
            
            # Visualize result if debug mode is enabled
            if args.debug >= 1:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=rgb_img, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(rgb_img, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                vis_path = f"{args.output_dir}/debug/track_vis/{timestamp}.png"
                cv2.imwrite(vis_path, vis)
                
                # Show visualization
                cv2.imshow("Pose Estimation Result", vis)
                print("✅ Pose estimation complete. Press any key to exit.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Export model with transformed pose if high debug level
            if args.debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f"{debug_dir}/model_tf.obj")
                
                xyz_map = depth2xyzmap(depth_img, K)
                valid = depth_img >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], rgb_img[valid])
                o3d.io.write_point_cloud(f"{debug_dir}/scene_complete.ply", pcd)
                print(f"✅ Debug 3D files exported to {debug_dir}")
            
        except Exception as e:
            print(f"❌ Error during pose estimation: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user. Exiting...")
        cv2.destroyAllWindows()
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        return 1
    
    print("✅ FoundationPose processing completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())