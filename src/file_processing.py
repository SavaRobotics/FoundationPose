import numpy as np
import cv2
import requests
import re
import os

class FileLoader:
    def __init__(self, base_url):
        self.base_url = base_url

    def load_camera_intrinsics(self, intrinsics_file, camera_section="LEFT_CAM_FHD1200"):
        """Load camera intrinsics from a file"""
        try:
            # Check if the file is in the old parameter-based format
            with open(intrinsics_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('['):
                    return self.convert_camera_intrinsics(intrinsics_file, camera_section)
            
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

    def convert_camera_intrinsics(self, intrinsics_file, camera_section="LEFT_CAM_FHD1200"):
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
    
    def load_rgb_image_from_file(self, file_path="data/samples/batch1/rgb/000000.png"):
        """Load RGB image from local file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"RGB file not found: {file_path}")
                
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load RGB image from {file_path}")
                
            print(f"✅ Loaded RGB image from {file_path}, shape: {img.shape}")
            return img
        except Exception as e:
            print(f"❌ Error loading RGB image: {str(e)}")
            return None
            
    def load_depth_from_file(self, file_path="data/samples/batch1/depth/000000.npy"):
        """Load depth data from local numpy file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Depth file not found: {file_path}")
                
            depth = np.load(file_path)
            
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
            
            return depth
        except Exception as e:
            print(f"❌ Error loading depth data: {str(e)}")
            return None
    
    def fetch_rgb_image(self, timeout=5):
        """Fetch rgb image from the /rgb endpoint"""
        try:
            response = requests.get(self.base_url + "/rgb", timeout=timeout)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img, None
            else:
                return None, f"Failed to fetch image: HTTP {response.status_code}"
        except requests.RequestException as e:
            return None, f"Error fetching image: {str(e)}"

    def fetch_depth(self, timeout=5):
        """Fetch raw depth data from the /depth endpoint"""
        try:
            response = requests.get(self.base_url + "/depth", timeout=timeout)
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

    def publish_pose(self, x, y, z, roll, pitch, yaw, timeout=5):
        """
        Publish pose data to the /pose endpoint
        
        Args:
            x, y, z: Position coordinates (float)
            roll, pitch, yaw: Orientation angles (float)
            timeout: Request timeout in seconds
            
        Returns:
            (success, error_message) tuple
        """
        try:
            # Format the pose data exactly as expected by the server: 6 comma-separated values
            pose_data = f"{x},{y},{z},{roll},{pitch},{yaw}"
            
            # Send the POST request with the formatted pose data
            response = requests.post(
                self.base_url + "/pose", 
                data=pose_data,
                headers={'Content-Type': 'text/plain'},
                timeout=timeout
            )
            
            if response.status_code == 200:
                print(f"✅ Successfully published pose: {pose_data}")
                return True, None
            else:
                error_msg = f"Failed to publish pose: HTTP {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return False, error_msg
                
        except requests.RequestException as e:
            error_msg = f"Error publishing pose: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg