import requests
import numpy as np
import cv2
from ...pipeline.base import Processor

class HttpImageFetcher(Processor):
    """Fetches RGB and depth images from HTTP endpoints."""
    
    def __init__(self, base_url, timeout=5, cache_dir=None):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.cache_dir = cache_dir
        
    def process(self, data):
        """Fetch images and update pipeline data."""
        # Store base URL in data
        data.base_url = self.base_url
        
        # Fetch RGB image
        print(f"Fetching RGB image from {self.base_url}/rgb")
        rgb_img, rgb_error = self._fetch_image(f"{self.base_url}/rgb")
        if rgb_error:
            data.add_error("HttpImageFetcher", rgb_error)
            return data
            
        # Fetch depth image
        print(f"Fetching depth image from {self.base_url}/depth")
        depth_img, depth_error = self._fetch_depth(f"{self.base_url}/depth")
        if depth_error:
            data.add_error("HttpImageFetcher", depth_error)
            return data
            
        # Fetch depth visualization (optional)
        print(f"Fetching depth visualization from {self.base_url}/depth_viz")
        depth_viz_img, depth_viz_error = self._fetch_image(f"{self.base_url}/depth_viz")
        if depth_viz_error:
            print(f"Warning: {depth_viz_error}")
            # Generate our own visualization
            depth_viz_img = self._visualize_depth(depth_img)
            
        # Update data
        data.rgb_image = rgb_img
        data.depth_image = depth_img
        data.depth_viz_image = depth_viz_img
        
        # Cache images if cache_dir is set
        if self.cache_dir:
            self._cache_images(data)
        
        return data
        
    def _fetch_image(self, url):
        """Fetch an image from the given URL."""
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img, None
            else:
                return None, f"Failed to fetch image: HTTP {response.status_code}"
        except requests.RequestException as e:
            return None, f"Error fetching image: {str(e)}"
    
    def _fetch_depth(self, url):
        """Fetch raw depth data from the /depth endpoint."""
        try:
            response = requests.get(url, timeout=self.timeout)
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
    
    def _cache_images(self, data):
        """Cache images to local files."""
        import os
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        timestamp = data.simple_timestamp
        
        # Define paths
        rgb_path = os.path.join(self.cache_dir, f"rgb_{timestamp}.png")
        depth_path = os.path.join(self.cache_dir, f"depth_{timestamp}.png")
        depth_npy_path = os.path.join(self.cache_dir, f"depth_{timestamp}.npy")
        depth_viz_path = os.path.join(self.cache_dir, f"depth_viz_{timestamp}.png")
        
        # Save RGB image
        if data.rgb_image is not None:
            cv2.imwrite(rgb_path, data.rgb_image)
            print(f"Cached RGB image to {rgb_path}")
        
        # Save depth image as both PNG and NPY
        if data.depth_image is not None:
            # Save as NPY for perfect precision
            np.save(depth_npy_path, data.depth_image)
            print(f"Cached raw depth data to {depth_npy_path}")
            
            # Save as PNG for visualization
            if data.depth_image.dtype == np.float32 or data.depth_image.dtype == np.float64:
                # Convert meters to mm for storage
                depth_scaled = (data.depth_image * 1000.0).astype(np.uint16)
                cv2.imwrite(depth_path, depth_scaled)
            else:
                cv2.imwrite(depth_path, data.depth_image)
            
            print(f"Cached depth image to {depth_path}")
        
        # Save depth visualization
        if data.depth_viz_image is not None:
            cv2.imwrite(depth_viz_path, data.depth_viz_image)
            print(f"Cached depth visualization to {depth_viz_path}")
        
    def visualize(self, data):
        """Display fetched images for debugging."""
        if data.rgb_image is not None:
            cv2.imshow("RGB Image", data.rgb_image)
            
        if data.depth_viz_image is not None:
            cv2.imshow("Depth Visualization", data.depth_viz_image)
            
        if data.depth_image is not None:
            # Create depth heatmap if not already available
            if data.depth_viz_image is None:
                depth_visual = self._visualize_depth(data.depth_image)
                cv2.imshow("Depth Heatmap", depth_visual)
        
        cv2.destroyAllWindows()