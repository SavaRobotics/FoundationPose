import os
import cv2
import numpy as np
import re

def ensure_directory(directory):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    return directory

def save_image(image, path):
    """Save an image to a file, ensuring the directory exists."""
    directory = os.path.dirname(path)
    ensure_directory(directory)
    return cv2.imwrite(path, image)

def load_image(path, flags=cv2.IMREAD_COLOR):
    """Load an image from a file."""
    if not os.path.exists(path):
        return None
    return cv2.imread(path, flags)

def load_depth(path):
    """Load a depth image, with handling for both PNG and NPY formats."""
    if not os.path.exists(path):
        return None
        
    # Check file extension
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        # Load from numpy file
        return np.load(path)
    else:
        # Load from image file
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        # Check if depth is likely in millimeters (values > 10)
        if depth is not None and np.max(depth) > 10 and depth.dtype != np.float32:
            # Convert to meters for internal processing
            depth = depth.astype(np.float32) / 1000.0
            
        return depth

def find_latest_sample(data_dir):
    """Find the latest sample in the data directory based on timestamp."""
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        return None
        
    # Look for RGB images
    rgb_dir = os.path.join(data_dir, "rgb")
    if not os.path.exists(rgb_dir):
        return None
        
    # Find all PNG files
    files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
    if not files:
        return None
        
    # Extract timestamps
    timestamps = []
    for f in files:
        # Try to extract timestamp (assuming format like 'YYYYMMDD_HHMMSS.png' or '000000.png')
        match = re.search(r'(\d+)', f)
        if match:
            timestamps.append(match.group(1))
    
    if not timestamps:
        return None
        
    # Find the latest timestamp
    latest = max(timestamps)
    
    # Return paths for RGB, depth, and mask
    result = {
        'timestamp': latest,
        'rgb': os.path.join(rgb_dir, f"{latest}.png")
    }
    
    # Check for depth files
    depth_dir = os.path.join(data_dir, "depth")
    depth_png = os.path.join(depth_dir, f"{latest}.png")
    depth_npy = os.path.join(depth_dir, f"{latest}.npy")
    
    if os.path.exists(depth_npy):
        result['depth_npy'] = depth_npy
    elif os.path.exists(depth_png):
        result['depth'] = depth_png
        
    # Check for mask file
    mask_dir = os.path.join(data_dir, "mask")
    mask_file = os.path.join(mask_dir, f"{latest}.png")
    if os.path.exists(mask_file):
        result['mask'] = mask_file
        
    return result