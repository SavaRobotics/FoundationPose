#!/usr/bin/env python3
"""
Simple HTTP Image Fetcher
------------------------
Fetches RGB and depth images from an HTTP server and saves them to disk.
"""
import requests
import numpy as np
import cv2
import os
import argparse

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
            
            # If depth is in millimeters, convert to meters
            if np.max(depth) > 10 and depth.dtype != np.float32:
                print("Converting depth from millimeters to meters...")
                depth = depth.astype(np.float32) / 1000.0
            
            return depth, None
        else:
            return None, f"Failed to fetch depth data: HTTP {response.status_code}"
    except requests.RequestException as e:
        return None, f"Error fetching depth data: {str(e)}"

def visualize_depth(depth_img):
    """Create a heatmap visualization of depth data for debugging"""
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Mark invalid regions (0 depth) as black
    depth_colormap[depth_img <= 0] = [0, 0, 0]
    
    return depth_colormap

def main():
    parser = argparse.ArgumentParser(description='Simple HTTP Image Fetcher')
    parser.add_argument('--url', required=True, help='Base URL of the HTTP server')
    parser.add_argument('--output-dir', default='data/samples', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize images')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'depth'), exist_ok=True)
    
    base_url = args.url.rstrip('/')
    print(f"Fetching images from {base_url}")
    
    # Fetch RGB image
    print("Fetching RGB image...")
    rgb_img, rgb_error = fetch_image(f"{base_url}/rgb")
    if rgb_img is None:
        print(f"Error: {rgb_error}")
        return 1
    
    # Fetch depth image
    print("Fetching depth image...")
    depth_img, depth_error = fetch_depth(f"{base_url}/depth")
    if depth_img is None:
        print(f"Error: {depth_error}")
        return 1
    
    # Create depth visualization
    depth_viz = visualize_depth(depth_img)
    
    # Save images
    timestamp = "000000"
    rgb_path = os.path.join(args.output_dir, 'rgb', f"{timestamp}.png")
    depth_png_path = os.path.join(args.output_dir, 'depth', f"{timestamp}.png")
    depth_npy_path = os.path.join(args.output_dir, 'depth', f"{timestamp}.npy")
    depth_viz_path = os.path.join(args.output_dir, 'depth', f"{timestamp}_viz.png")
    
    print(f"Saving images to {args.output_dir}...")
    cv2.imwrite(rgb_path, rgb_img)
    
    # Save depth as NPY (precise) and PNG (for viewing)
    np.save(depth_npy_path, depth_img)
    
    # Convert to mm for PNG storage
    if depth_img.dtype == np.float32:
        depth_png = (depth_img * 1000.0).astype(np.uint16)
        cv2.imwrite(depth_png_path, depth_png)
    else:
        cv2.imwrite(depth_png_path, depth_img)
    
    cv2.imwrite(depth_viz_path, depth_viz)
    
    print(f"Images saved successfully:")
    print(f"  RGB: {rgb_path}")
    print(f"  Depth NPY: {depth_npy_path}")
    print(f"  Depth PNG: {depth_png_path}")
    print(f"  Depth Viz: {depth_viz_path}")
    
    # Visualize if requested
    if args.visualize:
        cv2.imshow("RGB Image", rgb_img)
        cv2.imshow("Depth Visualization", depth_viz)
        cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())