#!/usr/bin/env python3
"""
FoundationPose HTTP Image Processor with Grounding DINO and SAM2
----------------------------------------------------------------
This script fetches a single RGB and depth image from an HTTP server,
detects sheet metal parts using Grounding DINO, segments them using SAM2,
and processes them through the FoundationPose model.
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
import torch

# Global flag for debugging mode - load images from files instead of HTTP
DEBUG_MODE = False

# Import FoundationPose components
# These imports assume you have the FoundationPose repo properly set up
try:
    from estimater import *
    from datareader import *
except ImportError:
    print("❌ Error: Cannot import FoundationPose modules. Please make sure the repository is correctly set up and in your PYTHONPATH.")
    sys.exit(1)

# Import Grounding DINO for object detection
try:
    from groundingdino.util.inference import load_model, load_image, predict
    from groundingdino.util.utils import clean_state_dict
    import groundingdino.datasets.transforms as T
    HAS_GROUNDINGDINO = True
except ImportError:
    print("⚠️ Warning: Cannot import Grounding DINO. Please install with: pip install groundingdino-py")
    HAS_GROUNDINGDINO = False

# Import SAM2 from Ultralytics for segmentation
try:
    from ultralytics import SAM
    HAS_SAM = True
except ImportError:
    print("⚠️ Warning: Cannot import Ultralytics SAM. Please install with: pip install ultralytics")
    HAS_SAM = False

# Set up logging
def set_logging_format():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Global variables
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
        f"{output_dir}/debug/detection",  # Directory for detection results
        f"{output_dir}/debug/segmentation",  # New directory for segmentation results
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

# Grounding DINO detection function with enhanced visualization
def detect_objects_with_grounding_dino(image, text_prompt, confidence_threshold=0.35, box_threshold=0.3, debug_dir=None):
    """
    Detect objects in an image using Grounding DINO with a text prompt
    
    Args:
        image: RGB image (numpy array)
        text_prompt: Text prompt for detection (e.g., "bent sheet metal")
        confidence_threshold: Confidence threshold for text-conditioning
        box_threshold: Box threshold for detection
        debug_dir: Directory to save debug visualizations
        
    Returns:
        boxes: Detected bounding boxes [x1, y1, x2, y2, score]
        phrases: Detected phrases for each box
    """
    if not HAS_GROUNDINGDINO:
        print("❌ Grounding DINO is not available. Please install with: pip install groundingdino-py")
        return None, None
    
    try:
        print(f"🔍 Running Grounding DINO detection with prompt: '{text_prompt}'...")
        
        # Show what DINO sees before running the model
        input_vis = image.copy()
        cv2.putText(input_vis, f"DINO Input - Prompt: '{text_prompt}'", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("DINO Input", input_vis)
        
        # Save the input visualization
        if debug_dir:
            input_path = f"{debug_dir}/detection/dino_input.png"
            cv2.imwrite(input_path, input_vis)
            print(f"💾 Saved DINO input visualization to {input_path}")
            
        cv2.waitKey(1000)  # Show for 1 second before continuing
        
        # Convert OpenCV's BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Debug the RGB conversion
        print(f"Image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")
        print(f"Sample RGB values at center: {image_rgb[image_rgb.shape[0]//2, image_rgb.shape[1]//2]}")
        
        # Save RGB image for direct inspection
        if debug_dir:
            rgb_debug_path = f"{debug_dir}/detection/dino_input_rgb.png"
            cv2.imwrite(rgb_debug_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
            print(f"💾 Saved RGB input for debugging to {rgb_debug_path}")
            
        # Method 1: Using direct PIL conversion (preferred)
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        temp_path = f"{debug_dir}/detection/temp_for_dino.jpg" if debug_dir else "temp_for_dino.jpg"
        pil_image.save(temp_path)
        print(f"💾 Saved temporary file for DINO at {temp_path}")
        
        # Load the Grounding DINO model
        model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                          "GroundingDINO/groundingdino/weights/groundingdino_swint_ogc.pth")
        
        # Load and preprocess the image with diagnostics
        print(f"Loading image from {temp_path} for GroundingDINO")
        image_source, image_orig = load_image(temp_path)
        print(f"GroundingDINO loaded image shape: {image_source.shape}")
        
        # Run prediction with lowered thresholds if prompt is a simple object name
        actual_box_threshold = box_threshold
        actual_conf_threshold = confidence_threshold
        
        # Lower thresholds for simple object names like "orange", "apple", etc.
        if len(text_prompt.split()) == 1 and text_prompt.islower():
            actual_box_threshold = max(0.1, box_threshold - 0.1)
            actual_conf_threshold = max(0.1, confidence_threshold - 0.1)
            print(f"Simple object name detected. Lowering thresholds to: box={actual_box_threshold}, conf={actual_conf_threshold}")
        
        # Try both the original prompt and with "a/an" prefix
        prompts_to_try = [text_prompt]
        if len(text_prompt.split()) == 1:
            article = "an" if text_prompt[0].lower() in "aeiou" else "a"
            prompts_to_try.append(f"{article} {text_prompt}")
        
        # Try each prompt
        all_boxes = []
        all_logits = []
        all_phrases = []
        
        for prompt in prompts_to_try:
            print(f"Trying prompt: '{prompt}'")
            
            # Load and preprocess the image with diagnostics
            print(f"Loading image from {temp_path} for GroundingDINO")
            image_source, image_transformed = load_image(temp_path)  # image_transformed is a tensor
            print(f"GroundingDINO loaded image shape: {image_source.shape}")

            # Run prediction with lowered thresholds if prompt is a simple object name
            actual_box_threshold = box_threshold
            actual_conf_threshold = confidence_threshold

            # Lower thresholds for simple object names like "orange", "apple", etc.
            if len(text_prompt.split()) == 1 and text_prompt.islower():
                actual_box_threshold = max(0.1, box_threshold - 0.1)
                actual_conf_threshold = max(0.1, confidence_threshold - 0.1)
                print(f"Simple object name detected. Lowering thresholds to: box={actual_box_threshold}, conf={actual_conf_threshold}")

            # Pass the device explicitly and use image_transformed (the tensor) instead of image_source (the numpy array)
            boxes, logits, phrases = predict(
                model=model,
                image=image_transformed,  # Use the transformed tensor from load_image
                caption=prompt,
                box_threshold=actual_box_threshold,
                text_threshold=actual_conf_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"  # Explicitly specify device
            )
            
            if len(boxes) > 0:
                print(f"✅ Found {len(boxes)} results with prompt '{prompt}'")
                all_boxes.extend(boxes)
                all_logits.extend(logits)
                all_phrases.extend([prompt] * len(boxes))
                break
            else:
                print(f"❌ No results found with prompt '{prompt}'")
        
        # If no results were found with any prompt, use the original results
        if len(all_boxes) == 0:
            print("⚠️ No objects detected with any prompt. Using original results.")
            all_boxes = boxes
            all_logits = logits
            all_phrases = phrases
        
        # Convert boxes to the format [x1, y1, x2, y2, score]
        result_boxes = []
        phrases_to_return = []
        
        for i in range(len(all_boxes)):
            x1, y1, x2, y2 = all_boxes[i].tolist()
            
            # Convert normalized coordinates to absolute coordinates
            h, w = image.shape[:2]
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h
            
            # Get confidence score
            score = all_logits[i].item() if isinstance(all_logits[i], torch.Tensor) else all_logits[i]
            
            result_boxes.append([x1, y1, x2, y2, score])
            phrases_to_return.append(all_phrases[i])
            
            print(f"  Box {i+1}: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], score: {score:.4f}, phrase: '{all_phrases[i]}'")
        
        # Visualize detection results
        if len(result_boxes) > 0:
            print(f"✅ DINO detected {len(result_boxes)} objects!")
        else:
            print("⚠️ DINO didn't detect any objects with the given prompt.")
        
        # Create and display visualization of the boxes
        detection_vis = visualize_detections(image, result_boxes, phrases_to_return)
        cv2.imshow("DINO Detection Results", detection_vis)
        
        # Save the detection visualization
        if debug_dir:
            detection_path = f"{debug_dir}/detection/dino_detection.png"
            cv2.imwrite(detection_path, detection_vis)
            print(f"💾 Saved DINO detection results to {detection_path}")
            
        cv2.waitKey(1000)  # Show for 1 second before continuing
        
        return result_boxes, phrases_to_return
    
    except Exception as e:
        print(f"❌ Error during Grounding DINO detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Visualize detection boxes
def visualize_detections(image, boxes, phrases):
    """
    Visualize detected bounding boxes and phrases on the image
    
    Args:
        image: RGB image (numpy array)
        boxes: Detected bounding boxes [x1, y1, x2, y2, score]
        phrases: Detected phrases for each box
        
    Returns:
        Visualization image
    """
    vis_img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, score = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{phrases[i]}: {score:.2f}"
        cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw the center point (will be used for SAM)
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(vis_img, (center_x, center_y), 5, (255, 0, 0), -1)
    
    return vis_img

# SAM2 segmentation function with point prompts
def segment_objects_with_sam2(image, boxes, debug_dir=None):
    """
    Segment objects using SAM2 with bounding boxes and center points as prompts
    
    Args:
        image: RGB image (numpy array)
        boxes: Detected bounding boxes [x1, y1, x2, y2, score]
        debug_dir: Directory to save debug visualizations
        
    Returns:
        masks: Binary masks for each detected object
    """
    if not HAS_SAM:
        print("❌ Ultralytics SAM is not available. Please install with: pip install ultralytics")
        return None
    
    try:
        print("🎭 Running SAM2 segmentation...")
        
        # Process each bounding box with SAM2
        all_masks = []
        box_vis_img = image.copy()  # Create a copy for visualization
        
        # Load the SAM model from Ultralytics
        model = SAM("sam_b.pt")
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2, _ = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate center point of the bounding box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Draw box and point on the input visualization
            cv2.rectangle(box_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(box_vis_img, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(box_vis_img, f"Object {idx+1}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show SAM input visualization
        cv2.putText(box_vis_img, "SAM Input - Boxes and Points", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("SAM Input", box_vis_img)
        
        # Save the input visualization
        if debug_dir:
            sam_input_path = f"{debug_dir}/segmentation/sam_input.png"
            cv2.imwrite(sam_input_path, box_vis_img)
            print(f"💾 Saved SAM input visualization to {sam_input_path}")
            
        cv2.waitKey(1000)  # Show for 1 second before continuing
        
        # Process each bounding box
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2, _ = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate center point of the bounding box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            print(f"Processing object {idx+1} with bounding box [{x1}, {y1}, {x2}, {y2}] and point [{center_x}, {center_y}]")
            
            # Use both the bounding box and the center point as prompts for SAM2
            results = model.predict(
                source=image,
                prompt="orange",  # Text prompt
                boxes=True,       # Enable box detection
                conf=0.1,         # Confidence threshold
                device="cuda" if torch.cuda.is_available() else "cpu",
                retina_masks=True,
                imgsz=1024,
                iou=0.9
            )
            
            # Convert results to binary mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Extract masks from results
            if hasattr(results[0], 'masks') and results[0].masks.data is not None and len(results[0].masks.data) > 0:
                # Convert mask data to numpy
                masks_tensor = results[0].masks.data
                for i in range(len(masks_tensor)):
                    mask_np = masks_tensor[i].cpu().numpy()
                    # Resize mask to match image dimensions
                    mask_resized = cv2.resize(
                        mask_np.astype(np.float32), 
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    # Convert to binary mask
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    mask = cv2.bitwise_or(mask, mask_binary)
                
                # Visualize this individual mask
                mask_vis = visualize_single_mask(image.copy(), mask)
                cv2.imshow(f"SAM Mask for Object {idx+1}", mask_vis)
                
                # Save individual mask visualization
                if debug_dir:
                    mask_path = f"{debug_dir}/segmentation/mask_object_{idx+1}.png"
                    cv2.imwrite(mask_path, mask_vis)
                    print(f"💾 Saved mask visualization for object {idx+1} to {mask_path}")
                
                cv2.waitKey(1000)  # Show for 1 second before continuing
                
                all_masks.append(mask)
            else:
                print(f"⚠️ No mask found for object {idx+1} at bounding box {[x1, y1, x2, y2]}")
                all_masks.append(None)
        
        # Show combined results
        if all_masks:
            combined_mask = combine_masks(all_masks, image.shape[:2])
            combined_vis = visualize_segmentations(image, all_masks)
            cv2.putText(combined_vis, "SAM Final Segmentation Results", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("SAM Results", combined_vis)
            
            # Save the combined visualization
            if debug_dir:
                combined_path = f"{debug_dir}/segmentation/sam_results.png"
                cv2.imwrite(combined_path, combined_vis)
                
                # Also save the raw mask
                raw_mask_path = f"{debug_dir}/segmentation/combined_mask.png"
                cv2.imwrite(raw_mask_path, combined_mask)
                
                print(f"💾 Saved combined SAM results to {combined_path}")
            
            cv2.waitKey(1000)  # Show for 1 second before continuing
        
        return all_masks
    
    except Exception as e:
        print(f"❌ Error during SAM2 segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Visualize a single mask
def visualize_single_mask(image, mask):
    """
    Visualize a single segmentation mask on the image
    
    Args:
        image: RGB image (numpy array)
        mask: Binary mask
        
    Returns:
        Visualization image
    """
    vis_img = image.copy()
    if mask is not None:
        # Create color overlay for mask
        mask_color = np.zeros_like(vis_img)
        mask_color[:,:,0] = 0   # B
        mask_color[:,:,1] = 0   # G
        mask_color[:,:,2] = 255 # R
        
        # Apply mask
        mask_bool = mask > 0
        mask_overlay = cv2.bitwise_and(mask_color, mask_color, mask=mask)
        vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.5, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (0, 255, 255), 2)
    
    return vis_img

# Visualize segmentation masks
def visualize_segmentations(image, masks):
    """
    Visualize segmentation masks on the image
    
    Args:
        image: RGB image (numpy array)
        masks: Binary masks for each detected object
        
    Returns:
        Visualization image
    """
    vis_img = image.copy()
    for i, mask in enumerate(masks):
        if mask is not None:
            # Create color overlay for mask - use different colors for multiple masks
            mask_color = np.zeros_like(vis_img)
            if i % 3 == 0:
                mask_color[:,:,0] = 0   # B
                mask_color[:,:,1] = 0   # G
                mask_color[:,:,2] = 255 # R
            elif i % 3 == 1:
                mask_color[:,:,0] = 0   # B
                mask_color[:,:,1] = 255 # G
                mask_color[:,:,2] = 0   # R
            else:
                mask_color[:,:,0] = 255 # B
                mask_color[:,:,1] = 0   # G
                mask_color[:,:,2] = 0   # R
            
            # Apply mask
            mask_bool = mask > 0
            mask_overlay = cv2.bitwise_and(mask_color, mask_color, mask=mask)
            vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.5, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (0, 255, 255), 2)
            
            # Add label
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(vis_img, f"Object {i+1}", (cX, cY), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return vis_img

# Combine masks if multiple objects are detected
def combine_masks(masks, shape):
    """
    Combine multiple masks into a single mask
    
    Args:
        masks: List of binary masks
        shape: Shape of the output mask (height, width)
        
    Returns:
        Combined binary mask
    """
    combined_mask = np.zeros(shape, dtype=np.uint8)
    
    for mask in masks:
        if mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return combined_mask

# Function to get a text prompt from the user for object detection
def get_text_prompt(default_prompt="bent sheet metal part"):
    """Get a text prompt from the user for object detection"""
    print("\n🔍 Object Detection Prompt:")
    print(f"   Default: '{default_prompt}'")
    user_prompt = input("   Enter custom prompt (or press Enter to use default): ").strip()
    
    if not user_prompt:
        return default_prompt
    
    return user_prompt

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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FoundationPose HTTP Image Processor with Grounding DINO and SAM2')
    parser.add_argument('--url', default="https://7250-162-218-227-129.ngrok-free.app", 
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
    # Arguments for detection and segmentation
    parser.add_argument('--prompt', default="bent sheet metal",
                        help='Text prompt for Grounding DINO detection (default: "bent sheet metal")')
    parser.add_argument('--confidence', type=float, default=0.25,  # Lower default threshold
                        help='Confidence threshold for detection (default: 0.25)')
    parser.add_argument('--box_threshold', type=float, default=0.2,  # Lower default threshold
                        help='Box threshold for detection (default: 0.2)')
    # Debug mode argument
    parser.add_argument('--debug_mode', action='store_true',
                        help='Enable debug mode to load images from files instead of HTTP')
    args = parser.parse_args()
    
    # Set global debug mode flag
    global DEBUG_MODE
    DEBUG_MODE = args.debug_mode
    
    try:
        # Set up logging
        set_logging_format()
        set_seed(0)
        
        # Create output directories
        dirs = create_output_dirs(args.output_dir)
        debug_dir = f"{args.output_dir}/debug"
        
        # Ensure the URL doesn't have a trailing slash
        base_url = args.url.rstrip('/')
        
        # Define timestamp for filenames - using a fixed value for a single image
        timestamp = "000000"
        
        # Get RGB and depth images either from HTTP or local files
        if DEBUG_MODE:
            # Debug mode - load images from local files
            print("🔍 DEBUG MODE: Loading images from local files...")
            
            # Load RGB image
            rgb_path = f"{args.output_dir}/rgb/{timestamp}.png"
            print(f"📸 Loading RGB image from {rgb_path}...")
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is None:
                print(f"❌ Error: Cannot load RGB image from {rgb_path}")
                return 1
                
            # Load depth image
            depth_path = f"{args.output_dir}/depth/{timestamp}.png"
            print(f"📏 Loading depth image from {depth_path}...")
            # In your debug mode loading section, replace:
            # depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            # with:
            depth_npy_path = f"{args.output_dir}/depth/{timestamp}.npy"
            if os.path.exists(depth_npy_path):
                # Load from numpy file if available (perfect precision)
                depth_img = np.load(depth_npy_path)
                print(f"📏 Loaded depth from NPY file: {depth_npy_path}")
            else:
                # Load from PNG (may have quantization)
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                # If we saved in mm, convert back to meters
                if np.max(depth_img) > 10:
                    depth_img = depth_img.astype(np.float32) / 1000.0
                print(f"📏 Loaded depth from PNG file: {depth_path}")
            if depth_img is None:
                print(f"❌ Error: Cannot load depth image from {depth_path}")
                return 1
                
            # If depth is in millimeters, convert to meters
            if np.max(depth_img) > 10 and depth_img.dtype != np.float32:
                print("Converting depth from millimeters to meters...")
                depth_img = depth_img.astype(np.float32) / 1000.0
                
            # Ensure the depth image is 2D
            if len(depth_img.shape) > 2:
                print("WARNING: Depth image has more than 2 dimensions, taking first channel...")
                depth_img = depth_img[:,:,0]
                
            # Apply a minimal threshold to filter out noise
            depth_img[depth_img < 0.001] = 0  # Filter out very close points that might be noise
                
            # Create depth visualization for display
            depth_viz_img = visualize_depth(depth_img)
        else:
            # Normal mode - fetch images from HTTP
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
                # Generate our own visualization
                depth_viz_img = visualize_depth(depth_img)

        # Display the depth visualization
        cv2.imshow("Depth Visualization", depth_viz_img)

        # Then display raw depth heatmap
        print("🌈 Displaying depth heatmap visualization...")
        depth_visual = visualize_depth(depth_img)
        cv2.imshow("Depth Heatmap", depth_visual)

        # Display the RGB image
        cv2.imshow("RGB Image", rgb_img)
        print("✅ Images loaded successfully. Press any key to continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Allow the user to type a custom prompt or use the default
        if args.prompt == "bent sheet metal":
            text_prompt = get_text_prompt(args.prompt)
        else:
            text_prompt = args.prompt
            
        print(f"🔍 Using '{text_prompt}' as the detection prompt")
        
        # Run Grounding DINO for object detection
        boxes, phrases = detect_objects_with_grounding_dino(
            rgb_img, 
            text_prompt, 
            confidence_threshold=args.confidence,
            box_threshold=args.box_threshold,
            debug_dir=debug_dir
        )
        
        # Check if detection was successful
        if not boxes or len(boxes) == 0:
            print(f"❌ No objects detected with prompt '{text_prompt}'. Exiting.")
            return 1
            
        # Run SAM2 for segmentation with bounding boxes and center points
        masks = segment_objects_with_sam2(rgb_img, boxes, debug_dir=debug_dir)
        
        # Check if segmentation was successful
        if not masks or len(masks) == 0:
            print("❌ No valid masks generated. Exiting.")
            return 1
            
        # Combine masks if multiple objects are detected
        mask_img = combine_masks(masks, rgb_img.shape[:2])
        
        # Save the RGB, depth and mask images
        rgb_path = f"{args.output_dir}/rgb/{timestamp}.png"
        depth_path = f"{args.output_dir}/depth/{timestamp}.png"
        mask_path = f"{args.output_dir}/mask/{timestamp}.png"
        
        print(f"💾 Saving images to {args.output_dir}...")
        cv2.imwrite(rgb_path, rgb_img)
        
        # Save depth data properly to preserve floating-point values
        if depth_img.dtype == np.float32 or depth_img.dtype == np.float64:
            # Option 1: Save as 16-bit PNG (scaled)
            depth_scaled = (depth_img * 1000.0).astype(np.uint16)  # Convert meters to mm for storage
            cv2.imwrite(depth_path, depth_scaled)
            
            # Option 2: Also save as numpy array for perfect precision
            depth_npy_path = f"{args.output_dir}/depth/{timestamp}.npy"
            np.save(depth_npy_path, depth_img)
            print(f"💾 Saved raw depth data as both PNG and NPY: {depth_path}, {depth_npy_path}")
        else:
            # If already integer type, save directly
            cv2.imwrite(depth_path, depth_img)
            print(f"💾 Saved raw depth data: {depth_path}")

        cv2.imwrite(mask_path, mask_img)

        # Save depth visualizations
        depth_viz_path = f"{args.output_dir}/debug/depth_viz_{timestamp}.png"
        depth_heatmap_path = f"{args.output_dir}/debug/depth_heatmap_{timestamp}.png"

        # Save depth visualization from server (if available) or the generated one
        cv2.imwrite(depth_viz_path, depth_viz_img)

        # Save the depth heatmap visualization that's generated locally
        depth_heatmap = visualize_depth(depth_img)
        cv2.imwrite(depth_heatmap_path, depth_heatmap)

        print(f"💾 Saved depth visualizations to {depth_viz_path} and {depth_heatmap_path}")
        
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
            print("2. Try using a different detection prompt")
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