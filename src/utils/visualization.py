import cv2
import numpy as np

def visualize_depth(depth_img):
    """Create a heatmap visualization of depth data."""
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

def visualize_mask(rgb_img, mask):
    """Visualize a mask on the RGB image."""
    vis_img = rgb_img.copy()
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

def visualize_detection_boxes(rgb_img, boxes, labels=None):
    """Visualize detection boxes on the RGB image."""
    vis_img = rgb_img.copy()
    
    if boxes is None or len(boxes) == 0:
        return vis_img
        
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, score = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label and score
        label_text = f"Object {i+1}"
        if labels is not None and i < len(labels):
            label_text = f"{labels[i]}"
            
        label_text += f": {score:.2f}"
        cv2.putText(vis_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw the center point
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(vis_img, (center_x, center_y), 5, (255, 0, 0), -1)
    
    return vis_img

def create_pipeline_visualization(data):
    """Create a comprehensive visualization of all pipeline stages."""
    if data.rgb_image is None:
        return None
        
    # Create a blank canvas
    h, w = data.rgb_image.shape[:2]
    vis_width = w * 2
    vis_height = h * 2
    visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    
    # Place RGB image in top-left corner
    visualization[:h, :w] = data.rgb_image
    
    # Place depth visualization in top-right corner
    if data.depth_viz_image is not None:
        depth_vis = cv2.resize(data.depth_viz_image, (w, h))
        visualization[:h, w:] = depth_vis
    elif data.depth_image is not None:
        depth_vis = visualize_depth(data.depth_image)
        if depth_vis is not None:
            depth_vis = cv2.resize(depth_vis, (w, h))
            visualization[:h, w:] = depth_vis
    
    # Place detection visualization in bottom-left corner
    if "detection" in data.debug_images and data.debug_images["detection"] is not None:
        det_vis = cv2.resize(data.debug_images["detection"], (w, h))
        visualization[h:, :w] = det_vis
    elif data.detection_boxes is not None:
        det_vis = visualize_detection_boxes(data.rgb_image, data.detection_boxes, data.detection_labels)
        det_vis = cv2.resize(det_vis, (w, h))
        visualization[h:, :w] = det_vis
    
    # Place pose visualization in bottom-right corner
    if "pose" in data.debug_images and data.debug_images["pose"] is not None:
        pose_vis = cv2.resize(data.debug_images["pose"], (w, h))
        visualization[h:, w:] = pose_vis
    elif "segmentation" in data.debug_images and data.debug_images["segmentation"] is not None:
        seg_vis = cv2.resize(data.debug_images["segmentation"], (w, h))
        visualization[h:, w:] = seg_vis
    elif data.mask is not None:
        mask_vis = visualize_mask(data.rgb_image, data.mask)
        mask_vis = cv2.resize(mask_vis, (w, h))
        visualization[h:, w:] = mask_vis
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualization, "RGB Image", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, "Depth Visualization", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, "Detection Results", (10, h + 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, "Pose Estimation", (w + 10, h + 30), font, 1, (255, 255, 255), 2)
    
    return visualization