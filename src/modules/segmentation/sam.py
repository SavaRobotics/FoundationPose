import cv2
import numpy as np
import os
from ...pipeline.base import Processor

class SAM2Segmenter(Processor):
    """Segments objects using SAM2 with bounding boxes as prompts."""
    
    def __init__(self, debug_dir=None):
        self.debug_dir = debug_dir
        
        # Check if SAM is available
        try:
            from ultralytics import SAM
            self.has_sam = True
            self._SAM = SAM
        except ImportError:
            print("⚠️ Warning: Cannot import Ultralytics SAM. Please install with: pip install ultralytics")
            self.has_sam = False
    
    def process(self, data):
        """Segment objects in the RGB image."""
        if not self.has_sam:
            data.add_error("SAM2Segmenter", "Ultralytics SAM is not available")
            return data
            
        if data.rgb_image is None:
            data.add_error("SAM2Segmenter", "No RGB image available")
            return data
            
        if data.detection_boxes is None or len(data.detection_boxes) == 0:
            data.add_error("SAM2Segmenter", "No detection boxes available")
            return data
            
        print("🎭 Running SAM2 segmentation...")
        
        # Create debug directory if specified
        if self.debug_dir:
            os.makedirs(os.path.join(self.debug_dir, "segmentation"), exist_ok=True)
        
        # Process each bounding box with SAM2
        all_masks = []
        box_vis_img = data.rgb_image.copy()
        
        # Draw boxes on the input visualization
        for idx, box in enumerate(data.detection_boxes):
            x1, y1, x2, y2, _ = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate center point of the bounding box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Draw box and point on the input visualization
            cv2.rectangle(box_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(box_vis_img, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(box_vis_img, f"Object {idx+1}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save the input visualization
        if self.debug_dir:
            sam_input_path = os.path.join(self.debug_dir, "segmentation", "sam_input.png")
            cv2.imwrite(sam_input_path, box_vis_img)
            print(f"💾 Saved SAM input visualization to {sam_input_path}")
            
        try:
            # Load the SAM model from Ultralytics
            model = self._SAM("sam_b.pt")
            
            # Process the entire image with SAM2
            results = model.predict(
                source=data.rgb_image,
                boxes=True,       # Enable box detection
                conf=0.1,         # Confidence threshold
                device="cuda" if torch.cuda.is_available() else "cpu",
                retina_masks=True,
                imgsz=1024,
                iou=0.9
            )
            
            # Process each detection box
            for idx, box in enumerate(data.detection_boxes):
                x1, y1, x2, y2, _ = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Create an empty mask for this object
                mask = np.zeros(data.rgb_image.shape[:2], dtype=np.uint8)
                
                # Extract masks from results that overlap with the detection box
                if hasattr(results[0], 'masks') and results[0].masks.data is not None and len(results[0].masks.data) > 0:
                    # Convert mask data to numpy
                    masks_tensor = results[0].masks.data
                    for i in range(len(masks_tensor)):
                        mask_np = masks_tensor[i].cpu().numpy()
                        
                        # Resize mask to match image dimensions
                        mask_resized = cv2.resize(
                            mask_np.astype(np.float32), 
                            (data.rgb_image.shape[1], data.rgb_image.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        # Convert to binary mask
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        
                        # Check overlap with the bounding box
                        box_mask = np.zeros(data.rgb_image.shape[:2], dtype=np.uint8)
                        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
                        
                        overlap = cv2.bitwise_and(mask_binary, box_mask)
                        overlap_ratio = np.count_nonzero(overlap) / np.count_nonzero(mask_binary) if np.count_nonzero(mask_binary) > 0 else 0
                        
                        # If sufficient overlap, add to the object mask
                        if overlap_ratio > 0.5:
                            mask = cv2.bitwise_or(mask, mask_binary)
                
                # Add mask to the list
                if np.count_nonzero(mask) > 0:
                    all_masks.append(mask)
                    
                    # Visualize this individual mask
                    mask_vis = self._visualize_single_mask(data.rgb_image.copy(), mask)
                    
                    # Save individual mask visualization
                    if self.debug_dir:
                        mask_path = os.path.join(self.debug_dir, "segmentation", f"mask_object_{idx+1}.png")
                        cv2.imwrite(mask_path, mask_vis)
                        
                else:
                    print(f"⚠️ No mask found for object {idx+1} at bounding box {[x1, y1, x2, y2]}")
                    all_masks.append(None)
            
            # Combine masks if any were found
            if all_masks:
                # Store all individual masks
                data.masks = all_masks
                
                # Create a combined mask for FoundationPose
                combined_mask = self._combine_masks(all_masks, data.rgb_image.shape[:2])
                data.mask = combined_mask
                
                # Create visualization
                combined_vis = self._visualize_segmentations(data.rgb_image, all_masks)
                data.save_debug_image("segmentation", combined_vis)
                
                # Save visualizations
                if self.debug_dir:
                    combined_path = os.path.join(self.debug_dir, "segmentation", "sam_results.png")
                    cv2.imwrite(combined_path, combined_vis)
                    
                    # Also save the raw mask
                    raw_mask_path = os.path.join(self.debug_dir, "segmentation", "combined_mask.png")
                    cv2.imwrite(raw_mask_path, combined_mask)
                    
                print(f"✅ SAM2 segmented {len(all_masks)} objects")
            else:
                data.add_error("SAM2Segmenter", "No valid masks generated")
            
            return data
            
        except Exception as e:
            data.add_error("SAM2Segmenter", f"Error during segmentation: {str(e)}")
            import traceback
            traceback.print_exc()
            return data
    
    def _visualize_single_mask(self, image, mask):
        """Visualize a single segmentation mask on the image."""
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
    
    def _visualize_segmentations(self, image, masks):
        """Visualize multiple segmentation masks on the image."""
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
    
    def _combine_masks(self, masks, shape):
        """Combine multiple masks into a single mask."""
        combined_mask = np.zeros(shape, dtype=np.uint8)
        
        for mask in masks:
            if mask is not None:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask
    
    def visualize(self, data):
        """Visualize segmentation results."""
        if "segmentation" in data.debug_images and data.debug_images["segmentation"] is not None:
            cv2.imshow("SAM Segmentation Results", data.debug_images["segmentation"])
            cv2.destroyAllWindows()