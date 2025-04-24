import cv2
import numpy as np
import os
from ...pipeline.base import Processor

class ManualMasker(Processor):
    """Creates a mask using interactive painting."""
    
    def __init__(self, brush_size=10, debug_dir=None):
        self.brush_size = brush_size
        self.debug_dir = debug_dir
        self.drawing = False
        self.mask = None
        self.image_for_mask = None
        self.exit_requested = False
    
    def process(self, data):
        """Create a mask for the RGB image using interactive painting."""
        if data.rgb_image is None:
            data.add_error("ManualMasker", "No RGB image available")
            return data
            
        print("🎭 Creating mask for the object...")
        
        # Create debug directory if specified
        if self.debug_dir:
            os.makedirs(os.path.join(self.debug_dir, "mask"), exist_ok=True)
        
        # Create the mask
        mask_img = self._create_mask(data.rgb_image)
        
        if mask_img is None:
            data.add_error("ManualMasker", "Mask creation cancelled")
            return data
            
        # Check mask statistics
        mask_pixels = np.count_nonzero(mask_img)
        total_pixels = mask_img.size
        mask_percent = (mask_pixels / total_pixels) * 100
        print(f"Mask stats: {mask_pixels}/{total_pixels} pixels are True ({mask_percent:.2f}%)")
        
        # Check if mask overlaps with valid depth
        if data.depth_image is not None:
            valid_depth = data.depth_image > 0  # Assuming 0 means invalid depth
            overlap = np.logical_and(mask_img.astype(bool), valid_depth)
            overlap_count = np.count_nonzero(overlap)
            print(f"Mask-depth overlap: {overlap_count} valid depth points within mask")
            
            if overlap_count == 0:
                print("❌ ERROR: No valid depth points within the mask region!")
                print("Suggestions:")
                print("1. Check if the depth camera is capturing data for that object")
                print("2. Try creating a mask over a different area with valid depth")
                print("3. Verify that the depth and RGB images are properly aligned")
                data.add_error("ManualMasker", "No valid depth points within mask region")
        
        # Save the mask visualization
        if self.debug_dir:
            # Create a visualization of the mask on the RGB image
            mask_vis = self._visualize_mask(data.rgb_image, mask_img)
            mask_vis_path = os.path.join(self.debug_dir, "mask", "manual_mask_vis.png")
            cv2.imwrite(mask_vis_path, mask_vis)
            
            # Save the raw mask
            mask_path = os.path.join(self.debug_dir, "mask", "manual_mask.png")
            cv2.imwrite(mask_path, mask_img)
            
            print(f"💾 Saved mask to {mask_path}")
        
        # Store the mask in pipeline data
        data.mask = mask_img
        data.masks = [mask_img]  # For consistency with automatic segmentation
        
        # Store visualization
        mask_vis = self._visualize_mask(data.rgb_image, mask_img)
        data.save_debug_image("mask", mask_vis)
        
        return data
    
    def _paint_mask(self, event, x, y, flags, param):
        """Mouse callback function for mask painting."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)  # Fill
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        
        # Handle brush size adjustment with mouse wheel
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.brush_size = min(self.brush_size + 2, 50)  # Max 50
            else:
                self.brush_size = max(self.brush_size - 2, 2)   # Min 2
            print(f"🖌 Brush Size: {self.brush_size}")
    
    def _create_mask(self, rgb_image):
        """Create a mask for the given RGB image using interactive painting."""
        self.image_for_mask = rgb_image.copy()
        self.mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        self.drawing = False
        
        window_name = "Paint Mask (Press 's' to save and continue, ESC to cancel)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._paint_mask)
        
        print("\n🖌 Mask Creation Instructions:")
        print("- Left-click and drag to paint the mask")
        print("- Mouse wheel to adjust brush size")
        print("- Press 's' key (on the OpenCV window, not terminal) to save the mask and continue")
        print("- Press ESC key (on the OpenCV window, not terminal) to cancel")
        print("- Note: You must click on the OpenCV window first, then press the keys\n")
        
        while not self.exit_requested:
            try:
                # Create overlay of mask on image
                mask_overlay = cv2.addWeighted(
                    self.image_for_mask, 0.6, 
                    cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), 0.4, 0
                )
                
                # Add instructions
                cv2.putText(mask_overlay, f"Brush Size: {self.brush_size}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(mask_overlay, "Press 's' to save, ESC to cancel", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(window_name, mask_overlay)
                
                # Use a short timeout to check for exit_requested flag periodically
                key = cv2.waitKey(100) & 0xFF
                
                if key == 27:  # ESC key to cancel
                    cv2.destroyWindow(window_name)
                    return None
                elif key == ord('s'):  # 's' key to save
                    cv2.destroyWindow(window_name)
                    return self.mask
                    
            except KeyboardInterrupt:
                print("\n⚠️ Keyboard interrupt detected! Exiting mask creation...")
                cv2.destroyWindow(window_name)
                return None
                
        # This will be reached if exit_requested is set by signal handler
        cv2.destroyWindow(window_name)
        return None
    
    def _visualize_mask(self, image, mask):
        """Visualize a mask on the image."""
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
    
    def visualize(self, data):
        """Visualize mask results."""
        if "mask" in data.debug_images and data.debug_images["mask"] is not None:
            cv2.imshow("Manual Mask", data.debug_images["mask"])
            print("Mask shown. Press any key to continue.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()