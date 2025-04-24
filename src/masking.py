import cv2
import numpy as np
import os

class MaskingTool:
    def __init__(self):
        self.drawing = False
        self.brush_size = 10
        self.mask = None
        self.image = None
    
    def load_image(self, image_input):
        # Handle both file paths and numpy arrays
        if isinstance(image_input, str):
            self.image = cv2.imread(image_input)
            if self.image is None:
                print(f"❌ Error: Cannot load image {image_input}")
                return False
        elif isinstance(image_input, np.ndarray):
            self.image = image_input.copy()
        else:
            print(f"❌ Error: Unsupported image input type: {type(image_input)}")
            return False
            
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        return True
    
    def paint_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def change_brush_size(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.brush_size = min(self.brush_size + 2, 50)
            else:
                self.brush_size = max(self.brush_size - 2, 2)
            print(f"🖌 Brush Size: {self.brush_size}")
    
    def run(self, image_input, mask_save_path):
        if not self.load_image(image_input):
            return None
        
        cv2.namedWindow("Paint Mask", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Paint Mask", lambda event, x, y, flags, param: self.paint_mask(event, x, y, flags, param))
        
        while True:
            mask_overlay = cv2.addWeighted(self.image, 0.6, cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
            cv2.imshow("Paint Mask", mask_overlay)
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC key to exit
                break
            elif key == 115:  # 's' key to save
                os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                cv2.imwrite(mask_save_path, self.mask)
                print(f"✅ Mask saved to {mask_save_path}")
                break
        
        cv2.destroyAllWindows()
        return self.mask