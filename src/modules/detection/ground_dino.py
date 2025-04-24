import cv2
import numpy as np
import torch
import os
from ...pipeline.base import Processor

class GroundingDinoDetector(Processor):
    """Detects objects using Grounding DINO with a text prompt."""
    
    def __init__(self, text_prompt, confidence_threshold=0.35, box_threshold=0.3, debug_dir=None):
        self.text_prompt = text_prompt
        self.confidence_threshold = confidence_threshold
        self.box_threshold = box_threshold
        self.debug_dir = debug_dir
        
        # Check if Grounding DINO is available
        try:
            from groundingdino.util.inference import load_model, load_image, predict
            from groundingdino.util.utils import clean_state_dict
            import groundingdino.datasets.transforms as T
            self.has_groundingdino = True
            self._load_model_fn = load_model
            self._load_image_fn = load_image
            self._predict_fn = predict
        except ImportError:
            print("⚠️ Warning: Cannot import Grounding DINO. Please install with: pip install groundingdino-py")
            self.has_groundingdino = False
        
    def process(self, data):
        """Detect objects in the RGB image."""
        if not self.has_groundingdino:
            data.add_error("GroundingDinoDetector", "Grounding DINO is not available")
            return data
            
        if data.rgb_image is None:
            data.add_error("GroundingDinoDetector", "No RGB image available")
            return data
            
        print(f"🔍 Running Grounding DINO detection with prompt: '{self.text_prompt}'...")
        
        # Create debug directory if specified
        if self.debug_dir:
            os.makedirs(os.path.join(self.debug_dir, "detection"), exist_ok=True)
        
        # Show what DINO sees before running the model
        input_vis = data.rgb_image.copy()
        cv2.putText(input_vis, f"DINO Input - Prompt: '{self.text_prompt}'", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save the input visualization
        if self.debug_dir:
            input_path = os.path.join(self.debug_dir, "detection", "dino_input.png")
            cv2.imwrite(input_path, input_vis)
            print(f"💾 Saved DINO input visualization to {input_path}")
            
        # Convert OpenCV's BGR to RGB
        image_rgb = cv2.cvtColor(data.rgb_image, cv2.COLOR_BGR2RGB)
        
        # Save RGB image for direct inspection
        if self.debug_dir:
            rgb_debug_path = os.path.join(self.debug_dir, "detection", "dino_input_rgb.png")
            cv2.imwrite(rgb_debug_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            
        # Save as temporary file for DINO
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        temp_path = os.path.join(self.debug_dir, "detection", "temp_for_dino.jpg") if self.debug_dir else "temp_for_dino.jpg"
        pil_image.save(temp_path)
        
        try:
            # Load the Grounding DINO model
            model = self._load_model_fn("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                                      "GroundingDINO/groundingdino/weights/groundingdino_swint_ogc.pth")
            
            # Load and preprocess the image
            image_source, image_transformed = self._load_image_fn(temp_path)
            
            # Try different prompts if needed
            prompts_to_try = [self.text_prompt]
            if len(self.text_prompt.split()) == 1:
                article = "an" if self.text_prompt[0].lower() in "aeiou" else "a"
                prompts_to_try.append(f"{article} {self.text_prompt}")
            
            # Try each prompt
            all_boxes = []
            all_logits = []
            all_phrases = []
            
            for prompt in prompts_to_try:
                # Adjust thresholds for simple prompts
                actual_box_threshold = self.box_threshold
                actual_conf_threshold = self.confidence_threshold
                
                if len(prompt.split()) == 1 and prompt.islower():
                    actual_box_threshold = max(0.1, self.box_threshold - 0.1)
                    actual_conf_threshold = max(0.1, self.confidence_threshold - 0.1)
                
                # Run prediction
                boxes, logits, phrases = self._predict_fn(
                    model=model,
                    image=image_transformed,
                    caption=prompt,
                    box_threshold=actual_box_threshold,
                    text_threshold=actual_conf_threshold,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                if len(boxes) > 0:
                    all_boxes.extend(boxes)
                    all_logits.extend(logits)
                    all_phrases.extend([prompt] * len(boxes))
                    break
            
            # If no results were found with any prompt, use the original results
            if len(all_boxes) == 0:
                all_boxes = boxes
                all_logits = logits
                all_phrases = phrases
            
            # Convert boxes to the format [x1, y1, x2, y2, score]
            result_boxes = []
            phrases_to_return = []
            
            for i in range(len(all_boxes)):
                x1, y1, x2, y2 = all_boxes[i].tolist()
                
                # Convert normalized coordinates to absolute coordinates
                h, w = data.rgb_image.shape[:2]
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
                
                # Get confidence score
                score = all_logits[i].item() if isinstance(all_logits[i], torch.Tensor) else all_logits[i]
                
                result_boxes.append([x1, y1, x2, y2, score])
                phrases_to_return.append(all_phrases[i])
                
                print(f"  Box {i+1}: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], score: {score:.4f}, phrase: '{all_phrases[i]}'")
            
            # Create and display visualization of the boxes
            if len(result_boxes) > 0:
                print(f"✅ DINO detected {len(result_boxes)} objects!")
                detection_vis = self._visualize_detections(data.rgb_image, result_boxes, phrases_to_return)
                
                # Save the detection visualization
                if self.debug_dir:
                    detection_path = os.path.join(self.debug_dir, "detection", "dino_detection.png")
                    cv2.imwrite(detection_path, detection_vis)
                    
                # Store detection results in the data
                data.detection_boxes = result_boxes
                data.detection_labels = phrases_to_return
                data.save_debug_image("detection", detection_vis)
            else:
                print("⚠️ DINO didn't detect any objects with the given prompt.")
                data.add_error("GroundingDinoDetector", f"No objects detected with prompt '{self.text_prompt}'")
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return data
            
        except Exception as e:
            data.add_error("GroundingDinoDetector", f"Error during detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return data
    
    def _visualize_detections(self, image, boxes, phrases):
        """Visualize detected bounding boxes and phrases on the image."""
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
        
    def visualize(self, data):
        """Visualize detection results."""
        if "detection" in data.debug_images and data.debug_images["detection"] is not None:
            cv2.imshow("DINO Detection Results", data.debug_images["detection"])
            print("Detection results shown. Press any key to continue.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()