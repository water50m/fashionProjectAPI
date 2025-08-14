#!/usr/bin/env python3
"""
Python script for YOLO prediction with FastAPI
Handles image upload via formData and returns structured predictions
"""

import json
import os
import cv2
import numpy as np
from typing import List, Dict, Any
 
 

# YOLO imports (assuming YOLOv8)
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    raise


class YOLOPredictor:
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize YOLO predictor with config"""
        self.config = self.load_config(config_path)
        self.model = self.load_model()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise BaseException(f"Config file not found: {config_path}")
        except json.JSONDecodeError:
            raise BaseException(f"Invalid JSON in config file: {config_path}")
    
    def load_model(self) -> YOLO:
        """Load YOLO model from config"""
        try:
            model_path = os.path.join(
                self.config.get("AI_MODEL_PATH", ""),
                self.config.get("AI_MODEL_NAME", "")
            )
            
            if not os.path.exists(model_path):
                raise BaseException(f"Model file not found: {model_path}")
            
            model = YOLO(model_path)
            print(f"Model loaded successfully from: {model_path}")
            return model
            
        except Exception as e:
            raise BaseException(f"Failed to load model: {str(e)}")
    
    def extract_dominant_colors(self, image: np.ndarray, bbox: List[float], k: int = 3) -> List[str]:
        """Extract dominant colors from bounding box region"""
        try:
            h, w = image.shape[:2]
            
            # Convert YOLO format to pixel coordinates
            x_center, y_center, width, height = bbox[1:5]  # Skip class_id
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Extract region of interest
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return ["unknown"]
            
            # Reshape for k-means clustering
            roi_reshaped = roi.reshape(-1, 3)
            
            # Simple dominant color extraction using histogram
            colors = []
            for channel in range(3):
                hist = cv2.calcHist([roi], [channel], None, [8], [0, 256])
                dominant_bin = np.argmax(hist)
                colors.append(dominant_bin * 32)  # Convert bin to color value
            
            # Convert BGR to color name (simplified)
            color_name = self.bgr_to_color_name(colors)
            return [color_name]
            
        except Exception as e:
            print(f"Error extracting colors: {e}")
            return ["unknown"]
    
    def bgr_to_color_name(self, bgr: List[int]) -> str:
        """Convert BGR values to color name"""
        b, g, r = bgr
        
        # Simple color classification
        if r > 180 and g < 100 and b < 100:
            return "red"
        elif g > 180 and r < 100 and b < 100:
            return "green"
        elif b > 180 and r < 100 and g < 100:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and b > 150 and g < 100:
            return "magenta"
        elif g > 150 and b > 150 and r < 100:
            return "cyan"
        elif r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            if r > 128:
                return "gray"
            else:
                return "dark_gray"
        elif r > 139 and g > 69 and b < 50:
            return "brown"
        elif r > 255 and g > 165 and b < 50:
            return "orange"
        else:
            return "other"
    
    def format_yolo_bbox(self, detection: Any) -> List[float]:
        """Format bounding box to YOLO label format"""
        try:
            # Get bounding box in YOLO format (normalized)
            bbox = detection.boxes.xywhn[0].cpu().numpy()  # [x_center, y_center, width, height]
            class_id = int(detection.boxes.cls[0].cpu().numpy())
            confidence = float(detection.boxes.conf[0].cpu().numpy())
            
            # YOLO label format: [class_id, x_center, y_center, width, height]
            return [class_id, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            
        except Exception as e:
            print(f"Error formatting bbox: {e}")
            return []
    
    def format_yolo_keypoints(self, detection: Any) -> List[float]:
        """Format keypoints to YOLO label format"""
        try:
            if not hasattr(detection, 'keypoints') or detection.keypoints is None:
                return []
            
            keypoints = detection.keypoints.xyn[0].cpu().numpy()  # Normalized keypoints
            class_id = int(detection.boxes.cls[0].cpu().numpy())
            
            # YOLO keypoint format: [class_id, x1, y1, v1, x2, y2, v2, ...]
            formatted_keypoints = [class_id]
            
            for kpt in keypoints:
                if len(kpt) >= 2:
                    x, y = kpt[0], kpt[1]
                    visibility = kpt[2] if len(kpt) > 2 else 1.0  # Default visibility
                    formatted_keypoints.extend([float(x), float(y), float(visibility)])
            
            return formatted_keypoints
            
        except Exception as e:
            print(f"Error formatting keypoints: {e}")
            return []
    
    def predict_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Run prediction on image and return formatted results"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run inference
            results = self.model(image_path)
            
            predictions = []
            
            for result in results:
                if result.boxes is not None:
                    for i, detection in enumerate(result.boxes):
                        # Get class name
                        class_id = int(detection.cls[0].cpu().numpy())
                        class_name = self.model.names.get(class_id, f"class_{class_id}")
                        
                        # Format bounding box
                        bbox_label = self.format_yolo_bbox(type('Detection', (), {
                            'boxes': type('Boxes', (), {
                                'xywhn': [detection.xywhn[0]],
                                'cls': [detection.cls[0]],
                                'conf': [detection.conf[0]]
                            })()
                        })())
                        
                        # Extract colors
                        colors = self.extract_dominant_colors(image, bbox_label)
                        
                        # Format keypoints (if available)
                        keypoints_label = []
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            keypoints_label = self.format_yolo_keypoints(type('Detection', (), {
                                'keypoints': result.keypoints,
                                'boxes': type('Boxes', (), {
                                    'cls': [detection.cls[0]]
                                })()
                            })())
                        
                        prediction = {
                            'classname': class_name,
                            'colors': colors,
                            'Keypoints_label': keypoints_label,
                            'B_Box_label': bbox_label
                        }
                        
                        predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

