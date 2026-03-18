import os
import re
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
from ultralytics import YOLO
import easyocr
from huggingface_hub import hf_hub_download
import warnings

# Suppress annoying warnings to keep the terminal clean
warnings.filterwarnings("ignore")

# --- PIE CHART MODEL ARCHITECTURE ---
class PieRegressor(nn.Module):
    def __init__(self):
        super(PieRegressor, self).__init__()
        self.backbone = models.resnet18()
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)

# --- MAIN EXTRACTION ENGINE ---
class GraphExtractor:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.hf_repo_id = "ShadowGard3n/graphvision"
        
        print("🧠 Booting up GraphVision AI Models from Hugging Face...")
        
        # 1. Download Weights
        classifier_path = hf_hub_download(repo_id=self.hf_repo_id, filename="graph_classifier_real.pth")
        pie_model_path = hf_hub_download(repo_id=self.hf_repo_id, filename="pie_regressor_stable.pth")
        yolo_path = hf_hub_download(repo_id=self.hf_repo_id, filename="bar.pt")
        dot_yolo_path = hf_hub_download(repo_id=self.hf_repo_id, filename="dot_line.pt")
        
        # 2. Setup Chart Classifier
        self.CLASS_NAMES = ['dot_line', 'hbar_categorical', 'line', 'pie', 'vbar_categorical'] 
        self.classifier = models.resnet18() 
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 5)  
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 3. Setup YOLO Bar Detector
        self.yolo_model = YOLO(yolo_path)

        # 4. Setup YOLO Dot line Detector
        self.dot_yolo_model = YOLO(dot_yolo_path)

        # 5. Setup Pie Regressor
        self.pie_model = PieRegressor()
        self.pie_model.load_state_dict(torch.load(pie_model_path, map_location=self.device))
        self.pie_model.to(self.device)
        self.pie_model.eval()

        # 6. Setup OCR Text Reader
        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available() or torch.backends.mps.is_available())

    def _extract_number(self, text_str):
        matches = re.findall(r'-?\d+\.?\d*', text_str.replace(',', ''))
        if matches:
            return float(matches[0])
        
        text_clean = text_str.strip().upper()
        if text_clean == 'O': return 0.0
        if text_clean == 'S': return 5.0
        if text_clean in ['I', 'L']: return 1.0
        return None
    
    @staticmethod
    def _color_distance(c1, c2):
        """Calculate Euclidean distance between two BGR colors."""
        return np.linalg.norm(np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32))

    @staticmethod
    def _get_robust_scale(axis_nums, axis_key='y'):
        """Robust scaling that ignores OCR typos."""
        if len(axis_nums) < 2:
            return 1.0, 0.0

        coords = np.array([n[axis_key] for n in axis_nums])
        vals = np.array([n['val'] for n in axis_nums])

        median_val = np.median(vals)
        if median_val > 0:
            for i in range(len(vals)):
                if vals[i] > median_val * 5: 
                    vals[i] /= 10.0

        slopes = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                if abs(coords[i] - coords[j]) > 10: 
                    slope = (vals[i] - vals[j]) / (coords[i] - coords[j])
                    slopes.append(slope)
        
        if not slopes:
            return 1.0, 0.0
            
        best_m = np.median(slopes)
        intercepts = vals - best_m * coords
        best_c = np.median(intercepts)
        
        return best_m, best_c

    def extract(self, image_path):
        """
        Main entry point. Classifies the image and routes it to the correct extraction logic.
        """
        if not os.path.exists(image_path):
            return json.dumps({"error": f"Image not found at {image_path}"})

        print(f"\n🚀 Analyzing: {image_path}")
        print("-" * 40)
        
        # --- Classify Chart Type ---
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            _, predicted = torch.max(outputs, 1)
            chart_type = self.CLASS_NAMES[predicted.item()]
            
        print(f"📊 Detected Chart Type: {chart_type.upper()}")

        # --- Route to specific extractors ---
        if 'bar' in chart_type:
            return self._extract_bar_chart(image_path, chart_type)
        elif 'pie' in chart_type:
            return self._extract_pie_chart(image_path)
        elif chart_type == 'dot_line':
            return self._extract_dot_line_chart(image_path)
        else:
            return json.dumps({
                "chart_type": chart_type, 
                "error": f"Extraction for {chart_type} is currently under development."
            }, indent=4)
        
    def _extract_dot_line_chart(self, image_path):
        results = self.dot_yolo_model(image_path, conf=0.5, iou=0.4, imgsz=1024, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            return json.dumps({"chart_type": "dot_line", "error": "No dots detected."})

        img_cv = cv2.imread(image_path)
        # img_upscaled = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        h, w = img_cv.shape[:2]

        scale = 2.0 if w < 800 else 1.0

        if scale == 2.0:
            img_upscaled = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            img_upscaled = img_cv.copy()

        ocr_results = self.ocr_reader.readtext(img_upscaled)
        
        numbers = []
        text_labels = []
        raw_legend_candidates = []

        for (bbox, text, prob) in ocr_results:
            orig_tl_x, orig_tl_y = bbox[0][0] / scale, bbox[0][1] / scale
            orig_br_x, orig_br_y = bbox[2][0] / scale, bbox[2][1] / scale
            
            cx = (orig_tl_x + orig_br_x) / 2.0
            cy = (orig_tl_y + orig_br_y) / 2.0
            
            val = self._extract_number(text)
            if val is not None:
                numbers.append({'val': val, 'x': cx, 'y': cy})
            else:
                text_labels.append({'text': text, 'x': cx, 'y': cy})
                raw_legend_candidates.append({
                    'text': text, 
                    'x': cx,
                    'tl_x': orig_tl_x, 
                    'cy': cy, 
                    'height': orig_br_y - orig_tl_y
                })

        leftmost_dot = boxes[:, 0].min()
        bottommost_dot = boxes[:, 3].max()

        y_axis_nums = [n for n in numbers if n['x'] < leftmost_dot]
        y_m, y_c = self._get_robust_scale(y_axis_nums, axis_key='y')

        x_axis_nums = [n for n in numbers if n['y'] > bottommost_dot]
        x_m, x_c = self._get_robust_scale(x_axis_nums, axis_key='x')

        legend_colors = {}
        legend_texts = set()
        ignore_words = ['xaxis_label', 'yaxis_label', 'xaxis label', 'yaxis label', 'title', 'x_axis', 'y_axis']
        
        for item in raw_legend_candidates:
            clean_text = item['text'].strip()
            if clean_text.lower() in ignore_words:
                continue
                
            sample_x = int(item['tl_x'] - item['height'] * 0.8) 
            sample_y = int(item['cy'])
            
            if sample_x > 0 and sample_y > 0 and sample_x < img_cv.shape[1] and sample_y < img_cv.shape[0]:
                bgr_color = img_cv[sample_y, sample_x]
                b, g, r = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
                
                is_not_white = (b + g + r) < 700      
                is_not_black = (b + g + r) > 50       
                is_colorful = max(b, g, r) - min(b, g, r) > 20 
                
                if is_not_white and is_not_black and is_colorful:
                    legend_colors[clean_text] = bgr_color
                    legend_texts.add(item['text'])

        extracted_points = []

        for box in boxes:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            
            real_x = (x_m * cx) + x_c
            real_y = (y_m * cy) + y_c
            
            dot_color = img_cv[cy, cx]
            best_class = "Unknown"
            
            if legend_colors:
                best_class = min(legend_colors.keys(), key=lambda k: self._color_distance(dot_color, legend_colors[k]))

            extracted_points.append({
                "class": best_class,
                "x": float(round(real_x, 2)), 
                "y": float(round(real_y, 2))
            })

        extracted_points.sort(key=lambda p: p['x'])

        title, x_axis_label, y_axis_label = None, None, None

        if text_labels:
            top_most = min(text_labels, key=lambda l: l['y'])
            if top_most['y'] < boxes[:, 1].min(): 
                title = top_most['text']
            
            remaining = [l for l in text_labels if l != top_most and l['text'] not in legend_texts]
            
            if remaining:
                bottom_most = max(remaining, key=lambda l: l['y'])
                x_axis_label = bottom_most['text']
                left_most = min(remaining, key=lambda l: l['x'])
                y_axis_label = left_most['text']

        output = {
            "chart_type": "dot_line",
            "title": title,
            "x_axis_label": x_axis_label,
            "y_axis_label": y_axis_label,
            "total_points": len(extracted_points),
            "data": extracted_points
        }

        return json.dumps(output, indent=4)

    def _extract_bar_chart(self, image_path, chart_type):
        results = self.yolo_model(image_path, conf=0.8, iou=0.4, imgsz=1024, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            return json.dumps({"chart_type": chart_type, "error": "No bars detected."})
            
        img_cv = cv2.imread(image_path)
        # img_upscaled = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # ocr_results = self.ocr_reader.readtext(img_upscaled)

        h, w = img_cv.shape[:2]

        scale = 2.0 if w < 800 else 1.0

        if scale == 2.0:
            img_upscaled = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            img_upscaled = img_cv.copy()
        
        img_for_ocr = img_upscaled.copy()

        for box in boxes:
            # Scale YOLO boxes up if the image was upscaled
            scale = 2 if w < 800 else 1 
            x1, y1, x2, y2 = [int(coord * scale) for coord in box]
            
            # Fill the bar area with white (255, 255, 255)
            cv2.rectangle(img_for_ocr, (x1, y1), (x2, y2), (255, 255, 255), -1)

        ocr_results = self.ocr_reader.readtext(img_for_ocr)
        numbers = []
        text_labels = []
        
        # def add_or_update_label(new_text, new_cx, new_cy):
        #     if len(new_text.strip()) == 1 and not new_text.isalnum():
        #         return
        #     existing = next((l for l in text_labels if abs(l['x'] - new_cx) < 30 and abs(l['y'] - new_cy) < 30), None)
        #     if existing:
        #         if len(new_text) > len(existing['text']):
        #             existing['text'] = new_text
        #     else:
        #         text_labels.append({'text': new_text, 'x': new_cx, 'y': new_cy})

        # Update the definition to accept is_rotated
        def add_or_update_label(new_text, new_cx, new_cy, is_rotated=False):
            if len(new_text.strip()) == 1 and not new_text.isalnum():
                return
            existing = next((l for l in text_labels if abs(l['x'] - new_cx) < 30 and abs(l['y'] - new_cy) < 30), None)
            if existing:
                if len(new_text) > len(existing['text']):
                    existing['text'] = new_text
                    existing['is_rotated'] = is_rotated
            else:
                # Save the rotation flag with the text
                text_labels.append({'text': new_text, 'x': new_cx, 'y': new_cy, 'is_rotated': is_rotated})
        
        # Horizontal Pass
        for (bbox, text, prob) in ocr_results:
            cx = ((bbox[0][0] + bbox[1][0]) / 2) / scale
            cy = ((bbox[0][1] + bbox[2][1]) / 2) / scale
            val = self._extract_number(text)
            if val is not None:
                numbers.append({'val': val, 'x': cx, 'y': cy})
            else:
                add_or_update_label(text, cx, cy, is_rotated=False)

        # Rotated Passes for VBAR
        # if 'vbar' in chart_type.lower():
        h_up, w_up = img_upscaled.shape[:2]
            
        img_rot_ccw = cv2.rotate(img_upscaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for (bbox, text, prob) in self.ocr_reader.readtext(img_rot_ccw):
                cx_rot = ((bbox[0][0] + bbox[1][0]) / 2)
                cy_rot = ((bbox[0][1] + bbox[2][1]) / 2)
                cx = (w_up - cy_rot) / scale
                cy = cx_rot / scale
                if self._extract_number(text) is None:
                    add_or_update_label(text, cx, cy, is_rotated=True)
                        
        img_rot_cw = cv2.rotate(img_upscaled, cv2.ROTATE_90_CLOCKWISE)
        for (bbox, text, prob) in self.ocr_reader.readtext(img_rot_cw):
                cx_rot = ((bbox[0][0] + bbox[1][0]) / 2)
                cy_rot = ((bbox[0][1] + bbox[2][1]) / 2)
                cx = cy_rot / scale
                cy = (h_up - cx_rot) / scale
                if self._extract_number(text) is None:
                    add_or_update_label(text, cx, cy, is_rotated=True)

        final_data = []
        ignore_words = ['xaxis_label', 'yaxis_label', 'xaxis label', 'yaxis label', 'title', 'y_axis', 'x_axis']
        
        if 'hbar' in chart_type.lower():
            lowest_bar_bottom = boxes[:, 3].max()
            axis_nums = sorted([n for n in numbers if n['y'] > lowest_bar_bottom - 20], key=lambda d: d['x'])
            
            if len(axis_nums) >= 2:
                units_per_pixel = (axis_nums[-1]['val'] - axis_nums[0]['val']) / (axis_nums[-1]['x'] - axis_nums[0]['x'])
                zero_x_pixel = axis_nums[0]['x'] - (axis_nums[0]['val'] / units_per_pixel)
            else:
                units_per_pixel = 1.0
                zero_x_pixel = boxes[:, 0].min() 

            img_w = img_cv.shape[1]
            max_label_distance = img_w * 0.4

            sorted_boxes = sorted(boxes, key=lambda b: b[1]) 
            for box in sorted_boxes:
                x1, y1, x2, y2 = box
                bar_cy = (y1 + y2) / 2
                if (x2 - x1) < 10 or (y2 - y1) < 5: continue 


                # Get ALL labels to the left of the bar that are vertically aligned with it (within 15 pixels)
                aligned_labels = [
                    l for l in text_labels 
                    if l['x'] < x1 
                    and (x1 - l['x']) < max_label_distance 
                    and abs(l['y'] - bar_cy) < 15 
                    and not l.get('is_rotated', False)
                    and l['text'].lower().replace('_', ' ') not in ignore_words
                ]
                
                if aligned_labels:
                    # Sort the fragments left-to-right based on their x position
                    aligned_labels.sort(key=lambda l: l['x'])
                    # Join them together with a space
                    label_text = " ".join([l['text'] for l in aligned_labels])
                else:
                    label_text = "Unknown Category"

                # possible_labels = [
                #     l for l in text_labels 
                #     if l['x'] < x1 and (x1 - l['x']) < max_label_distance and l['text'].lower().replace('_', ' ') not in ignore_words
                # ]
                
                # # possible_labels = [
                # #     l for l in text_labels 
                # #     if l['x'] < x1 and (x1 - l['x']) < 150 and l['text'].lower().replace('_', ' ') not in ignore_words
                # # ]
                
                # if possible_labels:
                #     best_label = min(possible_labels, key=lambda l: abs(l['y'] - bar_cy))
                #     label_text = best_label['text']
                # else:
                #     label_text = "Unknown Category"
                    
                real_val = (x2 - zero_x_pixel) * units_per_pixel
                final_data.append((label_text, round(real_val, 2)))

        elif 'vbar' in chart_type.lower():
            leftmost_bar_edge = boxes[:, 0].min()
            axis_nums = sorted([n for n in numbers if n['x'] < leftmost_bar_edge + 20], key=lambda d: d['y'], reverse=True)
            
            if len(axis_nums) >= 2:
                units_per_pixel = abs((axis_nums[-1]['val'] - axis_nums[0]['val']) / (axis_nums[-1]['y'] - axis_nums[0]['y']))
                zero_y_pixel = axis_nums[0]['y'] + (axis_nums[0]['val'] / units_per_pixel)
            else:
                units_per_pixel = 1.0
                zero_y_pixel = boxes[:, 3].max() 

            sorted_boxes = sorted(boxes, key=lambda b: b[0]) 
            for box in sorted_boxes:
                x1, y1, x2, y2 = box
                bar_cx = (x1 + x2) / 2
                if (x2 - x1) < 5 or (y2 - y1) < 10: continue 
                
                possible_labels = [
                    l for l in text_labels 
                    if l['y'] > y2 and (l['y'] - y2) < 150 and l['text'].lower().replace('_', ' ') not in ignore_words
                ]
                
                if possible_labels:
                    best_label = min(possible_labels, key=lambda l: abs(l['x'] - bar_cx))
                    label_text = best_label['text']
                else:
                    label_text = "Unknown Category"
                    
                real_val = (zero_y_pixel - y1) * units_per_pixel
                final_data.append((label_text, round(real_val, 2)))

        x_axis_label, y_axis_label = None, None
        if text_labels:
            category_texts = [item[0] for item in final_data]
            bottom_most = max(text_labels, key=lambda l: l['y'])
            if bottom_most['text'] not in category_texts and bottom_most['text'].lower() != 'title':
                x_axis_label = bottom_most['text']
                
            left_most = min(text_labels, key=lambda l: l['x'])
            if left_most['text'] not in category_texts and left_most['text'].lower() != 'title':
                y_axis_label = left_most['text']

        output_dict = {
            "chart_type": chart_type,
            "x_axis_label": x_axis_label,
            "y_axis_label": y_axis_label,
            "data": [{"category": label, "value": value} for label, value in final_data]
        }
        return json.dumps(output_dict, indent=4)

    def _extract_pie_chart(self, image_path):
        img_pil = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.pie_model(input_tensor).squeeze().cpu().numpy() * 100.0
        
        cv_img = cv2.imread(image_path)
        h, w, _ = cv_img.shape
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        all_text_results = self.ocr_reader.readtext(gray, mag_ratio=2.5)
        
        title = "Untitled"
        raw_legend_names = [] 
        
        for bbox, text, conf in all_text_results:
            clean_text = text.strip()
            if not clean_text: continue
                
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            y_pct = y_center / h
            
            if y_pct < 0.15:
                title = clean_text
            elif y_pct > 0.15:
                if len(clean_text) > 2 and not clean_text.replace('.', '', 1).isdigit():
                    if clean_text.lower() == "grav": clean_text = "Gray"
                    raw_legend_names.append((y_pct, clean_text)) 
                    
        legend_names = [item[1] for item in sorted(raw_legend_names, key=lambda i: i[0])]
        
        num_slices = len(legend_names)
        if num_slices == 0:
            valid_preds = [v for v in preds if v > 1.5]
            num_slices = len(valid_preds)
            
        num_slices = min(num_slices, 10)
        slice_preds = preds[:num_slices]
        
        total_pred = sum(slice_preds)
        if total_pred > 0:
            normalized_preds = [(v / total_pred) * 100.0 for v in slice_preds]
        else:
            normalized_preds = slice_preds
        
        final_slices = {}
        for i in range(num_slices):
            slice_name = legend_names[i] if i < len(legend_names) else f"Unknown_Slice_{i+1}"
            slice_value = round(float(normalized_preds[i]), 2) if i < len(normalized_preds) else 0.0
            # slice_value = round(normalized_preds[i], 2) if i < len(normalized_preds) else 0.0
            final_slices[slice_name] = slice_value

        return json.dumps({"chart_type": "pie", "title": title, "data": final_slices}, indent=4)




## Fast but inaccurate

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from PIL import Image
# from ultralytics import YOLO
# import easyocr
# import warnings
# import numpy as np
# import cv2
# import os
# from huggingface_hub import hf_hub_download
# import matplotlib.pyplot as plt

# # Suppress annoying warnings
# warnings.filterwarnings("ignore")

# # --- CUSTOM PIE REGRESSOR DEFINITION ---
# class PieRegressor(nn.Module):
#     def __init__(self):
#         super(PieRegressor, self).__init__()
#         self.backbone = models.resnet18(weights=None)
#         num_ftrs = self.backbone.fc.in_features
#         self.backbone.fc = nn.Linear(num_ftrs, 10)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.backbone(x)
#         return self.sigmoid(x)


# # --- MAIN EXTRACTOR CLASS ---
# class GraphExtractor:
#     def __init__(self, hf_repo_id="ShadowGard3n/graphvision"):
#         """
#         Initializes the STEM Sight AI Models by fetching weights directly from Hugging Face.
#         """
#         print(f"🧠 Booting up STEM Sight AI Models from {hf_repo_id}...")
        
#         self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#         print(f"⚡ Using device: {self.device}")

#         print("📥 Checking model weights (downloading if not cached)...")
#         classifier_path = hf_hub_download(repo_id=hf_repo_id, filename="graph_classifier_real.pth")
#         pie_model_path = hf_hub_download(repo_id=hf_repo_id, filename="pie_regressor_stable.pth")
#         yolo_path = hf_hub_download(repo_id=hf_repo_id, filename="best.pt") 

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device.type != 'cpu'), verbose=False)

#         self.classifier = models.resnet18()
#         self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 5)
#         self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
#         self.classifier.to(self.device)
#         self.classifier.eval()
#         self.class_names = ['vbar_categorical', 'hbar_categorical', 'line', 'pie', 'dot_line']

#         self.yolo_model = YOLO(yolo_path)

#         self.pie_model = PieRegressor().to(self.device)
#         self.pie_model.load_state_dict(torch.load(pie_model_path, map_location=self.device))
#         self.pie_model.eval()
        
#         print("✅ All models loaded successfully!\n")

#     def identify_graph_type(self, image_path):
#         img_pil = Image.open(image_path).convert('RGB')
#         img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             outputs = self.classifier(img_tensor)
#             _, predicted = torch.max(outputs, 1)
#             chart_type = self.class_names[predicted.item()]
            
#         return chart_type

#     # Notice the new `show=False` parameter
#     def extract_data(self, image_path, show=False):
#         if not os.path.exists(image_path):
#             return {"error": f"Image not found at {image_path}"}

#         chart_type = self.identify_graph_type(image_path)
        
#         response = {
#             "chart_type": chart_type,
#             "data": None,
#             "status": "Success"
#         }

#         # Pass the `show` parameter down to the extractors
#         if chart_type in ['hbar_categorical', 'vbar_categorical']:
#             response["data"] = self._extract_bars(image_path, chart_type, show)
#         elif chart_type == 'pie':
#             response["data"] = self._extract_pie(image_path, show)
#         elif chart_type in ['line', 'dot_line']:
#             response["status"] = "Pending Implementation"
#             response["message"] = f"Extraction for {chart_type} is not yet integrated."
#         else:
#             response["status"] = "Unknown Chart Type"
            
#         return response

#     def _extract_bars(self, image_path, chart_type, show):
#         results = self.yolo_model(image_path, conf=0.8, iou=0.4, imgsz=1024, verbose=False)
#         boxes = results[0].boxes.xyxy.cpu().numpy()
        
#         # --- DISPLAY THE GRAPH WITH YOLO BOXES ---
#         if show:
#             annotated_img = results[0].plot() 
#             annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
#             plt.figure(figsize=(10, 6))
#             plt.imshow(annotated_img_rgb)
#             file_name = os.path.basename(image_path)
#             plt.title(f"STEM Sight Extraction: {file_name} ({chart_type.upper()})")
#             plt.axis('off')
#             plt.show()

#         if len(boxes) == 0:
#             return {"error": "No bars detected"}
            
#         ocr_results = self.ocr_reader.readtext(image_path)
#         numbers, text_labels = [], []
        
#         for (bbox, text, prob) in ocr_results:
#             cx = (bbox[0][0] + bbox[1][0]) / 2
#             cy = (bbox[0][1] + bbox[2][1]) / 2
#             clean_text = text.replace(',', '').replace('.', '').strip()
            
#             if clean_text.isdigit():
#                 numbers.append({'val': float(clean_text), 'x': cx, 'y': cy})
#             else:
#                 text_labels.append({'text': text, 'x': cx, 'y': cy})

#         final_data = {}
        
#         if chart_type == 'hbar_categorical':
#             axis_nums = sorted([n for n in numbers if n['y'] > boxes[:, 3].max() - 50], key=lambda d: d['x'])
#             units_per_pixel = (axis_nums[-1]['val'] - axis_nums[0]['val']) / (axis_nums[-1]['x'] - axis_nums[0]['x']) if len(axis_nums) >= 2 else 1.0
#             sorted_boxes = sorted(boxes, key=lambda b: b[1])
            
#             for box in sorted_boxes:
#                 x1, y1, x2, y2 = box
#                 bar_cy = (y1 + y2) / 2
#                 pixel_val = x2 - x1
#                 if pixel_val < 10 or (y2-y1) < 5: continue
                
#                 possible_labels = [l for l in text_labels if l['x'] < x1]
#                 label_text = min(possible_labels, key=lambda l: abs(l['y'] - bar_cy))['text'] if possible_labels else "Unknown"
#                 # Ensure clean standard python int
#                 final_data[label_text] = int(float(pixel_val * units_per_pixel))

#         elif chart_type == 'vbar_categorical':
#             axis_nums = sorted([n for n in numbers if n['x'] < boxes[:, 0].min() + 50], key=lambda d: d['y'], reverse=True)
#             units_per_pixel = abs((axis_nums[-1]['val'] - axis_nums[0]['val']) / (axis_nums[-1]['y'] - axis_nums[0]['y'])) if len(axis_nums) >= 2 else 1.0
#             sorted_boxes = sorted(boxes, key=lambda b: b[0])
            
#             for box in sorted_boxes:
#                 x1, y1, x2, y2 = box
#                 bar_cx = (x1 + x2) / 2
#                 pixel_val = y2 - y1
#                 if pixel_val < 10 or (x2-x1) < 5: continue
                
#                 possible_labels = [l for l in text_labels if l['y'] > y2]
#                 label_text = min(possible_labels, key=lambda l: abs(l['x'] - bar_cx))['text'] if possible_labels else "Unknown"
#                 # Ensure clean standard python int
#                 final_data[label_text] = int(float(pixel_val * units_per_pixel))

#         return final_data

#     def _extract_pie(self, image_path, show):
#         img_pil = Image.open(image_path).convert('RGB')
#         input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             preds = self.pie_model(input_tensor).squeeze().cpu().numpy() * 100.0
        
#         cv_img = cv2.imread(image_path)
#         h, w, _ = cv_img.shape
#         gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#         all_text_results = self.ocr_reader.readtext(gray, mag_ratio=2.5)
        
#         title = "Untitled"
#         raw_legend_names = [] 
        
#         for bbox, text, conf in all_text_results:
#             clean_text = text.strip()
#             if not clean_text: continue
                
#             x_center, y_center = (bbox[0][0] + bbox[2][0]) / 2, (bbox[0][1] + bbox[2][1]) / 2
#             y_pct = y_center / h
            
#             if y_pct < 0.15:
#                 title = clean_text
#             elif y_pct >= 0.15:
#                 if len(clean_text) > 2 and not clean_text.replace('.', '', 1).isdigit():
#                     if clean_text.lower() == "grav": clean_text = "Gray"
#                     raw_legend_names.append((y_pct, clean_text)) 
                    
#         legend_names = [item[1] for item in sorted(raw_legend_names, key=lambda i: i[0])]
        
#         num_slices = len(legend_names)
#         if num_slices == 0:
#             num_slices = len([v for v in preds if v > 1.5])
            
#         num_slices = min(num_slices, 10) 
#         slice_preds = sorted(preds[:num_slices], reverse=True)
        
#         total_pred = sum(slice_preds)
#         normalized_preds = [(v / total_pred) * 100.0 for v in slice_preds] if total_pred > 0 else slice_preds
        
#         final_slices = {}
#         for i in range(num_slices):
#             slice_name = legend_names[i] if i < len(legend_names) else f"Category_{i+1}"
#             val = normalized_preds[i] if i < len(normalized_preds) else 0.0
            
#             # CRITICAL FIX: Cast to standard Python float to prevent JSON float32 crash
#             final_slices[slice_name] = float(round(val, 2))

#         # --- DISPLAY THE PIE GRAPH ---
#         if show:
#             plt.figure(figsize=(8, 5))
#             plt.imshow(img_pil)
#             plt.title(f"Analyzed Pie Chart: {title}")
#             plt.axis('off')
#             plt.show()

#         return {"title": title, "slices": final_slices}


