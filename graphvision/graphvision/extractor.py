import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from ultralytics import YOLO
import easyocr
import warnings
import numpy as np
import cv2
import os
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# Suppress annoying warnings
warnings.filterwarnings("ignore")

# --- CUSTOM PIE REGRESSOR DEFINITION ---
class PieRegressor(nn.Module):
    def __init__(self):
        super(PieRegressor, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)


# --- MAIN EXTRACTOR CLASS ---
class GraphExtractor:
    def __init__(self, hf_repo_id="ShadowGard3n/graphvision"):
        """
        Initializes the STEM Sight AI Models by fetching weights directly from Hugging Face.
        """
        print(f"🧠 Booting up STEM Sight AI Models from {hf_repo_id}...")
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚡ Using device: {self.device}")

        print("📥 Checking model weights (downloading if not cached)...")
        classifier_path = hf_hub_download(repo_id=hf_repo_id, filename="graph_classifier_real.pth")
        pie_model_path = hf_hub_download(repo_id=hf_repo_id, filename="pie_regressor_stable.pth")
        yolo_path = hf_hub_download(repo_id=hf_repo_id, filename="best.pt") 

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device.type != 'cpu'), verbose=False)

        self.classifier = models.resnet18()
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 5)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        self.class_names = ['vbar_categorical', 'hbar_categorical', 'line', 'pie', 'dot_line']

        self.yolo_model = YOLO(yolo_path)

        self.pie_model = PieRegressor().to(self.device)
        self.pie_model.load_state_dict(torch.load(pie_model_path, map_location=self.device))
        self.pie_model.eval()
        
        print("✅ All models loaded successfully!\n")

    def identify_graph_type(self, image_path):
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            _, predicted = torch.max(outputs, 1)
            chart_type = self.class_names[predicted.item()]
            
        return chart_type

    # Notice the new `show=False` parameter
    def extract_data(self, image_path, show=False):
        if not os.path.exists(image_path):
            return {"error": f"Image not found at {image_path}"}

        chart_type = self.identify_graph_type(image_path)
        
        response = {
            "chart_type": chart_type,
            "data": None,
            "status": "Success"
        }

        # Pass the `show` parameter down to the extractors
        if chart_type in ['hbar_categorical', 'vbar_categorical']:
            response["data"] = self._extract_bars(image_path, chart_type, show)
        elif chart_type == 'pie':
            response["data"] = self._extract_pie(image_path, show)
        elif chart_type in ['line', 'dot_line']:
            response["status"] = "Pending Implementation"
            response["message"] = f"Extraction for {chart_type} is not yet integrated."
        else:
            response["status"] = "Unknown Chart Type"
            
        return response

    def _extract_bars(self, image_path, chart_type, show):
        results = self.yolo_model(image_path, conf=0.8, iou=0.4, imgsz=1024, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # --- DISPLAY THE GRAPH WITH YOLO BOXES ---
        if show:
            annotated_img = results[0].plot() 
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 6))
            plt.imshow(annotated_img_rgb)
            file_name = os.path.basename(image_path)
            plt.title(f"STEM Sight Extraction: {file_name} ({chart_type.upper()})")
            plt.axis('off')
            plt.show()

        if len(boxes) == 0:
            return {"error": "No bars detected"}
            
        ocr_results = self.ocr_reader.readtext(image_path)
        numbers, text_labels = [], []
        
        for (bbox, text, prob) in ocr_results:
            cx = (bbox[0][0] + bbox[1][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            clean_text = text.replace(',', '').replace('.', '').strip()
            
            if clean_text.isdigit():
                numbers.append({'val': float(clean_text), 'x': cx, 'y': cy})
            else:
                text_labels.append({'text': text, 'x': cx, 'y': cy})

        final_data = {}
        
        if chart_type == 'hbar_categorical':
            axis_nums = sorted([n for n in numbers if n['y'] > boxes[:, 3].max() - 50], key=lambda d: d['x'])
            units_per_pixel = (axis_nums[-1]['val'] - axis_nums[0]['val']) / (axis_nums[-1]['x'] - axis_nums[0]['x']) if len(axis_nums) >= 2 else 1.0
            sorted_boxes = sorted(boxes, key=lambda b: b[1])
            
            for box in sorted_boxes:
                x1, y1, x2, y2 = box
                bar_cy = (y1 + y2) / 2
                pixel_val = x2 - x1
                if pixel_val < 10 or (y2-y1) < 5: continue
                
                possible_labels = [l for l in text_labels if l['x'] < x1]
                label_text = min(possible_labels, key=lambda l: abs(l['y'] - bar_cy))['text'] if possible_labels else "Unknown"
                # Ensure clean standard python int
                final_data[label_text] = int(float(pixel_val * units_per_pixel))

        elif chart_type == 'vbar_categorical':
            axis_nums = sorted([n for n in numbers if n['x'] < boxes[:, 0].min() + 50], key=lambda d: d['y'], reverse=True)
            units_per_pixel = abs((axis_nums[-1]['val'] - axis_nums[0]['val']) / (axis_nums[-1]['y'] - axis_nums[0]['y'])) if len(axis_nums) >= 2 else 1.0
            sorted_boxes = sorted(boxes, key=lambda b: b[0])
            
            for box in sorted_boxes:
                x1, y1, x2, y2 = box
                bar_cx = (x1 + x2) / 2
                pixel_val = y2 - y1
                if pixel_val < 10 or (x2-x1) < 5: continue
                
                possible_labels = [l for l in text_labels if l['y'] > y2]
                label_text = min(possible_labels, key=lambda l: abs(l['x'] - bar_cx))['text'] if possible_labels else "Unknown"
                # Ensure clean standard python int
                final_data[label_text] = int(float(pixel_val * units_per_pixel))

        return final_data

    def _extract_pie(self, image_path, show):
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
                
            x_center, y_center = (bbox[0][0] + bbox[2][0]) / 2, (bbox[0][1] + bbox[2][1]) / 2
            y_pct = y_center / h
            
            if y_pct < 0.15:
                title = clean_text
            elif y_pct >= 0.15:
                if len(clean_text) > 2 and not clean_text.replace('.', '', 1).isdigit():
                    if clean_text.lower() == "grav": clean_text = "Gray"
                    raw_legend_names.append((y_pct, clean_text)) 
                    
        legend_names = [item[1] for item in sorted(raw_legend_names, key=lambda i: i[0])]
        
        num_slices = len(legend_names)
        if num_slices == 0:
            num_slices = len([v for v in preds if v > 1.5])
            
        num_slices = min(num_slices, 10) 
        slice_preds = sorted(preds[:num_slices], reverse=True)
        
        total_pred = sum(slice_preds)
        normalized_preds = [(v / total_pred) * 100.0 for v in slice_preds] if total_pred > 0 else slice_preds
        
        final_slices = {}
        for i in range(num_slices):
            slice_name = legend_names[i] if i < len(legend_names) else f"Category_{i+1}"
            val = normalized_preds[i] if i < len(normalized_preds) else 0.0
            
            # CRITICAL FIX: Cast to standard Python float to prevent JSON float32 crash
            final_slices[slice_name] = float(round(val, 2))

        # --- DISPLAY THE PIE GRAPH ---
        if show:
            plt.figure(figsize=(8, 5))
            plt.imshow(img_pil)
            plt.title(f"Analyzed Pie Chart: {title}")
            plt.axis('off')
            plt.show()

        return {"title": title, "slices": final_slices}