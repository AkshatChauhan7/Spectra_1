# GraphVision AI 📊👁️

GraphVision AI is a lightweight, powerful computer vision library for automatic graph classification and structured data extraction.

Built with PyTorch and EasyOCR, it is designed to look at an image of a chart, instantly recognize what kind of graph it is, and extract its labels and values into a clean, developer-friendly JSON format.

---

## ✨ Key Features

### 🚀 Zero-Configuration
Models and weights are automatically downloaded from Hugging Face the first time you run it. No manual weight management required.

### 🧠 Intelligent Routing
Automatically classifies the input image (Pie, Vertical Bar, Horizontal Bar, Line, etc.) and routes it to the correct extraction algorithm.

### 🖼 Robust Input Handling
Pass a file path (`String`), an OpenCV image (`NumPy array`), or a `PIL Image` directly into the analyzer.

### 🔍 Smart OCR Masking
Uses contrast filtering and spatial mapping to accurately match text labels with their corresponding graphical data points.

---

## 📦 Installation

Install directly from PyPI:

```bash
pip install graphvision-ai
```


## 🚀 Quick Start

Extracting data from a graph takes less than 5 lines of code:


```
from graphvision.extractor import GraphExtractor

try:
    # 1. Initialize your engine (this will download weights if needed)
    vision_engine = GraphExtractor()
    
    # Path to your test image
    image_to_test = "hbar2.png" 
    
    # 2. Run the extraction using the new method name
    print(f"\n🚀 Extracting data from {image_to_test}...")
    result_json_string = vision_engine.extract(image_to_test)
    
    # 3. Print the result (it's already a nicely formatted string!)
    print("\n✅ Extraction Successful!")
    print(result_json_string)

except Exception as e:
    print(f"\n❌ Error during testing: {e}")
```

## 📄 Example Output

```
{
    "type": "pie",
    "title": "Favorite Programming Languages",
    "data": {
        "Python": 45.2,
        "JavaScript": 25.1,
        "C++": 15.4,
        "Java": 14.3
    }
}
```


## 📈 Supported Graph Types

Currently, GraphVision AI supports high-accuracy extraction for:

- `pie` — Pie Charts  
- `vbar_categorical` — Vertical Bar Charts  
- `hbar_categorical` — Horizontal Bar Charts  

Line and Dot-Line charts coming soon.