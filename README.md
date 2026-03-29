# Spectra: AI-Powered Multi-Expert Chart Analysis Platform

Spectra is an advanced **accessibility-first** platform designed to assist visually impaired students by converting complex data visualizations into descriptive audio and text summaries. The core engine, **STEM Sight**, utilizes a **multi-layered AI architecture** combining YOLO object detection, optical character recognition, and specialized Donut transformer models for precise chart understanding.

## 🎯 Project Vision

Spectra democratizes access to scientific charts and graphs by leveraging cutting-edge AI to:
- **Detect and classify** different chart types (Vertical Bars, Horizontal Bars, Line Charts, Pie Charts, Dot/Line Scatter)
- **Extract precise numerical data** using YOLO detection + EasyOCR text recognition
- **Generate conversational summaries** using Donut fine-tuned models
- **Deliver audio explanations** through natural language generation and text-to-speech

## 🏗️ Technical Architecture

Spectra's architecture follows a **three-stage pipeline**:

```
Image Upload
    ↓
Stage 1: Chart Classification (ResNet-18)
    ↓
Stage 2: Data Extraction (YOLO + EasyOCR)
    ↓
Stage 3: Natural Language Generation (Donut Fine-tuned Models)
    ↓
Audio Output (Text-to-Speech)
```

---

## 🔧 Component Breakdown

### 1. **Chart Classification** (ResNet-18)
- Classifies input images into: `VBAR`, `HBAR`, `Line`, `Pie`, `Dot/Line Scatter`
- Pre-trained ResNet-18 backbone fine-tuned on curated dataset
- Ensures routing to the correct specialized extraction pipeline

### 2. **Data Extraction Layer** (YOLO + EasyOCR)

#### YOLO Object Detection
- **Purpose**: Detect and localize chart elements
  - Bar segments in bar charts
  - Axis ticks and labels
  - Legend items
  - Data point markers in scatter plots
- **Models Used**:
  - `bar.pt` - Bar chart element detection (YOLO11n-seg)
  - `dot_line.pt` - Dot/Line chart element detection (YOLO11n-seg)
- **Output**: Bounding boxes with confidence scores for each detected element

#### EasyOCR Text Recognition
- **Purpose**: Extract text from chart axes, labels, and legends
- **Configuration**: Optimized for English text with high recall
- **Features**:
  - Robust to various font sizes and orientations
  - Handles rotated text in complex charts
  - Filters OCR noise through regex-based number extraction

#### Number Extraction & Axis Mapping
- **Smart OCR Cleaning**: Converts OCR text to numerical values
- **Robust Scaling**: Uses multiple reference points to establish axis scale
- **Spatial Reasoning**: Maps pixel coordinates to data values using detected axis positions

#### Why YOLO + EasyOCR is the Core Strength

This hybrid approach is **fundamentally superior** to end-to-end deep learning models like DePlot or Pix2Struct:

- **Direct Supervision**: YOLO learns exact visual boundaries; EasyOCR focuses purely on text recognition
- **Interpretability**: Each component's output is inspectable, making debugging and improvement straightforward
- **Modularity**: Swap YOLO versions, upgrade EasyOCR, or replace axis mapping independently
- **Robustness**: Specialized training on chart elements beats generalist vision models
- **Efficiency**: Modular design allows selective GPU/CPU usage for different stages

### 3. **Question-Answering & Summarization** (Donut Fine-tuning)

**Why Donut?** Donut (Document Understanding Transformer) excels at structured data extraction from visual inputs. Unlike generic VLMs, Donut is fine-tuned specifically for chart reasoning tasks.

#### Training Strategy: Multi-Expert Architecture
Rather than a single generalist model, we train **specialized experts** for different chart types:

- **VBAR Expert** - Vertical Bar Chart Specialist
- **HBAR Expert** - Horizontal Bar Chart Specialist  
- **Line Expert** - Trend/Line Chart Specialist
- **Pie Expert** - Pie Chart Specialist
- **Dot/Line Expert** - Scatter Plot Specialist

Each expert is fine-tuned starting from **STEM Sight Master weights**, ensuring:
- Knowledge transfer across chart types
- Stability through transfer learning
- Reduced hallucination through specialized training

#### Fine-Tuning Approach
```python
1. Start from: Naver Donut Base (Vision Encoder-Decoder)
2. Task Format: <s_chartqa> {ground_truth_summary} </s_chartqa>
3. Loss Function: Cross-entropy with beam search inference
4. Inference Parameters:
   - num_beams=4 (Better quality)
   - repetition_penalty=2.5 (Reduce hallucination)
   - no_repeat_ngram_size=3 (Prevent loops)
   - max_length=512 (Capture detailed explanations)
```

---

## 📊 STEM Sight: Model Training & Dataset Summary

**STEM Sight** uses a **Multi-Expert Vision-Encoder-Decoder (Donut)** architecture to convert complex charts and graphs into accessible summaries.  
By training specialized **"experts" for different plot types**, the system achieves higher accuracy in **spatial reasoning and data extraction**.

## 📈 Model Performance Summary

All specialized models were trained starting from the **VBAR Master weights** to utilize the **Stability Reset strategy**, ensuring a strong baseline for all chart types.

| Chart Type | Base Dataset | Epochs | Training Loss | Validation Loss | Status |
|-----------|-------------|--------|---------------------|-----------------------|--------|
| Vertical Bar (VBAR) | PlotQA / ChartQA | 30 | 0.2105 | 0.1920 | ✅ Master |
| Line Chart | PlotQA / ChartQA | 30 | 0.2380 | 0.2155 | ✅ Expert |
| Horizontal Bar (HBAR) | PlotQA | 30 | 0.1874 | 0.1710 | ✅ Expert |
| Pie Chart | PlotQA | 25 | 0.2241 | 0.2089 | ✅ Expert |
| Dot/Line Scatter | PlotQA | 25 | 0.2156 | 0.1998 | ✅ Expert |

### Training Dataset Details
- **Primary Source**: PlotQA (Standardized chart reasoning with clean layouts)
- **Secondary Source**: ChartQA (Real-world complex styling and annotations)
- **Samples per Type**: 10,000 curated samples to prevent catastrophic forgetting
- **Format**: Donut-ready JSONL with image + ground truth pairs
- **Label Structure**: `<s_chartqa> {chart_analysis_summary} </s_chartqa>`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration) OR Apple Silicon (MPS support) OR CPU fallback
- 8GB+ RAM (16GB+ recommended for full model inference)

### Option 1: Using PyPI Package (Recommended for Integration)

The core vision extraction engine is published on PyPI and can be installed independently:

```bash
pip install graphvision-ai==0.2.4
```

**Quick Start with GraphVision:**
```python
from graphvision import GraphExtractor

# Initialize the extraction engine
extractor = GraphExtractor()

# Extract chart data from an image
result = extractor.extract("path/to/chart.png")

# Result format (JSON):
# {
#   "chart_type": "vbar_categorical",
#   "data": [...],
#   "x_axis_label": "...",
#   "y_axis_label": "...",
#   "title": "..."
# }
```

📦 **PyPI Package**: https://pypi.org/project/graphvision-ai/

### Option 2: Running the Backend API Locally

The backend API wraps GraphVision and adds LLM-powered summarization using Groq's Llama 3 model.

#### Installation

```bash
cd Spectra-Backend

# Install dependencies
pip install -r requirements.txt

# Set up Groq API Key (get free key from https://console.groq.com)
export GROQ_API_KEY="your-groq-api-key-here"

# Start the FastAPI server
python main.py
```

The server will start on `http://127.0.0.1:8000`

#### API Documentation

Once running, visit: **http://127.0.0.1:8000/docs** for interactive Swagger documentation

**Endpoint**: `POST /analyze-graph`

```bash
curl -X POST "http://127.0.0.1:8000/analyze-graph" \
  -H "Content-Type: application/octet-stream" \
  -d @chart.png
```

**Response**: A conversational text summary of the chart (optimized for text-to-speech)

### Option 3: Deployed Backend (Hugging Face Spaces)

A production-ready version is hosted on Hugging Face Spaces:

🔗 **Backend API**: https://shadowgard3n-spectra-backend.hf.space/docs

Use this for integration without setting up locally. The API is identical to the local version.

### Option 4: Chrome Extension (Frontend)

The Chrome extension provides a convenient UI for real-time chart analysis on webpages.

#### Installation

1. Navigate to `chrome://extensions/` in Chrome
2. Enable **Developer Mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `Spectra-Frontend` folder

#### Usage

1. Click the **STEM Sight icon** in the Chrome toolbar
2. Navigate to a webpage with charts/graphs
3. The extension will:
   - Identify images on the page
   - Send them to the backend API for analysis
   - Speak the results aloud using Web Speech API
4. **Press `Escape`** to stop reading

#### Configuration

Edit `Spectra-Frontend/background.js` to change the backend URL:

```javascript
// Local backend (default)
const API_URL = "http://127.0.0.1:8000/analyze-graph";

// Or use the deployed version
// const API_URL = "https://shadowgard3n-spectra-backend.hf.space/analyze-graph";
```

---

## 📁 Project Structure

```
Spectra/
├── README.md                    # This file
│
├── graphvision/                 # PyPI Package (Core Engine)
│   ├── graphvision/
│   │   ├── __init__.py
│   │   ├── extractor.py         # GraphExtractor class
│   │   └── __pycache__/
│   ├── pyproject.toml           # PyPI metadata
│   └── weights/                 # Model weights (auto-downloaded from HF)
│
├── Spectra-Backend/             # FastAPI Server
│   ├── main.py                  # API endpoints
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile               # Docker deployment config
│   └── README.md
│
├── Spectra-Frontend/            # Chrome Extension
│   ├── manifest.json            # Extension metadata
│   ├── background.js            # Backend communication
│   ├── content.js               # Page content injection
│   ├── style.css
│   └── index.html
│
├── notebooks/                   # Jupyter notebooks for training
│   ├── STEM_Sight_Horizontal.ipynb      # HBAR fine-tuning
│   ├── STEM_Sight_VBar_Training.ipynb   # VBAR fine-tuning
│   ├── STEM_Sight_Line.ipynb            # Line chart training
│   └── ... (other experimentation notebooks)
│
├── ChartQA_Dataset/             # Training data (test/train/val splits)
├── PlotQA_Dataset/              # Training data (standardized plots)
└── FigureQA_Dataset/            # Training data (complex figures)
```

---

## 🎓 Training Your Own Models

If you want to fine-tune Donut on your own chart data:

### Dataset Format

Each dataset should have:
- `train/` and `validation/` folders with `png/` subdirectories
- `train/metadata.jsonl` and `validation/metadata.jsonl` with lines like:

```json
{"file_name": "chart_001.png", "ground_truth": "{\"gt_parse\": \"Chart shows an increase from 10 to 50...\"}"}
```

### Training Script

See `notebooks/STEM_Sight_Horizontal.ipynb` for a complete example:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer
from datasets import load_dataset

# Load base model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Configure model
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_chartqa>'])[0]

# Load your data
dataset = load_dataset("imagefolder", data_dir="/path/to/dataset")

# Train
training_args = TrainingArguments(
    output_dir="./my_expert_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-6,
    eval_strategy="steps",
    eval_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
```

---

## 💡 How It Works: End-to-End

### Example Flow

1. **User uploads chart image**
   ```
   Input: PNG of a vertical bar chart
   ```

2. **Stage 1 - Classification**
   ```
   ResNet-18 classifier → "vbar_categorical"
   Routes to VBAR extraction pipeline
   ```

3. **Stage 2 - Data Extraction (YOLO + EasyOCR)** ⭐ Core Architecture
   ```
   a) YOLO Detection:
      - Detects bar segments, axis labels, legend items
      - Returns bounding boxes with confidence scores
      - Models: bar.pt (YOLO11n-seg), dot_line.pt (YOLO11n-seg)
   
   b) EasyOCR Recognition:
      - Extracts text from detected regions
      - Reads bar labels, axis values, titles
      - Optimized for English text
   
   c) Spatial Reasoning:
      - Maps pixel coordinates to data values
      - Establishes axis scale from reference points
      - Result: {"Sales": 45.2, "Revenue": 78.5, ...}
   ```

4. **Stage 3 - Language Generation**
   ```
   Input: Extracted data from YOLO + EasyOCR
   Donut Model (VBAR Expert): 
   → "The chart shows sales performance across four quarters. 
      Q1 had the highest value at 45 units, while Q4 had the 
      lowest at 12 units."
   ```

5. **Stage 4 - Output**
   ```
   a) Text output (for Chrome extension)
   b) Text-to-Speech (browser Web Speech API)
   c) Accessible audio for users
   ```

---

## 🔌 API Integration Examples

### Using with JavaScript (Web)

```javascript
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch("http://127.0.0.1:8000/analyze-graph", {
  method: "POST",
  body: formData
});

const explanation = await response.text();
console.log(explanation);
```

### Using with Python

```python
import requests

with open("chart.png", "rb") as img:
    response = requests.post(
        "http://127.0.0.1:8000/analyze-graph",
        files={"file": img}
    )

print(response.text)
```

### Using with GraphVision (No API needed)

```python
from graphvision import GraphExtractor

extractor = GraphExtractor()
result_json = extractor.extract("chart.png")

# Parse and use result
import json
data = json.loads(result_json)
print(f"Chart Type: {data['chart_type']}")
print(f"Data: {data['data']}")
```

---

## 🌐 Performance & Accuracy Metrics

### 📊 YOLO + EasyOCR Accuracy Results

Comprehensive evaluation on FigureQA dataset (799 processed images across 4 chart types):

#### Type 1: Global Chart Text Extraction (Title, Axis Labels)

| Chart Type | Title Accuracy | X-Axis Label | Y-Axis Label |
|-----------|--------|--------|--------|
| **Vertical Bar** | 100.00% | 90.91% | 65.00% |
| **Horizontal Bar** | 100.00% | 91.76% | 86.30% |
| **Pie** | 77.17% | N/A | N/A |
| **Dot/Line** | 100.00% | 88.02% | 67.10% |

#### Type 2: Data Category Recognition (Labels on bars, pie slices, data points)

| Chart Type | Recall | Precision | F1 Score |
|-----------|--------|--------|--------|
| **Vertical Bar** | 90.05% | 93.53% | **0.9176** |
| **Horizontal Bar** | 93.58% | 99.64% | **0.9652** ⭐ |
| **Pie** | 86.84% | 96.47% | **0.9140** |
| **Dot/Line** | 71.17% | 80.27% | **0.7545** |

#### Type 3: Data Value Accuracy (Numerical extraction from charts)

| Chart Type | 5% Error Threshold | 10% Error Threshold |
|-----------|--------|--------|
| **Vertical Bar** | 55.66% recall | 63.43% recall |
| **Horizontal Bar** | 87.77% recall ⭐ | 88.45% recall ⭐ |
| **Dot/Line** | 63.48% recall | 66.43% recall |
| **Pie** | N/A | N/A |

### ⚡ Speed Advantage: YOLO + EasyOCR vs. DePlot

**YOLO + EasyOCR is ~5x faster than DePlot** while maintaining strong accuracy:

- **Modular Pipeline**: Each component optimized independently for speed
- **Lightweight Models**: YOLO11n-seg vs. heavy end-to-end architectures
- **Efficient Text Extraction**: EasyOCR for dedicated OCR vs. vision-language models
- **Direct Mapping**: Pixel-to-data conversion faster than neural regression

### Component Architecture

| Component | Purpose | Technology |
|-----------|---------|----------|
| **Chart Classification** | Identify chart type | ResNet-18 |
| **YOLO Detection** | Localize chart elements | YOLO11n-seg (bar.pt, dot_line.pt) |
| **EasyOCR Recognition** | Extract text from charts | EasyOCR English |
| **Spatial Mapping** | Convert pixels to data values | Custom coordinate mapping |
| **Donut QA** (Optional) | Generate natural language explanations | Vision Encoder-Decoder |
| **LLM Summarization** (Optional) | Conversational summaries | Groq Llama 3 (Cloud) |

### Hardware Tiers

| Tier | GPU | RAM | Typical Use Case |
|------|-----|-----|----------|
| **Local (CPU)** | None | 4GB | Development, single charts |
| **Local (GPU)** | NVIDIA RTX 3060+ | 16GB | Real-time applications |
| **Apple Silicon (M1/M4)** | MPS | 8GB | MacBook deployment |
| **Cloud (Spaces)** | A40 GPU | 32GB | Production deployment |
| **Chrome Extension** | N/A | 4GB | Browser-based, lightweight |

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Add support for more chart types (heatmaps, 3D charts, etc.)
- [ ] Improve OCR for handwritten labels
- [ ] Add multilingual support
- [ ] Optimize model inference time
- [ ] Create mobile app version

## 📄 License

Spectra is released under the **MIT License**. See `LICENSE` for details.

---

## 🎯 Future Roadmap

- **v2.0**: Mobile app (iOS/Android) with offline inference
- **v2.5**: Real-time chart generation from live data feeds
- **v3.0**: Multimodal learning with audio descriptions
- **v3.5**: Embedded chart generation for inaccessible documents

---

## 📞 Support

- **Issues & Bugs**: GitHub Issues
- **Documentation**: This README + Jupyter notebooks
- **API Docs**: https://shadowgard3n-spectra-backend.hf.space/docs
- **PyPI Package**: https://pypi.org/project/graphvision-ai/

---

**Built with ❤️ for accessibility**
