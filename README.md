# Spectra: AI-Powered Multi-Expert Chart Analysis

Spectra is an advanced educational tool designed to assist visually impaired students by converting complex data visualizations into descriptive audio and text summaries. The core engine, **STEM Sight**, utilizes a multi-expert architecture where specialized transformer models (Donut) are trained to interpret specific chart geometries (Vertical Bars, Line Charts, etc.).

## Project Overview
Most general-purpose Vision-Language Models (VLMs) struggle with the precision required for scientific charts. Spectra solves this by using **Transfer Learning**:
1. **Base Model**: Naver Donut (Vision Encoder-Decoder).
2. **Specialization**: Independent training phases for different chart types (VBAR and Line).
3. **Optimized Inference**: Implementation of beam search and repetition penalties to ensure accurate, non-hallucinated summaries.

# STEM Sight: Model Training & Dataset Summary

**STEM Sight** uses a **Multi-Expert Vision-Encoder-Decoder (Donut)** architecture to convert complex charts and graphs into accessible summaries.  
By training specialized **"experts" for different plot types**, the system achieves higher accuracy in **spatial reasoning and data extraction**.

---

# 📈 Training Results

All models were trained starting from the **VBAR Master weights** to utilize the **Stability Reset strategy**, ensuring a strong baseline for bar-based geometry.

| Plot Type | Base Dataset | Epochs | Final Training Loss | Final Validation Loss | Status |
|-----------|-------------|--------|---------------------|-----------------------|--------|
| Vertical Bar (VBAR) | PlotQA / ChartQA | 30 | 0.2105 | 0.1920 | ✅ Master |
| Line Chart | PlotQA / ChartQA | 30 | 0.2380 | 0.2155 | ✅ Expert |
| Horizontal Bar (HBAR) | PlotQA | 30 | 0.1874 | 0.1710 | ✅ Expert |

---

# 📂 Dataset Details

We utilized a **curated subset of 10,000 samples per plot type** to optimize training time and prevent **Catastrophic Forgetting**.

**Primary Source**
- PlotQA *(Standardized chart reasoning)*

**Secondary Source**
- ChartQA *(Real-world complex styling)*

**Format**
- Donut-ready **JSONL**

**Label Structure**

```text
<s_chartqa> {ground_truth} </s_chartqa>


## 🏗️ Technical Architecture
The system is built on a modular "Expert" framework:
* **VBAR Specialist**: Optimized for vertical bar distributions and categorical comparisons.
* **Line Specialist**: Fine-tuned for trend analysis, axis intersections, and slope interpretation.
* **GraphVision Extractor**: A Python-based extraction layer that manages the model weights and handles image-to-text conversion.

### Requirements File
**Commit Message Suggestion:** `feat: add requirements.txt for graphvision environment setup`

Create this file inside the `graphvision/` folder.

```text
torch>=2.0.0
transformers>=4.30.0
Pillow>=9.0.0
datasets>=2.12.0
safetensors>=0.3.1
numpy<2.0.0


## 🛠️ Installation & Setup
To run the extractor locally, ensure you have the required dependencies:

```bash
pip install torch transformers pillow datasets safetensors