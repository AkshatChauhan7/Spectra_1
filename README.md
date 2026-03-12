# Spectra: AI-Powered Multi-Expert Chart Analysis

Spectra is an advanced educational tool designed to assist visually impaired students by converting complex data visualizations into descriptive audio and text summaries. The core engine, **STEM Sight**, utilizes a multi-expert architecture where specialized transformer models (Donut) are trained to interpret specific chart geometries (Vertical Bars, Line Charts, etc.).

## 🚀 Project Overview
Most general-purpose Vision-Language Models (VLMs) struggle with the precision required for scientific charts. Spectra solves this by using **Transfer Learning**:
1. **Base Model**: Naver Donut (Vision Encoder-Decoder).
2. **Specialization**: Independent training phases for different chart types (VBAR and Line).
3. **Optimized Inference**: Implementation of beam search and repetition penalties to ensure accurate, non-hallucinated summaries.

## 🏗️ Technical Architecture
The system is built on a modular "Expert" framework:
* **VBAR Specialist**: Optimized for vertical bar distributions and categorical comparisons.
* **Line Specialist**: Fine-tuned for trend analysis, axis intersections, and slope interpretation.
* **GraphVision Extractor**: A Python-based extraction layer that manages the model weights and handles image-to-text conversion.

## 🛠️ Installation & Setup
To run the extractor locally, ensure you have the required dependencies:

```bash
pip install torch transformers pillow datasets safetensors
