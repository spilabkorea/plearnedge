# 🔥A Label-Free Lightweight Prompt-Driven Cross-Modal Fire Detection on Robotic Edge Platforms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> A lightweight, prompt-driven cross-modal learning framework for real-time wildfire monitoring on resource-constrained edge devices.


## 🌟 Highlights

- **90% accuracy** on Kaggle Fire & Smoke dataset
- **2239 FPS** inference on Raspberry Pi 5 (0.45ms latency)
- **0.36 MB** model size with 8-bit quantization
- **LoRA-based** parameter-efficient fine-tuning
- **Zero-shot capability** for unseen environments


## 📊 Performance Comparison

| Method             | Acc. (%) | FPS  | Latency (ms) | Size (MB) |
|--------------------|----------|------|--------------|-----------|
| COCA               | 64       | 10   | 99.81        | 1010      |
| Few-shot LoRA      | 71       | 2062 | 0.48         | 0.46      |
| Few-shot Hybrid    | 46       | 2062 | 0.48         | 0.46      |
| GIT Caption        | 71       | 12   | 82.34        | 690       |
| CLIP-ViT           | **91**   | 273  | 3.65         | 6.70      |
| **P-LearnEdge (Ours)** | **90**   | **2239** | **0.45**     | **0.36**  |


## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git@github.com:spilabkorea/plearnedge.git
cd plearnedge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download the Kaggle Fire and Smoke dataset:
```bash
# Download from Google Drive
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&id=1L_TOG_sWp4xI9ojwe3YHu46VxmCS5xP8" -O dataset.zip

# Extract
unzip dataset.zip -d data/

Expected structure:

data/
├── fire/
│   ├── 0000000001.jpg
│   └── ...
└── smoke/
├── 0000000001.jpg
└── ...

### Training
```bash
# Basic training

python model/plearnedge.py


### Inference
```bash
# Single image prediction + FPS
python model/plearnedge_fps.py
```

## 📖 Model Architecture

P-LearnEdge combines:
1. **Lightweight CNN** (4 conv layers, 128D embeddings)
2. **LoRA layers** (r=4, α=16) for efficient adaptation
3. **CLIP-style** dual-encoder architecture
4. **Prompt-driven** zero-shot classification
```python
from models import FireClipModel

# Initialize model
model = FireClipModel(embedding_dim=128, num_classes=2)

# Inference
with torch.no_grad():
    logits = model(image)
    prediction = torch.argmax(logits, dim=1)
```

## 🔬 Reproducing Results

### Baseline Comparisons
```bash
# CLIP-ViT baseline
python baselines/clip.py 

# COCA baseline
python baselines/coca.py

# Few-shot hybrid
python baselines/fewshot_hybrid.py
```

## 📄 Citation

If you use P-LearnEdge in your research, please cite:
```bibtex
@article{plearnedge2025,
  title={A Label-Free Lightweight Prompt-Driven Cross-Modal Fire Detection on Robotic Edge Platforms},
  author={HyeYoung et al.},
  conference={ICRCV},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see LICENSE file.

## 📧 Contact

- **Author**: SPILab Research Team
- **Email**: support@spilab.kr
- **Website**: https://spilab.kr/

## 🙏 Acknowledgments

- Kaggle Fire and Smoke Dataset contributors
- OpenAI CLIP team for inspiration
- SPILab Corporation for support

---

**Note**: This is research code. For production deployment, please contact us for optimized versions.



