# PersonaTrace: Fine-Tuned Vision Transformer for Target Person Identification

## 🚀 Overview

**PersonaTrace** is an end-to-end computer vision system designed to **identify and track a specific person across an entire video** using minimal reference images. The project combines **YOLOv8-based person detection** with a **fine-tuned Vision Transformer (ViT)** to generate robust identity embeddings and perform accurate matching with temporal consistency.

This project is built with a strong focus on **real-world deployment**, **model persistence**, and **evaluation reproducibility**, ensuring that results remain stable across sessions without requiring repeated training or inference.

---

## 🎯 Key Features

* 🔍 **High-accuracy person detection** using YOLOv8
* 🧠 **Vision Transformer (ViT) fine-tuned for identity discrimination**
* 🧬 **Embedding-based person re-identification**
* ⏱️ **Temporal smoothing** for stable video-level predictions
* 💾 **Model & embeddings persistence** (no reruns required)
* 📊 **Precision, Recall & F1-score evaluation**
* ⚙️ Modular, deployment-ready pipeline

---

## 🧩 System Architecture

```
Video Input
   ↓
Frame Extraction
   ↓
YOLOv8 Person Detection
   ↓
Identity-Relevant Crop Filtering
   ↓
ViT Embedding Extraction
   ↓
Similarity Matching
   ↓
Temporal Smoothing
   ↓
Final Target Person Frames
```

---

## 📁 Project Structure

```
project-root/
│                   
├── embeddings.npy             # ViT embeddings (persistent)
├── matched_indices.npy        # Final matched frame indices
├── vit_finetuned.pth          # Fine-tuned ViT model (Git LFS)
├── yolov8n.pt / yolov8x.pt    # YOLOv8 detection models
├── main.ipynb                 # Complete pipeline notebook
├── requirements.txt
└── README.md
```

---

## 🧠 Model Details

### Person Detection

* **Model**: YOLOv8
* **Classes Used**: Person
* **Purpose**: Accurate bounding box extraction for human subjects

### Identity Modeling

* **Backbone**: Vision Transformer (ViT-B/16)
* **Pretraining**: ImageNet
* **Fine-Tuning**: Binary identity classification
* **Output**: Identity-aware embeddings

---

## 📊 Evaluation Metrics

The system is evaluated using ground-truth identity annotations:

* **Precision**: Measures false positives
* **Recall**: Measures missed detections
* **F1-Score**: Balanced performance metric

> Final fine-tuned model achieved **near-perfect F1-score**, demonstrating strong generalization across frames.

---

## ⚡ Quick Start

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/PersonaTrace.git
cd PersonaTrace
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Pipeline

Open `main.ipynb` and execute cells sequentially.

> All models and embeddings are pre-saved — **no retraining required**.

---

## 🛠️ Deployment Notes

* Supports **CPU and GPU** environments
* Compatible with **Windows / Linux / macOS**
* Designed for **offline reproducibility**
* Large model files handled via **Git LFS**

---

## 🌟 Use Cases

* Video surveillance & forensics
* Sports analytics
* Content-based video retrieval
* Human behavior analysis
* Smart video indexing

---

## 📌 Future Enhancements

* Multi-person identity tracking
* Real-time inference pipeline
* Web-based UI dashboard
* Cross-video identity linking

---

## 🤝 Acknowledgements

* Ultralytics YOLOv8
* PyTorch & timm
* Open-source CV community

---

## 📜 License

This project is released under the **MIT License**.

---

## 👤 Author
**NAMAN BANSAL**
Developed with ❤️ as an advanced computer vision system for identity-aware video understanding.

If you find this project useful, please ⭐ the repository!

