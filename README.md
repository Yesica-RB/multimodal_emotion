# 🌍 Multimodal Emotion Recognition

**Detecting emotions in travel Twitter posts using NLP, Computer Vision, Deep Learning, and Intelligent Systems**

> Universidad Intercontinental de la Empresa (UIE) · Grado en Ingeniería en Sistemas Inteligentes · 3er Curso  
> Yésica Ramírez Bernal · April 2026

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Notebooks Execution Order](#notebooks-execution-order)
- [Demo Application](#demo-application)
- [Dataset](#dataset)
- [Modules](#modules)
- [Technologies](#technologies)

---

## Overview

This project builds a **multimodal emotion recognition system** that classifies travel Twitter posts as **positive**, **negative**, or **neutral** by combining text and image analysis.

Neither text nor image alone is sufficient for reliable emotion detection on Twitter:
- A beach photo can look positive — but the caption might say *"worst trip ever"*
- *"Sick view"* is positive slang — but classical NLP treats it as negative

The solution combines **5 independent modules** from 4 academic subjects, unified by a **goal-based rational agent** that uses **Simulated Annealing** to find the optimal fusion weights.

**Final result: F1 = 0.76** vs random baseline of 0.33.

---

## System Architecture

```
Input: image + text caption (Twitter post)
         │
    ┌────┴────────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
TEXT MODULES                              IMAGE MODULES
─────────────────────                    ──────────────────
NLP Classic (LR)    F1=0.60              CV Classic (SVM) F1=0.37
RoBERTa LLM         F1=0.67              ResNet18 (TL)    F1=0.39
BERT Fine-tuned     F1=0.74
    │                                             │
    └────────────────┬────────────────────────────┘
                     │
          P₁, P₂, P₃, P₄, P₅ (probability vectors)
                     │
                     ▼
     ┌───────────────────────────────────┐
     │  Late Fusion — Rational Agent     │
     │  Weights optimised by SA          │
     │  LR=0.364, BERT=0.384,            │
     │  SVM=0.000, ResNet=0.147,         │
     │  RoBERTa=0.105                    │
     └───────────────────────────────────┘
                     │
                     ▼
     Output: positive / negative / neutral
```

---

## Results

| Module | Modality | F1 |
|---|---|---|
| Random Baseline | — | 0.33 |
| CV Classic (SVM) | Image | 0.37 |
| ResNet18 (TL) | Image | 0.39 |
| Naive Bayes | Text | 0.58 |
| Logistic Regression | Text | 0.60 |
| RoBERTa LLM (zero-shot) | Text | 0.67 |
| BERT Fine-tuned | Text | 0.74 |
| **Late Fusion (SA)** | **Both** | **0.76** |

Key findings:
- **Fusion beats all individual modules** — combining modalities works
- **Text >> Image** on Twitter (35-point gap) — captions carry more emotional signal
- **RoBERTa zero-shot > trained classical NLP** — domain pretraining beats task training

---

## Project Structure

```
multimodal_emotion/
│
├── data/
│   ├── raw/
│   │   ├── data/              ← MVSA-Single images (.jpg) and captions (.txt)
│   │   └── labelResultAll.txt ← raw annotation file
│   └── processed/
│       └── labels.csv         ← 4,869 rows: id, label, text, image_path
│
├── src/
│   ├── nlp_classic/
│   │   ├── __init__.py
│   │   ├── build_csv.py       ← builds labels.csv from raw data
│   │   ├── preprocessing.py   ← text cleaning pipeline
│   │   └── classifier.py      ← TF-IDF + NB + LR
│   ├── nlp_llm/
│   │   ├── __init__.py
│   │   └── llm_classifier.py  ← RoBERTa zero-shot evaluation
│   ├── cv_classic/
│   │   ├── __init__.py
│   │   └── feature_extractor.py ← HSV + Canny + k-means + SVM
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── late_fusion.py     ← Simulated Annealing weight optimisation
│   ├── nlp_bert/
│   │   └── __init__.py        ← BERT trained in Colab (see notebooks/03)
│   ├── dl_cnn/
│   │   └── __init__.py        ← ResNet18 trained in Colab (see notebooks/05)
│   └── demo.py                ← Gradio demo application
│
├── notebooks/
│   ├── 01_eda.ipynb           ← Exploratory Data Analysis
│   ├── 02_nlp_classic.ipynb   ← NLP Classic (Mac)
│   ├── 03_bert.ipynb          ← BERT Fine-tuning (Google Colab T4)
│   ├── 04_cv_classic.ipynb    ← Classical CV (Mac)
│   ├── 05_resnet.ipynb        ← ResNet18 (Google Colab T4)
│   ├── 06_fusion.ipynb        ← Late Fusion with SA (Mac)
│   └── 07_llm.ipynb           ← RoBERTa LLM (Mac)
│
├── results/
│   ├── metrics_nlp_classic.json
│   ├── metrics_bert.json
│   ├── metrics_cv_classic.json
│   ├── metrics_resnet.json
│   ├── metrics_llm.json
│   ├── metrics_fusion_sa.json
│   ├── resnet18_model.pth
│   └── figures/
│       ├── class_distribution.png
│       ├── caption_length.png
│       ├── nlp_classic_results.png
│       ├── bert_loss_curve.png
│       ├── cv_classic_results.png
│       ├── resnet_loss_curve.png
│       ├── gradcam_examples.png
│       ├── sa_convergence.png
│       ├── fusion_comparison.png
│       └── llm_comparison.png
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/               ← same figures as results/figures/
│
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### Requirements
- Python 3.11
- macOS (for text/CV modules) or Google Colab (for BERT/ResNet18)

### Install dependencies

```bash
# Clone the repository
git clone https://github.com/yesicarb/multimodal_emotion.git
cd multimodal_emotion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
gradio>=4.0.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=10.0.0
nltk>=3.8.0
```

---

## How to Run

### Step 1 — Build the dataset CSV (only once)

```bash
cd multimodal_emotion
source venv/bin/activate
python src/nlp_classic/build_csv.py
```

### Step 2 — Run the demo

```bash
python src/demo.py
```

The demo will load all models and open at `http://localhost:7860`. A public shareable URL is also generated.

---

## Notebooks Execution Order

> ⚠️ All 5 JSON result files must exist in `results/` before running notebook 06.

### On your Mac (in order):

```
01_eda.ipynb          → EDA figures
02_nlp_classic.ipynb  → results/metrics_nlp_classic.json
04_cv_classic.ipynb   → results/metrics_cv_classic.json   (~15 min)
07_llm.ipynb          → results/metrics_llm.json          (~10 min)
```

### On Google Colab (T4 GPU required):

```
03_bert.ipynb   → download metrics_bert.json to results/            (~20 min)
05_resnet.ipynb → download metrics_resnet.json + resnet18_model.pth (~15 min)
```

### Back on your Mac:

```
06_fusion.ipynb → results/metrics_fusion_sa.json  (~2 min)
```

---

## Demo Application

The Gradio demo (`src/demo.py`) provides:

- **Text input** — tweet caption
- **Image input** (optional) — travel photo
- **Prediction** — emotion label + confidence score
- **Module breakdown** — prediction from each of the 5 modules
- **Probability chart** — distribution over 3 classes
- **Grad-CAM map** — visual attention map (only when image is provided)

Example predictions:

| Caption | Prediction | Confidence |
|---|---|---|
| "Amazing sunset! Best trip ever #travel" | 🟢 POSITIVE | 68% |
| "Missed my flight, lost my luggage. Worst day." | 🔴 NEGATIVE | 43% |
| "Just arrived at the hotel." | ⚪ NEUTRAL | 37% |

---

## Dataset

**MVSA-Single** — Multimodal Visual-Textual Sentiment Analysis  
- 4,869 image-text pairs from Twitter (after filtering)
- Labels: positive (35.6%), neutral (39.4%), negative (25.0%)
- Mean caption length: 13.1 words
- Source: http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

---

## Modules

| Module | Subject | Technique | F1 |
|---|---|---|---|
| NLP Classic | NLP | TF-IDF + Logistic Regression + Grid Search | 0.60 |
| RoBERTa LLM | NLP | Twitter-pretrained RoBERTa (zero-shot) | 0.67 |
| BERT Fine-tuned | NLP | bert-base-uncased + AdamW, 3 epochs T4 GPU | 0.74 |
| CV Classic | Computer Vision | HSV + Canny + k-means + RBF-SVM | 0.37 |
| ResNet18 | Deep Learning | Transfer learning from ImageNet + Grad-CAM | 0.39 |
| **Late Fusion** | **Intelligent Systems** | **Rational agent + Simulated Annealing** | **0.76** |

---

## Technologies

| Technology | Version | Use |
|---|---|---|
| Python | 3.11 | Main language |
| PyTorch | 2.0 | BERT, ResNet18 |
| HuggingFace Transformers | 4.35 | BERT, RoBERTa |
| scikit-learn | 1.3 | TF-IDF, SVM, LR, Grid Search |
| OpenCV | 4.8 | HSV histogram, Canny, k-means, Grad-CAM |
| Gradio | 4.x | Demo interface |
| NLTK | 3.8 | Tokenisation, stopwords, stemming |
| Google Colab | T4 GPU | BERT and ResNet18 training |

---

*UIE · Grado en Ingeniería en Sistemas Inteligentes · Yésica Ramírez Bernal · 2026*
