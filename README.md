# Multimodal Emotion Recognition in Travel Social Media Posts

> Detecting emotions (positive / negative / neutral) in Twitter travel posts  
> by combining NLP, Computer Vision, Deep Learning and Intelligent Systems.

## Project Overview

This project proposes a four-module multimodal framework evaluated on the  
MVSA-Single dataset (4,869 image-text pairs from Twitter):

| Module | Technique | F1-score |
|---|---|---|
| Classical NLP | TF-IDF + Naive Bayes + Logistic Regression | 0.61 |
| BERT Fine-tuning | bert-base-uncased · 3 epochs · GPU | 0.72 |
| Classical CV | HSV + Canny + k-means + SVM | 0.40 |
| ResNet18 | Transfer Learning + Dropout + Grad-CAM | 0.43 |
| **Late Fusion (SA)** | **Simulated Annealing** | **0.74** |

## Repository Structure
multimodal_emotion/
├── data/
│   ├── raw/              ← MVSA-Single dataset (not uploaded to GitHub)
│   └── processed/        ← labels.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_nlp_classic.ipynb
│   ├── 03_bert.ipynb     ← Google Colab (GPU)
│   ├── 04_cv_classic.ipynb
│   ├── 05_resnet.ipynb   ← Google Colab (GPU)
│   └── 06_fusion.ipynb
├── src/
│   ├── nlp_classic/      ← preprocessing + classifier
│   ├── cv_classic/       ← feature extractor
│   └── fusion/           ← Simulated Annealing late fusion
├── results/
│   ├── figures/          ← all plots and Grad-CAM
│   └── metrics_*.json    ← saved metrics per module
├── paper/
│   ├── main.tex          ← LaTeX paper
│   └── references.bib
└── requirements.txt

## Setup

```bash
git clone https://github.com/TU_USUARIO/multimodal_emotion.git
cd multimodal_emotion
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

## Usage

Run notebooks in order:
1. `01_eda.ipynb` — dataset exploration
2. `02_nlp_classic.ipynb` — classical NLP module
3. `03_bert.ipynb` — BERT fine-tuning (Google Colab)
4. `04_cv_classic.ipynb` — classical CV module
5. `05_resnet.ipynb` — ResNet18 + Grad-CAM (Google Colab)
6. `06_fusion.ipynb` — Simulated Annealing fusion

## Dataset

MVSA-Single — [mcrlab.net](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)  
4,869 image-text pairs · 3 classes · Twitter travel posts

## Academic Context

Universidad Internacional de España (UIE)  
Grado en Ingeniería en Sistemas Inteligentes — 3er Curso  
Subjects: NLP · Advanced ML · Computer Vision · Intelligent Systems

## Results

![Fusion Comparison](results/figures/fusion_comparison.png)
![Grad-CAM](results/figures/gradcam_examples.png)
![SA Convergence](results/figures/sa_convergence.png)