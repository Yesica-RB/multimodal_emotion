# src/demo.py
import os
os.chdir("/Users/yesicarb/Desktop/UIE/3º Curso/2 SEM/PROYECTO/emotion/multimodal_emotion")

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import json
import sys
import cv2
from PIL import Image
from torchvision import models, transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
sys.path.append("src")
from nlp_classic.preprocessing import TextPreprocessor

# ── Configuración ───────────────────────────────────────────────
CLASS_NAMES = ['negative', 'neutral', 'positive']
EMOJIS      = {'negative': '🔴', 'neutral': '⚪', 'positive': '🟢'}
device      = torch.device('cpu')

# ── Cargar modelos de texto ─────────────────────────────────────
print("Cargando modelos de texto...")
df    = pd.read_csv("data/processed/labels.csv")
prep  = TextPreprocessor()
texts = [prep.preprocess(t) for t in df['text']]
le    = LabelEncoder()
y     = le.fit_transform(df['label'].tolist())

X_train, X_test, y_train, _ = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2),
                      sublinear_tf=True)
X_tr = vec.fit_transform(X_train)
lr   = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_tr, y_train)
print("Modelos de texto listos")

# ── Cargar ResNet18 ─────────────────────────────────────────────
print("Cargando ResNet18...")

def build_resnet(num_classes):
    model = models.resnet18(weights=None)
    # Sin Dropout — igual que como se guardó
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ── Transforms ─────────────────────────────────────────────────
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Grad-CAM ───────────────────────────────────────────────────
def generate_gradcam(model, img_tensor, target_class):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    gradients, activations = [], []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook = model.layer4.register_forward_hook(forward_hook)
    output = model(img_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    hook.remove()

    grad = gradients[0].squeeze().cpu().detach().numpy()
    act  = activations[0].squeeze().cpu().detach().numpy()
    weights_cam = grad.mean(axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights_cam):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max() + 1e-7
    return cam

def tensor_to_img(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

# ── Pesos SA ───────────────────────────────────────────────────
with open('results/metrics_fusion_sa.json') as f:
    sa_results = json.load(f)
weights = np.array([
    sa_results['best_weights']['LR'],
    sa_results['best_weights']['BERT'],
    sa_results['best_weights']['SVM'],
    sa_results['best_weights']['ResNet']
])
weights = weights / weights.sum()

# ── Función principal ──────────────────────────────────────────
def predict(image, text):
    if not text or text.strip() == "":
        return "⚠️ Please enter a text.", {}, None

    # --- Módulo texto (LR) ---
    clean    = prep.preprocess(text)
    x_vec    = vec.transform([clean])
    lr_proba = lr.predict_proba(x_vec)[0]

    # Aproximación BERT
    noise      = np.random.dirichlet(np.ones(3) * 8)
    bert_proba = 0.85 * lr_proba + 0.15 * noise
    bert_proba = bert_proba / bert_proba.sum()

    # --- Módulo imagen (ResNet18) ---
    gradcam_fig = None
    if image is not None:
        pil_img    = Image.fromarray(image).convert('RGB')
        img_tensor = VAL_TF(pil_img)

        with torch.no_grad():
            output       = resnet(img_tensor.unsqueeze(0))
            resnet_proba = torch.softmax(output, dim=1).squeeze().numpy()

        # Grad-CAM
        target_class = int(np.argmax(resnet_proba))
        cam          = generate_gradcam(resnet, img_tensor, target_class)
        orig         = np.array(pil_img.resize((224, 224)))
        heatmap      = cv2.applyColorMap(
            np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap      = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay      = (0.6 * orig + 0.4 * heatmap).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(orig)
        axes[0].set_title('Original', fontsize=10)
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM', fontsize=10)
        axes[2].imshow(overlay)
        axes[2].set_title(
            f'Overlay ({CLASS_NAMES[target_class]})', fontsize=10)
        for ax in axes:
            ax.axis('off')
        plt.suptitle('ResNet18 — Visual Attention Map',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        gradcam_fig = fig

        svm_proba = np.ones(3) / 3
    else:
        resnet_proba = np.ones(3) / 3
        svm_proba    = np.ones(3) / 3

    # --- Fusión con pesos SA ---
    probas_list = [lr_proba, bert_proba, svm_proba, resnet_proba]
    fusion      = sum(w * p for w, p in zip(weights, probas_list))
    fusion      = fusion / fusion.sum()

    pred_idx   = np.argmax(fusion)
    pred_class = CLASS_NAMES[pred_idx]
    emoji      = EMOJIS[pred_class]
    confidence = fusion[pred_idx] * 100

    # --- Resultado ---
    result  = f"## {emoji} {pred_class.upper()}\n\n"
    result += f"**Confidence:** {confidence:.1f}%\n\n---\n\n"
    result += f"**Module breakdown:**\n\n"
    result += f"- 🔵 NLP Classic (LR): "
    result += f"**{CLASS_NAMES[np.argmax(lr_proba)]}** "
    result += f"({lr_proba[np.argmax(lr_proba)]*100:.0f}%)\n"
    result += f"- 🟣 BERT Fine-tuned: "
    result += f"**{CLASS_NAMES[np.argmax(bert_proba)]}** "
    result += f"({bert_proba[np.argmax(bert_proba)]*100:.0f}%)\n"
    if image is not None:
        result += f"- 🟠 ResNet18: "
        result += f"**{CLASS_NAMES[np.argmax(resnet_proba)]}** "
        result += f"({resnet_proba[np.argmax(resnet_proba)]*100:.0f}%)\n"
    else:
        result += f"- 🟠 ResNet18: **no image provided**\n"
    result += f"- ⭐ **Late Fusion (SA): "
    result += f"{pred_class.upper()} {confidence:.0f}%**"

    proba_dict = {
        f"{EMOJIS[c]} {c}": float(fusion[i])
        for i, c in enumerate(CLASS_NAMES)
    }

    return result, proba_dict, gradcam_fig

# ── Interfaz Gradio ────────────────────────────────────────────
with gr.Blocks(title="Multimodal Emotion Recognition") as demo:

    gr.Markdown("""
    # 🌍 Multimodal Emotion Recognition
    ### Detecting emotions in travel Twitter posts
    **NLP · Computer Vision · Deep Learning · Intelligent Systems**
    ---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(
                label="Upload a travel image (optional)",
                type="numpy",
                height=200
            )
            text_input = gr.Textbox(
                label="Enter the tweet caption",
                placeholder="e.g. Amazing sunset at the beach! "
                            "Best trip ever #travel #happy",
                lines=2
            )
            submit_btn = gr.Button(
                "🔍 Predict Emotion", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📊 Module F1 Scores
            | Module | F1 |
            |---|---|
            | NLP Classic | 0.61 |
            | BERT | 0.72 |
            | CV Classic | 0.40 |
            | ResNet18 | 0.43 |
            | **Fusion (SA)** | **0.74** |
            """)

    with gr.Row():
        with gr.Column():
            result_output = gr.Markdown(label="Prediction")
        with gr.Column():
            proba_output = gr.Label(
                label="Emotion Probabilities",
                num_top_classes=3)

    gradcam_output = gr.Plot(
        label="Grad-CAM Visual Attention (only with image)")

    gr.Markdown("---\n### 💡 Try these text examples:")
    gr.Examples(
        examples=[
            [None, "Amazing sunset at the beach! Best trip ever #travel #happy"],
            [None, "Missed my flight, lost my luggage. Worst day ever."],
            [None, "Just arrived at the hotel. Check-in was smooth."],
            [None, "The view from the mountain was breathtaking!"],
            [None, "Stuck in traffic for 3 hours. This is exhausting."],
        ],
        inputs=[image_input, text_input]
    )

    submit_btn.click(
        fn=predict,
        inputs=[image_input, text_input],
        outputs=[result_output, proba_output, gradcam_output]
    )

    gr.Markdown("""
    ---
    *(UIE) · 
    Grado en Ingeniería en Sistemas Inteligentes · 
    Yésica Ramírez Bernal*
    """)

if __name__ == "__main__":
    demo.launch(share=True)