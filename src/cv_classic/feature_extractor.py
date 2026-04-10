# src/cv_classic/feature_extractor.py
import cv2
import numpy as np
import json
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def extract_hsv_histogram(image_path, bins=32):
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros(bins * 3)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats = []
    ranges = [(0, 180), (0, 256), (0, 256)]
    for ch, (lo, hi) in enumerate(ranges):
        hist = cv2.calcHist([img_hsv], [ch], None, [bins], [lo, hi])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        feats.append(hist)
    return np.concatenate(feats)

def extract_edge_density(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.array([0.0, 0.0])
    img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    edges    = cv2.Canny(img_blur, 50, 150)
    density  = np.sum(edges > 0) / edges.size
    return np.array([density, img.std() / 255.0])

def extract_kmeans_colors(image_path, k=4):
    """Segmentación k-means para extraer colores dominantes."""
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros(k * 3)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels  = img_rgb.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    
    # Proporción de cada cluster como feature
    counts = np.bincount(labels.flatten(), minlength=k)
    proportions = counts / counts.sum()
    
    # Combinar color dominante + proporción
    centers_norm = centers.flatten() / 255.0
    return np.concatenate([centers_norm, proportions])

def extract_features(image_path):
    hsv    = extract_hsv_histogram(image_path)
    edge   = extract_edge_density(image_path)
    kmeans = extract_kmeans_colors(image_path)
    return np.concatenate([hsv, edge, kmeans])

def run_cv_classic(df, img_col='image_path', label_col='label',
                   save_path='results/metrics_cv_classic.json'):
    os.makedirs('results', exist_ok=True)
    print("Extrayendo features de imagen...")

    features, labels = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        feat = extract_features(row[img_col])
        features.append(feat)
        labels.append(row[label_col])
        if i % 500 == 0:
            print(f"  {i}/{len(df)} imágenes procesadas")

    X  = np.array(features)
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Entrenando SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    probas = svm.predict_proba(X_test)

    print("\n=== CV Clásico (SVM) ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results = {
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'probas':   probas.tolist()
    }
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {save_path}")

    return svm, scaler, le, y_test, probas