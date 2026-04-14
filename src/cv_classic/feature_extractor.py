# src/cv_classic/feature_extractor.py
# Classical Computer Vision module.
# Extracts three types of visual features per image:
#   1. HSV colour histogram  → captures emotional colour information
#   2. Canny edge density    → captures visual complexity
#   3. K-means colour palette → captures dominant colours and their proportions
#
# These features are concatenated into a single vector
# and fed into an RBF-SVM classifier.

import cv2
import numpy as np
import json
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def extract_hsv_histogram(image_path: str, bins: int = 32) -> np.ndarray:
    """Compute a normalised HSV colour histogram for one image.

    HSV is preferred over RGB because it separates colour (hue)
    from brightness, making it more robust to lighting changes.

    Args:
        image_path: Path to the image file.
        bins:       Number of bins per channel (default 32).

    Returns:
        A vector of shape (bins * 3,) = (96,) with normalised frequencies.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros(bins * 3)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats   = []

    # H channel range: 0–180, S and V channels: 0–256
    ranges = [(0, 180), (0, 256), (0, 256)]
    for ch, (lo, hi) in enumerate(ranges):
        hist = cv2.calcHist([img_hsv], [ch], None, [bins], [lo, hi])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)   # normalise to sum = 1
        feats.append(hist)

    return np.concatenate(feats)


def extract_edge_density(image_path: str) -> np.ndarray:
    """Compute edge density using the Canny detector.

    High edge density → visually complex or chaotic image (often negative).
    Low edge density  → calm, simple image (often positive or neutral).

    Steps:
        1. Convert to grayscale.
        2. Apply Gaussian blur (5x5) to reduce noise.
        3. Run Canny edge detection with thresholds 50 and 150.
        4. Compute the fraction of edge pixels over total pixels.

    Returns:
        A vector of shape (2,): [edge_density, pixel_std / 255].
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.array([0.0, 0.0])

    img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    edges    = cv2.Canny(img_blur, 50, 150)
    density  = np.sum(edges > 0) / edges.size   # fraction of edge pixels

    return np.array([density, img.std() / 255.0])


def extract_kmeans_colors(image_path: str, k: int = 4) -> np.ndarray:
    """Extract the dominant colours of an image using k-means clustering.

    Pixels are grouped into k clusters. Each cluster centre represents
    a dominant colour. The proportion of pixels in each cluster
    captures how much of the image is covered by that colour.

    Args:
        image_path: Path to the image file.
        k:          Number of colour clusters (default 4).

    Returns:
        A vector of shape (k*3 + k,) = (16,):
        [centre_RGB_normalised (12), cluster_proportions (4)].
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros(k * 4)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels  = img_rgb.reshape(-1, 3).astype(np.float32)

    # k-means stopping criteria: max 20 iterations or epsilon = 0.5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    # Compute proportion of pixels belonging to each cluster
    counts      = np.bincount(labels.flatten(), minlength=k)
    proportions = counts / counts.sum()

    centers_norm = centers.flatten() / 255.0   # normalise RGB to [0, 1]
    return np.concatenate([centers_norm, proportions])


def extract_features(image_path: str) -> np.ndarray:
    """Combine all three feature sets into a single feature vector.

    Final vector size:
        HSV histogram : 96 dimensions
        Edge density  :  2 dimensions
        K-means       : 16 dimensions
        Total         : 114 dimensions
    """
    hsv    = extract_hsv_histogram(image_path)
    edge   = extract_edge_density(image_path)
    kmeans = extract_kmeans_colors(image_path)
    return np.concatenate([hsv, edge, kmeans])


def run_cv_classic(df,
                   img_col:   str = 'image_path',
                   label_col: str = 'label',
                   save_path: str = 'results/metrics_cv_classic.json'):
    """Extract visual features and train an RBF-SVM classifier.

    Pipeline:
        1. Extract HSV + edge + k-means features for every image.
        2. Split into train / test (80/20, stratified).
        3. Normalise features with StandardScaler (zero mean, unit variance).
        4. Train SVM with RBF kernel (C=10, gamma='scale').
        5. Evaluate and save results to JSON.

    Args:
        df:        DataFrame with image_path and label columns.
        img_col:   Column name for image paths.
        label_col: Column name for emotion labels.
        save_path: Path to save the JSON results file.

    Returns:
        svm, scaler, le, y_test, probas
    """
    os.makedirs('results', exist_ok=True)
    print("Extracting image features...")

    features, labels = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        feat = extract_features(row[img_col])
        features.append(feat)
        labels.append(row[label_col])
        if i % 500 == 0:
            print(f"  {i}/{len(df)} images processed")

    X  = np.array(features)
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # 80/20 stratified split — same random state as all other modules
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalise: SVM is sensitive to feature scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # RBF-SVM: C=10 (strong margin penalty), gamma='scale' (auto-tuned)
    print("Training SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    probas = svm.predict_proba(X_test)

    print("\n=== Classical CV (SVM) ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save results — probas are needed for late fusion
    results = {
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'probas':   probas.tolist()
    }
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")

    return svm, scaler, le, y_test, probas
