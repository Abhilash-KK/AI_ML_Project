"""
Customer Feedback Sentiment Analysis - Model Training Pipeline
Trains a Logistic Regression classifier with TF-IDF vectorization on
the Amazon Fine Food Reviews dataset.

Compatible with Windows, Linux, and MacOS environments.
Includes automated fallback for data loading if Windows Smart App Control
blocks pandas compiled extensions.
"""

import os
import re
import json
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. SETUP PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Reviews.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. DATA CLEANING UTILITY
# --------------------------------------------------
def clean_text(text: str) -> str:
    """Preprocess customer review text for TF-IDF feature extraction."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove special characters and digits, keep alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_sentiment(score: float) -> str:
    """Convert 1-5 star ratings to 3 sentiment categories."""
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"


# --------------------------------------------------
# 3. LOAD DATASET (With Windows App Control Resilience)
# --------------------------------------------------
print("\n" + "=" * 60)
print("CUSTOMER FEEDBACK SENTIMENT ANALYSIS - TRAINING PIPELINE")
print("=" * 60)
print(f"Dataset path: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH}'.\n"
        "Please download 'Reviews.csv' from Kaggle Amazon Fine Food Reviews:\n"
        "https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews\n"
        "and place it inside the 'dataset/' folder."
    )

texts = []
scores = []

# Try pandas first; if Windows Smart App Control blocks pandas C-extensions,
# fallback to pyarrow or python csv seamlessly.
use_pandas = False
try:
    import pandas as pd
    use_pandas = True
    print("Using Pandas for dataset loading...")
except Exception as e:
    print(f"[Notice] Pandas load skipped ({e.__class__.__name__}). Using high-performance PyArrow/CSV loader.")

if use_pandas:
    df = pd.read_csv(DATA_PATH, usecols=["Score", "Text"])
    df.dropna(subset=["Score", "Text"], inplace=True)
    df["Text"] = df["Text"].astype(str).str.strip()
    df = df[df["Text"] != ""]
    df.drop_duplicates(subset=["Text"], inplace=True)
    df["Sentiment"] = df["Score"].apply(map_sentiment)
    
    # Balanced sampling: up to 20,000 per class
    SAMPLE_SIZE = 20000
    df = (
        df.groupby("Sentiment", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), SAMPLE_SIZE), random_state=42))
        .reset_index(drop=True)
    )
    raw_texts = df["Text"].tolist()
    labels = df["Sentiment"].tolist()
else:
    # High-speed PyArrow loader with full deduplication & balanced sampling
    try:
        import pyarrow.csv as pv
        print("Reading Reviews.csv with PyArrow...")
        table = pv.read_csv(
            DATA_PATH,
            read_options=pv.ReadOptions(use_threads=True),
            convert_options=pv.ConvertOptions(include_columns=["Score", "Text"])
        )
        raw_scores = table["Score"].to_pylist()
        raw_text_list = table["Text"].to_pylist()
    except Exception:
        import csv
        print("Reading Reviews.csv with Python standard csv module...")
        raw_scores = []
        raw_text_list = []
        with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s = float(row.get("Score", 0))
                    t = row.get("Text", "")
                    raw_scores.append(s)
                    raw_text_list.append(t)
                except Exception:
                    continue

    seen = set()
    by_sentiment = {"Negative": [], "Neutral": [], "Positive": []}
    for s, t in zip(raw_scores, raw_text_list):
        if s is None or t is None:
            continue
        t_clean = str(t).strip()
        if not t_clean or t_clean in seen:
            continue
        seen.add(t_clean)
        sent = map_sentiment(float(s))
        by_sentiment[sent].append(t_clean)

    SAMPLE_SIZE = 20000
    random.seed(42)
    raw_texts = []
    labels = []
    for sentiment, items in by_sentiment.items():
        k = min(len(items), SAMPLE_SIZE)
        sampled = random.sample(items, k) if len(items) >= k else items
        print(f"Sampled {len(sampled)} reviews for class '{sentiment}' (available: {len(items)})")
        for item in sampled:
            raw_texts.append(item)
            labels.append(sentiment)

print(f"\nTotal sampled reviews: {len(raw_texts)}")
label_counts = {lbl: labels.count(lbl) for lbl in set(labels)}
print(f"Sentiment distribution: {label_counts}")

# --------------------------------------------------
# 4. TEXT PREPROCESSING
# --------------------------------------------------
print("\nCleaning text data (lowercase, removing HTML, URLs, special characters)...")
cleaned_texts = [clean_text(t) for t in raw_texts]

# Filter out any reviews that become empty after cleaning
valid_data = [(txt, lbl) for txt, lbl in zip(cleaned_texts, labels) if txt]
X = [d[0] for d in valid_data]
y = [d[1] for d in valid_data]

print(f"Valid cleaned samples after preprocessing: {len(X)}")

# --------------------------------------------------
# 5. STRATIFIED TRAIN / TEST SPLIT
# --------------------------------------------------
print("\nSplitting dataset into 80% training and 20% testing (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Testing set:  {len(X_test)} samples")

# --------------------------------------------------
# 6. TF-IDF FEATURE EXTRACTION
# --------------------------------------------------
print("\nExtracting features using TF-IDF Vectorizer...")
print("Parameters: max_features=15000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True")

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Fit strictly on training set to prevent data leakage
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF training matrix shape: {X_train_tfidf.shape}")
print(f"TF-IDF testing matrix shape:  {X_test_tfidf.shape}")

# --------------------------------------------------
# 7. MODEL TRAINING: LOGISTIC REGRESSION
# --------------------------------------------------
print("\nTraining Logistic Regression model...")
print("Parameters: max_iter=1000, class_weight='balanced', random_state=42")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)
print("Model training successfully completed!")

# --------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------
print("\n" + "=" * 60)
print("EVALUATION ON UNSEEN TEST SET")
print("=" * 60)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
class_labels = ["Negative", "Neutral", "Positive"]

report_text = classification_report(y_test, y_pred, labels=class_labels, zero_division=0)
report_dict = classification_report(y_test, y_pred, labels=class_labels, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=class_labels)

print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(report_text)
print("Confusion Matrix:")
print(cm)

# --------------------------------------------------
# 9. SAVE EVALUATION ARTIFACTS
# --------------------------------------------------
# Save Confusion Matrix figure
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap="Blues", values_format="d", ax=ax)
plt.title("Customer Feedback Sentiment - Confusion Matrix")
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"\n[Saved] Confusion matrix figure -> {cm_path}")

# Save JSON evaluation metrics
eval_payload = {
    "model_name": "Logistic Regression with TF-IDF",
    "dataset": "Amazon Fine Food Reviews",
    "total_samples": len(X),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "accuracy": round(float(accuracy), 4),
    "precision_macro": round(float(report_dict["macro avg"]["precision"]), 4),
    "recall_macro": round(float(report_dict["macro avg"]["recall"]), 4),
    "f1_score_macro": round(float(report_dict["macro avg"]["f1-score"]), 4),
    "precision_weighted": round(float(report_dict["weighted avg"]["precision"]), 4),
    "recall_weighted": round(float(report_dict["weighted avg"]["recall"]), 4),
    "f1_score_weighted": round(float(report_dict["weighted avg"]["f1-score"]), 4),
    "class_labels": class_labels,
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "class_distribution": {lbl: y.count(lbl) for lbl in class_labels}
}

json_path = os.path.join(MODEL_DIR, "evaluation_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(eval_payload, f, indent=4)
print(f"[Saved] Evaluation metrics JSON -> {json_path}")

# Save trained model and vectorizer
model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
print(f"[Saved] Sentiment model -> {model_path}")
print(f"[Saved] TF-IDF vectorizer -> {vectorizer_path}")

# --------------------------------------------------
# 10. VERIFY WITH SAMPLE PREDICTIONS
# --------------------------------------------------
print("\n" + "=" * 60)
print("VERIFICATION ON SAMPLE REVIEWS")
print("=" * 60)

test_samples = [
    "The product quality is excellent. I really love it.",
    "The product is terrible. The quality is very poor.",
    "The product is okay. It works as expected."
]

for sample in test_samples:
    cl = clean_text(sample)
    feat = vectorizer.transform([cl])
    pred = model.predict(feat)[0]
    probs = model.predict_proba(feat)[0]
    idx = list(model.classes_).index(pred)
    conf = probs[idx] * 100
    print(f"\nReview:    \"{sample}\"")
    print(f"Cleaned:   \"{cl}\"")
    print(f"Predicted: {pred} (Confidence: {conf:.2f}%)")

print("\n" + "=" * 60)
print("MODEL TRAINING AND ARTIFACT EXPORT COMPLETE!")
print("=" * 60)
