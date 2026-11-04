"""
Customer Feedback Sentiment Analysis - ML Prediction Module
Handles text preprocessing, model loading, and sentiment inference.
"""

import os
import re
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Base paths relative to this file
MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULES_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
EVALUATION_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")

# Module-level cache for loaded model and vectorizer
_CACHED_MODEL = None
_CACHED_VECTORIZER = None
_CACHED_EVALUATION = None


def clean_text(text: str) -> str:
    """
    Clean and preprocess review text for TF-IDF feature extraction.
    1. Lowercase text
    2. Strip HTML tags
    3. Remove URLs
    4. Remove special characters and digits, keeping alphabetic characters
    5. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)
    # Normalize multiple whitespace to single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model_artifacts(models_dir: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Load the trained Logistic Regression model and TF-IDF vectorizer.
    Uses cached instances if already loaded.
    """
    global _CACHED_MODEL, _CACHED_VECTORIZER

    if _CACHED_MODEL is not None and _CACHED_VECTORIZER is not None:
        return _CACHED_MODEL, _CACHED_VECTORIZER

    m_dir = models_dir or MODELS_DIR
    model_file = os.path.join(m_dir, "sentiment_model.pkl")
    vec_file = os.path.join(m_dir, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_file) or not os.path.exists(vec_file):
        raise FileNotFoundError(
            f"Trained model artifacts not found in '{m_dir}'.\n"
            f"Expected files:\n  - {model_file}\n  - {vec_file}\n"
            "Please run 'python train_model.py' first to train and save the model."
        )

    _CACHED_MODEL = joblib.load(model_file)
    _CACHED_VECTORIZER = joblib.load(vec_file)
    return _CACHED_MODEL, _CACHED_VECTORIZER


def load_evaluation_results(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load the JSON evaluation metrics saved during model training."""
    global _CACHED_EVALUATION
    if _CACHED_EVALUATION is not None:
        return _CACHED_EVALUATION

    m_dir = models_dir or MODELS_DIR
    eval_file = os.path.join(m_dir, "evaluation_results.json")

    if not os.path.exists(eval_file):
        raise FileNotFoundError(
            f"Evaluation metrics file not found at '{eval_file}'.\n"
            "Please execute 'python train_model.py' to generate genuine evaluation results."
        )

    with open(eval_file, "r", encoding="utf-8") as f:
        _CACHED_EVALUATION = json.load(f)

    return _CACHED_EVALUATION


def predict_sentiment(
    text: str,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Predict the sentiment of a single customer review.

    Args:
        text: The customer review string.
        model: Optional pre-loaded Logistic Regression model.
        vectorizer: Optional pre-loaded TF-IDF Vectorizer.

    Returns:
        Dict containing original text, cleaned text, predicted sentiment,
        confidence score, class probabilities, and status.

    Raises:
        ValueError: If input review is empty or contains only whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Customer review cannot be empty. Please provide review text.")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Customer review contains no meaningful alphabetic content after preprocessing.")

    if model is None or vectorizer is None:
        model, vectorizer = load_model_artifacts()

    # Transform text to TF-IDF feature vector
    features = vectorizer.transform([cleaned])

    # Predict class and probabilities
    prediction = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)
    pred_idx = classes.index(prediction)
    confidence = float(probabilities[pred_idx] * 100.0)

    prob_dict = {
        cls_name: round(float(prob * 100.0), 2)
        for cls_name, prob in zip(classes, probabilities)
    }

    return {
        "status": "success",
        "original_text": text,
        "cleaned_text": cleaned,
        "sentiment": prediction,
        "confidence": round(confidence, 2),
        "probabilities": prob_dict
    }


def predict_batch(
    texts: List[str],
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Perform batch inference on a list of customer reviews.
    Optimized for high throughput by vectorizing the whole batch at once.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model_artifacts()

    results = []
    valid_indices = []
    cleaned_batch = []

    for idx, raw_t in enumerate(texts):
        if not isinstance(raw_t, str) or not raw_t.strip():
            results.append({
                "status": "error",
                "original_text": str(raw_t) if raw_t is not None else "",
                "cleaned_text": "",
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
                "error": "Empty or invalid review"
            })
            continue

        c = clean_text(raw_t)
        if not c:
            results.append({
                "status": "error",
                "original_text": raw_t,
                "cleaned_text": "",
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
                "error": "No alphabetic content"
            })
            continue

        valid_indices.append(idx)
        cleaned_batch.append(c)
        results.append(None)  # Placeholder to be filled

    if cleaned_batch:
        features = vectorizer.transform(cleaned_batch)
        preds = model.predict(features)
        probs = model.predict_proba(features)
        classes = list(model.classes_)

        for local_i, global_i in enumerate(valid_indices):
            pred_cls = str(preds[local_i])
            pred_idx = classes.index(pred_cls)
            conf = float(probs[local_i][pred_idx] * 100.0)
            prob_dict = {
                c_name: round(float(p * 100.0), 2)
                for c_name, p in zip(classes, probs[local_i])
            }
            results[global_i] = {
                "status": "success",
                "original_text": texts[global_i],
                "cleaned_text": cleaned_batch[local_i],
                "sentiment": pred_cls,
                "confidence": round(conf, 2),
                "probabilities": prob_dict
            }

    return results
