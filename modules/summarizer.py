"""
Customer Feedback Sentiment Analysis - Review Batch Summarizer Module
Handles CSV parsing, text column auto-detection, batch ML analysis,
and CSV export for downstream reporting.
"""

import io
import csv
from typing import List, Dict, Any, Tuple, Optional
from .prediction import predict_batch, load_model_artifacts


def parse_reviews_csv(file_or_path) -> List[str]:
    """
    Parse uploaded CSV or file path to extract customer reviews.
    Auto-detects common review column headers.

    Raises:
        ValueError: If file is empty, has invalid format, or lacks text columns.
    """
    lines = []
    if isinstance(file_or_path, str):
        try:
            with open(file_or_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Could not open CSV file: {str(e)}")
    elif hasattr(file_or_path, "read"):
        raw_bytes = file_or_path.read()
        if isinstance(raw_bytes, bytes):
            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = raw_bytes.decode("latin-1", errors="replace")
        else:
            content = str(raw_bytes)
        # Reset file pointer if needed
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
    else:
        raise ValueError("Unsupported input type for CSV parsing.")

    content = content.strip()
    if not content:
        raise ValueError("The uploaded CSV file is completely empty.")

    # Read using python csv
    reader = csv.reader(io.StringIO(content))
    try:
        rows = list(reader)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV format: {str(e)}")

    if not rows or len(rows) < 2:
        raise ValueError("CSV file must contain a header row and at least one data row.")

    headers = [h.strip() for h in rows[0]]
    if not headers or all(not h for h in headers):
        raise ValueError("CSV header row is invalid or empty.")

    # Look for candidate review text column names
    target_names = [
        "text", "review", "reviews", "review_text", "reviewtext",
        "comments", "comment", "feedback", "body", "summary",
        "description", "customer_review", "content"
    ]
    target_idx = None
    for idx, header in enumerate(headers):
        if header.lower() in target_names:
            target_idx = idx
            break

    # If no standard header name matches, inspect columns for text length
    if target_idx is None:
        best_col = -1
        max_avg_len = 0
        for col_i in range(len(headers)):
            col_samples = [
                r[col_i].strip() for r in rows[1:min(len(rows), 25)]
                if len(r) > col_i and isinstance(r[col_i], str)
            ]
            if col_samples:
                avg_len = sum(len(s) for s in col_samples) / len(col_samples)
                if avg_len > max_avg_len:
                    max_avg_len = avg_len
                    best_col = col_i
        if best_col != -1 and max_avg_len >= 8:
            target_idx = best_col

    if target_idx is None:
        raise ValueError(
            f"Could not identify a review text column among headers: {headers}. "
            "Please ensure your CSV has a column named 'Text', 'Review', 'Comments', or 'Feedback'."
        )

    reviews = []
    for row in rows[1:]:
        if len(row) > target_idx:
            val = row[target_idx].strip()
            if val:
                reviews.append(val)

    if not reviews:
        raise ValueError("No valid customer review text entries found in the selected CSV column.")

    return reviews


def analyze_uploaded_reviews(
    file_or_path,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Parse CSV, run batch ML sentiment prediction, and aggregate distribution metrics.
    """
    reviews = parse_reviews_csv(file_or_path)

    if model is None or vectorizer is None:
        model, vectorizer = load_model_artifacts()

    # Predict batch
    analyzed_records = predict_batch(reviews, model=model, vectorizer=vectorizer)

    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for rec in analyzed_records:
        sentiment = rec.get("sentiment", "Neutral")
        if sentiment in counts:
            counts[sentiment] += 1

    total = len(analyzed_records)
    percentages = {
        k: round((v / max(total, 1)) * 100.0, 2)
        for k, v in counts.items()
    }

    # Calculate average confidence per sentiment
    avg_conf = {}
    for s_label in counts.keys():
        confs = [r["confidence"] for r in analyzed_records if r.get("sentiment") == s_label]
        avg_conf[s_label] = round(sum(confs) / max(len(confs), 1), 2) if confs else 0.0

    return {
        "total_reviews": total,
        "sentiment_counts": counts,
        "sentiment_percentages": percentages,
        "average_confidence": avg_conf,
        "records": analyzed_records
    }


def export_analysis_csv(records: List[Dict[str, Any]]) -> str:
    """Generate CSV string containing all analyzed reviews, sentiments, and probability scores."""
    output = io.StringIO()
    fieldnames = [
        "Review_Index",
        "Original_Review",
        "Cleaned_Review",
        "Predicted_Sentiment",
        "Confidence_Percent",
        "Positive_Probability_Percent",
        "Negative_Probability_Percent",
        "Neutral_Probability_Percent"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for idx, r in enumerate(records, start=1):
        probs = r.get("probabilities", {})
        writer.writerow({
            "Review_Index": idx,
            "Original_Review": r.get("original_text", ""),
            "Cleaned_Review": r.get("cleaned_text", ""),
            "Predicted_Sentiment": r.get("sentiment", ""),
            "Confidence_Percent": r.get("confidence", 0.0),
            "Positive_Probability_Percent": probs.get("Positive", 0.0),
            "Negative_Probability_Percent": probs.get("Negative", 0.0),
            "Neutral_Probability_Percent": probs.get("Neutral", 0.0)
        })

    return output.getvalue()
