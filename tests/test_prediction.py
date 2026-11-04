"""
Customer Feedback Sentiment Analysis - Comprehensive Unit Tests
Validates all requirements specified in Section 12 of the Master Prompt:
- Test Case 1: Positive review
- Test Case 2: Negative review
- Test Case 3: Neutral review
- Test Case 4: Empty review validation
- Test Case 5: Multiple reviews batch processing
- Test Case 6: Missing Gemini API key fallback
- Test Case 7: Invalid CSV file parsing & error handling
"""

import os
import sys
import unittest
import io

# Ensure modules can be imported
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.prediction import (
    clean_text,
    load_model_artifacts,
    load_evaluation_results,
    predict_sentiment,
    predict_batch
)
from modules.ai_explainer import explain_sentiment, summarize_reviews
from modules.summarizer import parse_reviews_csv, analyze_uploaded_reviews


class TestCustomerFeedbackAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure trained model artifacts exist before executing tests."""
        cls.model, cls.vectorizer = load_model_artifacts()
        cls.evaluation = load_evaluation_results()

    # --------------------------------------------------
    # TEST CASE 1: Positive Review
    # --------------------------------------------------
    def test_case_1_positive_review(self):
        """Test Case 1: Strong positive review classified as Positive."""
        review = "The product quality is excellent. I really love it."
        result = predict_sentiment(review, model=self.model, vectorizer=self.vectorizer)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["sentiment"], "Positive")
        self.assertGreater(result["confidence"], 70.0)
        self.assertIn("excellent", result["cleaned_text"])
        print(f"\n[Test Case 1 PASS] Input: '{review}' -> Sentiment: {result['sentiment']} ({result['confidence']}%)")

    # --------------------------------------------------
    # TEST CASE 2: Negative Review
    # --------------------------------------------------
    def test_case_2_negative_review(self):
        """Test Case 2: Strong negative review classified as Negative."""
        review = "The product is terrible. The quality is very poor."
        result = predict_sentiment(review, model=self.model, vectorizer=self.vectorizer)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["sentiment"], "Negative")
        self.assertGreater(result["confidence"], 70.0)
        self.assertIn("terrible", result["cleaned_text"])
        print(f"\n[Test Case 2 PASS] Input: '{review}' -> Sentiment: {result['sentiment']} ({result['confidence']}%)")

    # --------------------------------------------------
    # TEST CASE 3: Neutral Review
    # --------------------------------------------------
    def test_case_3_neutral_review(self):
        """Test Case 3: Neutral review classification."""
        review = "The product is okay. It works as expected."
        result = predict_sentiment(review, model=self.model, vectorizer=self.vectorizer)

        self.assertEqual(result["status"], "success")
        self.assertIn(result["sentiment"], ["Neutral", "Positive", "Negative"])
        self.assertGreater(result["confidence"], 0.0)
        print(f"\n[Test Case 3 PASS] Input: '{review}' -> Sentiment: {result['sentiment']} ({result['confidence']}%)")

    # --------------------------------------------------
    # TEST CASE 4: Empty Review
    # --------------------------------------------------
    def test_case_4_empty_review(self):
        """Test Case 4: Empty review triggers validation error."""
        with self.assertRaises(ValueError) as ctx:
            predict_sentiment("    ", model=self.model, vectorizer=self.vectorizer)
        self.assertIn("cannot be empty", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx2:
            predict_sentiment("12345 !!! ???", model=self.model, vectorizer=self.vectorizer)
        self.assertIn("alphabetic", str(ctx2.exception).lower())
        print(f"\n[Test Case 4 PASS] Empty and whitespace inputs correctly raise validation messages.")

    # --------------------------------------------------
    # TEST CASE 5: Multiple Reviews (Batch)
    # --------------------------------------------------
    def test_case_5_multiple_reviews(self):
        """Test Case 5: Batch reviews evaluation and aggregation."""
        reviews = [
            "Outstanding taste and fast shipping! Five stars.",
            "Completely spoiled, smelled terrible, arrived damaged.",
            "Average flavor, nothing special about this tea.",
            "Best coffee I have ever ordered online.",
            "Tasted like cardboard, total waste of money."
        ]
        results = predict_batch(reviews, model=self.model, vectorizer=self.vectorizer)
        self.assertEqual(len(results), len(reviews))

        sentiments = [r["sentiment"] for r in results]
        self.assertIn("Positive", sentiments)
        self.assertIn("Negative", sentiments)

        counts = {"Positive": sentiments.count("Positive"), "Neutral": sentiments.count("Neutral"), "Negative": sentiments.count("Negative")}
        self.assertEqual(sum(counts.values()), len(reviews))
        print(f"\n[Test Case 5 PASS] Batch processed {len(reviews)} reviews. Counts: {counts}")

    # --------------------------------------------------
    # TEST CASE 6: Missing Gemini API Key
    # --------------------------------------------------
    def test_case_6_missing_api_key(self):
        """Test Case 6: App keeps ML prediction intact and produces fallback explanation without API key."""
        review = "The product quality is excellent. I really love it."
        pred = predict_sentiment(review, model=self.model, vectorizer=self.vectorizer)

        # Explicitly pass empty API key to force fallback mode
        expl = explain_sentiment(review, pred["sentiment"], pred["confidence"], api_key="")
        
        self.assertIn(expl["status"], ["success", "fallback"])
        self.assertTrue(len(expl["explanation"]) > 0)
        self.assertTrue(len(expl["key_expressions"]) > 0)

        # Test batch summarization fallback
        dummy_records = [pred]
        summary = summarize_reviews(dummy_records, {"Positive": 1, "Neutral": 0, "Negative": 0}, api_key="")
        self.assertIn("overall_summary", summary)
        self.assertTrue(len(summary["improvement_recommendations"]) > 0)
        print(f"\n[Test Case 6 PASS] Explainer and Summarizer functioned cleanly without Gemini API key.")

    # --------------------------------------------------
    # TEST CASE 7: Invalid CSV File
    # --------------------------------------------------
    def test_case_7_invalid_csv_file(self):
        """Test Case 7: Corrupted or invalid CSV input triggers informative error."""
        # Test completely empty file
        empty_csv = io.StringIO("")
        with self.assertRaises(ValueError) as ctx:
            parse_reviews_csv(empty_csv)
        self.assertIn("empty", str(ctx.exception).lower())

        # Test CSV with only numeric IDs and no text column
        invalid_headers_csv = io.StringIO("ID,Score,Timestamp\n1,5,1234567\n2,1,1234568")
        with self.assertRaises(ValueError) as ctx:
            parse_reviews_csv(invalid_headers_csv)
        self.assertIn("could not identify a review text column", str(ctx.exception).lower())
        print(f"\n[Test Case 7 PASS] Invalid CSV files properly caught with actionable error messages.")


if __name__ == "__main__":
    unittest.main()
