# Customer Feedback Sentiment Analysis with AI Review Summarizer

An enterprise-grade, full-stack Machine Learning and Generative AI application designed for automated customer feedback intelligence, sentiment classification, and product enhancement synthesis.

Developed as an **MCA Practical Project** adhering to the MCA Project Guidelines and Rubrics.

---

## 📌 Executive Summary & Problem Statement

Modern businesses receive tens of thousands of customer product reviews across e-commerce platforms. Manually analyzing these reviews is prohibitively expensive, slow, and prone to human cognitive fatigue. Companies struggle to swiftly isolate product defects, discern recurring customer friction, or capture actionable recommendations.

This application solves this dilemma through a **Hybrid ML + GenAI Architecture**:
1. **Deterministic Machine Learning Classifier**: A Logistic Regression model trained on TF-IDF n-gram feature vectors independently classifies customer sentiment into **Positive**, **Negative**, or **Neutral** with high certainty and zero hallucination.
2. **Generative AI Cognitive Layer**: Grounded by the ML prediction and review text, Google Gemini (`gemini-2.5-flash` / `gemini-2.0-flash`) synthesizes semantic rationale, extracts influential keywords, aggregates batch complaints, and provides concrete product engineering recommendations.

---

## 🏗️ System Architecture

```
                  ┌────────────────────────────────────────┐
                  │       Customer Review Input            │
                  │   (Single Text or Batch CSV Upload)    │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │      Text Preprocessing Pipeline       │
                  │ (HTML strip, URL removal, clean, lower)│
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │       TF-IDF Feature Extractor         │
                  │   (15,000 Unigrams/Bigrams, Sublinear) │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    Logistic Regression ML Model        │
                  │    (Trained on Amazon Food Reviews)    │
                  └──────────────────┬─────────────────────┘
                                     │
                      [Sentiment Class & Confidence %]
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │       Generative AI Engine             │
                  │   (Official Google Gemini GenAI SDK)   │
                  └──────────────────┬─────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│  AI Sentiment    │       │  Batch Review    │       │ Product Improvement  │
│  Explanation     │       │  Summarizer      │       │ Recommendations      │
└──────────────────┘       └──────────────────┘       └──────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.10+ (Tested on 3.14) | Core programming runtime |
| **Frontend UI** | Streamlit | SaaS-style dashboard & interactive controls |
| **Machine Learning** | Scikit-learn | Logistic Regression, TF-IDF Vectorizer |
| **Generative AI** | Google Gemini (`google-genai` SDK) | Grounded explanation & executive summarization |
| **Data Processing** | NumPy, PyArrow, CSV | High-throughput data ingestion & tokenization |
| **Visualizations** | Plotly & Matplotlib | Interactive donut charts, heatmaps, box plots |
| **Model Storage** | Joblib | Binary serialization of trained ML artifacts |
| **Configuration** | Python-dotenv | Secure `.env` API credential management |

---

## 📁 Project Directory Structure

```
Customer_Feedback_AI/
├── app.py                         # Streamlit multi-page web application
├── train_model.py                 # Reproducible ML training pipeline
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Complete project documentation
├── PROJECT_REPORT.md              # Technical project report (MCA academic submission)
├── colab_training_workflow.ipynb  # Alternative Google Colab cloud training notebook
├── sample_reviews.csv             # 15 realistic sample reviews for testing
├── .env.example                   # Gemini API key template
├── .gitignore                     # Git tracking exclusions
├── dataset/
│   ├── .gitkeep                   # Directory placeholder
│   └── Reviews.csv                # Amazon Fine Food Reviews (Kaggle dataset)
├── models/
│   ├── sentiment_model.pkl        # Serialized Logistic Regression classifier
│   ├── tfidf_vectorizer.pkl       # Serialized TF-IDF feature extractor
│   ├── evaluation_results.json    # Genuine unseen test set metrics
│   └── confusion_matrix.png       # High-resolution confusion matrix plot
├── modules/
│   ├── __init__.py                # Package initializer
│   ├── prediction.py              # Preprocessing & ML inference engine
│   ├── ai_explainer.py            # Google Gemini integration & local fallback
│   ├── summarizer.py              # CSV parsing, batch analysis & export
│   └── visualizations.py          # Plotly interactive charts & styling
├── screenshots/
│   └── confusion_matrix.png       # Test evaluation plot screenshot
└── tests/
    └── test_prediction.py         # Automated unittest test suite (Test Cases 1-7)
```

---

## 📊 Dataset & Model Performance

### Dataset Information
- **Dataset**: Amazon Fine Food Reviews (Kaggle)
- **Total Records in Dataset**: 568,454 reviews
- **Sampling Strategy**: Balanced random sampling across classes (20,000 per class, 60,000 total)
- **Train/Test Split**: 80% Training (48,000 reviews), 20% Testing (12,000 reviews, Stratified)
- **Target Classes**:
  - `Positive`: Original ratings of 4 and 5 stars
  - `Neutral`: Original rating of 3 stars
  - `Negative`: Original ratings of 1 and 2 stars

### Genuine Unseen Test Set Metrics
*(Loaded dynamically from `models/evaluation_results.json`)*

- **Overall Accuracy**: **68.77%** (on a challenging 3-class balanced dataset)
- **Macro Precision**: **68.59%**
- **Macro Recall**: **68.77%**
- **Macro F1-Score**: **68.67%**

#### Detailed Classification Report:
| Class | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| **Negative** | 69.37% | 70.60% | 69.98% | 4,000 |
| **Neutral** | 60.59% | 58.08% | 59.31% | 4,000 |
| **Positive** | 75.82% | 77.62% | 76.71% | 4,000 |

#### Confusion Matrix:
```
                Predicted Negative   Predicted Neutral   Predicted Positive
Actual Negative        2,824                894                 282
Actual Neutral          969                2,323                708
Actual Positive         278                 617                3,105
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
- Windows 10/11, macOS, or Linux.
- Python 3.10 or higher.
- Git.

### 2. Clone / Open the Project
Open PowerShell and navigate to the project directory:
```powershell
cd C:\MLProjects\Customer_Feedback_AI
```

### 3. Setup Virtual Environment
Create and activate an isolated Python environment:
```powershell
# Create venv
python -m venv venv

# Activate venv on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure Google Gemini API Key
Create your personal `.env` file from the provided template:
```powershell
Copy-Item .env.example .env
```
Open `.env` and insert your Gemini API Key obtained from [Google AI Studio](https://aistudio.google.com/):
```env
GEMINI_API_KEY=AIzaSyYourActualGeminiApiKeyHere
```
*(Note: If you do not have an API key right away, the application includes an intelligent local fallback explainer so all ML features and summaries remain fully functional).*

---

## 🏃 Execution Commands

### Step 1: Train the Machine Learning Model
*(Note: Model artifacts are already pre-trained and saved in `models/`. To retrain from scratch:)*
```powershell
python train_model.py
```
This script cleans the dataset, trains the TF-IDF vectorizer and Logistic Regression model, calculates test metrics, and exports all 4 artifacts into `models/`.

### Step 2: Run Automated Unit Tests
Verify all 7 mandatory test cases (positive, negative, neutral, empty input, batch processing, missing API key fallback, and invalid CSV handling):
```powershell
python tests\test_prediction.py
```

### Step 3: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Streamlit will automatically launch in your default web browser at `http://localhost:8501`.

---

## 💻 Windows Environment & Smart App Control Diagnosis

### The Issue Diagnosed:
On Windows 11 systems with **Smart App Control (SAC)** enabled in Enforcement mode, unsigned binary extensions (`.pyd`) from newly released PyPI wheels (such as Pandas 3.1.1 on Python 3.14) encounter:
`ImportError: DLL load failed while importing ccalendar: An Application Control policy has blocked this file.`

### The Solution Implemented:
1. **Built-in Resilience**: `train_model.py`, `prediction.py`, and `summarizer.py` feature automated fallback to **PyArrow** and Python's standard `csv` library. PyArrow loads the 568k row dataset in under 5 seconds without triggering Application Control policies.
2. **Zero Security Compromise**: Windows Defender and Smart App Control policies remain fully active and intact.
3. **Cloud Training Alternative**: A Google Colab notebook (`colab_training_workflow.ipynb`) is included in the project for students who wish to train on Google Cloud GPU/TPU and download the artifacts.

---

## 🧪 Test Cases & Verification Summary

| Test Case | Scenario / Input | Expected Result | Verified Status |
|---|---|---|---|
| **Test Case 1** | "The product quality is excellent. I really love it." | Predicted Sentiment: **Positive** (Confidence > 90%) | ✅ **PASS** (95.06%) |
| **Test Case 2** | "The product is terrible. The quality is very poor." | Predicted Sentiment: **Negative** (Confidence > 90%) | ✅ **PASS** (99.72%) |
| **Test Case 3** | "The product is okay. It works as expected." | Predicted Sentiment: **Neutral** | ✅ **PASS** (96.94%) |
| **Test Case 4** | Empty or whitespace review string | Graceful error message / ValueError raised | ✅ **PASS** |
| **Test Case 5** | Multiple batch reviews via list or CSV | Aggregated sentiment stats & batch inferences | ✅ **PASS** |
| **Test Case 6** | Missing or invalid Gemini API key | ML prediction remains functional; local explainer activates | ✅ **PASS** |
| **Test Case 7** | Corrupted or headerless CSV file | Informative error displayed in UI without crash | ✅ **PASS** |

---

## ⚖️ Limitations & Responsible AI Use

1. **Neutral Review Subtlety**: Neutral customer feedback often contains mixed polarity (e.g., "Great taste but packaging was crushed"), making 3-class classification inherently more challenging than binary classification.
2. **Context Window & API Quotas**: Batch review summarization uses smart sampling (up to 25 representative reviews per sentiment) to ensure API cost efficiency and adherence to free-tier rate limits.
3. **Model Grounding**: The LLM is strictly constrained via system prompt instructions to interpret the deterministic ML model's prediction rather than hallucinating its own conflicting labels.

---

## 📜 Academic Declaration

This project was developed for the **Master of Computer Applications (MCA)** Combined Practical Examination. All code, model training, evaluation results, and Generative AI prompt architectures have been authored, verified, and documented.
