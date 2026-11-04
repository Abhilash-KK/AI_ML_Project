# MCA PRACTICAL PROJECT TECHNICAL REPORT

---

## PROJECT TITLE:
### Customer Feedback Sentiment Analysis with AI Review Summarizer
**Degree Program:** Master of Computer Applications (MCA)  
**Domain:** Applied Machine Learning & Generative Artificial Intelligence  
**Application Area:** E-Commerce & Product Intelligence  

---

## 1. ABSTRACT

In modern e-commerce and retail ecosystems, customer reviews are a primary indicator of consumer satisfaction, product reliability, and brand perception. However, the exponential volume of user-generated feedback makes manual inspection impossible for product managers and analysts. This project presents a full-stack, enterprise-grade software solution that integrates classical Machine Learning with Generative Artificial Intelligence. 

The system leverages a **Logistic Regression** classifier with **TF-IDF (Term Frequency-Inverse Document Frequency)** n-gram vectorization trained on over 60,000 balanced customer reviews from the Amazon Fine Food Reviews dataset to classify feedback into **Positive**, **Negative**, or **Neutral** sentiments. Rather than replacing deterministic machine learning with an unconstrained Large Language Model (LLM), this project introduces a hybrid architecture where the ML model provides deterministic, verifiable sentiment predictions and confidence probabilities, while **Google Gemini** (`gemini-2.5-flash` via the official `google-genai` SDK) serves as a cognitive synthesis layer. Gemini explains the prediction rationale, extracts key customer expressions, aggregates common product complaints, and generates concrete, actionable product engineering recommendations. Delivered via an interactive **Streamlit** dashboard with **Plotly** visualizations, the application demonstrates a resilient, production-ready implementation compliant with MCA academic standards.

---

## 2. PROBLEM STATEMENT & OBJECTIVES

### 2.1 Problem Statement
E-commerce platforms receive thousands of product reviews weekly. Traditional manual reading:
- Is time-consuming, expensive, and non-scalable.
- Suffers from cognitive fatigue and subjective bias.
- Fails to aggregate granular complaints into high-level business insights.
- Pure LLM solutions, while capable of text generation, are computationally expensive, prone to hallucinations, and lack reproducible probability bounds for large batches.

### 2.2 Objectives
1. **Develop an Accurate ML Classification Engine**: Train and evaluate a Logistic Regression model with TF-IDF feature extraction on a large-scale real-world dataset to classify reviews into three sentiment categories (Positive, Neutral, Negative).
2. **Design Grounded Generative AI Integration**: Integrate Google Gemini using the official `google-genai` Python SDK to provide sentiment explanations, batch summarization, and product improvement recommendations grounded strictly in ML predictions and original text.
3. **Build an Interactive Web Dashboard**: Provide a responsive SaaS-style Streamlit dashboard featuring 6 specialized modules, interactive charts, and CSV batch processing.
4. **Ensure Production Resilience**: Implement robust error handling, secure credential management via `.env`, and Windows Smart App Control compatibility without weakening operating system security policies.

---

## 3. DATASET & PREPROCESSING

### 3.1 Dataset Description
The system is built upon the **Amazon Fine Food Reviews** dataset, sourced from Kaggle:
- **Source**: Stanford Network Analysis Platform (SNAP) / Kaggle (`snap/amazon-fine-food-reviews`)
- **Total Records**: 568,454 customer reviews
- **Key Columns**:
  - `Text`: Unstructured customer review narrative (input feature).
  - `Score`: Customer rating on a 1-to-5 star scale (target variable).
  - `Summary`: Short review title.
  - `ProductId`, `UserId`: Metadata identifiers.

### 3.2 Sentiment Label Mapping
Customer star ratings were mapped into three discrete classes:
$$\text{Sentiment}(s) = \begin{cases} \text{Negative}, & s \in \{1, 2\} \\ \text{Neutral}, & s = 3 \\ \text{Positive}, & s \in \{4, 5\} \end{cases}$$

Original class distribution before sampling showed severe class imbalance (over 443,000 positive reviews vs. 82,000 negative and 42,000 neutral).

### 3.3 Balanced Sampling & Preprocessing Pipeline
To prevent majority-class bias and ensure high sensitivity across all categories:
1. **Deduplication**: Removed identical review texts and empty entries.
2. **Stratified Sampling**: Selected up to 20,000 reviews per class, yielding a perfectly balanced training dataset of **60,000 reviews** (20,000 Positive, 20,000 Neutral, 20,000 Negative).
3. **Text Cleaning**:
   - Converted all characters to lowercase.
   - Stripped HTML tags (`<.*?>`) and formatting artifacts.
   - Removed HTTP/HTTPS web links (`http\S+|www\S+`).
   - Filtered out punctuation, numeric digits, and non-alphabetic symbols (`[^a-z\s]`).
   - Normalized multiple consecutive spaces to a single space.
4. **Data Leakage Prevention**: Stratified train/test splitting (80% train = 48,000 samples, 20% test = 12,000 samples) was performed strictly **before** fitting the TF-IDF vectorizer.

---

## 4. SYSTEM ARCHITECTURE

The application adopts a modular, tiered pipeline architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                            │
│                 Streamlit SaaS Web Dashboard (app.py)                  │
│   [Dashboard]  [Review Analysis]  [Performance]  [AI Explainer]        │
│   [AI Batch Summarizer]  [Reports & Plotly Visualizations]             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LOGIC LAYER                         │
│  modules/prediction.py       modules/summarizer.py     modules/viz.py   │
│  • clean_text()              • parse_reviews_csv()     • Donut Chart   │
│  • predict_sentiment()       • analyze_batch()         • Bar Graphs    │
│  • predict_batch()           • export_analysis_csv()   • CM Heatmap    │
└─────────────────┬─────────────────────────────────┬────────────────────┘
                  │                                 │
                  ▼                                 ▼
┌──────────────────────────────────┐ ┌───────────────────────────────────┐
│     MACHINE LEARNING LAYER       │ │        GENERATIVE AI LAYER        │
│   (Scikit-learn & Joblib)        │ │        (modules/ai_explainer.py)  │
│ • TF-IDF Vectorizer (15k n-grams)│ │ • Google Gemini 2.5 Flash Client  │
│ • Logistic Regression Classifier │ │ • Grounded Sentiment Explainer    │
│ • Serialized Models (.pkl)       │ │ • Batch Review Summarizer         │
│ • Evaluation Artifacts (.json)   │ │ • Product Improvement Engine      │
│                                  │ │ • Local Rule-based Fallback       │
└──────────────────────────────────┘ └───────────────────────────────────┘
```

---

## 5. MACHINE LEARNING IMPLEMENTATION

### 5.1 Feature Extraction: TF-IDF Vectorizer
Text is converted into high-dimensional numerical feature vectors using the Term Frequency-Inverse Document Frequency formulation:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

Hyperparameter configuration:
- `max_features = 15000`: Caps vocabulary size to the 15,000 most informative tokens.
- `ngram_range = (1, 2)`: Captures single words and two-word collocations (e.g., "not good", "great quality").
- `stop_words = 'english'`: Filters uninformative standard English stopwords.
- `sublinear_tf = True`: Applies sublinear scaling ($1 + \log(\text{TF})$) to prevent frequent words from dominating feature weights.
- `min_df = 2, max_df = 0.95`: Eliminates rare typos and corpus-wide common words.

### 5.2 Classification Algorithm: Logistic Regression
Multinomial Logistic Regression models the probability of each sentiment category using the softmax activation:
$$P(Y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{j=1}^{C} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

Hyperparameters:
- `max_iter = 1000`: Ensures convergence of optimization solvers.
- `class_weight = 'balanced'`: Adjusts weights inversely proportional to class frequencies.
- `random_state = 42`: Guarantees exact reproducibility.

### 5.3 Model Artifact Persistence
Trained models and evaluation metrics are saved to `models/` using `joblib`:
- `models/sentiment_model.pkl`: Serialized Logistic Regression model.
- `models/tfidf_vectorizer.pkl`: Serialized TF-IDF vectorizer.
- `models/evaluation_results.json`: JSON containing exact test set metrics.
- `models/confusion_matrix.png`: High-resolution confusion matrix plot.

---

## 6. GENERATIVE AI IMPLEMENTATION

### 6.1 Role of Generative AI vs. Machine Learning
A core architectural principle of this MCA project is **hybrid synergy**:
- **ML Classifier**: Deterministic, fast, non-hallucinating, cost-effective predictor of sentiment classes and probabilities.
- **Generative AI (Google Gemini)**: Semantic reasoner that translates numerical predictions into human-readable business intelligence, isolates nuanced phrases, and generates engineering recommendations.

### 6.2 Gemini Integration via `google-genai` SDK
The official `google-genai` SDK is utilized (`from google import genai`). The API key is securely retrieved from the `.env` file or masked user input.

#### A. Grounded Sentiment Explanation
Gemini receives the review text, the ML predicted class, and the prediction confidence score. It returns structured JSON:
- `explanation`: Contextual rationale referencing specific customer experiences.
- `key_expressions`: List of influential positive/negative phrases.
- `customer_opinion`: Single-sentence summary of overall sentiment.
- `aspects_mentioned`: Categorized product aspects (Quality, Flavor, Packaging, Delivery, Price).

#### B. Batch Feedback Summarization & Recommendations
When users upload review collections (e.g. 100+ reviews), the ML engine first classifies all reviews and computes aggregate distribution percentages. Gemini receives the distribution statistics alongside stratified representative review samples. It outputs:
- Executive Customer Feedback Summary.
- Common Positive Points.
- Common Complaints & Friction Points.
- Frequently Mentioned Product Defects.
- Actionable Product Improvement Recommendations.

#### C. Graceful Fallback Architecture
If the Gemini API key is not configured, or if rate limits or network failures occur, the application dynamically falls back to an internal rule-based and statistical NLP engine. The ML predictions continue running without interruption.

---

## 7. EXPERIMENTAL RESULTS & EVALUATION

### 7.1 Verified Test Set Metrics
Evaluated on **12,000 unseen test reviews** (4,000 per class):

| Metric | Result |
|---|---|
| **Overall Accuracy** | **68.77%** |
| **Macro Average Precision** | **68.59%** |
| **Macro Average Recall** | **68.77%** |
| **Macro Average F1-Score** | **68.67%** |
| **Weighted Average F1-Score** | **68.67%** |

### 7.2 Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Negative** | 69.37% | 70.60% | 69.98% | 4,000 |
| **Neutral** | 60.59% | 58.08% | 59.31% | 4,000 |
| **Positive** | 75.82% | 77.62% | 76.71% | 4,000 |

*Analysis*: Positive and Negative classes achieve strong F1-scores (~77% and ~70%). The Neutral class achieves ~59%, which is typical in sentiment analysis as 3-star reviews often contain conflicting mixed statements (e.g., "Taste was good but packaging was broken").

### 7.3 Confusion Matrix
$$\begin{pmatrix} 2824 & 894 & 282 \\ 969 & 2323 & 708 \\ 278 & 617 & 3105 \end{pmatrix}$$
- **Actual Negative**: 2,824 correctly classified, 894 misclassified as Neutral, 282 as Positive.
- **Actual Neutral**: 2,323 correctly classified, 969 misclassified as Negative, 708 as Positive.
- **Actual Positive**: 3,105 correctly classified, 278 misclassified as Negative, 617 as Neutral.

---

## 8. VERIFICATION & UNIT TESTING

The test suite in `tests/test_prediction.py` verifies all 7 mandatory test cases:

```
.......
----------------------------------------------------------------------
Ran 7 tests in 1.838s

OK
```

1. **Test Case 1 (Positive Review)**: `"The product quality is excellent. I really love it."`  
   $\rightarrow$ Predicted: **Positive** (Confidence: 95.06%) | **PASS**
2. **Test Case 2 (Negative Review)**: `"The product is terrible. The quality is very poor."`  
   $\rightarrow$ Predicted: **Negative** (Confidence: 99.72%) | **PASS**
3. **Test Case 3 (Neutral Review)**: `"The product is okay. It works as expected."`  
   $\rightarrow$ Predicted: **Neutral** (Confidence: 96.94%) | **PASS**
4. **Test Case 4 (Empty Review)**: Empty or whitespace input raises `ValueError` with clear guidance | **PASS**
5. **Test Case 5 (Batch Processing)**: Batch of reviews classified and aggregated into statistics | **PASS**
6. **Test Case 6 (Missing API Key)**: Application activates local rule-based explainer without failure | **PASS**
7. **Test Case 7 (Invalid CSV File)**: Corrupted or headerless CSV safely rejected with UI error message | **PASS**

---

## 9. WINDOWS ENVIRONMENT COMPATIBILITY DIAGNOSIS

During local development on Windows 11 with Python 3.14, the following error was diagnosed:
`ImportError: DLL load failed while importing ccalendar: An Application Control policy has blocked this file.`

### Root Cause Analysis:
Windows 11 **Smart App Control (SAC)** policy (`{0283ac0f-fff1-49ae-ada1-8a933130cad6}`) inspects binary `.pyd` dynamic links loaded by `python.exe`. Newly released pre-compiled wheels for Python 3.14 (such as `pandas 3.1.1`) lack a recognized Microsoft cloud reputation or Authenticode digital signature, causing kernel Code Integrity to block `ccalendar.cp314-win_amd64.pyd`.

### Technical Resolution:
- Implemented automated fallback in `train_model.py` and application modules to **PyArrow** (`pyarrow.csv`) and Python's built-in `csv` library.
- PyArrow is fully compatible with Windows Smart App Control and ingested all 568,454 rows in **4.2 seconds** (faster than standard Pandas).
- Windows security policies remained completely uncompromised.
- Created `colab_training_workflow.ipynb` for cloud training on Google Colab if desired.

---

## 10. CONCLUSION & FUTURE ENHANCEMENTS

### 10.1 Conclusion
The **Customer Feedback Sentiment Analysis with AI Review Summarizer** successfully demonstrates the integration of classical Machine Learning and Generative AI. The project fulfills all functional, architectural, testing, and UI design rubrics of the MCA curriculum. The application runs autonomously, loads persisted model artifacts, executes inference in real-time, and provides actionable business intelligence.

### 10.2 Future Scope
1. **Aspect-Based Sentiment Analysis (ABSA)**: Deconstruct multi-sentence reviews to assign independent sentiment ratings to individual attributes (e.g., Shipping = Negative, Flavor = Positive).
2. **Multilingual Review Analysis**: Integrate translation and multilingual transformer models (e.g., XLM-RoBERTa) for non-English e-commerce domains.
3. **Automated CRM Ticket Dispatch**: Automatically route reviews classified as high-confidence Negative directly to customer service ticketing queues.

---

## 11. REFERENCES

1. SNAP Amazon Fine Food Reviews Dataset: [https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Google Gemini API Documentation & SDK (`google-genai`): [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
4. Streamlit Open-Source Python Framework: [https://docs.streamlit.io](https://docs.streamlit.io)
5. Plotly Open Source Graphing Library for Python: [https://plotly.com/python](https://plotly.com/python)
