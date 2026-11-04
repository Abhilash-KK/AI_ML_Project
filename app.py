"""
Customer Feedback Sentiment Analysis with AI Review Summarizer
Streamlit Web Application
An MCA Academic Project integrating Scikit-learn ML with Google Gemini Generative AI.
"""

import os
import sys
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project modules are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.prediction import (
    clean_text,
    load_model_artifacts,
    load_evaluation_results,
    predict_sentiment,
    predict_batch
)
from modules.ai_explainer import (
    explain_sentiment,
    summarize_reviews,
    get_api_key
)
from modules.summarizer import (
    analyze_uploaded_reviews,
    export_analysis_csv,
    parse_reviews_csv
)
from modules.visualizations import (
    plot_sentiment_donut,
    plot_sentiment_bars,
    plot_confidence_distribution,
    plot_interactive_confusion_matrix,
    SENTIMENT_COLORS
)

# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Feedback AI | ML + GenAI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# MODERN SAAS CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #dbeafe !important;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: transform 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    .badge-positive {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #a7f3d0;
    }
    .badge-negative {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #fecaca;
    }
    .badge-neutral {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #fde68a;
    }

    .insight-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------
if "current_review" not in st.session_state:
    st.session_state.current_review = "The product quality is excellent. I really love it."
if "single_prediction" not in st.session_state:
    st.session_state.single_prediction = None
if "batch_analysis" not in st.session_state:
    st.session_state.batch_analysis = None
if "api_key" not in st.session_state:
    st.session_state.api_key = get_api_key() or ""

# --------------------------------------------------
# LOAD MODEL & EVALUATION ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def get_model():
    return load_model_artifacts()

@st.cache_data
def get_eval():
    return load_evaluation_results()

model_loaded = False
load_error_msg = None
try:
    model, vectorizer = get_model()
    eval_metrics = get_eval()
    model_loaded = True
except Exception as e:
    load_error_msg = str(e)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=64)
    st.title("Sentiment AI")
    st.caption("MCA Combined Practical Project")
    
    st.markdown("---")
    navigation = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Analyze Review",
            "📈 ML Model Performance",
            "🤖 AI Sentiment Explainer",
            "📝 AI Review Summarizer",
            "📑 Reports & Visualization"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("🔑 Gemini API Settings")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Reads from .env by default. Enter your API key from Google AI Studio."
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    if st.session_state.api_key:
        st.success("API Key configured", icon="✅")
    else:
        st.info("API Key optional (Local explainer fallback active)", icon="ℹ️")

    st.markdown("---")
    st.markdown("**System Architecture:**")
    st.caption("• Classifier: Logistic Regression\n• Features: TF-IDF (15k N-grams)\n• GenAI: Google Gemini Flash\n• Dataset: Amazon Fine Food Reviews")

# Check if model loaded successfully
if not model_loaded:
    st.error(f"**Model Load Error:**\n{load_error_msg}")
    st.warning("Please train the model first by executing in PowerShell:\n`python train_model.py`")
    st.stop()


# --------------------------------------------------
# MODULE 1: DASHBOARD
# --------------------------------------------------
if navigation == "📊 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>Customer Feedback Sentiment Intelligence Dashboard</h1>
        <p>Real-time analytics combining Machine Learning classification with Generative AI product synthesis</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset & Training Distribution Stats
    dist = eval_metrics.get("class_distribution", {"Positive": 20000, "Neutral": 20000, "Negative": 20000})
    total_samples = eval_metrics.get("total_samples", sum(dist.values()))
    acc = eval_metrics.get("accuracy", 0.6877) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Reviews Analyzed</div>
            <div class="metric-value" style="color: #1e3a8a;">{total_samples:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Positive Reviews</div>
            <div class="metric-value" style="color: #059669;">{dist.get('Positive', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Neutral Reviews</div>
            <div class="metric-value" style="color: #d97706;">{dist.get('Neutral', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Negative Reviews</div>
            <div class="metric-value" style="color: #dc2626;">{dist.get('Negative', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value" style="color: #2563eb;">{acc:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive charts row
    c_left, c_right = st.columns(2)
    with c_left:
        fig_donut = plot_sentiment_donut(dist)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        fig_bar = plot_sentiment_bars(dist)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Integration overview section
    st.markdown("### 🔄 End-to-End Hybrid Architecture")
    st.info(
        "**Hybrid ML + GenAI Paradigm:** "
        "The Machine Learning component (TF-IDF + Logistic Regression) acts as the deterministic classifier, "
        "predicting sentiment probabilities without hallucination. Google Gemini acts as the generative cognitive layer, "
        "interpreting the review context and synthesizing customer insights without replacing the classifier."
    )


# --------------------------------------------------
# MODULE 2: REVIEW ANALYSIS (Single Review)
# --------------------------------------------------
elif navigation == "🔍 Analyze Review":
    st.markdown("""
    <div class="main-header">
        <h1>Single Review Sentiment Analyzer</h1>
        <p>Clean, vectorize, and classify customer feedback using the trained Logistic Regression model</p>
    </div>
    """, unsafe_allow_html=True)

    # Sample Buttons
    st.write("**Quick Test Samples:**")
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        if st.button("Sample 1: Positive Review"):
            st.session_state.current_review = "The product quality is excellent. I really love it."
    with s_col2:
        if st.button("Sample 2: Negative Review"):
            st.session_state.current_review = "The product is terrible. The quality is very poor."
    with s_col3:
        if st.button("Sample 3: Neutral Review"):
            st.session_state.current_review = "The product is okay. It works as expected."

    review_input = st.text_area(
        "Enter customer review text:",
        value=st.session_state.current_review,
        height=130,
        placeholder="Type or paste a customer product review..."
    )

    if st.button("🚀 Analyze Sentiment", type="primary"):
        if not review_input.strip():
            st.warning("Please enter review text before analyzing.")
        else:
            try:
                pred_result = predict_sentiment(review_input, model=model, vectorizer=vectorizer)
                st.session_state.single_prediction = pred_result
                st.session_state.current_review = review_input
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    if st.session_state.single_prediction:
        res = st.session_state.single_prediction
        st.markdown("---")
        st.subheader("Analysis Results")

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown("**Original Customer Review:**")
            st.write(f"> *{res['original_text']}*")
            st.markdown("**Preprocessed (Cleaned) Text:**")
            st.code(res["cleaned_text"], language="text")

        with res_col2:
            s_label = res["sentiment"]
            badge_class = f"badge-{s_label.lower()}"
            st.markdown(f"""
            <div style="margin-bottom: 0.8rem;">
                <span class="{badge_class}" style="font-size: 1.3rem; padding: 0.5rem 1.2rem;">
                    Predicted Sentiment: {s_label}
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Model Confidence", f"{res['confidence']:.2f}%")

            # Probability breakdown bars
            probs = res["probabilities"]
            st.write("**Probability Distribution:**")
            for cat, pct in probs.items():
                st.write(f"- **{cat}**: {pct:.1f}%")
                st.progress(pct / 100.0)

        # Quick action to explain with Gemini
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 Explain this prediction with Generative AI"):
            # Set review in session and navigate
            st.session_state.current_review = res["original_text"]
            st.info("Switch to the '🤖 AI Sentiment Explainer' tab from the sidebar to inspect the detailed Gemini interpretation!")


# --------------------------------------------------
# MODULE 3: ML MODEL PERFORMANCE
# --------------------------------------------------
elif navigation == "📈 ML Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>Machine Learning Model Evaluation</h1>
        <p>Verified performance metrics loaded directly from models/evaluation_results.json</p>
    </div>
    """, unsafe_allow_html=True)

    # Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Overall Accuracy", f"{eval_metrics['accuracy'] * 100:.2f}%")
    with m2:
        st.metric("Macro Precision", f"{eval_metrics['precision_macro'] * 100:.2f}%")
    with m3:
        st.metric("Macro Recall", f"{eval_metrics['recall_macro'] * 100:.2f}%")
    with m4:
        st.metric("Macro F1-Score", f"{eval_metrics['f1_score_macro'] * 100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Classification Report
    st.subheader("📋 Classification Report by Sentiment Class")
    report_dict = eval_metrics.get("classification_report", {})
    
    table_data = []
    for cls_name in ["Negative", "Neutral", "Positive"]:
        if cls_name in report_dict:
            c_info = report_dict[cls_name]
            table_data.append({
                "Sentiment Class": cls_name,
                "Precision": f"{c_info['precision'] * 100:.2f}%",
                "Recall": f"{c_info['recall'] * 100:.2f}%",
                "F1-Score": f"{c_info['f1-score'] * 100:.2f}%",
                "Support (Test Samples)": int(c_info["support"])
            })

    st.table(table_data)

    st.markdown("---")
    # Confusion Matrix Section
    cm_left, cm_right = st.columns([1, 1])
    with cm_left:
        st.subheader("Interactive Confusion Matrix")
        cm_data = eval_metrics.get("confusion_matrix", [])
        if cm_data:
            fig_cm = plot_interactive_confusion_matrix(cm_data, eval_metrics.get("class_labels", ["Negative", "Neutral", "Positive"]))
            st.plotly_chart(fig_cm, use_container_width=True)

    with cm_right:
        st.subheader("Saved Artifact: confusion_matrix.png")
        cm_png_path = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
        if os.path.exists(cm_png_path):
            st.image(cm_png_path, caption="Confusion Matrix generated during model training", use_container_width=True)
        else:
            st.info("Confusion matrix image not found. Please train model.")

    st.markdown("---")
    st.subheader("⚙️ Pipeline Technical Specifications")
    st.markdown(f"""
    - **Dataset**: Amazon Fine Food Reviews (568,454 original records)
    - **Balanced Training Subset**: {eval_metrics.get('training_samples', 48000):,} samples (80%)
    - **Independent Testing Subset**: {eval_metrics.get('testing_samples', 12000):,} samples (20%, Stratified)
    - **Feature Extractor**: Scikit-learn `TfidfVectorizer` (max_features=15000, n-grams=(1,2), sublinear_tf=True)
    - **Classifier**: `LogisticRegression` (class_weight='balanced', max_iter=1000, random_state=42)
    """)


# --------------------------------------------------
# MODULE 4: AI SENTIMENT EXPLAINER
# --------------------------------------------------
elif navigation == "🤖 AI Sentiment Explainer":
    st.markdown("""
    <div class="main-header">
        <h1>Generative AI Sentiment Explainer</h1>
        <p>Grounding Gemini AI with the independent Machine Learning prediction and original review context</p>
    </div>
    """, unsafe_allow_html=True)

    review_text = st.text_area(
        "Enter customer review to explain:",
        value=st.session_state.current_review,
        height=110
    )

    if st.button("🔍 Generate AI Explanation", type="primary"):
        if not review_text.strip():
            st.warning("Please provide a customer review.")
        else:
            with st.spinner("Classifying with ML model and querying Gemini AI..."):
                # Step 1: Independent ML prediction
                pred = predict_sentiment(review_text, model=model, vectorizer=vectorizer)
                # Step 2: GenAI explanation
                expl = explain_sentiment(
                    review_text,
                    pred["sentiment"],
                    pred["confidence"],
                    api_key=st.session_state.api_key
                )

                st.markdown("---")
                # Top badges
                b_col1, b_col2 = st.columns([1, 1])
                with b_col1:
                    badge_cls = f"badge-{pred['sentiment'].lower()}"
                    st.markdown(f"""
                    <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <strong>ML Model Prediction:</strong>
                        <span class="{badge_cls}">{pred['sentiment']}</span>
                        <span style="color: #6b7280; font-size: 0.95rem; margin-left: 0.5rem;">({pred['confidence']:.2f}% confidence)</span>
                    </div>
                    """, unsafe_allow_html=True)
                with b_col2:
                    st.markdown(f"""
                    <div style="text-align: right; color: #4b5563; font-size: 0.9rem;">
                        <strong>Engine:</strong> <code>{expl.get('source', 'Gemini')}</code>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Explanation Card
                st.markdown(f"""
                <div class="insight-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e3a8a;">💡 Sentiment Rationale</h4>
                    <p style="font-size: 1.05rem; line-height: 1.6; margin: 0; color: #1f2937;">{expl['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Expressions and Customer Opinion
                e_left, e_right = st.columns([1, 1])
                with e_left:
                    st.markdown("#### 🔑 Influential Phrases & Keywords")
                    expressions = expl.get("key_expressions", [])
                    if expressions:
                        for exp in expressions:
                            st.markdown(f"- `{exp}`")
                    else:
                        st.write("No specific keywords isolated.")

                with e_right:
                    st.markdown("#### 👤 Customer Core Opinion")
                    st.info(expl.get("customer_opinion", "Customer opinion analyzed."))

                    aspects = expl.get("aspects_mentioned", [])
                    if aspects:
                        st.markdown("**Product Aspects Mentioned:**")
                        st.write(", ".join([f"**{a}**" for a in aspects]))


# --------------------------------------------------
# MODULE 5: AI REVIEW SUMMARIZER
# --------------------------------------------------
elif navigation == "📝 AI Review Summarizer":
    st.markdown("""
    <div class="main-header">
        <h1>AI Review Batch Summarizer & Advisor</h1>
        <p>Upload a collection of customer reviews for automated ML classification and Generative AI product synthesis</p>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_sm = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload CSV containing reviews", type=["csv"])
    with col_sm:
        st.write("**Or use included sample:**")
        load_sample = st.button("📂 Load 15 Sample Reviews")

    file_to_process = None
    if uploaded_file is not None:
        file_to_process = uploaded_file
    elif load_sample:
        sample_path = os.path.join(BASE_DIR, "sample_reviews.csv")
        if os.path.exists(sample_path):
            file_to_process = sample_path
        else:
            st.error("Sample CSV file not found on disk.")

    if file_to_process is not None:
        try:
            with st.spinner("Parsing CSV and running ML predictions on all reviews..."):
                analysis_res = analyze_uploaded_reviews(file_to_process, model=model, vectorizer=vectorizer)
                st.session_state.batch_analysis = analysis_res

            st.success(f"Successfully processed {analysis_res['total_reviews']} reviews using the ML pipeline!", icon="✅")
        except Exception as e:
            st.error(f"Failed to process CSV: {str(e)}")

    if st.session_state.batch_analysis is not None:
        b_data = st.session_state.batch_analysis
        counts = b_data["sentiment_counts"]
        pcts = b_data["sentiment_percentages"]
        total = b_data["total_reviews"]

        # Batch Metrics
        bm1, bm2, bm3, bm4 = st.columns(4)
        with bm1:
            st.metric("Total Reviews", f"{total}")
        with bm2:
            st.metric("Positive", f"{counts['Positive']} ({pcts['Positive']}%)")
        with bm3:
            st.metric("Neutral", f"{counts['Neutral']} ({pcts['Neutral']}%)")
        with bm4:
            st.metric("Negative", f"{counts['Negative']} ({pcts['Negative']}%)")

        st.markdown("<br>", unsafe_allow_html=True)
        # Visual distribution
        v_left, v_right = st.columns([1, 1])
        with v_left:
            st.plotly_chart(plot_sentiment_donut(counts), use_container_width=True)
        with v_right:
            st.plotly_chart(plot_sentiment_bars(counts), use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 Generative AI Synthesis & Product Advisor")

        with st.spinner("Synthesizing batch feedback with Google Gemini AI..."):
            summary_result = summarize_reviews(
                b_data["records"],
                counts,
                api_key=st.session_state.api_key
            )

        # Executive Summary
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #1e3a8a;">📑 Executive Customer Feedback Summary</h4>
            <p style="font-size: 1.05rem; line-height: 1.6; margin: 0; color: #1f2937;">{summary_result['overall_summary']}</p>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #6b7280;">Source: <code>{summary_result.get('source', 'Gemini')}</code></div>
        </div>
        """, unsafe_allow_html=True)

        # Insights Grid
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### ✅ Common Positive Feedback")
            for pos_item in summary_result.get("common_positive_opinions", []):
                st.markdown(f"- {pos_item}")

            st.markdown("#### ⚠️ Common Complaints & Friction Points")
            for neg_item in summary_result.get("common_negative_opinions", []):
                st.markdown(f"- {neg_item}")

        with g2:
            st.markdown("#### 🔍 Frequently Mentioned Product Problems")
            for prob in summary_result.get("frequent_problems", []):
                st.markdown(f"- {prob}")

            st.markdown("#### 🎯 Actionable Improvement Recommendations")
            for rec in summary_result.get("improvement_recommendations", []):
                st.markdown(f"- **{rec}**")

        st.markdown("---")
        st.subheader("📥 Export Analyzed Data")
        csv_download = export_analysis_csv(b_data["records"])
        st.download_button(
            label="Download Analyzed Reviews (CSV)",
            data=csv_download,
            file_name="analyzed_customer_reviews.csv",
            mime="text/csv",
            type="primary"
        )


# --------------------------------------------------
# MODULE 6: REPORTS AND VISUALIZATION
# --------------------------------------------------
elif navigation == "📑 Reports & Visualization":
    st.markdown("""
    <div class="main-header">
        <h1>Reports, Statistical Diagnostics & Downloads</h1>
        <p>Comprehensive analytical views, model confidence spreads, and batch export history</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.batch_analysis is not None:
        records = st.session_state.batch_analysis["records"]
        st.subheader("📊 Model Confidence Distribution Across Sentiments")
        st.caption("Inspect prediction certainty and detect borderline reviews")
        fig_conf = plot_confidence_distribution(records)
        st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown("---")
        st.subheader("📄 Detailed Review Classification Table")
        
        display_data = []
        for r in records[:50]:  # Display top 50 in table
            display_data.append({
                "Sentiment": r.get("sentiment"),
                "Confidence": f"{r.get('confidence', 0):.1f}%",
                "Review Text": r.get("original_text")[:90] + ("..." if len(r.get("original_text", "")) > 90 else "")
            })
        st.dataframe(display_data, use_container_width=True)

        csv_download = export_analysis_csv(records)
        st.download_button(
            label="📥 Download Full Analysis CSV",
            data=csv_download,
            file_name="complete_sentiment_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No batch dataset has been uploaded in this session yet. You can upload a CSV in '📝 AI Review Summarizer' to view full reports, or inspect overall training statistics below.")
        
        st.subheader("Model Training Statistics Overview")
        st.json(eval_metrics)
