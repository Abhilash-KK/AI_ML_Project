"""
Customer Feedback Sentiment Analysis - Visualizations Module
Provides Plotly interactive charts and Matplotlib figures for analytics.
"""

from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go

# Brand color palette
SENTIMENT_COLORS = {
    "Positive": "#2ecc71",  # Vibrant emerald green
    "Neutral": "#f39c12",   # Warm amber/orange
    "Negative": "#e74c3c"   # Clean crimson red
}


def plot_sentiment_donut(counts: Dict[str, int]) -> go.Figure:
    """Create an interactive Plotly donut chart of the sentiment breakdown."""
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SENTIMENT_COLORS.get(lbl, "#95a5a6") for lbl in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=14, family="sans-serif")
            )
        ]
    )
    fig.update_layout(
        title=dict(text="<b>Customer Sentiment Distribution</b>", font=dict(size=18)),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=380
    )
    return fig


def plot_sentiment_bars(counts: Dict[str, int]) -> go.Figure:
    """Create an interactive Plotly horizontal or vertical bar chart with counts."""
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SENTIMENT_COLORS.get(lbl, "#95a5a6") for lbl in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=values,
                textposition="auto",
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(0,0,0,0.1)", width=1.5)
                )
            )
        ]
    )
    fig.update_layout(
        title=dict(text="<b>Review Volume by Sentiment Category</b>", font=dict(size=18)),
        xaxis=dict(title="Sentiment Category"),
        yaxis=dict(title="Number of Reviews"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380
    )
    return fig


def plot_confidence_distribution(records: List[Dict[str, Any]]) -> go.Figure:
    """Plot distribution of prediction confidence across sentiment categories."""
    data_list = []
    for r in records:
        if r.get("status") == "success":
            data_list.append({
                "Sentiment": r.get("sentiment", "Unknown"),
                "Confidence": r.get("confidence", 0.0)
            })

    if not data_list:
        fig = go.Figure()
        fig.update_layout(title="No valid records for confidence distribution")
        return fig

    sentiments = [d["Sentiment"] for d in data_list]
    confs = [d["Confidence"] for d in data_list]

    fig = px.box(
        x=sentiments,
        y=confs,
        color=sentiments,
        color_discrete_map=SENTIMENT_COLORS,
        labels={"x": "Sentiment Category", "y": "Confidence Score (%)"},
        title="<b>Model Confidence Spread by Sentiment</b>"
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        height=380
    )
    return fig


def plot_interactive_confusion_matrix(cm_data: List[List[int]], labels: List[str]) -> go.Figure:
    """Render interactive Plotly heatmap for Confusion Matrix inspection."""
    z_values = cm_data
    text_values = [[str(val) for val in row] for row in z_values]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=labels,
            y=labels,
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            colorscale="Blues",
            hoverongaps=False
        )
    )
    fig.update_layout(
        title=dict(text="<b>Interactive Confusion Matrix</b>", font=dict(size=18)),
        xaxis=dict(title="<b>Predicted Sentiment</b>"),
        yaxis=dict(title="<b>Actual Sentiment</b>", autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=420
    )
    return fig
