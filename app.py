"""Streamlit app: NSL-KDD Intrusion Detection (Normal vs Attack).

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import (
    METRIC_COLUMNS,
    compute_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_roc_curve,
    predict_with_scores,
)
from src.preprocessing import extract_target, transform_features
from src.utils import (
    CLASS_NAMES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    PREPROCESSOR_PATH,
    TEST_DATA_PATH,
    load_artifact,
    read_uploaded_csv,
)

st.set_page_config(
    page_title="NSL-KDD Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

PAGES = {
    "🏠  Overview": "Overview",
    "📈  Model Evaluation": "Model Evaluation",
    "⚖️  Model Comparison": "Model Comparison",
    "ℹ️  About": "About",
}
COLOR_NORMAL = "#2f9e77"
COLOR_ATTACK = "#e5484d"
MODEL_COLORS = ["#1f6feb", "#2f9e77", "#f59f00", "#e5484d", "#8250df"]

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e3e8ef; border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(16, 24, 40, .06);
}
div[data-testid="stMetricLabel"] p {color: #5b6b7f; font-weight: 600;}
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.hero {
    background: linear-gradient(120deg, #0f3d91 0%, #1f6feb 60%, #2bb3a3 100%);
    color: #fff; padding: 28px 32px; border-radius: 16px; margin-bottom: 22px;
}
.hero h1 {color: #fff; margin: 0; padding: 0; font-size: 2rem;}
.hero p {color: #e6efff; margin: 8px 0 0 0; font-size: 1.02rem;}
.pill {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    background: #e8f1ff; color: #1f4fb4; font-size: .8rem; font-weight: 600;
    margin-right: 6px;
}
.pill-ok {background: #e3f6ee; color: #1b7a57;}
.pill-warn {background: #fff4db; color: #9a6700;}
.section-title {font-size: 1.15rem; font-weight: 700; margin: 6px 0 2px 0;}
.stat-card {
    background: #ffffff; border: 1px solid #e3e8ef; border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(16, 24, 40, .06);
    height: 100%;
}
.stat-label {color: #5b6b7f; font-weight: 600; font-size: .9rem;}
.stat-name {font-size: 1.25rem; font-weight: 700; line-height: 1.25; margin: 4px 0;}
.stat-value {color: #1b7a57; font-weight: 600; font-size: .95rem;}
.step-num {
    display: inline-block; width: 28px; height: 28px; line-height: 28px;
    border-radius: 50%; background: #1f6feb; color: #fff; text-align: center;
    font-weight: 700; margin-right: 8px;
}
section[data-testid="stSidebar"] {border-right: 1px solid #e3e8ef;}
</style>
"""


# --------------------------------------------------------------------------- #
# Cached loaders / computations
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_model(name: str):
    """Load a trained model from ``model/``."""
    return load_artifact(MODEL_DIR / MODEL_FILES[name])


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """Load the fitted encoders and scaler."""
    return load_artifact(PREPROCESSOR_PATH)


@st.cache_data(show_spinner="Preprocessing uploaded data...")
def prepare_features(df: pd.DataFrame):
    """Encode/scale the uploaded data; return (features, binary target|None)."""
    return transform_features(df, load_preprocessor()), extract_target(df)


@st.cache_data(show_spinner="Running model on uploaded data...")
def run_model(name: str, features: pd.DataFrame):
    """Predict with one model; return (hard predictions, attack scores)."""
    return predict_with_scores(load_model(name), features)


@st.cache_data(show_spinner=False)
def load_training_metrics():
    """Metrics saved at training time (hold-out split), if available."""
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_sample_csv() -> bytes:
    """Bundled sample test file."""
    return TEST_DATA_PATH.read_bytes()


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str) -> None:
    """Gradient page header."""
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None) -> None:
    """Section heading with optional caption."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def style_table(df: pd.DataFrame, highlight: bool = True):
    """Format metric columns and highlight the best value of each metric."""
    styler = df.style.format({c: "{:.4f}" for c in METRIC_COLUMNS if c in df})
    if highlight:
        styler = styler.highlight_max(
            subset=[c for c in METRIC_COLUMNS if c in df],
            props="background-color: #d3f3e3; color: #0b3d27; font-weight: 700;",
        )
    return styler


def style_figure(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply a consistent light look to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def confusion_matrix_figure(cm: np.ndarray) -> go.Figure:
    """Annotated confusion-matrix heatmap (counts and row percentages)."""
    row_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            showscale=False,
            xgap=3,
            ygap=3,
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z:,}<extra></extra>",
        )
    )
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=CLASS_NAMES[j],
                y=CLASS_NAMES[i],
                text=f"<b>{cm[i, j]:,}</b><br>{row_pct[i, j]:.1f}%",
                showarrow=False,
                font=dict(size=18, color="white" if cm[i, j] > threshold else "#1b2430"),
            )
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return style_figure(fig, 400)


def roc_figure(curves: dict) -> go.Figure:
    """ROC chart for one or several models: {name: (fpr, tpr, auc)}."""
    fig = go.Figure()
    for idx, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.4f})",
                line=dict(width=2.5, color=MODEL_COLORS[idx % len(MODEL_COLORS)]),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance",
            line=dict(dash="dash", color="#98a2b3"),
        )
    )
    fig.update_xaxes(title="False Positive Rate", range=[0, 1])
    fig.update_yaxes(title="True Positive Rate", range=[0, 1.02])
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )
    return style_figure(fig, 400)


def traffic_donut(normal: int, attack: int) -> go.Figure:
    """Donut chart of predicted traffic classes."""
    fig = go.Figure(
        go.Pie(
            labels=CLASS_NAMES,
            values=[normal, attack],
            hole=0.62,
            marker=dict(colors=[COLOR_NORMAL, COLOR_ATTACK]),
            textinfo="percent",
            sort=False,
        )
    )
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.05))
    return style_figure(fig, 260)


def stat_card(label: str, name: str, value: str) -> str:
    """HTML card that wraps long model names instead of truncating them."""
    return (
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-name">{name}</div>'
        f'<div class="stat-value">▲ {value}</div></div>'
    )


def show_metric_cards(metrics: dict) -> None:
    """Render the six metrics as cards."""
    for col, key in zip(st.columns(len(METRIC_COLUMNS)), METRIC_COLUMNS):
        col.metric(key, f"{metrics[key]:.4f}")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar():
    """Navigation, data upload and model selection."""
    st.sidebar.markdown("## 🛡️ NSL-KDD IDS")
    st.sidebar.caption("Network intrusion detection · Normal vs Attack")
    page_label = st.sidebar.radio(
        "Navigation", list(PAGES), label_visibility="collapsed"
    )

    st.sidebar.divider()
    st.sidebar.markdown("**1 · Test data**")
    uploaded = st.sidebar.file_uploader(
        "Upload test dataset (CSV)", type=["csv"], label_visibility="collapsed"
    )
    use_sample = False
    if TEST_DATA_PATH.exists():
        use_sample = st.sidebar.toggle("Use bundled sample", value=False)
        st.sidebar.download_button(
            "⬇ Download sample test_data.csv",
            data=load_sample_csv(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )

    st.sidebar.markdown("**2 · Model**")
    model_name = st.sidebar.selectbox(
        "Model", list(MODEL_FILES), label_visibility="collapsed"
    )

    df = None
    try:
        if uploaded is not None:
            df = read_uploaded_csv(uploaded)
        elif use_sample:
            df = read_uploaded_csv(load_sample_csv())
    except Exception as exc:  # noqa: BLE001 - show any parsing problem to user
        st.sidebar.error(f"Could not read the CSV file: {exc}")

    st.sidebar.divider()
    if df is not None:
        st.sidebar.markdown(
            f'<span class="pill pill-ok">Loaded</span> {len(df):,} rows × '
            f"{df.shape[1]} cols",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="pill pill-warn">No data</span> upload a CSV or enable '
            "the sample",
            unsafe_allow_html=True,
        )
    return PAGES[page_label], df, model_name


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    """Landing page with project summary, steps and training leaderboard."""
    hero(
        "Network Intrusion Detection",
        "Classify NSL-KDD network connections as <b>Normal</b> or "
        "<b>Attack</b> with five machine learning models.",
    )
    training = load_training_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", "125,973")
    c2.metric("Features", 41)
    c3.metric("Models", len(MODEL_FILES))
    if training is not None:
        best = training.sort_values("MCC", ascending=False).iloc[0]
        c4.markdown(
            stat_card("Best model", best["ML Model"], f"MCC {best['MCC']:.4f}"),
            unsafe_allow_html=True,
        )

    st.write("")
    section("How to use")
    steps = [
        ("1", "Upload data", "Add a test CSV in the sidebar, or switch on the "
              "bundled sample."),
        ("2", "Pick a model", "Choose one of five classifiers from the "
              "dropdown."),
        ("3", "Explore results", "Open <b>Model Evaluation</b> or <b>Model "
              "Comparison</b> for metrics and charts."),
    ]
    for col, (num, title, text) in zip(st.columns(3), steps):
        with col.container(border=True):
            st.markdown(
                f'<span class="step-num">{num}</span><b>{title}</b>',
                unsafe_allow_html=True,
            )
            st.markdown(text, unsafe_allow_html=True)

    if training is not None:
        st.write("")
        section(
            "Training results",
            "Hold-out test split (25,195 records) from the training pipeline.",
        )
        st.dataframe(
            style_table(training), width="stretch", hide_index=True
        )

    with st.expander("Expected CSV format"):
        st.markdown(
            "The file must contain the 41 NSL-KDD feature columns "
            "(`duration`, `protocol_type`, `service`, `flag`, ...). Add a "
            "`label` column (attack name, `normal`/`attack` or 0/1) to get "
            "evaluation metrics; without it only predictions are shown. "
            "Header-less original NSL-KDD files are also accepted."
        )


def page_evaluation(df: pd.DataFrame, model_name: str) -> None:
    """Metrics, confusion matrix, report, ROC and predictions for one model."""
    hero(
        f"Model Evaluation · {model_name}",
        "Performance of the selected model on the uploaded test data.",
    )
    try:
        features, y_true = prepare_features(df)
        y_pred, y_score = run_model(model_name, features)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not evaluate the uploaded data: {exc}")
        return

    attacks = int((y_pred == 1).sum())
    normals = int((y_pred == 0).sum())
    total = len(y_pred)

    section("Prediction results")
    left, right = st.columns([2, 1])
    with left:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total records", f"{total:,}")
        c2.metric("Normal traffic", f"{normals:,}", f"{normals / total:.1%}",
                  delta_color="off")
        c3.metric("Attacks detected", f"{attacks:,}", f"{attacks / total:.1%}",
                  delta_color="off")
    with right:
        st.plotly_chart(traffic_donut(normals, attacks), width="stretch")

    predictions = df.copy()
    predictions["predicted_class"] = [CLASS_NAMES[i] for i in y_pred]

    if y_true is None:
        st.warning(
            "No `label` column found in the uploaded file, so evaluation "
            "metrics cannot be computed. Predictions are shown below."
        )
        st.dataframe(predictions.head(200), width="stretch")
        return

    metrics = compute_metrics(y_true, y_pred, y_score)
    st.write("")
    section("Evaluation metrics")
    show_metric_cards(metrics)

    st.write("")
    tab_cm, tab_report, tab_preds = st.tabs(
        ["Confusion matrix & ROC", "Classification report", "Predictions"]
    )
    with tab_cm:
        left, right = st.columns(2)
        with left:
            section("Confusion matrix")
            st.plotly_chart(
                confusion_matrix_figure(get_confusion_matrix(y_true, y_pred)),
                width="stretch",
            )
        with right:
            section("ROC curve")
            fpr, tpr = get_roc_curve(y_true, y_score)
            st.plotly_chart(
                roc_figure({model_name: (fpr, tpr, metrics["AUC"])}),
                width="stretch",
            )
    with tab_report:
        report = get_classification_report(y_true, y_pred)
        report.loc["accuracy", "support"] = total
        st.dataframe(
            report.style.format(
                {"precision": "{:.4f}", "recall": "{:.4f}",
                 "f1-score": "{:.4f}", "support": "{:,d}"}
            ),
            width="stretch",
        )
    with tab_preds:
        mismatch = predictions.assign(actual_class=[CLASS_NAMES[i] for i in y_true])
        only_errors = st.toggle("Show misclassified records only", value=False)
        shown = mismatch[mismatch["predicted_class"] != mismatch["actual_class"]] \
            if only_errors else mismatch
        st.caption(f"{len(shown):,} records")
        st.dataframe(shown.head(500), width="stretch")
        st.download_button(
            "⬇ Download predictions (CSV)",
            data=mismatch.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


def page_comparison(df: pd.DataFrame) -> None:
    """Compare all five models on the uploaded data."""
    hero(
        "Model Comparison",
        "All five models evaluated on the same uploaded test data.",
    )
    try:
        features, y_true = prepare_features(df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not process the uploaded data: {exc}")
        return
    if y_true is None:
        st.warning("A `label` column is required to compare models.")
        return

    rows, curves = [], {}
    progress = st.progress(0.0, text="Evaluating models...")
    for i, name in enumerate(MODEL_FILES, start=1):
        y_pred, y_score = run_model(name, features)
        metrics = compute_metrics(y_true, y_pred, y_score)
        rows.append({"ML Model": name, **metrics})
        fpr, tpr = get_roc_curve(y_true, y_score)
        curves[name] = (fpr, tpr, metrics["AUC"])
        progress.progress(i / len(MODEL_FILES), text=f"Evaluated {name}")
    progress.empty()
    results = pd.DataFrame(rows)

    best = results.sort_values(["MCC", "F1"], ascending=False).iloc[0]
    st.success(
        f"🏆 Best overall model (by MCC): **{best['ML Model']}** — "
        f"MCC {best['MCC']:.4f}, F1 {best['F1']:.4f}, "
        f"Accuracy {best['Accuracy']:.4f}"
    )

    section("Best model per metric")
    for col, metric in zip(st.columns(len(METRIC_COLUMNS)), METRIC_COLUMNS):
        winner = results.loc[results[metric].idxmax()]
        col.markdown(
            stat_card(metric, winner["ML Model"], f"{winner[metric]:.4f}"),
            unsafe_allow_html=True,
        )

    st.write("")
    tab_table, tab_bar, tab_roc = st.tabs(
        ["Metrics table", "Bar chart", "ROC curves"]
    )
    with tab_table:
        st.dataframe(style_table(results), width="stretch", hide_index=True)
        st.caption("Best value per metric is highlighted.")
        training = load_training_metrics()
        if training is not None:
            with st.expander("Metrics on the training-time hold-out split"):
                st.dataframe(
                    style_table(training), width="stretch", hide_index=True
                )
    with tab_bar:
        melted = results.melt(
            id_vars="ML Model", var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            melted, x="Metric", y="Score", color="ML Model", barmode="group",
            color_discrete_sequence=MODEL_COLORS, text_auto=".3f",
        )
        lower = max(0.0, np.floor(results[METRIC_COLUMNS].min().min() * 20) / 20 - 0.05)
        fig.update_yaxes(range=[lower, 1.02])
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(style_figure(fig, 480), width="stretch")
        st.caption(f"Y-axis starts at {lower:.2f} to make differences visible.")
    with tab_roc:
        st.plotly_chart(roc_figure(curves), width="stretch")


def page_about() -> None:
    """Project background."""
    hero("About", "Dataset, preprocessing and models used in this project.")
    left, right = st.columns(2)
    with left.container(border=True):
        section("Dataset")
        st.markdown(
            "- **NSL-KDD** (`KDDTrain+`), 125,973 connections\n"
            "- 41 features (38 numeric, 3 categorical)\n"
            "- Target: Normal = 0, Attack = 1 (all attack types merged)"
        )
        section("Preprocessing")
        st.markdown(
            "- Duplicate removal and missing-value imputation\n"
            "- Label encoding: `protocol_type`, `service`, `flag`\n"
            "- `StandardScaler` on all features\n"
            "- Stratified 80/20 split, `random_state=42`"
        )
    with right.container(border=True):
        section("Models")
        st.markdown(
            "- Logistic Regression\n"
            "- Decision Tree\n"
            "- K-Nearest Neighbors (k = 5)\n"
            "- Gaussian Naive Bayes\n"
            "- Support Vector Machine (RBF kernel)"
        )
        section("Metrics")
        st.markdown(
            "Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation "
            "Coefficient (Attack is the positive class)."
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """App entry point."""
    st.markdown(CSS, unsafe_allow_html=True)
    page, df, model_name = render_sidebar()

    if page == "Overview":
        page_overview()
    elif page == "About":
        page_about()
    elif df is None:
        hero(page, "Upload a test CSV to get started.")
        st.info(
            "👈 Upload a test CSV in the sidebar, or switch on **Use bundled "
            "sample**."
        )
    else:
        if page == "Model Evaluation":
            page_evaluation(df, model_name)
        else:
            page_comparison(df)
        with st.expander(
            f"Uploaded data preview ({df.shape[0]:,} rows × {df.shape[1]} columns)"
        ):
            st.dataframe(df.head(20), width="stretch")


if __name__ == "__main__":
    main()
