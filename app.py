import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
from collections import Counter


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HTML RENDERING FIX
# ============================================================
# Markdown treats indented lines and blank lines inside an HTML
# block as code, which makes raw <div> tags appear on screen.
# This removes indentation and blank lines from HTML strings.

_original_markdown = st.markdown


def _markdown_with_clean_html(body, *args, **kwargs):

    if kwargs.get("unsafe_allow_html") and isinstance(body, str):

        body = "\n".join(
            line.strip()
            for line in body.splitlines()
            if line.strip()
        )

    return _original_markdown(body, *args, **kwargs)


st.markdown = _markdown_with_clean_html


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
    ====================================================== */

    .stApp {
        background-color: #f4f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    header {
        visibility: hidden;
    }

    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    .sidebar-logo {
        text-align: center;
        padding: 10px 0 20px 0;
    }

    .sidebar-logo-icon {
        font-size: 3rem;
    }

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 5px;
    }

    .sidebar-subtitle {
        color: #9ca3af;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    .sidebar-info {
        background: #1f2937;
        border-radius: 12px;
        padding: 12px;
        margin-top: 15px;
        border: 1px solid #374151;
    }

    .sidebar-info-title {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 4px;
    }

    .sidebar-info-value {
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* ======================================================
       HERO
    ====================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 45%,
            #2563eb 100%
        );

        border-radius: 20px;
        padding: 2.3rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;

        box-shadow:
            0 10px 30px rgba(15, 23, 42, 0.15);
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 0.78rem;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: 2.4rem;
        font-weight: 750;
        margin: 0;
        color: white;
    }

    .hero p {
        margin-top: 10px;
        color: #dbeafe;
        font-size: 1.02rem;
        max-width: 800px;
        line-height: 1.6;
    }

    /* ======================================================
       PAGE HEADER
    ====================================================== */

    .page-header {
        margin-bottom: 1.2rem;
    }

    .page-title {
        font-size: 1.8rem;
        font-weight: 750;
        color: #111827;
        margin-bottom: 4px;
    }

    .page-description {
        color: #64748b;
        font-size: 0.95rem;
    }

    /* ======================================================
       KPI CARDS
    ====================================================== */

    .kpi-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 135px;

        box-shadow:
            0 4px 14px rgba(15,23,42,0.05);
    }

    .kpi-icon {
        font-size: 1.35rem;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 1.55rem;
        font-weight: 750;
        color: #111827;
    }

    .kpi-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* ======================================================
       CARDS
    ====================================================== */

    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.35rem;
        box-shadow:
            0 4px 14px rgba(15,23,42,0.04);
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 7px;
    }

    .card-text {
        color: #64748b;
        line-height: 1.65;
        font-size: 0.9rem;
    }

    /* ======================================================
       PIPELINE
    ====================================================== */

    .pipeline-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.3rem;
        box-shadow: 0 4px 14px rgba(15,23,42,0.04);
    }

    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 15px;
    }

    .pipeline-step {
        background: #f8fafc;
        border: 1px solid #dbe3ef;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.82rem;
        color: #334155;
        font-weight: 600;
    }

    .pipeline-arrow {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* ======================================================
       RESULT BOXES
    ====================================================== */

    .positive-result {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 1.3rem;
        margin-top: 1rem;
    }

    .negative-result {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 6px solid #ef4444;
        border-radius: 14px;
        padding: 1.3rem;
        margin-top: 1rem;
    }

    .result-title {
        font-size: 1.4rem;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .result-description {
        color: #475569;
        font-size: 0.9rem;
    }

    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        border-top: 1px solid #e5e7eb;
        padding-top: 20px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_excel(
        "imdb_combined.xlsx"
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # best_sentiment_model.joblib is a saved Pipeline:
    # TfidfVectorizer -> LinearSVC (the vectorizer is inside it)
    pipeline = joblib.load(
        "best_sentiment_model.joblib"
    )

    vectorizer = pipeline.named_steps["tfidf"]

    model = pipeline.named_steps["svm"]

    return vectorizer, model


# ============================================================
# LOAD APPLICATION
# ============================================================

try:

    df = load_data()

    vectorizer, model = load_model()

except Exception as e:

    st.error(
        "Unable to load the application files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "review",
    "sentiment"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(
        f"Missing columns: {missing}"
    )

    st.stop()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PREDICTION
# ============================================================

def predict_sentiment(review):

    cleaned = preprocess_text(
        review
    )

    vectorized = vectorizer.transform(
        [cleaned]
    )

    prediction = model.predict(
        vectorized
    )[0]

    confidence = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            vectorized
        )[0]

        confidence = float(
            np.max(probabilities)
        )

    return prediction, confidence


# ============================================================
# EDA PREPARATION
# ============================================================

@st.cache_data
def prepare_eda(data):

    data = data.copy()

    data["review_length"] = (
        data["review"]
        .astype(str)
        .str.len()
    )

    data["word_count"] = (
        data["review"]
        .astype(str)
        .str.split()
        .str.len()
    )

    return data


eda = prepare_eda(df)


# ============================================================
# TOP WORDS
# ============================================================

@st.cache_data
def top_words(
    data,
    sentiment=None,
    n=15
):

    if sentiment:

        texts = data.loc[
            data["sentiment"] == sentiment,
            "review"
        ].astype(str)

    else:

        texts = data[
            "review"
        ].astype(str)

    combined_text = " ".join(
        texts
    ).lower()

    words = re.findall(
        r"\b[a-z]+\b",
        combined_text
    )

    try:

        from sklearn.feature_extraction.text import (
            ENGLISH_STOP_WORDS
        )

        stop_words = set(
            ENGLISH_STOP_WORDS
        )

    except Exception:

        stop_words = set()

    words = [
        word
        for word in words
        if word not in stop_words
        and len(word) > 1
    ]

    counts = Counter(
        words
    ).most_common(n)

    return pd.DataFrame(
        counts,
        columns=[
            "word",
            "frequency"
        ]
    )


# ============================================================
# PROFESSIONAL CHART STYLE
# ============================================================

COLORS = {
    "pos": "#10b981",
    "neg": "#ef4444",
    "primary": "#2563eb",
    "ink": "#111827",
    "muted": "#64748b",
    "grid": "#e5e7eb"
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": COLORS["ink"],
    "axes.labelcolor": COLORS["muted"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})


def new_figure(width=6.4, height=3.8):

    fig, ax = plt.subplots(
        figsize=(width, height)
    )

    return fig, ax


def style_axes(ax, grid_axis="y"):

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["grid"])

    ax.grid(
        axis=grid_axis,
        color=COLORS["grid"],
        linewidth=0.8
    )

    ax.set_axisbelow(True)

    ax.tick_params(length=0)


def show_figure(fig):

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# ============================================================
# DATASET STATISTICS
# ============================================================

total_reviews = len(df)

positive_reviews = (
    df["sentiment"] == "pos"
).sum()

negative_reviews = (
    df["sentiment"] == "neg"
).sum()

positive_percentage = (
    positive_reviews /
    total_reviews *
    100
)

negative_percentage = (
    negative_reviews /
    total_reviews *
    100
)

average_words = (
    eda["word_count"].mean()
)

average_characters = (
    eda["review_length"].mean()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">

            <div class="sidebar-logo-icon">
                🎬
            </div>

            <div class="sidebar-title">
                IMDB Sentiment Analysis
            </div>

            <div class="sidebar-subtitle">
                Movie Review Classification System
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dashboard",
            "💬 Prediction",
            "📁 Data Explorer",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-info">

            <div class="sidebar-info-title">
                MODEL
            </div>

            <div class="sidebar-info-value">
                Linear SVM
            </div>

        </div>

        <div class="sidebar-info">

            <div class="sidebar-info-title">
                FEATURE EXTRACTION
            </div>

            <div class="sidebar-info-value">
                TF-IDF
            </div>

        </div>

        <div class="sidebar-info">

            <div class="sidebar-info-title">
                TASK
            </div>

            <div class="sidebar-info-value">
                Binary Sentiment Classification
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DASHBOARD HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        f"""
        <div class="hero">

            <div class="hero-badge">
                🤖 CLASSICAL NLP • MACHINE LEARNING
            </div>

            <h1>
                IMDB Sentiment Analysis
            </h1>

            <p>
                An interactive Natural Language Processing
                application that analyzes movie reviews and
                classifies them into positive or negative sentiment.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # KPI SECTION
    # ========================================================

    st.markdown(
        "### 📊 Dataset Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    kpis = [
        (
            c1,
            "🎬",
            f"{total_reviews:,}",
            "Total Reviews"
        ),

        (
            c2,
            "😊",
            f"{positive_reviews:,}",
            "Positive Reviews"
        ),

        (
            c3,
            "😞",
            f"{negative_reviews:,}",
            "Negative Reviews"
        ),

        (
            c4,
            "⚖️",
            f"{positive_percentage:.1f}%",
            "Positive Share"
        )
    ]

    for col, icon, value, label in kpis:

        with col:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-icon">
                        {icon}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # QUICK ANALYTICS
    # ========================================================

    st.markdown(
        "### 🔎 Quick Dataset Insights"
    )

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Average Words / Review",
        f"{average_words:.1f}"
    )

    q2.metric(
        "Average Characters / Review",
        f"{average_characters:.0f}"
    )

    q3.metric(
        "Sentiment Classes",
        "2"
    )


    # ========================================================
    # PIPELINE
    # ========================================================

    st.markdown(
        "### 🔄 NLP Processing Pipeline"
    )

    st.markdown(
        """
        <div class="pipeline-container">

            <div class="pipeline">

                <div class="pipeline-step">
                    📝 Review Input
                </div>

                <div class="pipeline-arrow">
                    →
                </div>

                <div class="pipeline-step">
                    🧹 Text Cleaning
                </div>

                <div class="pipeline-arrow">
                    →
                </div>

                <div class="pipeline-step">
                    🔢 TF-IDF
                </div>

                <div class="pipeline-arrow">
                    →
                </div>

                <div class="pipeline-step">
                    🤖 Linear SVM
                </div>

                <div class="pipeline-arrow">
                    →
                </div>

                <div class="pipeline-step">
                    🎯 Prediction
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # ABOUT CARDS
    # ========================================================

    st.markdown(
        "### 🧠 Application Overview"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🤖 Machine Learning Model
                </div>

                <div class="card-text">

                    The application uses TF-IDF to transform
                    textual reviews into numerical features.
                    Linear SVM then uses these features
                    to classify the review sentiment.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    📚 Natural Language Processing
                </div>

                <div class="card-text">

                    Reviews are converted to lowercase,
                    HTML tags and unnecessary characters are
                    removed, and the cleaned text is passed
                    through the trained vectorizer.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.info(
        "💡 Use **Prediction** in the sidebar to test "
        "the trained model with your own movie review."
    )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

elif page == "💬 Prediction":

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                💬 Prediction
            </div>

            <div class="page-description">
                Enter a movie review and analyse its predicted sentiment.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # INPUT
    # ========================================================

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                📝 Movie Review
            </div>

            <div class="card-text">
                Write or paste a movie review below.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    review = st.text_area(
        "Review",
        height=190,
        placeholder=(
            "Example: I absolutely loved this movie. "
            "The acting was excellent, the story was engaging, "
            "and I would definitely recommend it."
        ),
        label_visibility="collapsed"
    )


    # ========================================================
    # TEXT STATISTICS
    # ========================================================

    if review.strip():

        word_count = len(
            review.split()
        )

        character_count = len(
            review
        )

        a, b = st.columns(2)

        a.metric(
            "Words",
            word_count
        )

        b.metric(
            "Characters",
            character_count
        )


    # ========================================================
    # BUTTON
    # ========================================================

    analyze = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )


    if analyze:

        if not review.strip():

            st.warning(
                "Please enter a movie review first."
            )

        elif len(review.strip()) < 3:

            st.warning(
                "Please enter a longer review."
            )

        else:

            prediction, confidence = (
                predict_sentiment(
                    review
                )
            )


            # =================================================
            # RESULT
            # =================================================

            if prediction == "pos":

                st.markdown(
                    """
                    <div class="positive-result">

                        <div class="result-title">
                            😊 Positive Sentiment
                        </div>

                        <div class="result-description">
                            The trained model predicts that this
                            movie review expresses positive sentiment.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                label = "Positive"

            else:

                st.markdown(
                    """
                    <div class="negative-result">

                        <div class="result-title">
                            😞 Negative Sentiment
                        </div>

                        <div class="result-description">
                            The trained model predicts that this
                            movie review expresses negative sentiment.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                label = "Negative"


            # =================================================
            # CONFIDENCE
            # =================================================

            if confidence is not None:

                st.markdown(
                    "### 📊 Model Confidence"
                )

                st.progress(
                    confidence
                )

                st.metric(
                    "Predicted Probability",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "This value represents the model's predicted "
                    "probability for the selected class."
                )


            # =================================================
            # DETAILS
            # =================================================

            st.markdown(
                "### 🔎 Analysis Details"
            )

            d1, d2 = st.columns(2)

            with d1:

                with st.expander(
                    "View Preprocessed Text"
                ):

                    st.code(
                        preprocess_text(
                            review
                        )
                    )

            with d2:

                with st.expander(
                    "View Prediction Pipeline"
                ):

                    st.write(
                        "1. Review input"
                    )

                    st.write(
                        "2. Text preprocessing"
                    )

                    st.write(
                        "3. TF-IDF transformation"
                    )

                    st.write(
                        "4. Linear SVM"
                    )

                    st.write(
                        f"5. Prediction → **{label}**"
                    )


# ============================================================
# EDA ANALYTICS
# ============================================================

elif page == "📊 Dashboard":

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                📊 Dashboard
            </div>

            <div class="page-description">
                Exploratory data analysis and model performance for the
                IMDB review dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    median_words = eda["word_count"].median()

    k1, k2, k3, k4 = st.columns(4)

    for col, icon, value, label in [
        (k1, "🎬", f"{total_reviews:,}", "Total Reviews"),
        (k2, "⚖️", f"{positive_percentage:.1f}% / {negative_percentage:.1f}%", "Positive / Negative Split"),
        (k3, "📝", f"{average_words:.0f}", "Average Words per Review"),
        (k4, "📏", f"{median_words:.0f}", "Median Words per Review")
    ]:

        with col:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-icon">
                        {icon}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    tab_overview, tab_length, tab_vocab, tab_model = st.tabs(
        [
            "Overview",
            "Review Length",
            "Vocabulary",
            "Model Performance"
        ]
    )

    pos_words = eda.loc[eda["sentiment"] == "pos", "word_count"]
    neg_words = eda.loc[eda["sentiment"] == "neg", "word_count"]
    upper_limit = float(eda["word_count"].quantile(0.99))


    # ========================================================
    # TAB 1: OVERVIEW
    # ========================================================

    with tab_overview:

        c1, c2 = st.columns(2)

        with c1:

            counts = (
                eda["sentiment"]
                .value_counts()
                .reindex(["pos", "neg"])
            )

            fig = plt.figure(figsize=(6.4, 3.8))

            ax = fig.add_axes([0.0, 0.02, 0.52, 0.96])

            ax.pie(
                counts.values,
                colors=[COLORS["pos"], COLORS["neg"]],
                startangle=90,
                counterclock=False,
                wedgeprops=dict(
                    width=0.36,
                    edgecolor="white",
                    linewidth=3
                )
            )

            ax.text(
                0, 0.07,
                f"{total_reviews:,}",
                ha="center",
                fontsize=16,
                fontweight="bold",
                color=COLORS["ink"]
            )

            ax.text(
                0, -0.2,
                "reviews",
                ha="center",
                fontsize=10,
                color=COLORS["muted"]
            )

            fig.text(
                0.56, 0.93,
                "Sentiment Class Balance",
                fontsize=12,
                fontweight="bold",
                color=COLORS["ink"]
            )

            for position, key, name in (
                (0.62, "pos", "Positive"),
                (0.36, "neg", "Negative")
            ):

                fig.text(
                    0.56, position,
                    "\u25A0",
                    fontsize=16,
                    color=COLORS[key]
                )

                fig.text(
                    0.62, position + 0.035,
                    name,
                    fontsize=11,
                    fontweight="bold",
                    color=COLORS["ink"]
                )

                fig.text(
                    0.62, position - 0.045,
                    f"{counts[key]:,}  \u2022  "
                    f"{counts[key] / counts.sum() * 100:.1f}%",
                    fontsize=10,
                    color=COLORS["muted"]
                )

            st.pyplot(fig, width="stretch")

            plt.close(fig)

            st.caption(
                f"{positive_reviews:,} positive and "
                f"{negative_reviews:,} negative reviews: "
                "the classes are almost perfectly balanced."
            )

        with c2:

            means = (
                eda.groupby("sentiment")["word_count"]
                .mean()
                .reindex(["pos", "neg"])
            )

            medians = (
                eda.groupby("sentiment")["word_count"]
                .median()
                .reindex(["pos", "neg"])
            )

            x = np.arange(2)

            fig, ax = new_figure()

            b1 = ax.bar(
                x - 0.19,
                means.values,
                0.36,
                label="Mean",
                color=[COLORS["pos"], COLORS["neg"]]
            )

            b2 = ax.bar(
                x + 0.19,
                medians.values,
                0.36,
                label="Median",
                color=[COLORS["pos"], COLORS["neg"]],
                alpha=0.45
            )

            for bars in (b1, b2):

                for bar in bars:

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        f"{bar.get_height():.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=COLORS["ink"]
                    )

            ax.set_xticks(x)

            ax.set_xticklabels(["Positive", "Negative"])

            ax.set_ylabel("Words per review")

            ax.set_ylim(0, float(means.max()) * 1.28)

            ax.set_title("Mean vs Median Review Length")

            style_axes(ax)

            ax.text(
                0.99, 0.97,
                "solid = mean   faded = median",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=COLORS["muted"]
            )

            show_figure(fig)

            st.caption(
                f"Positive reviews average {means['pos']:.0f} words and "
                f"negative reviews {means['neg']:.0f}: review length "
                "says little about sentiment."
            )


    # ========================================================
    # TAB 2: REVIEW LENGTH
    # ========================================================

    with tab_length:

        c1, c2 = st.columns(2)

        with c1:

            bins = np.linspace(0, upper_limit, 45)

            fig, ax = new_figure()

            ax.hist(
                pos_words[pos_words <= upper_limit],
                bins=bins,
                alpha=0.6,
                color=COLORS["pos"],
                label="Positive"
            )

            ax.hist(
                neg_words[neg_words <= upper_limit],
                bins=bins,
                alpha=0.6,
                color=COLORS["neg"],
                label="Negative"
            )

            ax.axvline(
                eda["word_count"].median(),
                color=COLORS["ink"],
                linestyle="--",
                linewidth=1.2
            )

            ax.text(
                eda["word_count"].median() + upper_limit * 0.01,
                ax.get_ylim()[1] * 0.93,
                f"median {median_words:.0f}",
                fontsize=9,
                color=COLORS["ink"]
            )

            ax.set_xlabel("Words per review")

            ax.set_ylabel("Number of reviews")

            ax.set_title("Review Length Distribution by Sentiment")

            ax.legend(frameon=False)

            style_axes(ax)

            show_figure(fig)

            st.caption(
                "Most reviews are short to medium length with a long tail "
                "of very long ones (the longest 1% of reviews are left out for readability)."
            )

        with c2:

            fig, ax = new_figure()

            box = ax.boxplot(
                [pos_words, neg_words],
                vert=False,
                patch_artist=True,
                showfliers=False,
                widths=0.5,
                medianprops=dict(
                    color=COLORS["ink"],
                    linewidth=2
                ),
                whiskerprops=dict(color=COLORS["muted"]),
                capprops=dict(color=COLORS["muted"])
            )

            for patch, color in zip(
                box["boxes"],
                [COLORS["pos"], COLORS["neg"]]
            ):

                patch.set_facecolor(color)

                patch.set_alpha(0.65)

                patch.set_edgecolor(color)

            ax.set_yticks([1, 2])

            ax.set_yticklabels(["Positive", "Negative"])

            ax.set_xlabel("Words per review")

            ax.set_title("Word Count Spread by Sentiment")

            style_axes(ax, grid_axis="x")

            show_figure(fig)

            st.caption(
                "The boxes show the middle 50% of reviews. Both classes "
                "have a very similar spread."
            )

        fig, ax = new_figure(12.8, 3.2)

        chars_cap = float(eda["review_length"].quantile(0.99))

        ax.hist(
            eda.loc[eda["review_length"] <= chars_cap, "review_length"],
            bins=60,
            color=COLORS["primary"],
            alpha=0.85,
            edgecolor="white"
        )

        ax.set_xlabel("Characters per review")

        ax.set_ylabel("Number of reviews")

        ax.set_title("Review Length in Characters")

        style_axes(ax)

        show_figure(fig)


    # ========================================================
    # TAB 3: VOCABULARY
    # ========================================================

    with tab_vocab:

        c1, c2 = st.columns(2)

        for column, key, title, color in (
            (c1, "pos", "Top 15 Words in Positive Reviews", COLORS["pos"]),
            (c2, "neg", "Top 15 Words in Negative Reviews", COLORS["neg"])
        ):

            with column:

                words = (
                    top_words(eda, key, 15)
                    .sort_values("frequency")
                )

                fig, ax = new_figure(6.4, 4.6)

                bars = ax.barh(
                    words["word"],
                    words["frequency"],
                    color=color,
                    height=0.68
                )

                for bar in bars:

                    ax.text(
                        bar.get_width() + words["frequency"].max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(bar.get_width()):,}",
                        va="center",
                        fontsize=8,
                        color=COLORS["muted"]
                    )

                ax.set_xlim(0, words["frequency"].max() * 1.12)

                ax.set_xlabel("Frequency")

                ax.set_title(title)

                style_axes(ax, grid_axis="x")

                show_figure(fig)

        overall = top_words(eda, None, 20)

        fig, ax = new_figure(12.8, 3.8)

        bars = ax.bar(
            overall["word"],
            overall["frequency"],
            color=COLORS["primary"],
            width=0.7
        )

        ax.tick_params(axis="x", rotation=40)

        ax.set_ylabel("Frequency")

        ax.set_title("Top 20 Most Frequent Words (stop words removed)")

        style_axes(ax)

        show_figure(fig)

        st.caption(
            "Frequent words are mostly movie vocabulary shared by both "
            "classes, so frequency alone does not indicate sentiment."
        )


    # ========================================================
    # TAB 4: MODEL PERFORMANCE
    # ========================================================

    with tab_model:

        # Test-set results (9,917 reviews) from the Feature
        # Engineering & Model Development notebook.
        results = pd.DataFrame(
            {
                "Model": [
                    "Naive Bayes",
                    "Logistic Regression",
                    "Linear SVM"
                ],
                "Accuracy": [0.8889, 0.9074, 0.9119],
                "F1-score": [0.8891, 0.9087, 0.9131]
            }
        )

        c1, c2 = st.columns([3, 2])

        with c1:

            x = np.arange(len(results))

            fig, ax = new_figure(7.6, 4.0)

            b1 = ax.bar(
                x - 0.2,
                results["Accuracy"] * 100,
                0.38,
                label="Accuracy",
                color=COLORS["primary"]
            )

            b2 = ax.bar(
                x + 0.2,
                results["F1-score"] * 100,
                0.38,
                label="F1-score",
                color="#93c5fd"
            )

            for bars in (b1, b2):

                for bar in bars:

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f"{bar.get_height():.1f}",
                        ha="center",
                        fontsize=9,
                        color=COLORS["ink"]
                    )

            ax.set_xticks(x)

            ax.set_xticklabels(results["Model"])

            ax.set_ylim(85, 93)

            ax.set_ylabel("Score (%)")

            ax.set_title("Tuned Models Compared on the Test Set")

            ax.legend(frameon=False, loc="upper left")

            style_axes(ax)

            show_figure(fig)

            st.caption(
                "Linear SVM scored highest and is the model used in this "
                "application. The y-axis starts at 85% to make the "
                "differences visible."
            )

        with c2:

            tfidf_params = vectorizer.get_params()

            details = pd.DataFrame(
                {
                    "Setting": [
                        "Final model",
                        "Feature extraction",
                        "N-gram range",
                        "Minimum document frequency",
                        "Vocabulary size",
                        "SVM regularization (C)",
                        "Test accuracy",
                        "Test F1-score"
                    ],
                    "Value": [
                        "Linear SVM",
                        "TF-IDF",
                        str(tfidf_params["ngram_range"]),
                        str(tfidf_params["min_df"]),
                        f"{len(vectorizer.vocabulary_):,}",
                        str(model.C),
                        "91.19%",
                        "91.31%"
                    ]
                }
            )

            st.markdown("**Final model configuration**")

            st.dataframe(
                details,
                width="stretch",
                hide_index=True
            )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📁 Data Explorer":

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                📁 Data Explorer
            </div>

            <div class="page-description">
                Search, filter and inspect the IMDB review dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Rows",
        f"{total_reviews:,}"
    )

    c2.metric(
        "Positive",
        f"{positive_reviews:,}"
    )

    c3.metric(
        "Negative",
        f"{negative_reviews:,}"
    )


    st.markdown("---")


    # ========================================================
    # FILTERS
    # ========================================================

    f1, f2 = st.columns(2)

    with f1:

        sentiment_filter = st.selectbox(
            "Sentiment Filter",
            [
                "All",
                "Positive",
                "Negative"
            ]
        )

    with f2:

        number_rows = st.slider(
            "Rows to Display",
            5,
            100,
            10
        )


    # ========================================================
    # SEARCH
    # ========================================================

    search = st.text_input(
        "🔎 Search Reviews",
        placeholder=(
            "Search for a word or phrase..."
        )
    )


    # ========================================================
    # FILTER
    # ========================================================

    view = df.copy()


    if sentiment_filter == "Positive":

        view = view[
            view["sentiment"] == "pos"
        ]


    elif sentiment_filter == "Negative":

        view = view[
            view["sentiment"] == "neg"
        ]


    if search.strip():

        view = view[
            view["review"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]


    # ========================================================
    # RESULTS
    # ========================================================

    st.caption(
        f"Showing {min(len(view), number_rows):,} "
        f"of {len(view):,} matching reviews"
    )


    columns_to_show = [
        "review",
        "sentiment"
    ]

    if "split" in view.columns:

        columns_to_show.append(
            "split"
        )


    st.dataframe(
        view[
            columns_to_show
        ].head(number_rows),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT PAGE
# ============================================================

else:

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                ℹ️ About the Application
            </div>

            <div class="page-description">
                Information about the NLP system and its methodology.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🎯 Objective
                </div>

                <div class="card-text">

                    The objective of this application is to classify
                    IMDB movie reviews into two sentiment categories:
                    positive and negative.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🧠 NLP Method
                </div>

                <div class="card-text">

                    The system uses classical NLP preprocessing,
                    TF-IDF feature extraction and Linear SVM
                    for binary text classification.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "### 🔄 Methodology"
    )

    st.info(
        "Raw Review → Cleaning → TF-IDF Feature Extraction → "
        "Linear SVM → Sentiment Classification"
    )


    st.markdown(
        "### 📌 Application Components"
    )

    components = pd.DataFrame(
        {
            "Component": [
                "Dataset",
                "Preprocessing",
                "Feature Extraction",
                "Machine Learning Model",
                "Output"
            ],

            "Technology": [
                "IMDB Movie Reviews",
                "Python / Regular Expressions",
                "TF-IDF",
                "Linear SVM",
                "Positive / Negative"
            ]
        }
    )

    st.dataframe(
        components,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎬 <b>IMDB Sentiment Analysis</b>
        &nbsp; • &nbsp;
        Classical NLP Application
        &nbsp; • &nbsp;
        TF-IDF + Linear SVM

        <br><br>

        Developed as a Natural Language Processing project

    </div>
    """,
    unsafe_allow_html=True
)