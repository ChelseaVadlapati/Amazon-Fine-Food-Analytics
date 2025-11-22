import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from sklearn.exceptions import NotFittedError
from textwrap import dedent


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Amazon Fine Food Sentiment Analyzer",
    page_icon="🍽️",
    layout="centered"
)

# -----------------------------
# Load model + vectorizer
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    base_dir = Path(__file__).parent
    model_path = base_dir / "Models" / "best_sentiment_model.pkl"
    vect_path = base_dir / "Models" / "tfidf_vectorizer.pkl"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)

    missing_attrs = [attr for attr in ("vocabulary_", "idf_") if not hasattr(vectorizer, attr)]
    if missing_attrs:
        raise NotFittedError(
            "The TF-IDF vectorizer is not fitted. Re-run the modeling notebook to regenerate "
            "Models/tfidf_vectorizer.pkl before launching the app."
        )

    return model, vectorizer



try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error(
        "⚠️ There was a problem loading the model artifacts. "
        "Check that the files exist in the **Models/** folder."
    )
    st.exception(e)
    st.stop()

# -----------------------------
# Helper: predict sentiment
# -----------------------------
def predict_sentiment(review_text: str):
    """Return label, emoji, description, and confidence for a given review."""
    X = vectorizer.transform([review_text])
    pred = int(model.predict(X)[0])  # assumes 0 = negative, 1 = positive

    if pred == 1:
        label = "Positive review"
        emoji = "😄"
        color = "#16a34a"  # green
        desc = "The model predicts this review expresses a positive experience."
    else:
        label = "Negative review"
        emoji = "☹️"
        color = "#dc2626"  # red
        desc = "The model predicts this review expresses a negative experience."

    # Use decision_function as a rough confidence proxy if available
    confidence = None
    try:
        margin = float(model.decision_function(X)[0])
        # squash to 0–1 range
        confidence = float(1 / (1 + np.exp(-abs(margin))))
    except Exception:
        pass

    return {
        "label": label,
        "emoji": emoji,
        "color": color,
        "description": desc,
        "confidence": confidence,
    }

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🍽️ About this app")
    st.write(
        "This app uses a LinearSVC model trained on the **Amazon Fine Food Reviews** "
        "dataset to predict whether a review is **positive** or **negative**."
    )

    st.markdown("### 🔍 How to use")
    st.markdown(
        "- Paste or type an Amazon-style food review.\n"
        "- Click **Analyze Sentiment**.\n"
        "- View the model’s prediction and confidence.\n"
        "- Scroll down to see the analyzed review text."
    )

    st.markdown("---")
    st.markdown(
        "**Project:** Amazon Fine Food Analytics\n\n"
        "**Author:** Thilisitha Chelsea Vadlapati"
    )
    st.markdown(
        "[GitHub repo](https://github.com/ChelseaVadlapati/Amazon-Fine-Food-Analytics)"
    )

# -----------------------------
# Main layout
# -----------------------------
st.title("Amazon Fine Food Sentiment Analyzer")
st.caption("Paste a review, hit analyze, and see how the model interprets it.")

# Sample text helper
with st.expander("Need some sample text?"):
    st.markdown(
        "- *“The pasta sauce was rich and flavorful, I’ll definitely buy again.”*\n"
        "- *“The chips arrived stale and the bag was half empty.”*"
    )

review_text = st.text_area(
    "Enter a food product review:",
    height=180,
    placeholder="Type or paste an Amazon-style food review here...",
)

analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True)

if analyze_clicked:
    if not review_text.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        result = predict_sentiment(review_text)

        # Build a styled prediction card
        base_bg = "#022c22"  # dark teal
        border = result["color"]

        card_html = dedent(
            f"""
            <div style="
                margin-top: 1.5rem;
                padding: 1.25rem 1.5rem;
                border-radius: 12px;
                border: 1px solid {border};
                background: {base_bg};
            ">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.8rem;">{result["emoji"]}</span>
                    <div>
                        <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: {border};">
                            {result["label"]}
                        </p>
                        <p style="margin: 0.25rem 0 0; font-size: 0.9rem; color: #e5e7eb;">
                            {result["description"]}
                        </p>
                    </div>
                </div>
            """
        ).strip()

        if result["confidence"] is not None:
            conf_pct = int(result["confidence"] * 100)
            card_html += dedent(
                f"""
                <div style="margin-top: 0.75rem; padding: 0.5rem 0.75rem; border-radius: 8px; background: #020617;">
                    <p style="margin: 0; font-size: 0.9rem; color: #d1d5db;">
                        <strong>Confidence:</strong> approximately {conf_pct}% based on the model's decision margin.
                    </p>
                </div>
                """
            ).strip()

        card_html += "</div>"

        st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔎 Review text analyzed")
        st.write(review_text)
