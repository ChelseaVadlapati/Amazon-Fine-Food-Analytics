import streamlit as st
import joblib
import numpy as np

from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Amazon Fine Food Sentiment Analyzer",
    page_icon="🍽️",
    layout="centered"
)

# -----------------------------
# Load model & vectorizer
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_path = Path("Models/best_sentiment_model.pkl")
    vectorizer_path = Path("Models/tfidf_vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("⚠️ Failed to load model or vectorizer. Please check the file paths in `Models/`.")
    st.exception(e)
    st.stop()

# -----------------------------
# Helper: predict sentiment
# -----------------------------
def predict_sentiment(review_text: str):
    X = vectorizer.transform([review_text])
    pred = model.predict(X)[0]  # assumes 0 = negative, 1 = positive

    # Some models may return np.int64, convert to python int
    pred = int(pred)

    if pred == 1:
        label = "Positive"
        emoji = "😄"
        color = "#16a34a"  # green
        desc = "The model predicts this review expresses a positive experience."
    else:
        label = "Negative"
        emoji = "☹️"
        color = "#dc2626"  # red
        desc = "The model predicts this review expresses a negative experience."

    # We can use decision_function as a rough confidence proxy if available
    try:
        margin = model.decision_function(X)[0]
        confidence = float(1 / (1 + np.exp(-abs(margin))))  # squashed 0–1
    except Exception:
        confidence = None

    return {
        "label": label,
        "emoji": emoji,
        "color": color,
        "desc": desc,
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
        "- See the model’s prediction and confidence."
    )
    st.markdown("---")
    st.markdown("**Project:** Amazon Fine Food Analytics\n\n"
                "**Author:** Thilisitha Chelsea Vadlapati")

# -----------------------------
# Main layout
# -----------------------------
st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 0.2rem;">Amazon Fine Food Sentiment Analyzer</h1>
    <p style="text-align: center; color: #6b7280; margin-top: 0;">
        Paste a customer review and let the model classify it as positive or negative.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# Example presets
examples = [
    "This product was amazing! Fresh, tasty, and arrived on time.",
    "The cookies were stale and the packaging was damaged. Very disappointed.",
    "It's okay, not great but not terrible either."
]

with st.expander("Need some sample text?"):
    st.write("Click a button to load an example review:")
    cols = st.columns(len(examples))
    if cols[0].button("Example 1"):
        st.session_state["review_text"] = examples[0]
    if cols[1].button("Example 2"):
        st.session_state["review_text"] = examples[1]
    if cols[2].button("Example 3"):
        st.session_state["review_text"] = examples[2]

# Text input
default_text = st.session_state.get(
    "review_text",
    ""
)

review_text = st.text_area(
    "Enter a food product review:",
    value=default_text,
    height=180,
    placeholder="Type or paste a review here, e.g. 'The pasta sauce was rich and full of flavor...'"
)

analyze_btn = st.button("🔎 Analyze Sentiment", use_container_width=True)

# -----------------------------
# Prediction section
# -----------------------------
if analyze_btn:
    if not review_text.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(review_text.strip())

        # Nice colored card for result
        card_html = f"""
        <div style="
            border-radius: 0.75rem;
            padding: 1.25rem 1.5rem;
            margin-top: 1rem;
            background: {result['color']}15;
            border: 1px solid {result['color']}55;
        ">
            <div style="font-size: 1rem; color: #6b7280; margin-bottom: 0.25rem;">
                Model prediction
            </div>
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 1.8rem;">{result['emoji']}</span>
                <span style="font-size: 1.4rem; font-weight: 600; color: {result['color']};">
                    {result['label']} review
                </span>
            </div>
            <p style="margin-top: 0.6rem; color: #374151; font-size: 0.98rem;">
                {result['desc']}
            </p>
        """

        if result["confidence"] is not None:
            conf_pct = int(result["confidence"] * 100)
            card_html += f"""
            <p style="margin-top: 0.2rem; color: #4b5563; font-size: 0.9rem;">
                <strong>Confidence:</strong> approximately {conf_pct}% based on the model's decision margin.
            </p>
            """

        card_html += "</div>"

        st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔍 Review text analyzed")
        st.write(review_text)
