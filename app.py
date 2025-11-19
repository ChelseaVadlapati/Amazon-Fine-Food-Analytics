import streamlit as st
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('Models/best_sentiment_model.pkl')
tfidf = joblib.load('Models/tfidf_vectorizer.pkl')

st.title("Amazon Review Sentiment Checker")

st.write("Paste an Amazon-style review below and I'll predict if it's positive or negative.")

user_input = st.text_area("Review text", height=150)

if st.button("Predict sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        X = tfidf.transform([user_input])
        pred = model.predict(X)[0]
        proba = None
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            pass

        label = "Positive 👍" if pred == 1 else "Negative 👎"
        st.subheader(f"Prediction: {label}")
        if proba is not None:
            st.write(f"Confidence: {max(proba):.2f}")
