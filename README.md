# Amazon Fine Food Reviews — Sentiment Analysis & NLP Project
This project analyzes the Amazon Fine Food Reviews dataset and builds a sentiment classification pipeline using Python and Natural Language Processing. The goal is to understand how customers describe their experiences with food products and to predict whether a review is positive, neutral, or negative based on its text.

The work includes data cleaning, text preprocessing, exploratory analysis, feature engineering, and model training. Everything is organized so the project is easy to understand and run.

# Project Overview

The dataset contains thousands of food product reviews from Amazon. Each review includes text, rating scores, product details, helpfulness votes, and timestamps.

In this project, I focused on:

Cleaning messy text
Understanding review patterns
Identifying sentiment drivers
Converting text to numeric features
Training machine learning models for sentiment prediction
Evaluating the performance of each model

# Tech Stack

Python
Pandas & NumPy
Matplotlib & Seaborn
NLTK / spaCy
Scikit-learn
TF-IDF Vectorizer
Logistic Regression / Random Forest / Naive Bayes
Jupyter Notebook / VS Code

# Key Steps

1. Data Cleaning
Removed duplicates
Filled or removed missing values
Stripped HTML tags and punctuation
Normalized text (lowercase, tokenization, stopwords, lemmatization)

2. Exploratory Data Analysis
Rating distribution
Most common positive and negative words
Word clouds
Review trends by year
Helpful vs non-helpful reviews

3. Text Preprocessing
Tokenization
Lemmatization
Stopword removal
TF-IDF vectorization

4. Sentiment Labeling
Converted star ratings into sentiment categories:
1–2 stars → Negative
3 stars → Neutral
4–5 stars → Positive

5. Model Development
Multiple ML models were trained and compared:
Logistic Regression
Multinomial Naive Bayes
Random Forest

6. Evaluation Metrics
Accuracy
Precision
Recall
F1 score
Confusion Matrix

# Results
The LinearSVC model combined with TF-IDF features delivered the best performance in this project. It handled long reviews, informal language, punctuation-heavy text, and subtle tone differences with strong accuracy and balanced evaluation metrics.


## Model Performance  
![Accuracy](https://img.shields.io/badge/Accuracy-95.03%25-2ecc71?style=for-the-badge)
![Precision](https://img.shields.io/badge/Precision%20(Neg)-86.90%25-f1c40f?style=for-the-badge)
![Recall](https://img.shields.io/badge/Recall%20(Neg)-79.99%25-e67e22?style=for-the-badge)
![F1](https://img.shields.io/badge/F1%20(Neg)-83.32%25-3498db?style=for-the-badge)


The classification report showed high reliability on positive reviews (precision 0.96, recall 0.98) and consistent performance on negative reviews, which are typically harder to classify. The confusion matrix confirms that the model was effective in separating positive and negative sentiment across more than 105,000 test samples.

LinearSVC was selected as the final model because it achieved the strongest balance of accuracy, robustness, and speed on this high-dimensional text data.

