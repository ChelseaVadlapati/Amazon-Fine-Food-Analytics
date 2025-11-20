<p align="center">
  <img src="Assests/Banner.png" alt="Amazon Fine Food Analytics Banner" width="100%">
</p>

![Python](https://img.shields.io/badge/Python-3.9-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset Size](https://img.shields.io/badge/Dataset-500k+_reviews-orange)
![Model](https://img.shields.io/badge/Model-LinearSVC-purple)
![NLP](https://img.shields.io/badge/Category-NLP-yellow)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Key Steps](#key-steps)
4. [Results](#results)
5. [How to Run the Project](#how-to-run-the-project)
6. [Streamlit App Demo](#streamlit-app-demo)
7. [Future Improvements](#future-improvements)
8. [Author](#author)

# Project Overview
This project analyzes the Amazon Fine Food Reviews dataset and builds a complete sentiment-classification pipeline using Python and Natural Language Processing. The goal is to predict whether a review is positive or negative using TF-IDF features and machine learning.
The workflow covers:
Data cleaning
Exploratory analysis
Text preprocessing
Feature engineering
Model development
Evaluation
Streamlit UI for easy predictions

# Tech Stack
Python
Pandas / NumPy
Matplotlib / Seaborn
NLTK / spaCy
Scikit-learn
TF-IDF Vectorizer
LinearSVC
Streamlit
Jupyter Notebook / VS Code

# Key Steps
Data Cleaning
Remove duplicates
Handle missing values
Clean HTML, symbols, punctuation
Normalize text (lowercasing, lemmatization, stopwords)
Exploratory Data Analysis
Rating distribution
Frequent words
Word clouds
Trends by year
Helpfulness analysis
Text Preprocessing
Tokenization
Stopword removal
Lemmatization
TF-IDF vectorization
Model Development
Models tested:
Logistic Regression
Multinomial Naive Bayes
Random Forest
LinearSVC (best)
Evaluation Metrics
Accuracy
Precision
Recall
F1-score
Confusion Matrix

# Results
Model Performance

LinearSVC with TF-IDF achieved:
Positive class: Precision 0.96, Recall 0.98
Negative class: Precision 0.87, Recall 0.80
Confusion Matrix
[[11314   3283]
 [ 1971  86785]]



## Streamlit App Demo

<p align="center">
  <img src="Assests/App Demo.gif" alt="Streamlit App Demo" width="70%">
</p>


