# 📰 Fake News Detection App

A Streamlit web app that classifies news text as **FAKE** or **REAL** using an NLP + Machine Learning pipeline.

## Features
- Paste a headline or article text
- Predict **FAKE / REAL**
- Shows model confidence (if available)

## Tech Stack
- Python, Streamlit
- scikit-learn (TF-IDF + Logistic Regression)
- joblib (model persistence)

## How to run locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_and_save.py
streamlit run app.py
