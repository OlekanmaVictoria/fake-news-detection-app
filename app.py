import os
import joblib
import streamlit as st

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------- HEADER ----------
st.title("📰 Fake News Detection")
st.write(
    "Paste a news headline or article text below. "
    "The system predicts whether the content is **FAKE** or **REAL**."
)

# ---------- LOAD MODEL ----------
MODEL_PATH = "fake_news_pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please train the model first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------- EXAMPLES ----------
with st.expander("📌 Try an example"):
    st.markdown("""
**Example Fake:**  
> *Breaking: Scientists confirm the Earth will stop spinning next week.*

**Example Real:**  
> *The UK government announced new funding for renewable energy projects.*
""")

# ---------- INPUT ----------
text = st.text_area(
    "News text",
    height=220,
    placeholder="Paste headline or full article here..."
)

col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🔍 Predict")
with col2:
    clear_btn = st.button("🧹 Clear")

if clear_btn:
    st.rerun()

# ---------- PREDICTION ----------
if predict_btn:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred = model.predict([text])[0]
        label = "FAKE" if pred == 1 else "REAL"

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0][1]  # Fake probability

        if label == "FAKE":
            st.error("🚨 Prediction: **FAKE NEWS**")
        else:
            st.success("✅ Prediction: **REAL NEWS**")

        if proba is not None:
            st.markdown("### Confidence")
            st.progress(proba)
            st.caption(f"Fake probability: {proba:.2%}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This is a demonstration system for educational purposes "
    "and may produce incorrect predictions."
)
