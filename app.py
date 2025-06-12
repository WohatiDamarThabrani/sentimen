import streamlit as st
from transformers import pipeline

# Muat model hanya sekali
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

model = load_model()

st.title("🧠 Sentiment Analysis with DistilBERT")
st.write("Enter text below to predict its sentiment.")

# Input pengguna
text = st.text_area("Enter text:")

if st.button("Sentiment prediction"):
    if not text.strip():
        st.warning("Enter text first")
    else:
        result = model(text)[0]
        st.success(f"**Sentimen:** {result['label']}")
        st.info(f"**Confidence Score:** {result['score']:.4f}")
