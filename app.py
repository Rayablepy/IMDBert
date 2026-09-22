import streamlit as st

from loader import classify, get_device, get_model

st.set_page_config(page_title="IMDBert Sentiment Analysis", page_icon="🎬")

st.title("IMDBert Sentiment Analysis")
st.write(
    "DistilBERT fine-tuned on IMDB movie reviews to predict Positive/Negative sentiment."
)

@st.cache_resource
def load_model():
    device = get_device()
    model, tokenizer = get_model()
    model.to(device)
    return model, tokenizer, device

MODEL, TOKENIZER, DEVICE = load_model()
st.caption(f"Model loaded on: {DEVICE}")

review = st.text_area(
    "Movie review",
    placeholder="Type a movie review here...",
    height=150,
)

if st.button("Classify", type="primary") and review.strip():
    label, confidence = classify(review, MODEL, TOKENIZER, DEVICE)
    if label:
        score = float(confidence.rstrip("%")) / 100
        st.metric("Sentiment", label)
        st.progress(score, text=f"Confidence: {confidence}")
