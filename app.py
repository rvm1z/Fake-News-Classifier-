import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# UI
st.set_page_config(page_title="Fake News Classifier")
st.title("🧠 Fake News Classifier")
st.markdown("Enter a news article or headline below to check if it's **Fake** or **Real**.")

news = st.text_area("📰 Enter News Text Here")

if st.button("Classify"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([news])
        prediction = model.predict(vec)[0]
        st.success(f"### 🔍 Prediction: **{prediction.upper()}**")

