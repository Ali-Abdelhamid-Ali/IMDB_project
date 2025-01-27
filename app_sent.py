import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="wide")

@st.cache_resource  
def load_model():
    model_path = r"D:\AMIT\amit\ODC\W4\D3\IMDB Dataset\sentiment_analysis_model"
    return tf.keras.models.load_model(model_path)

@st.cache_resource 
def load_tokenizer():
    tokenizer_path = r"D:\AMIT\amit\ODC\W4\D3\IMDB Dataset\tokenizer.pkl"
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            return pickle.load(handle)
    else:
        st.warning("⚠️ Tokenizer file not found. A new tokenizer will be created, but it may not be accurate.")
        training_texts = ["I love this movie", "This movie was bad", "Amazing experience", "Worst movie ever"]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(training_texts)
        return tokenizer

model = load_model()
tokenizer = load_tokenizer()

def preprocess_text(text, tokenizer, max_length=100):
    text = text.strip().lower()
    encoded_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')
    return padded_text

st.sidebar.title("🔍 Input Options")
input_type = st.sidebar.radio("Choose input type:", ("📝 Direct Text", "📌 Predefined Text"))

st.title('📊 Sentiment Analysis ')

if input_type == "📝 Direct Text":
    user_text = st.text_area("✍️ Enter your text here:")
else:
    user_text = st.selectbox("🔹 Choose a sentence:", 
                              ["This is a great movie!", "I hated this film.", "It was an amazing experience!"])

if st.button('🔍 Analyze'):
    if user_text:
        padded_text = preprocess_text(user_text, tokenizer)
        prediction = model.predict(padded_text)
        prediction_value = float(prediction[0][0])

        if prediction_value >= 0.5:
            st.success("😊 The text expresses **positive** sentiment! 👍")
        else:
            st.error("😞 The text expresses **negative** sentiment! 👎")

        st.markdown("### 📈 Visual Result:")
        fig, ax = plt.subplots(figsize=(6, 3))  
        ax.bar(["Positive 😊", "Negative 😞"], [prediction_value, 1 - prediction_value], color=["green", "red"])
        ax.set_ylabel('Value')
        ax.set_title('🔍 Sentiment Analysis')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter text for analysis.")
