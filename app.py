# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1q_4Eue6Z8mnVVQAZ69ihZOFcsn48mBMe
"""

# app.py

import streamlit as st
import pickle
import re
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1) TÍTULO Y DESCRIPCIÓN -------------------------------------------------
st.set_page_config(
    page_title="SPAM DETECTOR",
    page_icon="✉️",
    layout="centered"
)

st.title("🔎 SPAM CLASSIFIER")
st.markdown(
    """
This app uses a pre-trained LSTM model to classify emails as **SPAM** or **HAM** (not spam).
Enter the email text and press "Classify."
    """
)

# 2) CARGAR TOKENIZER Y MODELO ---------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_tokenizer(path="tokenizer_spam.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def cargar_modelo(path="modelo_spam_lstm.h5"):
    return tf.keras.models.load_model(path)

tokenizer = cargar_tokenizer()
model = cargar_modelo()

# 3) FUNCIÓN DE LIMPIEZA ---------------------------------------------------
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)            
    texto = re.sub(r"\d+", "", texto)                 
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = texto.strip()
    return texto

def predecir_spam(texto_crudo, tokenizer, model, max_len=100):

    texto_limpio = limpiar_texto(texto_crudo)

    secuencia = tokenizer.texts_to_sequences([texto_limpio])

    seq_pad = pad_sequences(secuencia, maxlen=max_len, padding="post", truncating="post")

    prob_spam = model.predict(seq_pad)[0][0]
    etiqueta = "SPAM" if prob_spam > 0.5 else "HAM"

    return prob_spam, etiqueta

# 5) INTERFAZ PRINCIPAL ----------------------------------------------------
st.subheader("Introduce the message or email")
texto_entrada = st.text_area(
    label="✉️ Paste here the fulle message of your email",
    height=200,
    help="Type or paste the content of the email you want to classify as spam or ham."
)

if st.button("🚀 Classify"):
    if texto_entrada.strip() == "":
        st.warning("Please enter some text before rating.")
    else:
        with st.spinner("Analizing…"):
            prob, etiqueta = predecir_spam(texto_entrada, tokenizer, model)
        st.markdown("---")
        st.write(f"**Probability SPAM:** `{prob:.3f}`")
        if etiqueta == "SPAM":
            st.error("🏴 Prediction: **SPAM**")
        else:
            st.success("✅ Prediction: **HAM**")
