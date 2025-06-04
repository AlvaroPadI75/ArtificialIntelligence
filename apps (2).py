# app.py

import streamlit as st
import pickle
import re
import string
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Unified Demo App",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ("Spam Classification", "Pet Breed Classification", "iPhone Price Prediction")
)

# ============================================
# 1) SPAM CLASSIFICATION PAGE
# ============================================
@st.cache_resource(show_spinner=False)
def load_spam_model():
    """Load the trained LSTM spam‐vs‐ham model (.h5)."""
    model = tf.keras.models.load_model("modelo_spam_lstm.h5")
    return model

@st.cache_resource(show_spinner=False)
def load_spam_tokenizer():
    """Load the tokenizer used to preprocess spam text."""
    with open("tokenizer_spam.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def clean_text(text: str) -> str:
    """
    Very basic cleaning pipeline:
      1) Lowercase
      2) Remove HTML tags
      3) Remove punctuation
      4) Remove digits
      5) Remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                 # strip HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                   # digits → space
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text

def spam_page():
    st.title("📧 Spam vs. Ham Classifier")
    st.markdown(
        """
        Enter an email or message in the box below, and the LSTM model
        will predict whether it is **Spam** or **Not Spam (Ham)**.
        """
    )

    # 1) Load model & tokenizer
    model = load_spam_model()
    tokenizer = load_spam_tokenizer()

    # 2) Figure out maxlen from the model’s input shape
    try:
        maxlen = model.input_shape[1]
    except:
        maxlen = 100  # fallback if input_shape isn’t accessible

    user_text = st.text_area("Enter text to classify:", height=200)

    if st.button("Classify"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
            return

        # 3) Preprocess
        cleaned = clean_text(user_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")

        # 4) Predict
        pred_prob = model.predict(padded)[0][0]
        label = "Spam" if pred_prob >= 0.5 else "Not Spam (Ham)"

        st.markdown("---")
        st.write(f"**Prediction:** `{label}`")
        st.write(f"**Spam probability:** `{pred_prob:.3f}`")

# ============================================
# 2) PET BREED CLASSIFICATION PAGE
# ============================================
@st.cache_resource(show_spinner=False)
def load_pet_model():
    """Load the pretrained MobileNetV2‐based pet breed classifier (.h5)."""
    model = tf.keras.models.load_model("modelo_pets_mobilenetv2.h5")
    return model

# 37 class names in exactly the order used during training
PET_CLASS_NAMES = [
    "Abyssinian", "American Bobtail", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese",
    "Sphynx", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
    "Beagle", "Boxer", "Chihuahua", "English Cocker Spaniel", "English Setter",
    "German Shorthaired Pointer", "Great Pyrenees", "Havanese", "Japanese Chin",
    "Keeshond", "Leonberger", "Miniature Pinscher", "Newfoundland", "Pomeranian",
    "Pug", "Saint Bernard", "Samoyed", "Scotch Terrier", "Shiba Inu", "Siberian Husky",
    "Toy Poodle", "Yorkshire Terrier"
]

def preprocess_pet_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    - Convert PIL to RGB
    - Resize to (224, 224)
    - Convert to float32 and normalize to [0,1]
    - Expand dims to shape (1, 224, 224, 3)
    """
    img = image.convert("RGB")
    img = img.resize(target_size, resample=Image.BICUBIC)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def pet_page():
    st.title("🐾 Pet Breed Classifier (Oxford-IIIT Pet)")
    st.markdown(
        """
        Upload a cat or dog photo, and the MobileNetV2‐based model will predict
        its **breed** out of 37 possible classes.
        """
    )

    model = load_pet_model()

    uploaded_file = st.file_uploader(
        "Upload a JPG/PNG image of your pet:", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        x = preprocess_pet_image(image, target_size=(224, 224))
        preds = model.predict(x)         # shape: (1, 37)
        prob = float(np.max(preds))      # highest softmax probability
        idx = int(np.argmax(preds))      # index of the predicted class
        breed = PET_CLASS_NAMES[idx]

        st.markdown("---")
        st.write(f"**Predicted breed:** `{breed}`")
        st.write(f"**Probability:** `{prob:.3f}`")

        st.subheader("Probabilities for each class")
        probs_dict = {PET_CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(PET_CLASS_NAMES))}
        st.bar_chart(probs_dict)

# ============================================
# 3) IPHONE PRICE PREDICTION PAGE
# ============================================
@st.cache_resource(show_spinner=False)
def load_iphone_model():
    """
    Load the scikit‐learn pipeline for iPhone Price Classification.
    Wrap in try/except so we can display a helpful message if it fails.
    """
    try:
        with open("modelo_regresion_iphone.pkl", "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error("The file 'modelo_regresion_iphone.pkl' was not found. "
                 "Ensure it is in the same folder as app.py and pushed to GitHub.")
        st.stop()
    except pickle.UnpicklingError:
        st.error("Could not unpickle 'modelo_regresion_iphone.pkl'. "
                 "It may be corrupted or depend on an incompatible scikit-learn version.")
        st.stop()
    except Exception as e:
        st.error(f"Unknown error loading pickle: {e}")
        st.stop()

def parse_gdp(gdp_str: str) -> float | None:
    """
    Convert a string like "$27.72 trillion" or "$3.5 billion" or "$450,000,000"
    into a numeric value in plain dollars (float). Returns None if parsing fails.
    """
    if not gdp_str or not isinstance(gdp_str, str):
        return None
    s = gdp_str.replace("$", "").replace(",", "").strip().lower()
    if "trillion" in s:
        try:
            val = float(s.replace("trillion", "").strip())
            return val * 1e12
        except:
            return None
    if "billion" in s:
        try:
            val = float(s.replace("billion", "").strip())
            return val * 1e9
        except:
            return None
    try:
        return float(s)
    except:
        return None

def parse_pc_gdp(pc_str: str) -> float | None:
    """
    Convert a string like "$45,000" or "45000" into a float.
    Returns None on failure.
    """
    if not pc_str or not isinstance(pc_str, str):
        return None
    s = pc_str.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except:
        return None

def iphone_page():
    st.title("📱 iPhone Price Classification")
    st.markdown(
        """
        Enter the following information for a country, and the model
        will predict if iPhones there are **Expensive** (above median)
        or **Not Expensive** (below median), based on historical data.
        """
    )

    pipeline = load_iphone_model()

    st.subheader("Enter input features:")
    country   = st.text_input("Country (e.g. United States):", value="United States")
    tax       = st.number_input("Tax rate (e.g. 0.07 for 7%):", min_value=0.0, format="%.4f", value=0.07)
    gdp_str   = st.text_input("GDP (e.g. \"$27.72 trillion\" or \"$3.5 billion\"):",
                              value="$1 trillion")
    pcgdp_str = st.text_input("Per Capita GDP (e.g. \"$45,000\"):",
                              value="$45,000")

    if st.button("Predict Expensiveness"):
        gdp_val   = parse_gdp(gdp_str)
        pcgdp_val = parse_pc_gdp(pcgdp_str)

        if gdp_val is None or pcgdp_val is None:
            st.error("Could not parse GDP or Per Capita GDP. Use formats like "
                     "\"$27.72 trillion\" or \"$3.5 billion\", \"$45,000\".")
            return

        df_input = pd.DataFrame({
            "Tax":       [tax],
            "GDP_num":   [gdp_val],
            "PC_GDP_num":[pcgdp_val],
            "Country":   [country]
        })

        try:
            prob = pipeline.predict_proba(df_input)[0][1]  # prob of class “Expensive” (1)
            label = "Expensive" if prob >= 0.5 else "Not Expensive"
            st.markdown("---")
            st.write(f"**Prediction:** `{label}`")
            st.write(f"**Probability (Expensive):** `{prob:.3f}`")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ============================================
# MAIN: Render the chosen page
# ============================================
if page == "Spam Classification":
    spam_page()
elif page == "Pet Breed Classification":
    pet_page()
elif page == "iPhone Price Prediction":
    iphone_page()

