import streamlit as st
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
from PIL import Image 

# Enable wide layout
st.set_page_config(layout="wide")

# Custom CSS for image + layout
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        padding-top: 0rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    img.banner {
        border-radius: 10px;
        width: 100%;
        max-width: 600px;
    }
    .title-box {
        padding-left: 30px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .title-box h1 {
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
    }
    .title-box p {
        font-size: 1.1rem;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# Show banner at the top
banner_image = Image.open("banner.png")  
st.image(banner_image, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# Load models and encoders
emotion_model = load_model("emotion_model.h5")
gender_model = load_model("gender_model.h5")
le_emotion = joblib.load("le_emotion.pkl")
le_gender = joblib.load("le_gender.pkl")

# Feature extractor
def extract_features(file_path, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# Streamlit UI
#st.title("🎙️ Speech Emotion & Gender Recognition")
#st.markdown("Upload an audio file (.wav or .mp3) and the model will predict the speaker's **emotion** and **gender**.")
# Layout container with image and title
col1, col2 = st.columns([1.2, 2])  # Image on the left, title on the right

with col1:
    st.image("image.png", caption=None, use_container_width=True, output_format="auto", channels="RGB")

with col2:
    st.markdown("""
    <div class="title-box">
        <h1>🎙️ Speech Emotion & Gender Recognition</h1>
        <p>Upload an audio file (.wav or .mp3) and the model will predict the speaker's <b>emotion</b> and <b>gender</b>.</p>
    </div>
    """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Predict"):
        try:
            features = extract_features("temp.wav")
            features = np.expand_dims(features, axis=0)

            emotion_pred = emotion_model.predict(features)
            gender_pred = gender_model.predict(features)

            predicted_emotion = le_emotion.inverse_transform([np.argmax(emotion_pred)])[0]
            predicted_gender = le_gender.inverse_transform([np.argmax(gender_pred)])[0]

            st.success(f"**Emotion:** {predicted_emotion.capitalize()}")
            st.success(f"**Gender:** {predicted_gender.capitalize()}")

        except Exception as e:
            st.error("⚠️ Error during prediction: " + str(e))
