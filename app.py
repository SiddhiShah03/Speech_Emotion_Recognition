import streamlit as st
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
from PIL import Image 

# Enable wide layout
#st.set_page_config(layout="wide")

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
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎙️ Speech Emotion & Gender Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Upload an audio file (.wav or .mp3) and the model will predict the speaker's <b>emotion</b> and <b>gender</b>.</p>", unsafe_allow_html=True)

#uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
st.markdown("""
    <style>
        .stFileUploader { display: none; }
    </style>
    <div style="text-align: center;">
        <h3>Click below to upload an audio file</h3>
        <button style="font-size: 18px; padding: 10px 20px; cursor: pointer; border: 2px solid #4CAF50; background-color: #fff; color: #4CAF50;">
            Upload Audio
        </button>
    </div>
""", unsafe_allow_html=True)

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
