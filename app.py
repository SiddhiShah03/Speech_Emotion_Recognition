import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Speech Emotion and Gender Recognition", layout="wide")

# Load models and encoders
emotion_model = load_model("emotion_model.h5")
gender_model = load_model("gender_model.h5")
le_emotion = joblib.load("le_emotion.pkl")
le_gender = joblib.load("le_gender.pkl")

# Emojis for fun!
emotion_emojis = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😱", "disgust": "🤢", "surprised": "😲"
}
gender_emojis = {
    "male": "👨", "female": "👩"
}

def extract_features_from_audio_array(audio_array, sample_rate, max_pad_len=174):
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs
    
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    return extract_features_from_audio_array(audio, sample_rate)

# UI Design
st.markdown("""
    <h1 style='text-align: center; color: #4B0082;'>🎙️ Speech Emotion and Gender Recognition</h1>
    <p style='text-align: center;'>Upload or record your voice to detect your emotion and gender</p>
""", unsafe_allow_html=True)

st.image("banner.png", use_container_width=True)

# Upload block
st.header("🔼 Upload a WAV Audio File")
uploaded_file = st.file_uploader("Choose a file...", type=["wav","mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    features = extract_features("temp.wav")
    features = np.expand_dims(features, axis=0)

    emotion_pred = emotion_model.predict(features)
    gender_pred = gender_model.predict(features)

    predicted_emotion = le_emotion.inverse_transform([np.argmax(emotion_pred)])[0]
    predicted_gender = le_gender.inverse_transform([np.argmax(gender_pred)])[0]

    emotion_emoji = emotion_emojis.get(predicted_emotion.lower(), "")
    gender_emoji = gender_emojis.get(predicted_gender.lower(), "")

    st.markdown(f"""
        <h2 style='color: #8B4513; text-align: center;'>Prediction Results</h2>
        <div style='text-align: center;'>
            <div style="font-size: 22px; color: #6F4F37;">
                <p><b>Emotion:</b> <span style="color: #CD853F;">{predicted_emotion.capitalize()} {emotion_emoji}</span></p>
                <p><b>Gender:</b> <span style="color: #8B4513;">{predicted_gender.capitalize()} {gender_emoji}</span></p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 🎤 Live Audio Recording ---
st.header("🎤 Or Record Your Voice")

rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
media_stream_constraints = {"audio": True, "video": False}

class AudioProcessor:
    def __init__(self):
        self.audio_data = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.audio_data.extend(audio.flatten())
        return frame

ctx = webrtc_streamer(
    key="live-audio",
    mode=WebRtcMode.RECVONLY,
    rtc_configuration=rtc_configuration,
    media_stream_constraints=media_stream_constraints,
    async_processing=True,
)

if ctx.state.playing:
    st.info("Recording... Speak now!")

    if "audio_processor" not in ctx.session_state:
        ctx.session_state.audio_processor = AudioProcessor()

    if ctx.audio_receiver:
        ctx.audio_receiver._processor = ctx.session_state.audio_processor

    if st.button("Predict"):
    try:
        if "processor" not in ctx.session_state or not hasattr(ctx.session_state.processor, "audio_data"):
            st.warning("⚠️ No audio processor found. Please speak into the microphone first.")
        elif len(ctx.session_state.processor.audio_data) < 1000:
            st.warning("❗ No audio captured. Please speak clearly before clicking Predict.")
        else:
            raw_audio = np.array(ctx.session_state.processor.audio_data).astype(np.float32)
            sample_rate = 48000  # streamlit-webrtc default

            features = extract_features_from_audio_array(raw_audio, sample_rate)
            features = np.expand_dims(features, axis=0)

            emotion_pred = emotion_model.predict(features)
            gender_pred = gender_model.predict(features)

            predicted_emotion = le_emotion.inverse_transform([np.argmax(emotion_pred)])[0]
            predicted_gender = le_gender.inverse_transform([np.argmax(gender_pred)])[0]

            st.success(f"**Emotion:** {predicted_emotion.capitalize()} {emotion_emojis.get(predicted_emotion.lower(), '')}")
            st.success(f"**Gender:** {predicted_gender.capitalize()} {gender_emojis.get(predicted_gender.lower(), '')}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
