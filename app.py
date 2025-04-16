import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
from PIL import Image

# Load models and encoders
emotion_model = load_model("emotion_model.h5")
gender_model = load_model("gender_model.h5")
le_emotion = joblib.load("le_emotion.pkl")
le_gender = joblib.load("le_gender.pkl")

# Emojis
emotion_emojis = {
    "happy": "😊", "angry": "😡", "sad": "🥺", "neutral": "😐",
    "fearful": "😨", "disgust": "🤢", "surprised": "😲", "calm": "😌"
}
gender_emojis = {
    "male": "👨", "female": "👩"
}

# Feature extraction
def extract_features_from_array(audio, sr, max_pad_len=174):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

banner_image = Image.open("banner.png")  
st.image(banner_image, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #8B4513;'>🎙️ Speech Emotion & Gender Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #6F4F37;'>Upload or record audio and predict <b>emotion</b> and <b>gender</b>.</p>", unsafe_allow_html=True)

# ---------- AUDIO FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Predict from File"):
        audio, sr = librosa.load("temp.wav", sr=None)
        features = extract_features_from_array(audio, sr)
        features = np.expand_dims(features, axis=0)

        emotion_pred = emotion_model.predict(features)
        gender_pred = gender_model.predict(features)

        predicted_emotion = le_emotion.inverse_transform([np.argmax(emotion_pred)])[0]
        predicted_gender = le_gender.inverse_transform([np.argmax(gender_pred)])[0]

        st.success(f"**Emotion:** {predicted_emotion.capitalize()} {emotion_emojis.get(predicted_emotion.lower(), '')}")
        st.success(f"**Gender:** {predicted_gender.capitalize()} {gender_emojis.get(predicted_gender.lower(), '')}")

# ---------- LIVE AUDIO RECORDING ----------
st.markdown("---")
st.markdown("### 🎤 Record Audio")

# Audio buffer
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

class AudioProcessor:
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        st.session_state.audio_buffer.extend(audio.tolist())
        return frame

webrtc_streamer(
    key="live-audio",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

# Check if the recording device is available
if len(st.session_state.audio_buffer) == 0:
    st.warning("⚠️ No audio recorded yet. Please record some audio first.")
    
if st.button("Predict from Recording"):
    if len(st.session_state.audio_buffer) < 10000:
        st.warning("⚠️ Not enough audio recorded. Please record more.")
    else:
        audio_array = np.array(st.session_state.audio_buffer).astype(np.float32)
        sr = 48000  # default from streamlit-webrtc

        features = extract_features_from_array(audio_array, sr)
        features = np.expand_dims(features, axis=0)

        emotion_pred = emotion_model.predict(features)
        gender_pred = gender_model.predict(features)

        predicted_emotion = le_emotion.inverse_transform([np.argmax(emotion_pred)])[0]
        predicted_gender = le_gender.inverse_transform([np.argmax(gender_pred)])[0]

        st.success(f"**Emotion:** {predicted_emotion.capitalize()} {emotion_emojis.get(predicted_emotion.lower(), '')}")
        st.success(f"**Gender:** {predicted_gender.capitalize()} {gender_emojis.get(predicted_gender.lower(), '')}")

        # Clear buffer after prediction
        st.session_state.audio_buffer = []
