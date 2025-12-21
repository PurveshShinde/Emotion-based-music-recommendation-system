import os
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# ----------------------------------------------------
# 1. PATH CONFIG (CLOUD SAFE)
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "DATA_PATH": os.path.join(BASE_DIR, "muse_v3.csv"),
    "MODEL_WEIGHTS_PATH": os.path.join(BASE_DIR, "model.h5"),
    "HAARCASCADE_PATH": os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"),
}

# ----------------------------------------------------
# 2. CONSTANTS
# ----------------------------------------------------
EMOTION_DICT = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

MUSIC_MAP = {
    "Angry": "Angry",
    "Disgusted": "Angry",
    "Fearful": "Fearful",
    "Happy": "Happy",
    "Neutral": "Neutral",
    "Sad": "Sad",
    "Surprised": "Happy"
}

SAMPLING_PLAN = {
    1: [30],
    2: [30, 20],
    3: [55, 20, 15],
    4: [30, 29, 18, 9],
    5: [10, 7, 6, 5, 2],
}

# ----------------------------------------------------
# 3. CACHED LOADERS
# ----------------------------------------------------
@st.cache_data
def load_and_split_data(path):
    df = pd.read_csv(path)

    df.rename(columns={
        'lastfm_url': 'link',
        'track': 'name',
        'number_of_emotion_tags': 'emotional',
        'valence_tags': 'pleasant'
    }, inplace=True)

    df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
    df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

    size = len(df) // 5
    return {
        'Sad': df.iloc[:size],
        'Fearful': df.iloc[size:2*size],
        'Angry': df.iloc[2*size:3*size],
        'Neutral': df.iloc[3*size:4*size],
        'Happy': df.iloc[4*size:]
    }

@st.cache_resource
def load_model(path):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights(path)
    return model

@st.cache_resource
def load_cascade(path):
    return cv2.CascadeClassifier(path)

EMOTION_DFS = load_and_split_data(CONFIG["DATA_PATH"])
MODEL = load_model(CONFIG["MODEL_WEIGHTS_PATH"])
FACE_CASCADE = load_cascade(CONFIG["HAARCASCADE_PATH"])

# ----------------------------------------------------
# 4. HELPERS
# ----------------------------------------------------
def prioritize_emotions(emotions):
    mapped = [MUSIC_MAP[e] for e in emotions]
    return [e for e, _ in Counter(mapped).most_common()]

def get_recommendations(priority):
    data = []
    plan = SAMPLING_PLAN.get(min(len(priority), 5), [])

    for emo, n in zip(priority, plan):
        df = EMOTION_DFS.get(emo)
        if df is not None and not df.empty:
            data.append(df.sample(min(n, len(df))))

    return pd.concat(data) if data else pd.DataFrame()

# ----------------------------------------------------
# 5. UI
# ----------------------------------------------------
st.set_page_config(page_title="Emotion Music Recommender", layout="wide")

st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
h2, h4, h5, a {
    color: white !important;
    text-shadow: 2px 2px 4px black;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center'>🎶 Emotion-Based Music Recommender</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center'>Capture your emotion and get music!</h5>", unsafe_allow_html=True)
st.write("---")

# ----------------------------------------------------
# 6. CAMERA INPUT (CLOUD SAFE)
# ----------------------------------------------------
img = st.camera_input("📸 Capture your emotion")

if img is not None:
    bytes_data = np.asarray(bytearray(img.read()), dtype=np.uint8)
    frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    detected = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.reshape(1, 48, 48, 1)

        pred = MODEL.predict(roi, verbose=0)
        emotion = EMOTION_DICT[int(np.argmax(pred))]
        detected.append(emotion)

    if detected:
        priority = prioritize_emotions(detected)
        st.success(f"Detected emotion(s): {', '.join(priority)}")

        recs = get_recommendations(priority)
        st.write("---")
        st.markdown("<h4 style='text-align:center'>🎧 Recommended Songs</h4>", unsafe_allow_html=True)

        for i, row in recs.head(30).iterrows():
            st.markdown(
                f"<h4 style='text-align:center'><a href='{row['link']}' target='_blank'>{row['name']}</a></h4>"
                f"<h5 style='text-align:center'><i>{row['artist']}</i></h5><hr>",
                unsafe_allow_html=True
            )
    else:
        st.warning("No face detected. Try better lighting and face the camera.")
