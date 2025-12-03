import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# ----------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------
# Define paths once for easy updating
CONFIG = {
    "DATA_PATH": r"D:\Projects\Emotion-based-music-recommendation-system-main\muse_v3.csv",
    "MODEL_WEIGHTS_PATH": r'D:\Projects\Emotion-based-music-recommendation-system-main\model.h5',
    "HAARCASCADE_PATH": r'D:\Projects\Emotion-based-music-recommendation-system-main\haarcascade_frontalface_default.xml',
    "FRAME_LIMIT": 20, # Number of frames to scan (approx 1 second)
}

# Emotion Mapping for the Model
EMOTION_DICT = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Map model outputs (7 emotions) to your 5 music categories
MUSIC_MAP = {
    "Angry": "Angry",
    "Disgusted": "Angry",    # Map to Angry (High arousal, negative valence)
    "Fearful": "Fearful",    # Map to Fearful
    "Happy": "Happy",
    "Neutral": "Neutral",
    "Sad": "Sad",
    "Surprised": "Happy"     # Map to Happy (High arousal, positive valence)
}

# Sampling plan definition (more flexible structure)
SAMPLING_PLAN = {
    1: [30],
    2: [30, 20],
    3: [55, 20, 15],
    4: [30, 29, 18, 9],
    5: [10, 7, 6, 5, 2],
}

# ----------------------------------------------------
# 2. CACHED DATA AND RESOURCE LOADING (MAX EFFICIENCY)
# ----------------------------------------------------

@st.cache_data(show_spinner="Loading and preprocessing music data...")
def load_and_split_data(path):
    """Loads, preprocesses, and splits the music data using Streamlit caching."""
    df = pd.read_csv(path)
    
    # 1. Rename columns for cleaner access
    df.rename(columns={
        'lastfm_url': 'link',
        'track': 'name',
        'number_of_emotion_tags': 'emotional',
        'valence_tags': 'pleasant'
    }, inplace=True)
    df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
    
    # 2. Sort once and reset index
    df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

    # 3. Split into 5 equal chunks
    total_len = len(df)
    chunk_size = total_len // 5
    
    # Use iloc for faster, index-based slicing
    emotion_dfs = {
        'Sad': df.iloc[:chunk_size],
        'Fearful': df.iloc[chunk_size:2 * chunk_size],
        'Angry': df.iloc[2 * chunk_size:3 * chunk_size],
        'Neutral': df.iloc[3 * chunk_size:4 * chunk_size],
        'Happy': df.iloc[4 * chunk_size:]
    }
    return emotion_dfs

@st.cache_resource(show_spinner="Loading Keras model weights...")
def load_emotion_model(path):
    """Loads the CNN model architecture and weights using Streamlit resource caching."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights(path)
    # Compile the model for efficiency if you plan to retrain, 
    # but for inference, loading weights is sufficient.
    return model

@st.cache_resource(show_spinner="Loading face cascade classifier...")
def load_cascade(path):
    """Loads the Haar Cascade classifier using Streamlit resource caching."""
    return cv2.CascadeClassifier(path)

# Execute the loading functions (only runs once thanks to caching)
EMOTION_DFS = load_and_split_data(CONFIG["DATA_PATH"])
MODEL = load_emotion_model(CONFIG["MODEL_WEIGHTS_PATH"])
FACE_CASCADE = load_cascade(CONFIG["HAARCASCADE_PATH"])

# ----------------------------------------------------
# 3. OPTIMIZED CORE FUNCTIONS
# ----------------------------------------------------

def prioritize_emotions(detected_list):
    """
    Processes the raw list of detected emotions, maps them to music categories,
    and returns unique categories sorted by frequency (most frequent first).
    
    Time Complexity: O(N log K) where N is the number of frames, K is unique emotions.
    """
    # 1. Map all 7 detected emotions to the 5 music categories
    mapped_list = [MUSIC_MAP.get(e, 'Sad') for e in detected_list] 
    
    # 2. Count frequencies of the music categories
    emotion_counts = Counter(mapped_list)
    
    # 3. Sort by count (descending) and return only the emotion names
    sorted_emotions = sorted(emotion_counts.items(), key=lambda item: item[1], reverse=True)
    
    return [emotion for emotion, count in sorted_emotions]

def get_recommendations(prioritized_list):
    """
    Samples songs from the relevant DataFrames based on the prioritized list 
    and the pre-defined SAMPLING_PLAN.
    
    Time Complexity: O(1) for lookup + O(N) for sampling and concatenation, 
    where N is the total number of songs to be sampled (max ~90).
    """
    data_list = []
    
    length = len(prioritized_list)
    # Get the sampling plan for the detected number of unique emotions
    times = SAMPLING_PLAN.get(min(length, 5), []) 
    
    # Iterate through emotions and their assigned sample counts
    for music_emotion, num_samples in zip(prioritized_list, times):
        # O(1) dictionary lookup for the DataFrame
        df_emotion = EMOTION_DFS.get(music_emotion)
        
        if df_emotion is not None and not df_emotion.empty:
            # Ensure we don't try to sample more than available rows
            n_samples = min(num_samples, len(df_emotion))
            # Append the sampled DataFrame to the list
            data_list.append(df_emotion.sample(n=n_samples, replace=False))
            
    # Concatenate all sampled DataFrames once at the end (Efficient)
    if data_list:
        return pd.concat(data_list, ignore_index=True)
    else:
        return pd.DataFrame()

# ----------------------------------------------------
# 4. STREAMLIT UI SETUP (USING SESSION STATE)
# ----------------------------------------------------

st.set_page_config(layout="wide")

# Initialize Streamlit Session State for persistent variables
if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []
if 'processed_emotions' not in st.session_state:
    st.session_state.processed_emotions = []

# --- Custom CSS for Styling ---
page_bg_img = f"""
<style>
/* Optimized to use a small, efficient image or gradient */
.stApp {{
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    background-attachment: fixed; 
}}
h2, h5, h4, i, a {{
    color: white !important;
    text-shadow: 2px 2px 4px #000000;
}}
/* Styling for the music list links */
.stMarkdown a:hover {{
    color: #4CAF50 !important; /* Green on hover */
    text-decoration: none;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>🎶 Emotion-Based Music Recommender</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>Click on a song name to listen!</h5>", unsafe_allow_html=True)
st.write("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button('📸 SCAN MY EMOTION (Start Camera)', use_container_width=True):
        st.session_state.emotion_log.clear() # Reset for new scan
        st.session_state.processed_emotions.clear()
        
        FRAME_WINDOW = st.empty() # Placeholder for video feed
        count = 0
        cap = cv2.VideoCapture(0)
        
        # --- Real-time Scanning Loop ---
        while cap.isOpened() and count < CONFIG["FRAME_LIMIT"]:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces (The detection step is the quickest part)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            
            # --- Prediction Loop (The heaviest part) ---
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                
                try:
                    # Preprocess image for CNN
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    
                    # CNN Prediction
                    # Added 'verbose=0' to prevent Keras output flood, slightly faster
                    prediction = MODEL.predict(cropped_img, verbose=0) 
                    max_index = int(np.argmax(prediction))
                    detected_emotion = EMOTION_DICT[max_index]
                    
                    st.session_state.emotion_log.append(detected_emotion) # Log the raw result

                    cv2.putText(frame, detected_emotion, (x + 20, y - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    # Failsafe if prediction or resize fails
                    pass
            
            # Display frame (Minimal overhead)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB")
            
            count += 1
            
        cap.release()
        
        # --- Post-Scan Processing ---
        if st.session_state.emotion_log:
            # Use the optimized prioritization function
            st.session_state.processed_emotions = prioritize_emotions(st.session_state.emotion_log)
            st.success(f"✅ Scan complete! Detected emotions: {', '.join(st.session_state.processed_emotions)}")
        else:
            st.warning("⚠️ No face or emotions detected during the scan period.")

# ----------------------------------------------------
# 5. DISPLAY RESULTS
# ----------------------------------------------------

# This section runs quickly because the data is already in Session State
if st.session_state.processed_emotions:
    st.write("---")
    st.markdown("<h4 style='text-align: center;'>🎧 Top Recommended Songs</h4>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center; color: #4CAF50;'>*Based on prioritized emotion(s): {', '.join(st.session_state.processed_emotions)}*</h5>", unsafe_allow_html=True)
    st.write("---")

    # Get recommendations (fast O(N) operation)
    recommendation_df = get_recommendations(st.session_state.processed_emotions)
    
    if not recommendation_df.empty:
        # Iterate over the resulting DataFrame (Max 30 songs)
        for i, row in recommendation_df.head(30).iterrows():
            # Use f-strings and st.markdown for efficient, styled output
            st.markdown(
                f"""
                <h4 style='text-align: center;'><a href="{row['link']}" target="_blank">
                {(i % 30) + 1}. {row['name']}
                </a></h4>
                <h5 style='text-align: center; color: grey;'><i>by {row['artist']}</i></h5>
                <hr style='border-top: 1px solid #333333;'>
                """, 
                unsafe_allow_html=True
            )
    else:
        st.warning("No songs found for the detected emotion categories.")

        # .\.venv\Scripts\activate     
        # streamlit run app.py