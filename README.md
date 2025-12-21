# 🎶 Emotion-Based Music Recommendation System

An AI-powered web application that detects a user’s **facial emotion in real time** using computer vision and deep learning, then recommends a **curated music playlist** matching the detected mood.

This project demonstrates practical **Affective Computing** by combining **OpenCV**, **CNN-based emotion classification**, and a **Streamlit web interface**.

---

## 📸 Demo & Screenshots

### 🏠 App Home UI
Clean, minimal interface built using Streamlit.

![App Home UI](assets/main_ui.png)

---

### 📷 Emotion Detection (Camera Active)
Real-time face detection and emotion classification using OpenCV and a trained CNN model.

![Emotion Detection](assets/scan.png)

---

### 🎵 Music Recommendation Output
Detected emotion is mapped to a mood category and corresponding songs are fetched.

![Music Recommendation Output](assets/output.png)

---

## ✨ Key Features

- **Real-time Emotion Detection**  
  Uses Haar Cascade for instant face detection via webcam.

- **Deep Learning Model**  
  CNN-based classifier categorizes faces into **7 distinct emotions**.

- **Smart Mood Mapping**  
  Emotions are mapped into **5 mood categories** for better music relevance.

- **Curated Playlists**  
  Song recommendations are fetched from the `muse_v3.csv` dataset.

- **Interactive Web UI**  
  Fully responsive Streamlit-based interface.

---

## 📁 Project Structure

```text
Emotion-based-music-recommendation-system/
│
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── muse_v3.csv                    # Music dataset
├── model.h5                       # Pre-trained CNN model
├── haarcascade_frontalface_default.xml  # OpenCV face detector
│
└── assets/                        # README screenshots
    ├── main_ui.png
    ├── scan.png
    └── output.png

---
 
🛠️ Tech Stack

Language: Python 3.8+

Computer Vision: OpenCV (cv2)

Deep Learning: TensorFlow / Keras

Web Framework: Streamlit

Data Handling: Pandas, NumPy

🖥️ Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/PurveshShinde/Emotion-based-music-recommendation-system.git
cd Emotion-based-music-recommendation-system
2️⃣ Create a Virtual Environment (Recommended)

Windows

python -m venv venv
venv\Scripts\activate


macOS / Linux

python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py


The app will be available at:
👉 http://localhost:8501

🎥 How to Use

Launch the application.

Click 📸 SCAN MY EMOTION from the sidebar.

Allow webcam access in the browser.

Look at the camera for a few seconds.

View detected emotion and recommended songs.

⚙️ Configuration & Troubleshooting
.gitignore (Recommended)
venv/
.venv/
__pycache__/
.DS_Store

Common Issues

Webcam not opening

Ensure no other apps (Zoom, Teams) are using the camera.

Model file not found

Confirm model.h5 and haarcascade_frontalface_default.xml are in the root directory.

Slow first run

TensorFlow may take time to load the model initially.

🏁 Summary

This project showcases an end-to-end Emotion-Aware Recommendation System:

Facial expression capture using OpenCV

Emotion inference via Deep Learning

Intelligent mood-to-music mapping

Real-time interactive web experience

🎓 Ideal for:
AI/ML coursework, Computer Vision demos, Human–Computer Interaction (HCI), and portfolio projects.

📌 Future Improvements (Optional)

Spotify / YouTube API integration

Emotion confidence visualization

Mobile-friendly UI

Model optimization for faster inference

⭐ If you like this project, consider giving it a star!

---

If you want next:
- 🔹 **README badge section (Python, Streamlit, ML)**  
- 🔹 **Deployment section (Streamlit Cloud URL)**  
- 🔹 **Resume-optimized short README**

Just tell me what you want to add.
