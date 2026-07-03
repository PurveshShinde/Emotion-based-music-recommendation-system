# 🎶 Emotion-Based Music Recommendation System

An AI-powered web application that detects a user’s **facial emotion in real time** using computer vision and deep learning, then recommends a **curated music playlist** matching the detected mood.

This project demonstrates practical **Affective Computing** by bridging human-computer interaction (HCI), computer vision (**OpenCV**), deep learning (**TensorFlow/Keras**), and interactive web design (**Streamlit**).

---

## 📸 Demo & Screenshots

### 🏠 App Home UI
A clean, responsive interface built with Streamlit.
![App Home UI](assets/main.png)

### 📷 Emotion Detection (Camera Active)
Real-time face detection and 7-class facial emotion classification using a pre-trained Convolutional Neural Network (CNN).
![Emotion Detection](assets/scan.png)

### 🎵 Music Recommendation Output
The detected emotion is mapped to a mood category to instantly fetch and suggest relevant songs.
![Music Recommendation Output](assets/output.png)

---

## ✨ Key Features

* **Real-Time Face Detection:** Uses OpenCV's Haar Cascade classifier for high-speed, instant face detection via webcam.
* **Deep Learning Classification:** A CNN-based model categorizes facial expressions into **7 distinct emotions** (e.g., Happy, Sad, Angry, Neutral, etc.).
* **Smart Mood Mapping:** Aggregates the 7 raw emotions into **5 core mood categories** for more relevant and cohesive music recommendations.
* **Curated Playlists:** Fast, session-cached song retrieval powered by the `muse_v3.csv` dataset.
* **Interactive Web UI:** Fully responsive, accessible, and easy-to-use web interface.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **Computer Vision** | OpenCV (`haarcascade_frontalface_default.xml`) |
| **Deep Learning** | TensorFlow / Keras (`model.h5`) |
| **Data Processing** | Pandas, NumPy |

---

## 📁 Project Structure

```text
Emotion-based-music-recommendation-system/
│
├── app.py                               # Main Streamlit web application
├── requirements.txt                     # Python package dependencies
├── muse_v3.csv                          # Music recommendation dataset
├── model.h5                             # Pre-trained CNN emotion model
├── haarcascade_frontalface_default.xml  # OpenCV face detection rules
│
└── assets/                              # UI screenshots for documentation
    ├── main.png
    ├── scan.png
    └── output.png
