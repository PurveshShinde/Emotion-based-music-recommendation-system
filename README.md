🎶 Emotion-Based Music Recommendation System

This project detects a user’s facial emotion using a CNN model and recommends music accordingly. It uses:

Real-time face detection via Haar Cascade

Emotion classification using trained CNN model (model.h5)

A curated emotion-tagged music dataset (muse_v3.csv)

Streamlit frontend UI

Camera-based live inference

📂 Project Structure
.
│ app.py
│ requirements.txt
│ model.h5
│ muse_v3.csv
│ haarcascade_frontalface_default.xml

🧠 How it Works (Core Logic)

Camera captures user face frames

Haar classifier detects face region

CNN model predicts emotion from face

Detected emotions are tallied

Music entries are selected from dataset based on emotion category

Output is clickable song links

The main execution logic is in app.py, e.g. the emotion-scanner UI loop and recommendation generation.

app

The required dependencies are listed in requirements.txt:

requirements

🚀 Setup & Running the Project
1️⃣ Create Virtual Environment
Windows (CMD / PowerShell)
python -m venv venv

macOS / Linux
python3 -m venv venv

2️⃣ Activate the Environment
Windows:
venv\Scripts\activate

macOS / Linux:
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

Contains packages like streamlit, opencv-python-headless, tensorflow, etc.

requirements

4️⃣ Run the Application
streamlit run app.py

Once launched — click SCAN MY EMOTION and grant camera access.

📸 Requirements for Execution

Webcam access

Python 3.8–3.11

TensorFlow & OpenCV installed

Model & XML cascade are in the same project directory
(model.h5, haarcascade_frontalface_default.xml)
