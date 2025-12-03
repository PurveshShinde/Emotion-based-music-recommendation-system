🎶 Emotion-Based Music Recommendation System

An AI-powered web application that detects a user’s facial emotion in real-time using computer vision and deep learning, then recommends a curated music playlist to match their mood.

📸 Demo & Screenshots

1. App Home UI (assets/main.png)

The clean, minimal interface built with Streamlit.

2. Emotion Detection (Camera Active) (assets/scan.png)

Real-time face detection and emotion classification using OpenCV and CNN.

3. Music Recommendation Output (assets/output.png)

The system maps the detected emotion to a specific genre/vibes and fetches songs.

✨ Key Features

Real-time Detection: Uses Haar Cascade for immediate face detection via webcam.

Deep Learning Model: A trained CNN (Convolutional Neural Network) classifies faces into 7 distinct emotions.

Smart Mapping: Maps emotions to 5 mood categories for optimized song selection.

Curated Playlists: Fetches song data efficiently from the muse_v3.csv dataset.

Interactive UI: Fully responsive and interactive interface powered by Streamlit.

📁 Project Structure

Emotion-based-music-recommendation-system/
│
├── app.py # Main Streamlit application entry point
├── requirements.txt # List of Python dependencies
├── muse_v3.csv # Dataset containing song metadata
├── model.h5 # Pre-trained CNN emotion detection model
├── haarcascade_frontalface_default.xml # OpenCV face detection XML
│
└── assets/ # Images for README documentation
├── main_ui.png
├── scan.png
└── output.png

🛠️ Tech Stack

Language: Python 3.8+

Computer Vision: OpenCV (cv2)

Deep Learning: TensorFlow / Keras

Web Framework: Streamlit

Data Handling: Pandas, NumPy

🖥️ Setup & Installation Instructions

Follow these steps to run the project locally.

1. Clone the Repository

git clone [https://github.com/your-username/Emotion-based-music-recommendation-system.git](https://github.com/your-username/Emotion-based-music-recommendation-system.git)
cd Emotion-based-music-recommendation-system

2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

For Windows:

python -m venv venv
venv\Scripts\activate

For macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the Application

streamlit run app.py

Once running, the application will be available in your browser at:
http://localhost:8501

🎥 How to Use

Launch the App: Follow the setup instructions above to start the server.

Start Scanning: Click the "📸 SCAN MY EMOTION" button on the sidebar.

Grant Permissions: Allow the browser to access your webcam.

Hold Still: Look at the camera for a few seconds while the model analyzes your facial features.

Get Recommendations: Once the emotion is captured, the app will display your current mood and a list of recommended songs.

⚙️ Configuration & Troubleshooting

Git Ignore Setup

To prevent committing unnecessary files (like the virtual environment), ensure your .gitignore contains:

venv/
.venv/
**pycache**/
.DS_Store

Common Issues

Camera not opening: Ensure no other application (Zoom, Teams) is using the webcam.

Model not found: Ensure model.h5 and haarcascade_frontalface_default.xml are in the root directory.

Slow Performance: The first run might be slow as TensorFlow loads the model into memory.

🏁 Summary

This project demonstrates practical Affective Computing:

Captures facial expressions using OpenCV.

Infers emotion using a Deep Learning model.

Maps the result to a curated database.

Recommends content in a user-friendly UI.

Ideal for AI/ML coursework, Computer Vision research, and HCI (Human-Computer Interaction) demos.
