# Emotion-Based Music Recommendation System 🎶

This project captures a user's facial emotion using a live camera feed and recommends music that matches their emotional state. It uses a trained CNN model for emotion classification and a tagged dataset for emotion-driven song suggestions.

---

## Features

- Real-time face detection using Haar Cascade
- CNN-based facial emotion prediction
- Five-category emotion-to-music mapping
- Streamlit frontend UI
- Efficient caching for performance

---

## Project Structure

app.py
muse_v3.csv
model.h5
haarcascade_frontalface_default.xml
requirements.txt

yaml
Copy code

---

## Setup Instructions

### 1. Create virtual environment

#### Windows

```bash
python -m venv venv
macOS / Linux
bash
Copy code
python3 -m venv venv
2. Activate the environment
Windows
bash
Copy code
venv\Scripts\activate
macOS / Linux
bash
Copy code
source venv/bin/activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Run the application
bash
Copy code
streamlit run app.py
Streamlit will display a URL (e.g., http://localhost:8501).
Open it in a browser.

Camera Usage
Click:

java
Copy code
📸 SCAN MY EMOTION (Start Camera)
Grant webcam access, wait for scanning, and generated song suggestions will display.

Git Ignore Setup
Create a file named:

Copy code
.gitignore
Add:

Copy code
venv/
.venv/
If already committed earlier:

bash
Copy code
git rm -r --cached venv
git rm -r --cached .venv
git commit -m "Removed venv from repo"
Requirements
Python 3.8+

Webcam

TensorFlow & OpenCV installed via requirements.txt

Internet connection for clicking song links

Summary
This is an AI-powered real-time affective-computing system that:

captures facial expressions

interprets emotion

recommends mood-appropriate music
```
