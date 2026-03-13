Real-Time Facial Emotion Detection System

A real-time facial emotion detection system built using Python, OpenCV, DeepFace, PyQt5, and MySQL.
The system detects human emotions from a live webcam feed, analyzes facial expressions using deep learning, and stores the results in a database for further analysis.

This project demonstrates the integration of computer vision, machine learning, database management, and GUI development.

⸻

Features

Real-Time Emotion Detection
	•	Detects emotions directly from live webcam input
	•	Uses DeepFace deep learning model for emotion analysis
	•	Supports emotions:
	•	Happy
	•	Sad
	•	Angry
	•	Neutral

Face Detection
	•	Uses OpenCV Haar Cascade Classifier
	•	Detects faces before performing emotion analysis

Database Storage

Each detected emotion is stored in a MySQL database including:
	•	Timestamp
	•	Detected emotion
	•	Confidence score
	•	Face detection status
	•	Image path (optional)

GUI Interface
	•	Built using PyQt5
	•	Displays real-time emotion detection
	•	Shows webcam feed
	•	Interactive interface

⸻

Project Structure

Emotion-Detection-System
│
├── fist_accu.py        # Core emotion detection logic
├── fist_ui.py          # PyQt5 graphical interface
└── README.md


⸻

Technologies Used

Programming Language
	•	Python

Libraries
	•	OpenCV
	•	NumPy
	•	DeepFace
	•	PyQt5
	•	MySQL Connector

Database
	•	MySQL

Computer Vision
	•	Haar Cascade Face Detection
	•	Deep Learning Emotion Recognition

⸻

Installation

1. Clone the Repository

git clone https://github.com/yourusername/emotion-detection-system.git
cd emotion-detection-system


⸻

2. Install Required Libraries

pip install opencv-python
pip install numpy
pip install deepface
pip install mysql-connector-python
pip install pyqt5

Or install all together:

pip install opencv-python numpy deepface mysql-connector-python pyqt5


⸻

MySQL Database Setup

Make sure MySQL server is running.

Update database credentials inside fist_accu.py:

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'database': 'emotion_detection'
}

The program will automatically create the required database table:

emotion_records

Table structure:

Column	Description
id	Auto increment ID
timestamp	Detection time
emotion	Detected emotion
confidence	Confidence score
face_detected	Boolean value
image_path	Stored frame path


⸻

Running the Application

Run the GUI application:

python fist_ui.py

The program will:
	1.	Start the webcam
	2.	Detect faces in real-time
	3.	Analyze facial emotion using DeepFace
	4.	Display emotion on screen
	5.	Save detection results into MySQL database

Press:

q → Quit application
s → Save current frame


⸻

How It Works

Step 1: Capture Frame

The webcam continuously captures frames using OpenCV.

Step 2: Face Detection

Faces are detected using Haar Cascade Classifier.

Step 3: Emotion Analysis

Detected face regions are passed to DeepFace which predicts emotion probabilities.

Step 4: Emotion Classification

The system selects the emotion with the highest probability score.

Step 5: Data Storage

Emotion results are stored in the MySQL database.

⸻

Example Output

Emotion Detected: Happy
Confidence: 0.92
Timestamp: 2026-03-13 11:20:32
Face Detected: True


⸻

Future Improvements

Possible upgrades for this project:
	•	Multi-person emotion tracking
	•	Emotion analytics dashboard
	•	Emotion trend graphs
	•	Cloud database integration
	•	Emotion-based recommendation systems
	•	Mental health monitoring system
	•	AI-powered behavioral analysis

⸻

Applications

This system can be used in:
	•	Human Computer Interaction
	•	Smart classrooms
	•	Mental health monitoring
	•	Customer emotion analysis
	•	Security and surveillance systems
	•	Emotion-aware AI assistants

⸻

Author

Developed as part of an AI and Computer Vision learning project.
