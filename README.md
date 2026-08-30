# 🌿 BASMA — AI Classroom Analytics

BASMA is an AI-powered classroom analytics system designed to help teachers understand student attendance and classroom activities using **Computer Vision**.

The system analyzes uploaded classroom videos using **YOLO for activity detection** and **Face Recognition for student identification and attendance tracking**.

## ✨ Features

- 👤 Student Registration
- 🧑‍💻 Face Recognition
- 👥 Automated Attendance
- 🎥 Classroom Video Analysis
- 🤖 AI Activity Detection
- 📊 Classroom Analytics
- 📈 Attendance & Activity Reports
- 🎬 AI-annotated video output

## 🎯 Detected Classroom Activities

BASMA currently detects 8 classroom activities:

- 👏 Clapping
- 🧍 Facing-Forward
- 🙋 Hand-Raising
- 📖 Reading
- 😴 Sleeping
- 💬 Talking
- 📱 Using-Phone
- ✍️ Writing

## 🧠 How It Works

```text
Student Registration
        ↓
Student Photo
        ↓
Face Embedding
        ↓
Face Embeddings Database
        ↓
Upload Classroom Video
        ↓
 ┌───────────────┐
 │               │
 ↓               ↓
YOLO          Face Recognition
 ↓               ↓
Activities     Student ID
 ↓               ↓
Activity Log   Attendance
 └───────┬───────┘
         ↓
   Classroom Analytics
         ↓
   AI Annotated Video
```

## 🛠️ Technologies

* **Python**
* **Streamlit**
* **Ultralytics YOLO**
* **InsightFace**
* **OpenCV**
* **Pandas**
* **NumPy**
* **Plotly**

## 📁 Project Structure

```text
Basma_classroom_analytics/
│
├── app.py
├── requirements.txt
│
├── components/
│   ├── sidebar.py
│   ├── student_registration.py
│   ├── live_classroom.py
│   ├── cards.py
│   ├── charts.py
│   └── student_profile.py
│
├── utils/
│   ├── data_manager.py
│   ├── face_utils.py
│   └── email_utils.py
│
├── models/
│   └── basma_yolo.pt
│
├── data/
│   ├── students.csv
│   ├── attendance.csv
│   └── activity_log.csv
│
└── styles/
    └── basma_theme.css

```

## 🚀 Running the Project

Install the required packages:

```bash
pip install -r requirements.txt

```

Run the Streamlit application:

```bash
streamlit run app.py

```

## 📹 Classroom Video Analysis

Instead of relying on a live classroom camera, BASMA allows teachers to upload a recorded classroom video.

The system then:

1. Processes the uploaded video.
2. Detects classroom activities using YOLO.
3. Identifies registered students using Face Recognition.
4. Records student attendance.
5. Generates activity information.
6. Produces an AI-annotated video.
7. Displays classroom analytics.

## 👤 Student Registration

Before analyzing a classroom video, students can be registered with:

* Student ID
* Student name
* Parent email
* Parent phone
* Student photo

The student's face is converted into an embedding and stored for later recognition.

## 📊 Data

BASMA stores classroom information in CSV files:

### Students

```text
student_id
student_name
parent_email
parent_phone
photo_path
```

### Attendance

```text
student_id
date
first_seen
last_seen
status
```

### Activity Log

```text
student_id
date
time
activity
```

## 🌱 Project Goal

BASMA aims to transform classroom observations into useful data that can help teachers understand:

* Student attendance
* Classroom engagement
* Student activities
* Behavioral patterns
* Overall classroom performance

The goal is to provide teachers with a simple AI-powered tool for **data-driven classroom insights**.

## 🔮 Future Improvements

* Real-time classroom monitoring
* More classroom activity classes
* Improved multi-student tracking
* Advanced student performance analytics
* Automated parent notifications
* More detailed engagement reports
* Improved recognition in challenging lighting and camera angles

## 📌 Project Status

**Prototype / Capstone Project**

BASMA is currently being developed as a Computer Vision and Data Science capstone project.

---

### 🌿 BASMA

**AI-powered classroom analytics through Computer Vision.**
