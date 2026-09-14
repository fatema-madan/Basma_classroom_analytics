# BASMA — AI Classroom Analytics

BASMA is an AI-powered classroom analytics system that uses computer vision to understand what is happening in the classroom.

The system uses a classroom camera to capture an image and then analyzes it using AI.

BASMA can identify registered students, track attendance, and detect classroom activities.

---

## 🎯 Project Goal

The goal of BASMA is to help teachers get useful classroom insights by using AI to analyze classroom images.

BASMA combines:

- Student identification
- Attendance tracking
- Classroom activity detection
- Classroom analytics

---

## 💡 How BASMA Works

The BASMA process is simple:

**Camera → Capture Image → AI Analysis → Results**

1. The classroom camera captures an image.
2. Face Recognition identifies registered students.
3. YOLO detects classroom activities.
4. Attendance and activity information are recorded.
5. The results are shown in the BASMA application.

---

## 🤖 AI Models

### YOLO

We used YOLO, an AI model that acts like a smart eye for classroom images.

It detects classroom activities and sorts them into **8 categories**:

- Clapping
- Facing-Forward
- Hand-Raising
- Reading
- Sleeping
- Talking
- Using-Phone
- Writing

### Face Recognition

Face Recognition identifies registered students and helps track attendance.

Together, these technologies help BASMA automatically understand what is happening in the classroom.

---

## 📊 Dataset

The BASMA classroom activity dataset was prepared using **Roboflow**.

### Dataset at a Glance

- **Total images:** 9,596
- **Total objects:** 12,940
- **Average objects/image:** 1.35
- **Activity classes:** 8

### Activity Classes

- Clapping
- Facing-Forward
- Hand-Raising
- Reading
- Sleeping
- Talking
- Using-Phone
- Writing

### Dataset Source

**Roboflow — BASMA Data**

---

## ✨ Features

BASMA provides the following features:

- Student Registration
- Face Recognition
- Attendance Tracking
- Classroom Image Analysis
- Activity Detection
- Classroom Analytics
- Attendance Records
- Activity Records
- AI Analysis Results

---

## 📝 Student Registration

Teachers can register students by adding their information and a student photo.

The registered student information is used by the Face Recognition system to identify students during classroom analysis.

---

## 📷 Classroom Image Analysis

BASMA uses a classroom camera to capture an image.

The image is then analyzed by the AI models.

The system identifies registered students and detects classroom activities in the image.

---

## 📈 Classroom Analytics

BASMA collects the results of the analysis and presents useful classroom information.

The system records:

- Student attendance
- Student activity
- Date
- Time

This information can be used to understand classroom activity and attendance.

---

## 🚀 Deployed App

Try BASMA online:

**https://basmaclassa.streamlit.app/**

---

## 🎥 Demo

Watch BASMA in action:

**https://youtu.be/gkzCFMEXQaY**

---

## 🛠️ Technologies Used

- Python
- Streamlit
- YOLO
- Face Recognition
- OpenCV
- Pandas
- Google Sheets

---

## 📂 Project Structure

```text
basma/
│
├── assets/
│
├── components/
│   ├── sidebar.py
│   ├── student_registration.py
│   ├── live_classroom.py
│   ├── cards.py
│   ├── charts.py
│   └── student_profile.py
│
├── data/
│   ├── students.csv
│   ├── attendance.csv
│   └── activity_log.csv
│
├── models/
│   └── basma_yolo.pt
│
├── styles/
│   └── basma_theme.css
│
├── utils/
│   ├── data_manager.py
│   ├── face_utils.py
│   └── email_utils.py
│
├── Basma_notebook_final.ipynb
├── README.md
├── app.py
└── requirements.txt
````

---

## 🔮 Recommendations & Future Work

BASMA can be improved by:

* Adding more classroom activity classes
* Improving multi-student tracking
* Improving recognition in different lighting conditions
* Improving recognition from different camera angles
* Adding real-time classroom monitoring
* Adding more detailed engagement reports
* Adding automated parent notifications

---

## ⚠️ Limitations

The current system can be improved in challenging classroom conditions, such as different lighting and camera angles.

Student tracking can also be improved when there are multiple students in the classroom.

---

## 📌 Project Status

BASMA is a capstone project and an AI-powered classroom analytics prototype.

---

## 👩🏻‍💻 Project

**BASMA — AI Classroom Analytics**

