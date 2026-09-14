# BASMA — AI Classroom Analytics

BASMA is an AI-powered classroom analytics system that uses computer vision to understand what is happening in the classroom.

The system analyzes classroom images using artificial intelligence to identify registered students, track attendance, and detect different classroom activities.

---

## 🎯 Project Goal

The goal of BASMA is to help teachers gain useful insights into classroom activity by using AI and computer vision.

BASMA combines:

* Student identification
* Attendance tracking
* Classroom activity detection
* Classroom analytics

---

## 💡 How BASMA Works

The BASMA process is simple:

**Camera → Capture Image → AI Analysis → Results**

1. A classroom image is captured.
2. Face Recognition identifies registered students.
3. YOLO detects classroom activities.
4. Attendance and activity information are recorded.
5. The results are displayed in the BASMA application.

---

## 🤖 AI Models

### YOLO

BASMA uses YOLO as the computer vision model for detecting classroom activities.

The model detects and classifies the following 8 activities:

* 👏 Clapping
* 👀 Facing Forward
* 🙋 Hand Raising
* 📖 Reading
* 😴 Sleeping
* 🗣️ Talking
* 📱 Using Phone
* ✍️ Writing

### Face Recognition

Face Recognition is used to identify registered students and support attendance tracking.

Together, these AI technologies help BASMA automatically understand what is happening inside the classroom.

---

## 📊 Dataset

The BASMA classroom activity dataset was prepared and managed using Roboflow.

### Dataset at a Glance

* Total images: **9,596**
* Total objects: **12,940**
* Average objects per image: **1.35**
* Activity classes: **8**

### Activity Classes

* Clapping
* Facing-Forward
* Hand-Raising
* Reading
* Sleeping
* Talking
* Using-Phone
* Writing

### Dataset Source

The BASMA dataset is available on Roboflow:

🔗 **[View the BASMA Dataset on Roboflow](https://app.roboflow.com/fatema-yusuf/basma_data-31mu8)**

---

## ✨ Features

BASMA currently provides:

* 👤 Student Registration
* 🧠 Face Recognition
* 📅 Attendance Tracking
* 📷 Classroom Image Analysis
* 🤖 AI Activity Detection
* 📊 Classroom Analytics
* 📝 Attendance Records
* 📈 Activity Records
* 🔍 AI Analysis Results

---

## 📝 Student Registration

Teachers can register students by adding their information and a student photo.

The registered student information is used by the Face Recognition system to identify students during classroom analysis.

---

## 📷 Classroom Analysis

BASMA analyzes classroom images using computer vision and AI models.

The system can:

* Identify registered students
* Track attendance
* Detect classroom activities
* Record activity results
* Display classroom insights and analytics

---

## 📈 Classroom Analytics

BASMA collects the results of classroom analysis and presents useful information about classroom activity.

The system records:

* Student attendance
* Detected student activities
* Date
* Time

This information can help teachers better understand attendance and classroom behavior patterns.

---

## 🚀 Live Application

Try BASMA online:

🔗 **[Open BASMA Live App](https://basmaclassroom.streamlit.app/)**

---

## 🎥 Demo

Watch BASMA in action:

🔗 **[Watch the BASMA Demo on YouTube](https://youtu.be/gkzCFMEXQaY)**

---

## 🛠️ Technologies Used

* Python
* Streamlit
* YOLO
* Face Recognition
* OpenCV
* Pandas
* Google Sheets

---

## 📂 Project Structure

```text
basma_classroom_analytics/
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
├── requirements.txt
└── packages.txt
```

---

## 🔮 Future Work

BASMA can be further developed into a fully real-time intelligent classroom monitoring system.

### 📹 Live Classroom Camera

A future version of BASMA could connect directly to a live classroom camera.

The camera could continuously capture and analyze classroom activity in real time instead of analyzing individual images.

**Live Camera → AI Detection → Activity Analysis → Real-Time Alerts**

---

### 🤖 Real-Time Activity Detection

The system could continuously detect student activities directly from the live classroom video stream.

This could include detecting activities such as:

* Talking
* Using a phone
* Sleeping
* Reading
* Writing
* Raising a hand
* Other classroom behaviors

---

### 🔔 Teacher Alerts

BASMA could provide real-time notifications directly inside the application.

For example, when the system detects a student repeatedly:

* Talking during class
* Using a phone
* Sleeping
* Engaging in another predefined behavior

The teacher could receive an instant alert through the BASMA application.

This would allow teachers to be informed quickly about detected classroom behaviors without needing to manually monitor the analytics dashboard.

---

### 👨‍👩‍👧 Parent Notifications

A future version of BASMA could integrate a real-time messaging or notification system for parents.

When specific behaviors are detected repeatedly or exceed predefined thresholds, the system could automatically send a notification to the student's parent.

For example:

**Repeated Activity Detection → Behavior Threshold → Parent Notification**

Notifications could be delivered through a messaging service or another communication platform.

> **Note:** Parent email or messaging notifications are not currently active in the deployed version of BASMA.

---

### 🧠 Smart Behavior Monitoring

Future versions could track repeated behaviors over time.

Instead of generating an alert based on a single detection, BASMA could analyze behavior patterns and trigger alerts only when a predefined threshold is reached.

For example:

* A student is detected talking multiple times.
* A student repeatedly uses a phone during class.
* A student is frequently detected sleeping.

This could help reduce unnecessary alerts and provide more meaningful classroom insights.

---

### 👥 Improved Multi-Student Tracking

Future development could improve the ability to track and analyze multiple students simultaneously within a live classroom environment.

This could allow BASMA to maintain activity records for individual students throughout a classroom session.

---

### 📊 Enhanced Engagement Reports

Future versions could generate more detailed reports about:

* Student attendance
* Classroom engagement
* Activity patterns
* Repeated behaviors
* Individual student insights
* Overall classroom trends

---

### 🌍 Improved Performance in Different Environments

The AI models could be further improved to perform better under different:

* Lighting conditions
* Classroom layouts
* Camera angles
* Student positions
* Classroom sizes

---

## ⚠️ Current Limitations

BASMA is currently an AI-powered classroom analytics prototype.

The current version can be further improved in challenging classroom conditions, including:

* Different lighting conditions
* Different camera angles
* Multiple students appearing simultaneously
* Student occlusion
* Variations in classroom layouts

The current deployed version also does not yet provide automatic parent email or messaging notifications.

---

## 📌 Project Status

BASMA was developed as a Data Science capstone project and demonstrates how computer vision and artificial intelligence can be applied to classroom analytics.

The project is designed as a prototype and provides a foundation for future development into a fully real-time intelligent classroom monitoring and alert system.

---

## 👩🏻‍💻 Project

**BASMA — AI Classroom Analytics**

An AI-powered classroom analytics system using computer vision to support student attendance tracking, activity detection, and classroom insights.
