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
