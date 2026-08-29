import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ultralytics import YOLO

from utils.data_manager import (
    load_students,
    load_attendance,
    save_attendance,
    save_activity
)

from utils.face_utils import (
    face_app,
    load_embeddings
)


MODEL_PATH = Path("models/basma_yolo.pt")

FACE_INTERVAL = 10
FACE_THRESHOLD = 0.45
YOLO_CONFIDENCE = 0.40


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def find_student(face_embedding, embeddings):

    if not embeddings:
        return None

    face_embedding = np.array(
        face_embedding,
        dtype=np.float32
    )

    face_embedding = face_embedding / (
        np.linalg.norm(face_embedding) + 1e-8
    )

    best_student = None
    best_score = -1

    for student_id, saved_embedding in embeddings.items():

        saved_embedding = np.array(
            saved_embedding,
            dtype=np.float32
        )

        saved_embedding = saved_embedding / (
            np.linalg.norm(saved_embedding) + 1e-8
        )

        score = float(
            np.dot(
                face_embedding,
                saved_embedding
            )
        )

        if score > best_score:
            best_score = score
            best_student = str(student_id)

    if best_score >= FACE_THRESHOLD:
        return best_student

    return None


def get_student_name(student_id, students):

    if students.empty:
        return str(student_id)

    match = students[
        students["student_id"].astype(str)
        == str(student_id)
    ]

    if match.empty:
        return str(student_id)

    return str(
        match.iloc[0]["student_name"]
    )


def render_live_classroom():

    st.markdown(
        '<div class="page-title">'
        'Classroom Video Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Upload a classroom video to analyze attendance and student activities.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📹 Upload Classroom Video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is None:

        st.info(
            "Upload a classroom video to start the analysis."
        )

        return

    st.markdown("### 🎥 Video Preview")

    st.video(uploaded_file)

    if not st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    ):
        return

    analyze_video(uploaded_file)


def analyze_video(uploaded_file):

    progress = st.progress(0)

    status = st.empty()

    students = load_students()
    embeddings = load_embeddings()

    if not embeddings:

        st.error(
            "No face embeddings were found. "
            "Please register students first."
        )

        return

    model = load_model()

    suffix = Path(
        uploaded_file.name
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        input_path = Path(
            temp_file.name
        )

    output_path = Path(
        tempfile.mktemp(
            suffix=".mp4"
        )
    )

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():

        st.error(
            "Could not open the uploaded video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    detected_students = set()

    activity_counts = {}

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        current_time = datetime.now().strftime(
            "%H:%M:%S"
        )

        # =============================================
        # FACE RECOGNITION
        # =============================================

        if (
            frame_number % FACE_INTERVAL == 0
            or frame_number == 1
        ):

            try:

                faces = face_app.get(
                    frame
                )

                for face in faces:

                    student_id = find_student(
                        face.embedding,
                        embeddings
                    )

                    if student_id is None:
                        continue

                    detected_students.add(
                        student_id
                    )

                    save_attendance(
                        student_id=student_id,
                        date=today,
                        time=current_time
                    )

            except Exception as e:

                status.warning(
                    f"Face recognition warning: {e}"
                )

        # =============================================
        # YOLO
        # =============================================

        try:

            results = model.predict(
                source=frame,
                conf=YOLO_CONFIDENCE,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            result = results[0]

            annotated_frame = result.plot()

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    activity = model.names[
                        class_id
                    ]

                    activity_counts[
                        activity
                    ] = (
                        activity_counts.get(
                            activity,
                            0
                        ) + 1
                    )

        except Exception as e:

            annotated_frame = frame

            status.warning(
                f"YOLO warning: {e}"
            )

        writer.write(
            annotated_frame
        )

        if total_frames > 0:

            percentage = (
                frame_number
                / total_frames
            )

            progress.progress(
                min(
                    percentage,
                    1.0
                )
            )

            status.write(
                f"Analyzing video... "
                f"{int(percentage * 100)}%"
            )

    cap.release()

    writer.release()

    progress.progress(1.0)

    status.success(
        "Analysis completed successfully!"
    )

    # =============================================
    # SUMMARY
    # =============================================

    st.markdown(
        "### 📊 Analysis Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Students Detected",
            len(detected_students)
        )

    with col2:

        duration = (
            total_frames / fps
        )

        st.metric(
            "Video Duration",
            f"{duration:.1f}s"
        )

    with col3:

        total_activities = sum(
            activity_counts.values()
        )

        st.metric(
            "Activity Detections",
            total_activities
        )

    # =============================================
    # ATTENDANCE
    # =============================================

    st.markdown(
        "### 👥 Attendance"
    )

    attendance = load_attendance()

    today_attendance = attendance[
        attendance["date"].astype(str)
        == today
    ]

    if today_attendance.empty:

        st.warning(
            "No students were recognized in the video."
        )

    else:

        attendance_display = (
            today_attendance.copy()
        )

        names = []

        for student_id in (
            attendance_display["student_id"]
        ):

            names.append(
                get_student_name(
                    student_id,
                    students
                )
            )

        attendance_display.insert(
            1,
            "student_name",
            names
        )

        st.dataframe(
            attendance_display,
            use_container_width=True,
            hide_index=True
        )

    # =============================================
    # ANNOTATED VIDEO
    # =============================================

    st.markdown(
        "### 🎬 AI Annotated Video"
    )

    if output_path.exists():

        with open(
            output_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()

        st.video(
            video_bytes
        )

        st.download_button(
            "⬇️ Download Annotated Video",
            data=video_bytes,
            file_name="basma_annotated_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # =============================================
    # ACTIVITY SUMMARY
    # =============================================

    st.markdown(
        "### 📚 Activity Summary"
    )

    if activity_counts:

        activity_rows = []

        for activity, count in sorted(
            activity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            activity_rows.append(
                {
                    "Activity": activity,
                    "Detections": count
                }
            )

        activity_df = pd.DataFrame(
            activity_rows
        )

        st.dataframe(
            activity_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No activities were detected."
        )
