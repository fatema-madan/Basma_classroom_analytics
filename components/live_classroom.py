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
    save_attendance,
    save_activity,
)

from utils.face_utils import (
    face_app,
    load_embeddings,
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = Path("models/basma_yolo.pt")

ACTIVITIES = [
    "Clapping",
    "Facing-Forward",
    "Hand-Raising",
    "Reading",
    "Sleeping",
    "Talking",
    "Using-Phone",
    "Writing",
]

YOLO_CONFIDENCE = 0.40

# Face recognition does not need to run on every frame.
FACE_FRAME_INTERVAL = 5

# Save activity only once every few seconds
ACTIVITY_SAVE_INTERVAL = 3

# Face similarity threshold
FACE_THRESHOLD = 0.45


# =========================================================
# LOAD YOLO
# =========================================================

@st.cache_resource
def load_yolo_model():

    return YOLO(str(MODEL_PATH))


# =========================================================
# FACE MATCHING
# =========================================================

def find_student_from_embeddings(
    face_embedding,
    embeddings,
    threshold=FACE_THRESHOLD
):
    """
    Find the closest registered student using cosine similarity.
    """

    if not embeddings:
        return None

    face_embedding = np.asarray(
        face_embedding,
        dtype=np.float32
    )

    face_embedding = face_embedding / (
        np.linalg.norm(face_embedding) + 1e-8
    )

    best_student = None
    best_score = -1

    for student_id, saved_embedding in embeddings.items():

        saved_embedding = np.asarray(
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

    if best_score >= threshold:
        return best_student

    return None


# =========================================================
# IOU
# =========================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = max(
        0,
        box1[2] - box1[0]
    ) * max(
        0,
        box1[3] - box1[1]
    )

    area2 = max(
        0,
        box2[2] - box2[0]
    ) * max(
        0,
        box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# =========================================================
# FIND STUDENT FOR ACTIVITY
# =========================================================

def match_activity_to_face(
    activity_box,
    detected_faces
):
    """
    Match a YOLO activity box to the closest detected face.
    """

    best_student = None
    best_iou = 0

    for face_info in detected_faces:

        face_box = face_info["box"]

        iou = calculate_iou(
            activity_box,
            face_box
        )

        if iou > best_iou:

            best_iou = iou
            best_student = face_info["student_id"]

    return best_student


# =========================================================
# PROCESS VIDEO
# =========================================================

def process_video(
    input_path,
    output_path,
    progress_bar,
    status_text
):

    model = load_yolo_model()

    students = load_students()

    embeddings = load_embeddings()

    # -----------------------------------------------------
    # Open video
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open the uploaded video."
        )

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

    # -----------------------------------------------------
    # Output video
    # -----------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create the output video."
        )

    # -----------------------------------------------------
    # Tracking data
    # -----------------------------------------------------

    activity_counts = {
        activity: 0
        for activity in ACTIVITIES
    }

    student_activity_counts = {}

    detected_students = set()

    last_activity_saved = {}

    current_faces = []

    frame_number = 0

    # -----------------------------------------------------
    # Process frames
    # -----------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        current_time = frame_number / fps

        # =================================================
        # FACE RECOGNITION
        # =================================================

        if (
            frame_number % FACE_FRAME_INTERVAL == 0
            or frame_number == 1
        ):

            current_faces = []

            try:

                faces = face_app.get(
                    frame
                )

                for face in faces:

                    student_id = (
                        find_student_from_embeddings(
                            face.embedding,
                            embeddings
                        )
                    )

                    if student_id is None:
                        continue

                    bbox = face.bbox.astype(
                        int
                    )

                    face_box = [
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    ]

                    current_faces.append(
                        {
                            "student_id": student_id,
                            "box": face_box,
                        }
                    )

                    detected_students.add(
                        student_id
                    )

            except Exception as e:

                status_text.write(
                    f"Face recognition warning: {e}"
                )

        # =================================================
        # ATTENDANCE
        # =================================================

        today = datetime.now().strftime(
            "%Y-%m-%d"
        )

        current_clock_time = datetime.now().strftime(
            "%H:%M:%S"
        )

        for student_id in detected_students:

            try:

                first_seen = save_attendance(
                    student_id=student_id,
                    date=today,
                    time=current_clock_time
                )

                # Only first detection creates
                # the attendance record.
                if first_seen:

                    pass

            except Exception as e:

                status_text.write(
                    f"Attendance warning: {e}"
                )

        # =================================================
        # YOLO
        # =================================================

        try:

            results = model.predict(
                source=frame,
                conf=YOLO_CONFIDENCE,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            result = results[0]

            boxes = result.boxes

        except Exception as e:

            status_text.write(
                f"YOLO warning: {e}"
            )

            writer.write(frame)

            continue

        # =================================================
        # DRAW DETECTIONS
        # =================================================

        annotated_frame = result.plot()

        # =================================================
        # ACTIVITY PROCESSING
        # =================================================

        if boxes is not None:

            for box in boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                activity_name = model.names[
                    class_id
                ]

                if activity_name not in ACTIVITIES:
                    continue

                activity_counts[
                    activity_name
                ] += 1

                # -----------------------------------------
                # Bounding box
                # -----------------------------------------

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                activity_box = [
                    int(coordinates[0]),
                    int(coordinates[1]),
                    int(coordinates[2]),
                    int(coordinates[3]),
                ]

                # -----------------------------------------
                # Match activity to student
                # -----------------------------------------

                student_id = (
                    match_activity_to_face(
                        activity_box,
                        current_faces
                    )
                )

                # -----------------------------------------
                # Student activity count
                # -----------------------------------------

                if student_id is not None:

                    if student_id not in student_activity_counts:

                        student_activity_counts[
                            student_id
                        ] = {}

                    if activity_name not in (
                        student_activity_counts[
                            student_id
                        ]
                    ):

                        student_activity_counts[
                            student_id
                        ][activity_name] = 0

                    student_activity_counts[
                        student_id
                    ][activity_name] += 1

                    # -------------------------------------
                    # Save activity periodically
                    # -------------------------------------

                    activity_key = (
                        f"{student_id}_{activity_name}"
                    )

                    last_saved = (
                        last_activity_saved.get(
                            activity_key,
                            -999
                        )
                    )

                    if (
                        current_time - last_saved
                        >= ACTIVITY_SAVE_INTERVAL
                    ):

                        try:

                            save_activity(
                                student_id=student_id,
                                date=today,
                                time=current_clock_time,
                                activity=activity_name
                            )

                            last_activity_saved[
                                activity_key
                            ] = current_time

                        except Exception as e:

                            status_text.write(
                                f"Activity warning: {e}"
                            )

        # =================================================
        # WRITE FRAME
        # =================================================

        writer.write(
            annotated_frame
        )

        # =================================================
        # PROGRESS
        # =================================================

        if total_frames > 0:

            progress = (
                frame_number
                / total_frames
            )

            progress_bar.progress(
                min(
                    progress,
                    1.0
                )
            )

            status_text.write(
                f"Analyzing video... "
                f"{int(progress * 100)}%"
            )

    # -----------------------------------------------------
    # Release
    # -----------------------------------------------------

    cap.release()
    writer.release()

    progress_bar.progress(1.0)

    status_text.write(
        "Analysis completed."
    )

    return {
        "activity_counts": activity_counts,
        "student_activity_counts": student_activity_counts,
        "detected_students": detected_students,
        "fps": fps,
        "total_frames": total_frames,
        "duration": (
            total_frames / fps
            if fps > 0
            else 0
        )
    }


# =========================================================
# RENDER PAGE
# =========================================================

def render_live_classroom():

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        '<div class="page-title">'
        'Classroom Video Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Upload a classroom recording to analyze '
        'attendance and student activities.'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # UPLOAD
    # =====================================================

    uploaded_file = st.file_uploader(
        "Upload Classroom Video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ],
        help="Upload a classroom video for AI analysis."
    )

    if uploaded_file is None:

        st.info(
            "Upload a classroom video to start the analysis."
        )

        return

    # =====================================================
    # ORIGINAL VIDEO
    # =====================================================

    st.markdown(
        "### 🎥 Uploaded Video"
    )

    st.video(
        uploaded_file
    )

    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    analyze_button = st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    )

    if not analyze_button:
        return

    # =====================================================
    # SAVE UPLOADED VIDEO
    # =====================================================

    input_suffix = Path(
        uploaded_file.name
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=input_suffix
    ) as temp_input:

        temp_input.write(
            uploaded_file.getbuffer()
        )

        input_path = Path(
            temp_input.name
        )

    # =====================================================
    # OUTPUT VIDEO
    # =====================================================

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_path = Path(
        output_file.name
    )

    output_file.close()

    # =====================================================
    # PROGRESS
    # =====================================================

    st.markdown(
        "### 🤖 AI Analysis"
    )

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    try:

        results = process_video(
            input_path=input_path,
            output_path=output_path,
            progress_bar=progress_bar,
            status_text=status_text
        )

    except Exception as e:

        st.error(
            f"Video analysis failed: {e}"
        )

        return

    # =====================================================
    # RESULTS
    # =====================================================

    st.success(
        "Video analysis completed successfully!"
    )

    # =====================================================
    # METRICS
    # =====================================================

    duration = results[
        "duration"
    ]

    detected_students = results[
        "detected_students"
    ]

    activity_counts = results[
        "activity_counts"
    ]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Students Detected",
            len(detected_students)
        )

    with col2:

        st.metric(
            "Video Duration",
            f"{duration:.1f}s"
        )

    with col3:

        total_detections = sum(
            activity_counts.values()
        )

        st.metric(
            "Activity Detections",
            total_detections
        )

    # =====================================================
    # ANNOTATED VIDEO
    # =====================================================

    st.markdown(
        "### 🎬 AI Annotated Video"
    )

    if output_path.exists():

        with open(
            output_path,
            "rb"
        ) as video_file:

            video_bytes = (
                video_file.read()
            )

        st.video(
            video_bytes
        )

        st.download_button(
            label="⬇️ Download Annotated Video",
            data=video_bytes,
            file_name="basma_annotated_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # =====================================================
    # ACTIVITY SUMMARY
    # =====================================================

    st.markdown(
        "### 📊 Activity Summary"
    )

    activity_df = pd.DataFrame(
        {
            "Activity": list(
                activity_counts.keys()
            ),
            "Detections": list(
                activity_counts.values()
            )
        }
    )

    activity_df = activity_df.sort_values(
        "Detections",
        ascending=False
    )

    st.dataframe(
        activity_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # STUDENT SUMMARY
    # =====================================================

    st.markdown(
        "### 👥 Student Activity Summary"
    )

    student_activity_counts = results[
        "student_activity_counts"
    ]

    if not student_activity_counts:

        st.info(
            "No student-specific activities were matched."
        )

        return

    students = load_students()

    summary_rows = []

    for student_id, activities in (
        student_activity_counts.items()
    ):

        student_name = student_id

        if not students.empty:

            match = students[
                students[
                    "student_id"
                ].astype(str)
                == str(student_id)
            ]

            if not match.empty:

                student_name = match.iloc[0][
                    "student_name"
                ]

        for activity, count in activities.items():

            summary_rows.append(
                {
                    "Student": student_name,
                    "Activity": activity,
                    "Detections": count
                }
            )

    if summary_rows:

        student_df = pd.DataFrame(
            summary_rows
        )

        student_df = student_df.sort_values(
            [
                "Student",
                "Detections"
            ],
            ascending=[
                True,
                False
            ]
        )

        st.dataframe(
            student_df,
            use_container_width=True,
            hide_index=True
        )
