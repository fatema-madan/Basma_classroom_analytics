```python
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

# Face recognition every 10 frames
FACE_FRAME_INTERVAL = 10

# Minimum cosine similarity
FACE_THRESHOLD = 0.45

# Save the same activity for the same student
# only once every few seconds
ACTIVITY_SAVE_INTERVAL = 3


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():

    return YOLO(
        str(MODEL_PATH)
    )


# =========================================================
# FIND STUDENT
# =========================================================

def find_student_from_embeddings(
    face_embedding,
    embeddings,
    threshold=FACE_THRESHOLD
):
    """
    Compare a detected face embedding
    with registered student embeddings.
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
            best_student = str(
                student_id
            )

    if best_score >= threshold:

        return best_student

    return None


# =========================================================
# IOU
# =========================================================

def calculate_iou(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

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

    union = (
        area1
        + area2
        - intersection
    )

    if union <= 0:

        return 0

    return (
        intersection / union
    )


# =========================================================
# MATCH ACTIVITY TO STUDENT
# =========================================================

def match_activity_to_student(
    activity_box,
    detected_faces
):

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

            best_student = (
                face_info["student_id"]
            )

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

    # =====================================================
    # LOAD MODELS / DATA
    # =====================================================

    model = load_yolo_model()

    students = load_students()

    embeddings = load_embeddings()

    # =====================================================
    # OPEN VIDEO
    # =====================================================

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

    # =====================================================
    # OUTPUT VIDEO
    # =====================================================

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
            "Could not create output video."
        )

    # =====================================================
    # TRACKING
    # =====================================================

    activity_counts = {
        activity: 0
        for activity in ACTIVITIES
    }

    student_activity_counts = {}

    detected_students = set()

    current_faces = []

    last_activity_saved = {}

    frame_number = 0

    # =====================================================
    # DATE / TIME
    # =====================================================

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # =====================================================
    # PROCESS FRAMES
    # =====================================================

    while True:

        success, frame = cap.read()

        if not success:

            break

        frame_number += 1

        video_time = (
            frame_number / fps
        )

        current_time = datetime.now().strftime(
            "%H:%M:%S"
        )

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

                    # -------------------------------------
                    # ATTENDANCE
                    # -------------------------------------

                    save_attendance(
                        student_id=student_id,
                        date=today,
                        time=current_time
                    )

            except Exception as e:

                status_text.warning(
                    f"Face recognition warning: {e}"
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

        except Exception as e:

            status_text.warning(
                f"YOLO warning: {e}"
            )

            writer.write(
                frame
            )

            continue

        # =================================================
        # ANNOTATED FRAME
        # =================================================

        annotated_frame = (
            result.plot()
        )

        # =================================================
        # ACTIVITY DETECTION
        # =================================================

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                activity_name = (
                    model.names[class_id]
                )

                if (
                    activity_name
                    not in ACTIVITIES
                ):

                    continue

                # -----------------------------------------
                # Count activity
                # -----------------------------------------

                activity_counts[
                    activity_name
                ] += 1

                # -----------------------------------------
                # Activity box
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
                # Match to student
                # -----------------------------------------

                student_id = (
                    match_activity_to_student(
                        activity_box,
                        current_faces
                    )
                )

                if student_id is None:

                    continue

                # -----------------------------------------
                # Student activity counts
                # -----------------------------------------

                if (
                    student_id
                    not in student_activity_counts
                ):

                    student_activity_counts[
                        student_id
                    ] = {}

                if (
                    activity_name
                    not in student_activity_counts[
                        student_id
                    ]
                ):

                    student_activity_counts[
                        student_id
                    ][activity_name] = 0

                student_activity_counts[
                    student_id
                ][activity_name] += 1

                # -----------------------------------------
                # Save activity
                # -----------------------------------------

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
                    video_time - last_saved
                    >= ACTIVITY_SAVE_INTERVAL
                ):

                    try:

                        save_activity(
                            student_id=student_id,
                            date=today,
                            time=current_time,
                            activity=activity_name
                        )

                        last_activity_saved[
                            activity_key
                        ] = video_time

                    except Exception as e:

                        status_text.warning(
                            f"Activity warning: {e}"
                        )

        # =================================================
        # DRAW RECOGNIZED STUDENT NAMES
        # =================================================

        for face_info in current_faces:

            student_id = (
                face_info["student_id"]
            )

            box = face_info["box"]

            student_name = str(
                student_id
            )

            if not students.empty:

                match = students[
                    students[
                        "student_id"
                    ].astype(str)
                    == str(student_id)
                ]

                if not match.empty:

                    student_name = str(
                        match.iloc[0][
                            "student_name"
                        ]
                    )

            x1, y1, x2, y2 = box

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                student_name,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
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

    # =====================================================
    # CLOSE VIDEO
    # =====================================================

    cap.release()

    writer.release()

    progress_bar.progress(
        1.0
    )

    status_text.success(
        "Analysis completed successfully!"
    )

    return {
        "activity_counts":
            activity_counts,

        "student_activity_counts":
            student_activity_counts,

        "detected_students":
            detected_students,

        "duration":
            (
                total_frames / fps
                if fps > 0
                else 0
            )
    }


# =========================================================
# MAIN PAGE
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
        "📹 Upload Classroom Video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ],
        help="Upload a classroom recording."
    )

    if uploaded_file is None:

        st.info(
            "Upload a classroom video to start the analysis."
        )

        return

    # =====================================================
    # PREVIEW
    # =====================================================

    st.markdown(
        "### 🎥 Video Preview"
    )

    st.video(
        uploaded_file
    )

    # =====================================================
    # ANALYZE
    # =====================================================

    if not st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    ):

        return

    # =====================================================
    # SAVE INPUT VIDEO
    # =====================================================

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

    # =====================================================
    # OUTPUT
    # =====================================================

    output_path = Path(
        tempfile.mktemp(
            suffix=".mp4"
        )
    )

    # =====================================================
    # ANALYSIS
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
    # METRICS
    # =====================================================

    st.markdown(
        "### 📊 Analysis Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Students Detected",
            len(
                results[
                    "detected_students"
                ]
            )
        )

    with col2:

        st.metric(
            "Video Duration",
            f"{results['duration']:.1f}s"
        )

    with col3:

        total_activities = sum(
            results[
                "activity_counts"
            ].values()
        )

        st.metric(
            "Activity Detections",
            total_activities
        )

    # =====================================================
    # ATTENDANCE
    # =====================================================

    st.markdown(
        "### 👥 Attendance"
    )

    attendance = load_attendance()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    today_attendance = attendance[
        attendance["date"].astype(str)
        == today
    ]

    if today_attendance.empty:

        st.warning(
            "No students were recognized in the video."
        )

    else:

        display_attendance = (
            today_attendance.copy()
        )

        if not display_attendance.empty:

            student_names = []

            for student_id in (
                display_attendance[
                    "student_id"
                ]
            ):

                student_name = str(
                    student_id
                )

                if not load_students().empty:

                    match = load_students()[
                        load_students()[
                            "student_id"
                        ].astype(str)
                        == str(student_id)
                    ]

                    if not match.empty:

                        student_name = str(
                            match.iloc[0][
                                "student_name"
                            ]
                        )

                student_names.append(
                    student_name
                )

            display_attendance.insert(
                1,
                "student_name",
                student_names
            )

        st.dataframe(
            display_attendance,
            use_container_width=True,
            hide_index=True
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
            "⬇️ Download Annotated Video",
            data=video_bytes,
            file_name="basma_annotated_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # =====================================================
    # ACTIVITY SUMMARY
    # =====================================================

    st.markdown(
        "### 📚 Activity Summary"
    )

    activity_rows = []

    for activity in ACTIVITIES:

        activity_rows.append(
            {
                "Activity": activity,
                "Detections": results[
                    "activity_counts"
                ].get(
                    activity,
                    0
                )
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

    # =====================================================
    # STUDENT ACTIVITY
    # =====================================================

    st.markdown(
        "### 👤 Student Activity"
    )

    student_activity_counts = results[
        "student_activity_counts"
    ]

    student_rows = []

    students = load_students()

    for student_id, activities in (
        student_activity_counts.items()
    ):

        student_name = str(
            student_id
        )

        if not students.empty:

            match = students[
                students[
                    "student_id"
                ].astype(str)
                == str(student_id)
            ]

            if not match.empty:

                student_name = str(
                    match.iloc[0][
                        "student_name"
                    ]
                )

        for activity, count in (
            activities.items()
        ):

            student_rows.append(
                {
                    "Student": student_name,
                    "Activity": activity,
                    "Detections": count
                }
            )

    if student_rows:

        student_activity_df = (
            pd.DataFrame(
                student_rows
            )
        )

        student_activity_df = (
            student_activity_df.sort_values(
                "Detections",
                ascending=False
            )
        )

        st.dataframe(
            student_activity_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No student-specific activities were matched."
        )
```
