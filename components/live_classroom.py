from pathlib import Path
from datetime import datetime
import threading

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

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

FACE_INTERVAL = 10
FACE_THRESHOLD = 0.45
YOLO_CONFIDENCE = 0.40

# How often to save the same activity
ACTIVITY_SAVE_INTERVAL = 30


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# =========================================================
# FIND STUDENT
# =========================================================

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


# =========================================================
# GET STUDENT NAME
# =========================================================

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


# =========================================================
# WEBRTC CONFIG
# =========================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# =========================================================
# LIVE VIDEO PROCESSOR
# =========================================================

class BASMALiveProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = load_model()

        self.students = load_students()

        self.embeddings = load_embeddings()

        self.frame_count = 0

        self.detected_students = set()

        self.last_faces = []

        self.activity_counter = 0

        self.running = True

        self.lock = threading.Lock()


    # -----------------------------------------------------
    # PROCESS EVERY VIDEO FRAME
    # -----------------------------------------------------

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_count += 1


        # =================================================
        # YOLO ACTIVITY DETECTION
        # =================================================

        try:

            results = self.model.predict(
                source=img,
                conf=YOLO_CONFIDENCE,
                imgsz=640,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            annotated_frame = result.plot()

        except Exception:

            annotated_frame = img


        # =================================================
        # FACE RECOGNITION
        # =================================================

        if (
            self.frame_count % FACE_INTERVAL == 0
            or self.frame_count == 1
        ):

            try:

                faces = face_app.get(img)

                current_faces = []

                for face in faces:

                    student_id = find_student(
                        face.embedding,
                        self.embeddings
                    )

                    if student_id is None:
                        continue

                    student_name = get_student_name(
                        student_id,
                        self.students
                    )

                    current_faces.append(
                        {
                            "student_id": student_id,
                            "name": student_name,
                            "bbox": face.bbox
                        }
                    )

                    self.detected_students.add(
                        student_id
                    )

                    # -------------------------------------
                    # SAVE ATTENDANCE
                    # -------------------------------------

                    now = datetime.now()

                    save_attendance(
                        student_id=student_id,
                        date=now.strftime(
                            "%Y-%m-%d"
                        ),
                        time=now.strftime(
                            "%H:%M:%S"
                        )
                    )

                self.last_faces = current_faces

            except Exception:

                pass


        # =================================================
        # DRAW STUDENT NAME
        # =================================================

        for face_info in self.last_faces:

            bbox = face_info["bbox"]

            student_name = face_info["name"]

            x1, y1, x2, y2 = [
                int(value)
                for value in bbox
            ]

            # Make sure coordinates stay inside image

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(
                annotated_frame.shape[1] - 1,
                x2
            )
            y2 = min(
                annotated_frame.shape[0] - 1,
                y2
            )

            # ---------------------------------------------
            # Face box
            # ---------------------------------------------

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ---------------------------------------------
            # Student name
            # ---------------------------------------------

            label = f"{student_name} | Present"

            cv2.rectangle(
                annotated_frame,
                (
                    x1,
                    max(0, y1 - 30)
                ),
                (
                    x1 + max(
                        180,
                        len(label) * 9
                    ),
                    y1
                ),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                annotated_frame,
                label,
                (
                    x1 + 5,
                    y1 - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )


        # =================================================
        # SAVE ACTIVITY LOG
        # =================================================

        self.activity_counter += 1

        if (
            self.activity_counter
            >= ACTIVITY_SAVE_INTERVAL
        ):

            self.activity_counter = 0

            try:

                if result.boxes is not None:

                    now = datetime.now()

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        activity = self.model.names[
                            class_id
                        ]

                        # ---------------------------------
                        # If a student is recognized,
                        # associate activity with them.
                        #
                        # For now, assign the activity
                        # to all recognized students
                        # visible during the live session.
                        # ---------------------------------

                        for student_id in (
                            self.detected_students
                        ):

                            save_activity(
                                student_id=student_id,
                                date=now.strftime(
                                    "%Y-%m-%d"
                                ),
                                time=now.strftime(
                                    "%H:%M:%S"
                                ),
                                activity=activity
                            )

            except Exception:

                pass


        # =================================================
        # RETURN FRAME
        # =================================================

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# =========================================================
# RENDER LIVE CLASSROOM
# =========================================================

def render_live_classroom():

    st.markdown(
        '<div class="page-title">'
        'Live Classroom'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Monitor student attendance and classroom activities in real time.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # =====================================================
    # CAMERA
    # =====================================================

    st.markdown(
        "### 🎥 Classroom Camera"
    )

    st.info(
        "Click START and allow camera access "
        "when your browser asks."
    )


    ctx = webrtc_streamer(

        key="basma-live-classroom",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=BASMALiveProcessor,

        rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


    # =====================================================
    # STATUS
    # =====================================================

    if ctx.state.playing:

        st.success(
            "🟢 Live camera is running. "
            "BASMA is detecting activities and attendance."
        )

    else:

        st.warning(
            "🟡 Camera is stopped."
        )


    st.divider()


    # =====================================================
    # LIVE INFORMATION
    # =====================================================

    st.markdown(
        "### 🤖 AI Detection"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "YOLO"
        )

    with col2:

        st.metric(
            "Confidence",
            f"{YOLO_CONFIDENCE:.0%}"
        )

    with col3:

        st.metric(
            "Mode",
            "Real-Time"
        )


    # =====================================================
    # TODAY'S ATTENDANCE
    # =====================================================

    st.markdown(
        "### 👥 Today's Attendance"
    )

    try:

        attendance = load_attendance()

        students = load_students()

        today = datetime.now().strftime(
            "%Y-%m-%d"
        )

        if not attendance.empty:

            today_attendance = attendance[
                attendance["date"].astype(str)
                == today
            ].copy()

        else:

            today_attendance = pd.DataFrame()


        if today_attendance.empty:

            st.info(
                "No students have been recognized yet."
            )

        else:

            names = []

            for student_id in (
                today_attendance["student_id"]
            ):

                names.append(
                    get_student_name(
                        student_id,
                        students
                    )
                )

            today_attendance.insert(
                1,
                "student_name",
                names
            )

            st.dataframe(
                today_attendance,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:

        st.warning(
            f"Attendance display warning: {e}"
        )


    # =====================================================
    # CURRENT ACTIVITIES
    # =====================================================

    st.markdown(
        "### 📚 Activity Log"
    )

    try:

        from utils.data_manager import load_activity

        activities = load_activity()

        today = datetime.now().strftime(
            "%Y-%m-%d"
        )

        if not activities.empty:

            today_activities = activities[
                activities["date"].astype(str)
                == today
            ].copy()

            if not today_activities.empty:

                students = load_students()

                names = []

                for student_id in (
                    today_activities["student_id"]
                ):

                    names.append(
                        get_student_name(
                            student_id,
                            students
                        )
                    )

                today_activities.insert(
                    1,
                    "student_name",
                    names
                )

                st.dataframe(
                    today_activities.tail(20),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No activities recorded yet."
                )

        else:

            st.info(
                "No activities recorded yet."
            )

    except Exception as e:

        st.warning(
            f"Activity display warning: {e}"
        )

