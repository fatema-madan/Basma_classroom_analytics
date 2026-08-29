import time
from datetime import datetime

import av
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

from utils.data_manager import (
    load_students,
    save_attendance,
    save_activity
)

from utils.face_utils import (
    face_app,
    find_student
)

from utils.email_utils import send_attendance_email


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/basma_yolo.pt"

FACE_CHECK_INTERVAL_SECONDS = 3
ACTIVITY_SAVE_INTERVAL_SECONDS = 3


# =========================================================
# LOAD MODEL ONCE
# =========================================================

@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)


model = load_yolo_model()


# =========================================================
# CLASSROOM PROCESSOR
# =========================================================

class ClassroomProcessor(VideoProcessorBase):

    def __init__(self):

        self.last_face_check = 0
        self.last_activity_save = 0

        self.students = load_students()

    def recv(self, frame):

        # -------------------------------------------------
        # Convert frame
        # -------------------------------------------------

        image = frame.to_ndarray(format="bgr24")

        # -------------------------------------------------
        # YOLO
        # -------------------------------------------------

        results = model(
            image,
            conf=0.40,
            verbose=False
        )

        annotated = results[0].plot()

        now = time.time()

        current_datetime = datetime.now()

        today = current_datetime.strftime("%Y-%m-%d")
        now_str = current_datetime.strftime("%H:%M:%S")

        # -------------------------------------------------
        # SAVE ACTIVITIES
        # Only every few seconds
        # -------------------------------------------------

        if now - self.last_activity_save >= ACTIVITY_SAVE_INTERVAL_SECONDS:

            self.last_activity_save = now

            detected_activities = set()

            for box in results[0].boxes:

                class_id = int(box.cls[0])

                activity_name = model.names[class_id]

                detected_activities.add(activity_name)

            for activity_name in detected_activities:

                save_activity(
                    student_id="unknown",
                    date=today,
                    time=now_str,
                    activity=activity_name
                )

        # -------------------------------------------------
        # FACE RECOGNITION
        # Only every 3 seconds
        # -------------------------------------------------

        if now - self.last_face_check >= FACE_CHECK_INTERVAL_SECONDS:

            self.last_face_check = now

            try:

                faces = face_app.get(image)

                for face in faces:

                    student_id = find_student(
                        face.embedding
                    )

                    if student_id is None:
                        continue

                    match = self.students[
                        self.students["student_id"].astype(str)
                        == str(student_id)
                    ]

                    if match.empty:
                        continue

                    student_row = match.iloc[0]

                    is_first_detection_today = save_attendance(
                        student_id=student_id,
                        date=today,
                        time=now_str
                    )

                    if is_first_detection_today:

                        send_attendance_email(
                            parent_email=student_row["parent_email"],
                            student_name=student_row["student_name"],
                            time_str=now_str
                        )

            except Exception as e:

                print(
                    f"Face recognition error: {e}"
                )

        # -------------------------------------------------
        # Return annotated frame
        # -------------------------------------------------

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# =========================================================
# LIVE CLASSROOM PAGE
# =========================================================

def render_live_classroom():

    st.markdown(
        '<div class="page-title">Live Classroom</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Monitor classroom activity in real time.'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # WebRTC configuration
    # -----------------------------------------------------

    rtc_configuration = RTCConfiguration(
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

    # -----------------------------------------------------
    # Camera panel
    # -----------------------------------------------------

    with st.container(border=True):

        st.markdown(
            '<div class="panel-title">'
            'Classroom Camera'
            '</div>',
            unsafe_allow_html=True
        )

        webrtc_streamer(
            key="basma-classroom-camera",

            mode="SENDRECV",

            rtc_configuration=rtc_configuration,

            video_processor_factory=ClassroomProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True,

            desired_playing_state=True,
        )

        st.caption(
            "Click START to activate the classroom camera."
        )
