import time
from datetime import datetime

import av
import streamlit as st

from ultralytics import YOLO

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)

from utils.data_manager import (
    load_students,
    save_attendance,
    save_activity,
)

from utils.face_utils import (
    face_app,
    find_student,
)

from utils.email_utils import (
    send_attendance_email,
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/basma_yolo.pt"

FACE_CHECK_INTERVAL_SECONDS = 3
ACTIVITY_SAVE_INTERVAL_SECONDS = 3


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)


model = load_yolo_model()


# =========================================================
# CLASSROOM VIDEO PROCESSOR
# =========================================================

class ClassroomProcessor(VideoProcessorBase):

    def __init__(self):

        self.last_face_check = 0
        self.last_activity_save = 0

        try:
            self.students = load_students()
        except Exception as e:
            print(f"Could not load students: {e}")
            self.students = None

    # -----------------------------------------------------
    # PROCESS EACH CAMERA FRAME
    # -----------------------------------------------------

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        image = frame.to_ndarray(format="bgr24")

        # =================================================
        # YOLO DETECTION
        # =================================================

        try:

            results = model(
                image,
                conf=0.40,
                verbose=False
            )

            annotated = results[0].plot()

        except Exception as e:

            print(f"YOLO error: {e}")

            annotated = image

            results = None

        # Current time
        current_time = time.time()

        now = datetime.now()

        today = now.strftime("%Y-%m-%d")
        now_str = now.strftime("%H:%M:%S")

        # =================================================
        # SAVE ACTIVITY
        # =================================================

        if (
            results is not None
            and current_time - self.last_activity_save
            >= ACTIVITY_SAVE_INTERVAL_SECONDS
        ):

            self.last_activity_save = current_time

            detected_activities = set()

            try:

                for box in results[0].boxes:

                    class_id = int(box.cls[0])

                    activity_name = model.names[class_id]

                    detected_activities.add(
                        activity_name
                    )

                for activity_name in detected_activities:

                    save_activity(
                        student_id="unknown",
                        date=today,
                        time=now_str,
                        activity=activity_name,
                    )

            except Exception as e:

                print(
                    f"Activity saving error: {e}"
                )

        # =================================================
        # FACE RECOGNITION
        # =================================================

        if (
            current_time - self.last_face_check
            >= FACE_CHECK_INTERVAL_SECONDS
        ):

            self.last_face_check = current_time

            try:

                # If students could not be loaded,
                # skip attendance processing.
                if self.students is not None:

                    faces = face_app.get(image)

                    for face in faces:

                        student_id = find_student(
                            face.embedding
                        )

                        if student_id is None:
                            continue

                        match = self.students[
                            self.students[
                                "student_id"
                            ].astype(str)
                            == str(student_id)
                        ]

                        if match.empty:
                            continue

                        student_row = match.iloc[0]

                        # Save attendance
                        is_first_detection_today = (
                            save_attendance(
                                student_id=student_id,
                                date=today,
                                time=now_str,
                            )
                        )

                        # Send email only once
                        if is_first_detection_today:

                            send_attendance_email(
                                parent_email=student_row[
                                    "parent_email"
                                ],
                                student_name=student_row[
                                    "student_name"
                                ],
                                time_str=now_str,
                            )

            except Exception as e:

                print(
                    f"Face recognition error: {e}"
                )

        # =================================================
        # RETURN PROCESSED FRAME
        # =================================================

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )


# =========================================================
# LIVE CLASSROOM PAGE
# =========================================================

def render_live_classroom():

    # =====================================================
    # PAGE HEADER
    # =====================================================

    st.markdown(
        '<div class="page-title">'
        'Live Classroom'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Monitor classroom activity in real time.'
        '</div>',
        unsafe_allow_html=True,
    )

    # =====================================================
    # WEBRTC CONFIGURATION
    # =====================================================

    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                },
                {
                    "urls": [
                        "stun:stun1.l.google.com:19302"
                    ]
                },
            ]
        }
    )

    # =====================================================
    # CAMERA PANEL
    # =====================================================

    with st.container(border=True):

        st.markdown(
            '<div class="panel-title">'
            'Classroom Camera'
            '</div>',
            unsafe_allow_html=True,
        )

        # =================================================
        # WEBRTC CAMERA
        # =================================================

        webrtc_streamer(

            key="basma-classroom-camera",

            mode=WebRtcMode.SENDRECV,

            rtc_configuration=rtc_configuration,

            video_processor_factory=ClassroomProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            async_processing=True,

        )

        st.caption(
            "Click START to activate the classroom camera."
        )
