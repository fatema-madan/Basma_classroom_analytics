import time
from datetime import datetime

import av
import streamlit as st

from ultralytics import YOLO

from streamlit_webrtc import (
    webrtc_streamer,
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

FACE_CHECK_INTERVAL = 5
ACTIVITY_SAVE_INTERVAL = 3

# Process YOLO every N frames
# This prevents the camera from freezing on Streamlit Cloud.
YOLO_FRAME_SKIP = 2


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


model = load_model()


# =========================================================
# CLASSROOM PROCESSOR
# =========================================================

class ClassroomProcessor:

    def __init__(self):

        self.frame_count = 0

        self.last_face_check = 0
        self.last_activity_save = 0

        self.last_annotated_frame = None

        try:

            self.students = load_students()

        except Exception as e:

            print(
                f"Student loading error: {e}"
            )

            self.students = None

    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def process(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_count += 1

        current_time = time.time()

        now = datetime.now()

        today = now.strftime(
            "%Y-%m-%d"
        )

        now_str = now.strftime(
            "%H:%M:%S"
        )

        # =================================================
        # YOLO
        # =================================================

        if (
            self.frame_count % YOLO_FRAME_SKIP == 0
            or self.last_annotated_frame is None
        ):

            try:

                results = model.predict(
                    source=image,
                    conf=0.40,
                    imgsz=640,
                    verbose=False,
                    device="cpu",
                )

                annotated = results[0].plot()

                self.last_annotated_frame = annotated

            except Exception as e:

                print(
                    f"YOLO error: {e}"
                )

                annotated = image

                results = None

        else:

            annotated = self.last_annotated_frame

            results = None

        # =================================================
        # SAVE ACTIVITY
        # =================================================

        if (
            results is not None
            and
            current_time - self.last_activity_save
            >= ACTIVITY_SAVE_INTERVAL
        ):

            self.last_activity_save = current_time

            try:

                detected_activities = set()

                for box in results[0].boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    activity_name = model.names[
                        class_id
                    ]

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
                    f"Activity error: {e}"
                )

        # =================================================
        # FACE RECOGNITION
        # =================================================

        if (
            current_time - self.last_face_check
            >= FACE_CHECK_INTERVAL
        ):

            self.last_face_check = current_time

            try:

                if self.students is not None:

                    faces = face_app.get(
                        image
                    )

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

                        first_detection = (
                            save_attendance(
                                student_id=student_id,
                                date=today,
                                time=now_str,
                            )
                        )

                        if first_detection:

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
        # RETURN FRAME
        # =================================================

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )


# =========================================================
# LIVE CLASSROOM
# =========================================================

def render_live_classroom():

    # =====================================================
    # HEADER
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
    # WEBRTC CONFIG
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
    # CREATE PROCESSOR
    # =====================================================

    processor = ClassroomProcessor()

    # =====================================================
    # CAMERA
    # =====================================================

    with st.container(border=True):

        st.markdown(
            '<div class="panel-title">'
            'Classroom Camera'
            '</div>',
            unsafe_allow_html=True,
        )

        webrtc_streamer(
            key="basma-classroom-camera",

            mode=WebRtcMode.SENDRECV,

            rtc_configuration=rtc_configuration,

            video_frame_callback=processor.process,

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            async_processing=True,
        )

        st.caption(
            "Click START to activate the classroom camera."
        )
