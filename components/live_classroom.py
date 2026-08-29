import time
from datetime import datetime

import streamlit as st
import cv2

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utils.data_manager import (
    load_students,
    save_attendance,
    save_activity
)
from utils.face_utils import face_app, find_student
from utils.email_utils import send_attendance_email


MODEL_PATH = "models/basma_yolo.pt"

# Only run face recognition every N seconds per frame stream —
# it's a lot heavier than the YOLO activity model, so we don't
# want to run it on every single video frame.
FACE_CHECK_INTERVAL_SECONDS = 3


model = YOLO(MODEL_PATH)


class ClassroomProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_face_check = 0
        # Cache student lookup once per session instead of hitting
        # the CSV on every frame.
        self.students = load_students()

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # ---------------------------------------------
        # 1. Activity detection (existing behaviour)
        # ---------------------------------------------
        results = model(
            image,
            conf=0.40,
            verbose=False
        )

        annotated = results[0].plot()

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        now_str = now.strftime("%H:%M:%S")

        # Log each detected activity class
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            activity_name = model.names[class_id]

            save_activity(
                student_id="unknown",  # activity model doesn't ID students
                date=today,
                time=now_str,
                activity=activity_name
            )

        # ---------------------------------------------
        # 2. Face recognition + attendance + parent email
        #    (throttled — heavy to run every frame)
        # ---------------------------------------------
        current_time = time.time()

        if current_time - self.last_face_check >= FACE_CHECK_INTERVAL_SECONDS:

            self.last_face_check = current_time

            faces = face_app.get(image)

            for face in faces:

                student_id = find_student(face.embedding)

                if student_id is None:
                    continue

                match = self.students[
                    self.students["student_id"].astype(str) == str(student_id)
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

        return frame.from_ndarray(
            annotated,
            format="bgr24"
        )


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

    with st.container(border=True):

        st.markdown(
            '<div class="panel-title">Classroom Camera</div>',
            unsafe_allow_html=True
        )

        webrtc_streamer(
            key="basma-camera",
            video_processor_factory=ClassroomProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            }
        )
