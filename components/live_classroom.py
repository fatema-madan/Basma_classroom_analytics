import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from utils.face_utils import load_embeddings
from utils.data_manager import load_students
from utils.attendance import mark_attendance
from utils.activity_detection import save_activity


MODEL_PATH = Path("models/basma_yolo.pt")


def render_live_classroom():

    st.markdown(
        '<div class="page-title">Live Classroom</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Monitor students and classroom activities in real time."
        '</div>',
        unsafe_allow_html=True
    )

    students = load_students()
    embeddings = load_embeddings()

    if students.empty:

        st.warning(
            "No students are registered yet."
        )

        return

    if not MODEL_PATH.exists():

        st.error(
            "YOLO model was not found."
        )

        return

    model = YOLO(
        str(MODEL_PATH)
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    start_camera = st.checkbox(
        "Start Camera"
    )

    camera_placeholder = st.empty()

    status_placeholder = st.empty()

    if not start_camera:

        camera_placeholder.info(
            "Enable Start Camera to begin monitoring."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        st.error(
            "Could not open the camera."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        return

    st.success(
        "Camera is running."
    )

    while True:

        success, frame = camera.read()

        if not success:
            break

        # YOLO detection
        results = model(
            frame,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Get detected activities
        detected_activities = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                if confidence < 0.5:
                    continue

                class_name = model.names[
                    class_id
                ]

                detected_activities.append(
                    class_name
                )

        # Display camera
        camera_placeholder.image(
            cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            ),
            channels="RGB",
            use_container_width=True
        )

        # Display detected activities
        if detected_activities:

            activity_text = ", ".join(
                detected_activities
            )

            status_placeholder.info(
                f"Detected: {activity_text}"
            )

        else:

            status_placeholder.info(
                "No activity detected."
            )

    camera.release()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )