import streamlit as st
import cv2

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


MODEL_PATH = "models/basma_yolo.pt"


# Load YOLO model
model = YOLO(MODEL_PATH)


class ClassroomProcessor(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = model(
            image,
            conf=0.40,
            verbose=False
        )

        annotated_image = results[0].plot()

        return frame.from_ndarray(
            annotated_image,
            format="bgr24"
        )


def render_live_classroom():

    st.markdown(
        '<div class="page-title">'
        'Live Classroom'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Monitor classroom activity in real time.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        'Classroom Camera'
        '</div>',
        unsafe_allow_html=True
    )

    webrtc_streamer(
        key="basma-classroom",
        video_processor_factory=ClassroomProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )
