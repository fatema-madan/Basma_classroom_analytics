import av
import cv2
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("models/basma_yolo.pt")


# =========================================================
# LIVE CLASSROOM
# =========================================================

def render_live_classroom():

    st.title("🎥 BASMA — Live Classroom")

    st.write(
        "Monitor classroom activities in real time."
    )

    st.info(
        "Click START and allow camera access."
    )

    model = load_model()


    # =====================================================
    # VIDEO PROCESSOR
    # =====================================================

    class VideoProcessor:

        def recv(self, frame):

            # ---------------------------------------------
            # Get LIVE camera frame
            # ---------------------------------------------

            frm = frame.to_ndarray(
                format="bgr24"
            )


            # ---------------------------------------------
            # YOLO DETECTION
            # ---------------------------------------------

            results = model.predict(
                frm,
                conf=0.40,
                verbose=False
            )


            # ---------------------------------------------
            # DRAW BOUNDING BOXES
            # ---------------------------------------------

            if len(results) > 0:

                frm = results[0].plot()


            # ---------------------------------------------
            # Return the SAME live frame
            # with YOLO boxes on top
            # ---------------------------------------------

            return av.VideoFrame.from_ndarray(
                frm,
                format="bgr24"
            )


    # =====================================================
    # LIVE CAMERA
    # =====================================================

    webrtc_streamer(

        key="basma-live-classroom",

        video_processor_factory=VideoProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

        async_processing=True,
    )
