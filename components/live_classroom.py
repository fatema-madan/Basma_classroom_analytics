import av
import cv2
import streamlit as st

from streamlit_webrtc import webrtc_streamer


st.title("🎥 BASMA Camera Test")


cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


class VideoProcessor:

    def recv(self, frame):

        # Get camera frame
        frm = frame.to_ndarray(
            format="bgr24"
        )

        # Convert to grayscale
        gray = cv2.cvtColor(
            frm,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = cascade.detectMultiScale(
            gray,
            1.1,
            3
        )

        # Draw bounding boxes
        for x, y, w, h in faces:

            cv2.rectangle(
                frm,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            frm,
            format="bgr24"
        )


webrtc_streamer(
    key="basma-camera",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
