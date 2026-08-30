import av
import cv2
import streamlit as st


from streamlit_webrtc import webrtc_streamer


# =========================================================
# LIVE CLASSROOM
# =========================================================

def render_live_classroom():

    st.title("🎥 BASMA — Live Classroom")

    st.write(
        "Live classroom camera with real-time detection."
    )

    st.info(
        "Click START and allow camera access."
    )


    # =====================================================
    # VIDEO PROCESSOR
    # =====================================================

    class VideoProcessor:

        def recv(self, frame):

            # Get frame from browser camera
            frm = frame.to_ndarray(
                format="bgr24"
            )

            # ---------------------------------------------
            # TEST BOUNDING BOX
            # ---------------------------------------------
            # This is only to test that the live camera
            # and video processing are working.
            #
            # Later we replace this with YOLO.
            # ---------------------------------------------

            height, width, _ = frm.shape

            x1 = int(width * 0.25)
            y1 = int(height * 0.20)

            x2 = int(width * 0.75)
            y2 = int(height * 0.80)

            cv2.rectangle(
                frm,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frm,
                "BASMA LIVE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Return processed frame
            return av.VideoFrame.from_ndarray(
                frm,
                format="bgr24"
            )


    # =====================================================
    # WEBRTC CAMERA
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
