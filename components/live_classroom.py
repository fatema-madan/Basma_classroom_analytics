import av
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)


def video_frame_callback(frame: av.VideoFrame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24"
    )


def render_live_classroom():

    st.title("WebRTC Camera Test")

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

    webrtc_streamer(
        key="basma-camera-test",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=rtc_configuration,

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

        video_frame_callback=video_frame_callback,

        async_processing=True,
    )
