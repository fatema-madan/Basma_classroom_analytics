import av
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)


st.set_page_config(
    page_title="WebRTC Test",
    layout="centered",
)


st.title("WebRTC Camera Test")

st.write(
    "If the camera works, you should see the live video below."
)


def video_frame_callback(frame: av.VideoFrame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24"
    )


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


webrtc_ctx = webrtc_streamer(
    key="simple-camera-test",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration=rtc_configuration,

    media_stream_constraints={
        "video": True,
        "audio": False,
    },

    video_frame_callback=video_frame_callback,

    async_processing=True,
)
