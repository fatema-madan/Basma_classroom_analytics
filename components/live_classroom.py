import time
from datetime import datetime
import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from utils.data_manager import (
    load_students,
    save_attendance,
    save_activity
)
from utils.face_utils import face_app, find_student
from utils.email_utils import send_attendance_email

MODEL_PATH = "models/basma_yolo.pt"
FACE_CHECK_INTERVAL_SECONDS = 3

if "model" not in st.session_state:
    st.session_state.model = YOLO(MODEL_PATH)

model = st.session_state.model

class ClassroomProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_face_check = 0
        self.students = load_students()
        
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        results = model(
            image,
            conf=0.40,
            verbose=False
        )
        annotated = results[0].plot()
        
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        now_str = now.strftime("%H:%M:%S")
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            activity_name = model.names[class_id]
            save_activity(
                student_id="unknown",
                date=today,
                time=now_str,
                activity=activity_name
            )
        
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
        
        return frame.from_ndarray(annotated, format="bgr24")


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
    
    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        }
    )
    
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Classroom Camera</div>',
            unsafe_allow_html=True
        )
        
        webrtc_streamer(
            key="basma-camera",
            mode="SENDRECV",
            rtc_configuration=rtc_configuration,
            video_processor_factory=ClassroomProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )
