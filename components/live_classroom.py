import av
import cv2
import time
import threading
import streamlit as st

from datetime import datetime

from ultralytics import YOLO

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
    RTCConfiguration,
)

from utils.data_manager import (
    load_students,
    save_attendance,
    save_activity,
)

from utils.face_utils import (
    face_app,
    load_embeddings,
    find_student,
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/basma_yolo.pt"

YOLO_CONFIDENCE = 0.40

FACE_INTERVAL = 1.0

ATTENDANCE_INTERVAL = 5.0

ACTIVITY_INTERVAL = 5.0


# =========================================================
# WEBRTC CONFIGURATION
# =========================================================

RTC_CONFIGURATION = RTCConfiguration(
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


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():

    return YOLO(
        MODEL_PATH
    )


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class BASMAVideoProcessor(VideoProcessorBase):

    def __init__(self):

        # -------------------------------------------------
        # Models
        # -------------------------------------------------

        self.model = load_yolo_model()

        self.students = load_students()

        self.embeddings = load_embeddings()


        # -------------------------------------------------
        # Face recognition timing
        # -------------------------------------------------

        self.last_face_time = 0


        # -------------------------------------------------
        # Recognized students
        # -------------------------------------------------

        self.recognized_students = {}


        # -------------------------------------------------
        # Attendance
        # -------------------------------------------------

        self.attendance_saved = set()


        # -------------------------------------------------
        # Activity logging
        # -------------------------------------------------

        self.last_activity_saved = {}


        # -------------------------------------------------
        # Thread lock
        # -------------------------------------------------

        self.lock = threading.Lock()


    # =====================================================
    # GET STUDENT NAME
    # =====================================================

    def get_student_name(
        self,
        student_id
    ):

        if self.students is None:
            return str(student_id)


        if self.students.empty:
            return str(student_id)


        if "student_id" not in self.students.columns:
            return str(student_id)


        rows = self.students[
            self.students["student_id"]
            .astype(str)
            == str(student_id)
        ]


        if rows.empty:
            return str(student_id)


        if "student_name" not in rows.columns:
            return str(student_id)


        return str(
            rows.iloc[0]["student_name"]
        )


    # =====================================================
    # SAVE ATTENDANCE
    # =====================================================

    def save_student_attendance(
        self,
        student_id
    ):

        student_id = str(
            student_id
        )


        if student_id in self.attendance_saved:
            return


        now = datetime.now()

        date = now.strftime(
            "%Y-%m-%d"
        )

        current_time = now.strftime(
            "%H:%M:%S"
        )


        try:

            save_attendance(
                student_id=student_id,
                date=date,
                time=current_time
            )


            self.attendance_saved.add(
                student_id
            )


        except Exception as e:

            print(
                "Attendance error:",
                e
            )


    # =====================================================
    # FACE DETECTION + RECOGNITION
    # =====================================================

    def detect_faces(
        self,
        frame
    ):

        current_time = time.time()


        # Don't run face recognition on every frame
        if (
            current_time - self.last_face_time
            < FACE_INTERVAL
        ):

            return


        self.last_face_time = current_time


        try:

            faces = face_app.get(
                frame
            )

        except Exception as e:

            print(
                "Face detection error:",
                e
            )

            return


        new_faces = []


        for face in faces:

            # -------------------------------------------------
            # Face bounding box
            # -------------------------------------------------

            x1, y1, x2, y2 = [
                int(value)
                for value in face.bbox
            ]


            # -------------------------------------------------
            # Face recognition
            # -------------------------------------------------

            student_id = find_student(
                face.embedding
            )


            if student_id is not None:

                student_id = str(
                    student_id
                )


                student_name = (
                    self.get_student_name(
                        student_id
                    )
                )


                # ---------------------------------------------
                # Save attendance
                # ---------------------------------------------

                self.save_student_attendance(
                    student_id
                )


                # ---------------------------------------------
                # Save recognized face
                # ---------------------------------------------

                self.recognized_students[
                    student_id
                ] = {
                    "name": student_name,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "last_seen": current_time,
                }


                new_faces.append(
                    {
                        "name": student_name,
                        "student_id": student_id,
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "known": True,
                    }
                )


            else:

                new_faces.append(
                    {
                        "name": "Unknown",
                        "student_id": None,
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "known": False,
                    }
                )


        return new_faces


    # =====================================================
    # DRAW FACE BOXES
    # =====================================================

    def draw_faces(
        self,
        frame,
        faces
    ):

        for face in faces:

            x1, y1, x2, y2 = (
                face["box"]
            )


            if face["known"]:

                label = (
                    f'{face["name"]} | Present'
                )

                color = (
                    0,
                    255,
                    0
                )

            else:

                label = "Unknown"

                color = (
                    0,
                    165,
                    255
                )


            # -------------------------------------------------
            # Face bounding box
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # -------------------------------------------------
            # Label
            # -------------------------------------------------

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )


    # =====================================================
    # GET YOLO ACTIVITIES
    # =====================================================

    def get_activities(
        self,
        result
    ):

        activities = []


        if result is None:
            return activities


        if result.boxes is None:
            return activities


        for box in result.boxes:

            try:

                class_id = int(
                    box.cls[0]
                )


                activity = str(
                    self.model.names[
                        class_id
                    ]
                )


                if activity not in activities:

                    activities.append(
                        activity
                    )

            except Exception:

                continue


        return activities


    # =====================================================
    # SAVE ACTIVITIES
    # =====================================================

    def save_activities(
        self,
        activities
    ):

        if not activities:
            return


        if not self.recognized_students:
            return


        current_time = time.time()


        now = datetime.now()

        date = now.strftime(
            "%Y-%m-%d"
        )

        clock = now.strftime(
            "%H:%M:%S"
        )


        # -------------------------------------------------
        # For every recognized student
        # -------------------------------------------------

        for student_id in (
            self.recognized_students
        ):

            for activity in activities:

                key = (
                    f"{student_id}_{activity}"
                )


                previous_time = (
                    self.last_activity_saved.get(
                        key,
                        0
                    )
                )


                # Prevent duplicate logging
                if (
                    current_time
                    - previous_time
                    < ACTIVITY_INTERVAL
                ):

                    continue


                try:

                    save_activity(
                        student_id=student_id,
                        date=date,
                        time=clock,
                        activity=activity
                    )


                    self.last_activity_saved[
                        key
                    ] = current_time


                except Exception as e:

                    print(
                        "Activity save error:",
                        e
                    )


    # =====================================================
    # PROCESS LIVE FRAME
    # =====================================================

    def recv(
        self,
        frame
    ):

        # =================================================
        # CAMERA INPUT
        # =================================================

        image = frame.to_ndarray(
            format="bgr24"
        )


        # =================================================
        # YOLO TRACKING
        # =================================================

        try:

            results = self.model.track(

                image,

                persist=True,

                conf=YOLO_CONFIDENCE,

                tracker="bytetrack.yaml",

                verbose=False,

            )


            result = results[0]


            # Draw YOLO bounding boxes
            output = result.plot()


        except Exception as e:

            print(
                "YOLO error:",
                e
            )

            result = None

            output = image.copy()


        # =================================================
        # ACTIVITIES
        # =================================================

        activities = (
            self.get_activities(
                result
            )
        )


        # =================================================
        # FACE DETECTION
        # =================================================

        faces = self.detect_faces(
            image
        )


        # =================================================
        # DRAW FACE BOXES
        # =================================================

        if faces:

            self.draw_faces(
                output,
                faces
            )


        # =================================================
        # SAVE ACTIVITIES
        # =================================================

        self.save_activities(
            activities
        )


        # =================================================
        # LIVE STATUS
        # =================================================

        status = (
            f"Present: "
            f"{len(self.attendance_saved)}"
        )


        cv2.putText(
            output,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # =================================================
        # RETURN LIVE VIDEO
        # =================================================

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


# =========================================================
# RENDER LIVE CLASSROOM
# =========================================================

def render_live_classroom():

    st.markdown(
        "### 📸 Live Classroom"
    )


    st.write(
        "Start the camera to detect "
        "students, attendance, and "
        "classroom activities in real time."
    )


    # =====================================================
    # LIVE CAMERA INPUT
    # =====================================================

    webrtc_streamer(

        key="basma-live-classroom",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=(
            BASMAVideoProcessor
        ),

        rtc_configuration=(
            RTC_CONFIGURATION
        ),

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

        async_processing=True,
    )


    # =====================================================
    # INFORMATION
    # =====================================================

    st.divider()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Camera",
            "Live"
        )


    with col2:

        st.metric(
            "Detection",
            "YOLO + Face"
        )


    with col3:

        st.metric(
            "Tracking",
            "ByteTrack"
        )


    st.caption(
        "Live Camera → Face Recognition → "
        "Attendance + YOLO Activity Tracking"
    )
