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
    build_embeddings_from_students,
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/basma_yolo.pt"

YOLO_CONFIDENCE = 0.40

FACE_THRESHOLD = 0.45

FACE_INTERVAL = 1.0

ACTIVITY_INTERVAL = 3.0


# =========================================================
# LOAD YOLO
# =========================================================

@st.cache_resource
def load_yolo_model():

    return YOLO(
        MODEL_PATH
    )


# =========================================================
# LOAD FACE EMBEDDINGS
# =========================================================

@st.cache_resource
def load_face_data():

    students = load_students()

    embeddings = load_embeddings()


    # If embeddings file doesn't exist,
    # create it automatically.
    if not embeddings:

        embeddings = (
            build_embeddings_from_students(
                students
            )
        )


    return students, embeddings


# =========================================================
# STUDENT NAME
# =========================================================

def get_student_name(
    students,
    student_id
):

    if students.empty:

        return str(student_id)


    rows = students[
        students["student_id"]
        .astype(str)
        == str(student_id)
    ]


    if rows.empty:

        return str(student_id)


    return str(
        rows.iloc[0]["student_name"]
    )


# =========================================================
# WEBRTC
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
# VIDEO PROCESSOR
# =========================================================

class BASMAVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.model = load_yolo_model()

        (
            self.students,
            self.embeddings
        ) = load_face_data()


        # -----------------------------------------------
        # Attendance
        # -----------------------------------------------

        self.present_students = set()


        # -----------------------------------------------
        # Face recognition timing
        # -----------------------------------------------

        self.last_face_time = 0


        # -----------------------------------------------
        # Cached recognized faces
        # -----------------------------------------------

        self.recognized_faces = []


        # -----------------------------------------------
        # Activity timing
        # -----------------------------------------------

        self.last_activity_time = 0


        # -----------------------------------------------
        # Prevent duplicate activities
        # -----------------------------------------------

        self.last_activity_saved = {}


        # -----------------------------------------------
        # Thread safety
        # -----------------------------------------------

        self.lock = threading.Lock()


    # =====================================================
    # ATTENDANCE
    # =====================================================

    def mark_attendance(
        self,
        student_id
    ):

        student_id = str(
            student_id
        )


        if student_id in (
            self.present_students
        ):

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


            self.present_students.add(
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


        # Don't run InsightFace on every frame
        if (
            current_time
            - self.last_face_time
            < FACE_INTERVAL
        ):

            return


        self.last_face_time = (
            current_time
        )


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

            # -------------------------------------------
            # Face bounding box
            # -------------------------------------------

            x1, y1, x2, y2 = [
                int(value)
                for value in face.bbox
            ]


            # -------------------------------------------
            # Recognition
            # -------------------------------------------

            student_id = find_student(
                face.embedding,
                threshold=FACE_THRESHOLD
            )


            if student_id is not None:

                student_id = str(
                    student_id
                )

                name = get_student_name(
                    self.students,
                    student_id
                )


                # ---------------------------------------
                # Attendance
                # ---------------------------------------

                self.mark_attendance(
                    student_id
                )


                new_faces.append(
                    {
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "name": name,
                        "student_id": student_id,
                        "known": True,
                    }
                )


            else:

                new_faces.append(
                    {
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "name": "Unknown",
                        "student_id": None,
                        "known": False,
                    }
                )


        # Keep the latest face detections
        self.recognized_faces = (
            new_faces
        )


    # =====================================================
    # DRAW FACES
    # =====================================================

    def draw_faces(
        self,
        frame
    ):

        for face in (
            self.recognized_faces
        ):

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


            # -------------------------------------------
            # Face box
            # -------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # -------------------------------------------
            # Label background
            # -------------------------------------------

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
    # SAVE ACTIVITIES
    # =====================================================

    def save_detected_activities(
        self,
        activities
    ):

        if not activities:
            return


        if not self.present_students:
            return


        now_timestamp = time.time()


        if (
            now_timestamp
            - self.last_activity_time
            < ACTIVITY_INTERVAL
        ):

            return


        self.last_activity_time = (
            now_timestamp
        )


        now = datetime.now()

        date = now.strftime(
            "%Y-%m-%d"
        )

        current_time = now.strftime(
            "%H:%M:%S"
        )


        # -----------------------------------------------
        # Save activities for recognized students
        # -----------------------------------------------

        for student_id in (
            self.present_students
        ):

            for activity in activities:

                key = (
                    f"{student_id}_{activity}"
                )


                previous = (
                    self.last_activity_saved
                    .get(
                        key,
                        0
                    )
                )


                if (
                    now_timestamp
                    - previous
                    < ACTIVITY_INTERVAL
                ):

                    continue


                try:

                    save_activity(
                        student_id=student_id,
                        date=date,
                        time=current_time,
                        activity=activity
                    )


                    self.last_activity_saved[
                        key
                    ] = now_timestamp


                except Exception as e:

                    print(
                        "Activity error:",
                        e
                    )


    # =====================================================
    # LIVE FRAME
    # =====================================================

    def recv(
        self,
        frame
    ):

        # =================================================
        # 1. CAMERA FRAME
        # =================================================

        image = frame.to_ndarray(
            format="bgr24"
        )


        # =================================================
        # 2. YOLO TRACKING
        # =================================================
        #
        # Same main idea as the Aarohi tutorial:
        #
        # model.track()
        # persist=True
        # ByteTrack
        #
        # =================================================

        try:

            results = self.model.track(
                image,
                persist=True,
                conf=YOLO_CONFIDENCE,
                tracker="bytetrack.yaml",
                verbose=False
            )


            result = results[0]


            # Draw YOLO boxes
            output = result.plot()


        except Exception as e:

            print(
                "YOLO tracking error:",
                e
            )

            result = None

            output = image.copy()


        # =================================================
        # 3. GET ACTIVITIES
        # =================================================

        activities = []


        if result is not None:

            try:

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )


                        activity = str(
                            self.model.names[
                                class_id
                            ]
                        )


                        if activity not in (
                            activities
                        ):

                            activities.append(
                                activity
                            )


            except Exception as e:

                print(
                    "Activity reading error:",
                    e
                )


        # =================================================
        # 4. FACE DETECTION
        # =================================================

        self.detect_faces(
            image
        )


        # =================================================
        # 5. DRAW FACE BOXES
        # =================================================

        self.draw_faces(
            output
        )


        # =================================================
        # 6. SAVE ACTIVITY
        # =================================================

        self.save_detected_activities(
            activities
        )


        # =================================================
        # 7. LIVE STATUS
        # =================================================

        status = (
            f"Present: "
            f"{len(self.present_students)}"
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
        # 8. RETURN FRAME
        # =================================================

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


# =========================================================
# LIVE CLASSROOM PAGE
# =========================================================

def render_live_classroom():

    st.markdown(
        "### 📸 Live Classroom"
    )

    st.markdown(
        "BASMA detects student faces, "
        "attendance, and classroom activities "
        "in real time."
    )


    # =====================================================
    # START CAMERA
    # =====================================================

    webrtc_streamer(

        key="basma-classroom-stream",

        video_processor_factory=(
            BASMAVideoProcessor
        ),

        rtc_configuration=(
            RTC_CONFIGURATION
        ),

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


    # =====================================================
    # INFO
    # =====================================================

    st.caption(
        "Face Recognition → Attendance | "
        "YOLO Tracking → Activities"
    )
