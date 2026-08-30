import av
import cv2
import time
import threading
import numpy as np
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
    load_attendance,
    load_activity,
    save_attendance,
    save_activity,
)

from utils.face_utils import (
    face_app,
    load_embeddings,
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/basma_yolo.pt"

YOLO_CONFIDENCE = 0.40

FACE_THRESHOLD = 0.45

# Face recognition every 2 seconds
FACE_INTERVAL = 2.0

# Save activity every 3 seconds
ACTIVITY_INTERVAL = 3.0


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_my_model():

    return YOLO(MODEL_PATH)


# =========================================================
# LOAD STUDENT DATA
# =========================================================

@st.cache_resource
def load_student_data():

    students = load_students()

    embeddings = load_embeddings()

    return students, embeddings


# =========================================================
# FIND STUDENT FROM EMBEDDING
# =========================================================

def find_student(
    face_embedding,
    embeddings,
    threshold=FACE_THRESHOLD
):

    if embeddings is None:
        return None

    if len(embeddings) == 0:
        return None

    current_embedding = np.asarray(
        face_embedding,
        dtype=np.float32
    )

    current_norm = np.linalg.norm(
        current_embedding
    )

    if current_norm == 0:
        return None

    current_embedding = (
        current_embedding / current_norm
    )

    best_student = None

    best_score = -1


    for student_id, saved_embedding in embeddings.items():

        saved_embedding = np.asarray(
            saved_embedding,
            dtype=np.float32
        )

        saved_norm = np.linalg.norm(
            saved_embedding
        )

        if saved_norm == 0:
            continue

        saved_embedding = (
            saved_embedding / saved_norm
        )

        score = float(
            np.dot(
                current_embedding,
                saved_embedding
            )
        )

        if score > best_score:

            best_score = score

            best_student = student_id


    if best_score >= threshold:

        return best_student


    return None


# =========================================================
# GET STUDENT NAME
# =========================================================

def get_student_name(
    students,
    student_id
):

    if students is None:
        return str(student_id)

    if students.empty:
        return str(student_id)

    if "student_id" not in students.columns:
        return str(student_id)

    rows = students[
        students["student_id"].astype(str)
        == str(student_id)
    ]

    if rows.empty:
        return str(student_id)

    if "student_name" not in rows.columns:
        return str(student_id)

    return str(
        rows.iloc[0]["student_name"]
    )


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
# LIVE VIDEO PROCESSOR
# =========================================================

class BASMAVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        # -----------------------------------------------
        # Load model
        # -----------------------------------------------

        self.model = load_my_model()


        # -----------------------------------------------
        # Load students + face embeddings
        # -----------------------------------------------

        (
            self.students,
            self.embeddings
        ) = load_student_data()


        # -----------------------------------------------
        # Timing
        # -----------------------------------------------

        self.last_face_time = 0

        self.last_activity_time = 0


        # -----------------------------------------------
        # Students recognized during this session
        # -----------------------------------------------

        self.present_students = set()


        # -----------------------------------------------
        # Prevent duplicate activity records
        # -----------------------------------------------

        self.last_saved_activity = {}


        # -----------------------------------------------
        # Thread safety
        # -----------------------------------------------

        self.lock = threading.Lock()


    # =====================================================
    # SAVE ATTENDANCE
    # =====================================================

    def mark_present(
        self,
        student_id
    ):

        student_id = str(student_id)


        # Already marked during this session
        if student_id in self.present_students:

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
                "Attendance save error:",
                e
            )


    # =====================================================
    # FACE RECOGNITION
    # =====================================================

    def process_faces(
        self,
        image,
        output
    ):

        current_time = time.time()


        # -----------------------------------------------
        # Don't run face recognition on every frame
        # -----------------------------------------------

        if (
            current_time - self.last_face_time
            < FACE_INTERVAL
        ):

            return output


        self.last_face_time = current_time


        try:

            faces = face_app.get(image)

        except Exception as e:

            print(
                "Face recognition error:",
                e
            )

            return output


        # -----------------------------------------------
        # Process detected faces
        # -----------------------------------------------

        for face in faces:

            embedding = getattr(
                face,
                "embedding",
                None
            )

            if embedding is None:
                continue


            student_id = find_student(
                embedding,
                self.embeddings
            )


            # Unknown face
            if student_id is None:
                continue


            student_id = str(
                student_id
            )


            student_name = get_student_name(
                self.students,
                student_id
            )


            # -------------------------------------------
            # Face box
            # -------------------------------------------

            bbox = face.bbox

            x1, y1, x2, y2 = [
                int(value)
                for value in bbox
            ]


            # -------------------------------------------
            # Draw face box
            # -------------------------------------------

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # -------------------------------------------
            # Student name
            # -------------------------------------------

            label = (
                f"{student_name} | Present"
            )


            cv2.putText(
                output,
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
                (0, 255, 0),
                2
            )


            # -------------------------------------------
            # Mark attendance
            # -------------------------------------------

            self.mark_present(
                student_id
            )


        return output


    # =====================================================
    # SAVE ACTIVITIES
    # =====================================================

    def save_activities(
        self,
        activities
    ):

        if not activities:
            return


        if not self.present_students:
            return


        current_time = time.time()


        # -----------------------------------------------
        # Don't save too frequently
        # -----------------------------------------------

        if (
            current_time - self.last_activity_time
            < ACTIVITY_INTERVAL
        ):

            return


        self.last_activity_time = current_time


        now = datetime.now()

        date = now.strftime(
            "%Y-%m-%d"
        )

        clock = now.strftime(
            "%H:%M:%S"
        )


        # -----------------------------------------------
        # Save activities
        # -----------------------------------------------

        for student_id in self.present_students:

            for activity in activities:

                key = (
                    f"{student_id}_{activity}"
                )


                previous_time = (
                    self.last_saved_activity.get(
                        key,
                        0
                    )
                )


                # Avoid duplicate records
                if (
                    current_time - previous_time
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

                    self.last_saved_activity[key] = (
                        current_time
                    )

                except Exception as e:

                    print(
                        "Activity save error:",
                        e
                    )


    # =====================================================
    # RECEIVE LIVE FRAME
    # =====================================================

    def recv(
        self,
        frame: av.VideoFrame
    ):

        # =================================================
        # 1. CAMERA FRAME
        # =================================================

        image = frame.to_ndarray(
            format="bgr24"
        )


        # =================================================
        # 2. YOLO
        # =================================================

        try:

            results = self.model.predict(
                image,
                conf=YOLO_CONFIDENCE,
                verbose=False
            )


            result = results[0]


            # YOLO bounding boxes
            output = result.plot()


        except Exception as e:

            print(
                "YOLO error:",
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


                        if activity not in activities:

                            activities.append(
                                activity
                            )

            except Exception as e:

                print(
                    "Activity detection error:",
                    e
                )


        # =================================================
        # 4. FACE RECOGNITION
        # =================================================

        output = self.process_faces(
            image,
            output
        )


        # =================================================
        # 5. SAVE ACTIVITY
        # =================================================

        self.save_activities(
            activities
        )


        # =================================================
        # 6. LIVE STATUS
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
        # 7. RETURN LIVE FRAME
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
        "### 📸 Live Camera Detection"
    )

    st.write(
        "BASMA monitors classroom attendance "
        "and activities in real time."
    )


    # =====================================================
    # LIVE CAMERA
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
            "audio": False,
        },

        async_processing=True,
    )


    # =====================================================
    # STATUS INFORMATION
    # =====================================================

    st.divider()

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "AI Model",
            "BASMA YOLO"
        )


    with col2:

        st.metric(
            "Confidence",
            f"{YOLO_CONFIDENCE:.0%}"
        )


    with col3:

        st.metric(
            "Mode",
            "Live"
        )


    st.caption(
        "Live Camera → Face Recognition → "
        "Attendance + YOLO Activity Detection"
    )
