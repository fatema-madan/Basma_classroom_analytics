import cv2
import streamlit as st
import numpy as np

from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from zoneinfo import ZoneInfo

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

from utils.email_utils import (
    send_attendance_email,
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/basma_yolo.pt"

CONFIDENCE = 0.40

BAHRAIN_TIMEZONE = ZoneInfo(
    "Asia/Bahrain"
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO(
        MODEL_PATH
    )


# =========================================================
# LOAD FACE DATA
# =========================================================

@st.cache_resource
def load_face_data():

    students = load_students()

    embeddings = load_embeddings()

    return students, embeddings


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
# GET PARENT EMAIL
# =========================================================

def get_parent_email(
    students,
    student_id
):

    if students is None:
        return ""

    if students.empty:
        return ""

    if "parent_email" not in students.columns:
        return ""

    rows = students[
        students["student_id"]
        .astype(str)
        == str(student_id)
    ]

    if rows.empty:
        return ""

    email = rows.iloc[0].get(
        "parent_email",
        ""
    )

    if email is None:
        return ""

    return str(email).strip()


# =========================================================
# LIVE CLASSROOM
# =========================================================

def render_live_classroom():

    st.markdown(
        "### 📸 Live Camera"
    )

    st.write(
        "Take a classroom photo to detect "
        "students and activities."
    )

    # =====================================================
    # CAMERA
    # =====================================================

    camera_image = st.camera_input(
        "Open Camera"
    )

    if camera_image is None:

        st.info(
            "📷 Open the camera and take a photo."
        )

        return

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    try:

        image = Image.open(
            camera_image
        ).convert("RGB")

        image_bgr = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

    except Exception as e:

        st.error(
            f"❌ Could not read camera image: {e}"
        )

        return

    # =====================================================
    # LOAD MODELS
    # =====================================================

    try:

        model = load_model()

        students, embeddings = (
            load_face_data()
        )

    except Exception as e:

        st.error(
            f"❌ Could not load AI models: {e}"
        )

        return

    # =====================================================
    # BAHRAIN DATE & TIME
    # =====================================================

    now = datetime.now(
        BAHRAIN_TIMEZONE
    )

    current_date = now.strftime(
        "%Y-%m-%d"
    )

    current_time = now.strftime(
        "%H:%M:%S"
    )

    # =====================================================
    # AI ANALYSIS
    # =====================================================

    with st.spinner(
        "🤖 BASMA is analyzing..."
    ):

        # =================================================
        # YOLO
        # =================================================

        try:

            results = model.predict(
                image_bgr,
                conf=CONFIDENCE,
                verbose=False
            )

            result = results[0]

            output = result.plot()

        except Exception as e:

            st.error(
                f"❌ YOLO detection failed: {e}"
            )

            return

        # =================================================
        # ACTIVITIES
        # =================================================

        activities = []

        if result.boxes is not None:

            for box in result.boxes:

                try:

                    class_id = int(
                        box.cls[0]
                    )

                    activity = str(
                        model.names[class_id]
                    )

                    if activity not in activities:

                        activities.append(
                            activity
                        )

                except Exception:

                    continue

        # =================================================
        # FACE DETECTION
        # =================================================

        detected_students = []

        try:

            faces = face_app.get(
                image_bgr
            )

            for face in faces:

                # -----------------------------------------
                # FACE COORDINATES
                # -----------------------------------------

                x1, y1, x2, y2 = [
                    int(value)
                    for value in face.bbox
                ]

                # -----------------------------------------
                # FACE RECOGNITION
                # -----------------------------------------

                student_id = find_student(
                    face.embedding
                )

                # =========================================
                # REGISTERED STUDENT
                # =========================================

                if student_id is not None:

                    student_id = str(
                        student_id
                    )

                    student_name = (
                        get_student_name(
                            students,
                            student_id
                        )
                    )

                    # -------------------------------------
                    # ADD STUDENT ONCE
                    # -------------------------------------

                    already_detected = any(
                        student["student_id"]
                        == student_id
                        for student
                        in detected_students
                    )

                    if not already_detected:

                        detected_students.append(
                            {
                                "student_id": student_id,
                                "name": student_name
                            }
                        )

                        # =================================
                        # SAVE ATTENDANCE
                        # =================================

                        try:

                            first_seen_today = (
                                save_attendance(
                                    student_id=student_id,
                                    date=current_date,
                                    time=current_time
                                )
                            )

                            # =================================
                            # SEND PARENT EMAIL
                            # =================================

                            if first_seen_today:

                                parent_email = (
                                    get_parent_email(
                                        students,
                                        student_id
                                    )
                                )

                                if not parent_email:

                                    st.warning(
                                        f"⚠️ {student_name} "
                                        "has no parent email."
                                    )

                                else:

                                    email_sent = (
                                        send_attendance_email(
                                            parent_email=parent_email,
                                            student_name=student_name,
                                            time_str=current_time
                                        )
                                    )

                                    if email_sent:

                                        st.success(
                                            f"📧 Attendance email "
                                            f"sent to parent of "
                                            f"{student_name}."
                                        )

                                    else:

                                        st.error(
                                            f"❌ Attendance was saved "
                                            f"for {student_name}, "
                                            f"but the parent email "
                                            f"could not be sent."
                                        )

                        except Exception as attendance_error:

                            st.error(
                                "❌ Attendance error: "
                                f"{attendance_error}"
                            )

                    # -------------------------------------
                    # FACE BOX
                    # -------------------------------------

                    color = (
                        0,
                        255,
                        0
                    )

                    label = (
                        f"{student_name} | Present"
                    )

                # =========================================
                # UNKNOWN STUDENT
                # =========================================

                else:

                    color = (
                        0,
                        165,
                        255
                    )

                    label = "Unknown"

                # -----------------------------------------
                # DRAW FACE BOX
                # -----------------------------------------

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                cv2.putText(
                    output,
                    label,
                    (
                        x1,
                        max(
                            30,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

        except Exception as e:

            st.warning(
                f"Face detection error: {e}"
            )

    # =====================================================
    # SAVE ACTIVITIES
    # =====================================================

    if activities and detected_students:

        for student in detected_students:

            for activity in activities:

                try:

                    save_activity(
                        student_id=student[
                            "student_id"
                        ],

                        date=current_date,

                        time=current_time,

                        activity=activity
                    )

                except Exception as e:

                    st.warning(
                        f"Activity save error: {e}"
                    )

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.image(
        cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        ),
        caption=(
            "BASMA Detection Result"
        ),
        use_container_width=True
    )

    # =====================================================
    # ATTENDANCE
    # =====================================================

    st.markdown(
        "### 👥 Attendance"
    )

    if detected_students:

        for student in detected_students:

            st.success(
                f'🟢 {student["name"]} — Present'
            )

    else:

        st.warning(
            "No registered student detected."
        )

    # =====================================================
    # ACTIVITIES
    # =====================================================

    st.markdown(
        "### 🎯 Detected Activities"
    )

    if activities:

        for activity in activities:

            st.info(
                f"Activity: {activity}"
            )

    else:

        st.info(
            "No classroom activity detected."
        )
