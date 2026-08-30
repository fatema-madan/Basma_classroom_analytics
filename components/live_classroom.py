import cv2
import streamlit as st
import pandas as pd

from PIL import Image
from ultralytics import YOLO

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

CONFIDENCE = 0.40


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO(
        MODEL_PATH
    )


# =========================================================
# LOAD DATA
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
    # CAMERA INPUT
    # =====================================================

    camera_image = st.camera_input(
        "Open Camera"
    )


    # =====================================================
    # WAIT FOR PHOTO
    # =====================================================

    if camera_image is None:

        st.info(
            "📷 Open the camera and take a photo."
        )

        return


    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image = Image.open(
        camera_image
    ).convert("RGB")


    image_bgr = cv2.cvtColor(
        __import__("numpy").array(image),
        cv2.COLOR_RGB2BGR
    )


    # =====================================================
    # LOAD MODELS
    # =====================================================

    model = load_model()

    students, embeddings = (
        load_face_data()
    )


    # =====================================================
    # AI ANALYSIS
    # =====================================================

    with st.spinner(
        "🤖 BASMA is analyzing..."
    ):

        # -----------------------------------------------
        # YOLO
        # -----------------------------------------------

        results = model.predict(
            image_bgr,
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]


        # -----------------------------------------------
        # YOLO annotated image
        # -----------------------------------------------

        output = result.plot()


        # -----------------------------------------------
        # Activities
        # -----------------------------------------------

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

                x1, y1, x2, y2 = [
                    int(value)
                    for value in face.bbox
                ]


                # -----------------------------------------
                # Face Recognition
                # -----------------------------------------

                student_id = find_student(
                    face.embedding
                )


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
                    # Attendance
                    # -------------------------------------

                    try:

                        save_attendance(
                            student_id=student_id
                        )

                    except Exception as e:

                        print(
                            "Attendance error:",
                            e
                        )


                    detected_students.append(
                        {
                            "student_id": student_id,
                            "name": student_name
                        }
                    )


                    # -------------------------------------
                    # Face Box
                    # -------------------------------------

                    color = (
                        0,
                        255,
                        0
                    )


                    label = (
                        f"{student_name} | Present"
                    )


                else:

                    color = (
                        0,
                        165,
                        255
                    )

                    label = "Unknown"


                # -----------------------------------------
                # Draw Face Box
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
    # DISPLAY RESULT
    # =====================================================

    st.image(
        cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        ),
        caption="BASMA Detection Result",
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


        # -----------------------------------------------
        # Save activities
        # -----------------------------------------------

        for student in detected_students:

            for activity in activities:

                try:

                    save_activity(
                        student_id=student[
                            "student_id"
                        ],
                        activity=activity
                    )

                except Exception as e:

                    print(
                        "Activity save error:",
                        e
                    )

    else:

        st.info(
            "No classroom activity detected."
        )
