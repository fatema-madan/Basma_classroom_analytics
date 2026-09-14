import streamlit as st
from pathlib import Path

from utils.data_manager import load_students, save_student
from utils.face_utils import create_embedding, save_embedding


PHOTO_FOLDER = Path("student_images/student_photos")


def render_student_registration():

    st.markdown(
        '<div class="page-title">Student Registration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Register a student before starting classroom monitoring."
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # REGISTRATION FORM
    # =====================================================

    with st.container(border=True):

        # Required field note
        st.caption("* Required field")

        student_id = st.text_input(
            "Student ID *"
        )

        student_name = st.text_input(
            "Student Name *"
        )

        parent_email = st.text_input(
            "Parent Email *"
        )

        parent_phone = st.text_input(
            "Parent Phone *"
        )

        # =================================================
        # PHOTO SOURCE
        # =================================================

        st.markdown(
            "### 📸 Student Photo *"
        )

        photo_source = st.radio(
            "Choose how to add the student's photo:",
            [
                "Upload Photo",
                "Take Photo"
            ],
            horizontal=True
        )

        photo = None

        # =================================================
        # UPLOAD PHOTO
        # =================================================

        if photo_source == "Upload Photo":

            photo = st.file_uploader(
                "Upload Student Photo",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

        # =================================================
        # CAMERA
        # =================================================

        else:

            photo = st.camera_input(
                "Take Student Photo"
            )

        # =================================================
        # PHOTO PREVIEW
        # =================================================

        if photo is not None:

            st.image(
                photo,
                width=180,
                caption="Student Photo"
            )

        # =================================================
        # REGISTER BUTTON
        # =================================================

        register_button = st.button(
            "Register Student",
            type="primary"
        )

        # =================================================
        # REGISTER STUDENT
        # =================================================

        if register_button:

            # ---------------------------------------------
            # Validate fields
            # ---------------------------------------------

            if (
                not student_id
                or not student_name
                or not parent_email
                or not parent_phone
                or photo is None
            ):

                st.warning(
                    "Please complete all required fields "
                    "and provide a student photo."
                )

                return

            # ---------------------------------------------
            # Check duplicate Student ID
            # ---------------------------------------------

            students = load_students()

            if (
                not students.empty
                and students["student_id"]
                .astype(str)
                .eq(str(student_id))
                .any()
            ):

                st.error(
                    "This Student ID is already registered."
                )

                return

            # ---------------------------------------------
            # Create photo folder
            # ---------------------------------------------

            PHOTO_FOLDER.mkdir(
                parents=True,
                exist_ok=True
            )

            # ---------------------------------------------
            # Save photo
            # ---------------------------------------------

            photo_path = (
                PHOTO_FOLDER
                / f"{student_id}.jpg"
            )

            with open(
                photo_path,
                "wb"
            ) as file:

                file.write(
                    photo.getbuffer()
                )

            # ---------------------------------------------
            # Create face embedding
            # ---------------------------------------------

            embedding = create_embedding(
                str(photo_path)
            )

            # ---------------------------------------------
            # Check face
            # ---------------------------------------------

            if embedding is None:

                photo_path.unlink(
                    missing_ok=True
                )

                st.error(
                    "No clear face was found in the photo. "
                    "Please use a clear front-facing photo."
                )

                return

            # ---------------------------------------------
            # Save student
            # ---------------------------------------------

            save_student(
                student_id=student_id,
                student_name=student_name,
                parent_email=parent_email,
                parent_phone=parent_phone,
                photo_path=str(photo_path)
            )

            # ---------------------------------------------
            # Save embedding
            # ---------------------------------------------

            save_embedding(
                student_id,
                embedding
            )

            # ---------------------------------------------
            # Success
            # ---------------------------------------------

            st.success(
                f"{student_name} has been registered successfully!"
            )

            st.info(
                "The student is now ready for attendance "
                "and activity analysis."
            )
