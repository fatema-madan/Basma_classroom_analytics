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

    # Using a native Streamlit bordered container instead of a
    # hand-rolled "<div class='panel'>...</div>" split across two
    # separate st.markdown calls (that pattern leaves an empty ghost
    # box, since Streamlit renders each markdown call as its own
    # independent HTML fragment).
    with st.container(border=True):

        student_id = st.text_input("Student ID")

        student_name = st.text_input("Student Name")

        parent_email = st.text_input("Parent Email")

        parent_phone = st.text_input("Parent Phone")

        photo = st.file_uploader(
            "Student Photo",
            type=["jpg", "jpeg", "png"]
        )

        if photo is not None:
            st.image(
                photo,
                width=180
            )

        register_button = st.button(
            "Register Student",
            type="primary"
        )

        if register_button:

            if (
                not student_id
                or not student_name
                or not parent_email
                or not parent_phone
                or photo is None
            ):

                st.warning(
                    "Please complete all fields and upload a photo."
                )

            else:

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

                else:

                    PHOTO_FOLDER.mkdir(
                        parents=True,
                        exist_ok=True
                    )

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

                    embedding = create_embedding(
                        str(photo_path)
                    )

                    if embedding is None:

                        photo_path.unlink(
                            missing_ok=True
                        )

                        st.error(
                            "No clear face was found in the photo."
                        )

                    else:

                        save_student(
                            student_id=student_id,
                            student_name=student_name,
                            parent_email=parent_email,
                            parent_phone=parent_phone,
                            photo_path=str(photo_path)
                        )

                        save_embedding(
                            student_id,
                            embedding
                        )

                        st.success(
                            f"{student_name} has been registered successfully!"
                        )

                        st.info(
                            "The student is now ready for attendance and activity analysis."
                        )
