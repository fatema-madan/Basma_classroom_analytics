from pathlib import Path
import streamlit as st

from utils.data_manager import load_students


def render_sidebar():

    logo_path = Path("assets/basma_logo.png")
    small_logo_path = Path("assets/basma_logo_small.png")

    students = load_students()

    if students.empty:
        student_count = 0
    else:
        student_count = len(students)

    with st.sidebar:

        # BASMA Logo
        if logo_path.exists():
            st.image(
                str(logo_path),
                width=170
            )

        elif small_logo_path.exists():
            st.image(
                str(small_logo_path),
                width=160
            )

        else:
            st.markdown(
                """
                <div style="
                    text-align: center;
                    color: #4D7964;
                    margin-bottom: 25px;
                ">
                    <h2>BASMA</h2>
                    <small>AI Classroom Analytics</small>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="nav-title">Workspace</div>',
            unsafe_allow_html=True
        )

        selected_page = st.radio(
            "Navigation",
            [
                "📝 Student Registration",
                "🏠 Dashboard",
                "📹 Live Classroom",
                "👤 Student Profile",
                "📊 Analytics",
                "⚙️ Settings"
            ],
            label_visibility="collapsed"
        )

        st.markdown(
            f"""
            <div class="status-box">
                <span class="status-dot">●</span>
                &nbsp; System Online

                <br><br>

                👥 {student_count} registered students
            </div>
            """,
            unsafe_allow_html=True
        )

    return selected_page
