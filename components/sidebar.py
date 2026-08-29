from pathlib import Path
import streamlit as st

from utils.data_manager import load_students


def render_sidebar():

    logo_path = Path("assets/basma_logo.jpg")

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

        else:
            # NOTE: removed the broken "elif small_logo_path.exists():"
            # branch — small_logo_path was never defined anywhere in this
            # file, so this would crash with a NameError whenever the
            # main logo file was missing.
            st.markdown(
                '<div style="text-align:center;color:#4D7964;margin-bottom:25px;">'
                '<h2>BASMA</h2>'
                '<small>AI Classroom Analytics</small>'
                '</div>',
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

        # IMPORTANT: no leading indentation before the HTML lines,
        # otherwise Markdown treats it as a code block instead of HTML.
        st.markdown(
            f'<div class="status-box">'
            f'<span class="status-dot">●</span>'
            f'&nbsp; System Online'
            f'<br><br>'
            f'👥 {student_count} registered students'
            f'</div>',
            unsafe_allow_html=True
        )

    return selected_page
