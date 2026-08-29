import streamlit as st
from pathlib import Path

from components.sidebar import render_sidebar
from components.cards import render_metric_cards
from components.charts import (
    render_class_activity_chart,
    render_attendance_chart
)
from components.student_registration import (
    render_student_registration
)
from components.student_profile import (
    render_student_profile
)
from components.activity_timeline import (
    render_activity_timeline
)
from components.live_classroom import (
    render_live_classroom
)


# ==================================================
# Page Setup
# ==================================================

st.set_page_config(
    page_title="BASMA — AI Classroom Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# Load Theme
# ==================================================

theme_path = Path("styles/basma_theme.css")

if theme_path.exists():

    with open(
        theme_path,
        "r",
        encoding="utf-8"
    ) as file:

        css = file.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )


# ==================================================
# Sidebar
# ==================================================

selected_page = render_sidebar()


# ==================================================
# Dashboard
# ==================================================

if selected_page == "🏠 Dashboard":

    st.markdown(
        '<div class="page-title">'
        'Good Morning, Teacher 👋'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Here's today's classroom overview."
        '</div>',
        unsafe_allow_html=True
    )

    render_metric_cards()

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col, attendance_col = st.columns([2.2, 1])

    with chart_col:
        render_class_activity_chart()

    with attendance_col:
        render_attendance_chart()

    st.markdown("<br>", unsafe_allow_html=True)

    render_activity_timeline()


# ==================================================
# Live Classroom
# ==================================================

elif selected_page == "📹 Live Classroom":

    render_live_classroom()


# ==================================================
# Student Profile
# ==================================================

elif selected_page == "👤 Student Profile":

    render_student_profile()


# ==================================================
# Analytics
# ==================================================

elif selected_page == "📊 Analytics":

    st.markdown(
        '<div class="page-title">Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Analyze classroom activity and attendance."
        '</div>',
        unsafe_allow_html=True
    )

    render_class_activity_chart()

    st.markdown("<br>", unsafe_allow_html=True)

    render_attendance_chart()


# ==================================================
# Settings
# ==================================================

elif selected_page == "⚙️ Settings":

    st.markdown(
        '<div class="page-title">Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Manage BASMA application settings."
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        "Application"
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "BASMA AI Classroom Analytics"
    )

    st.write(
        "YOLO model:"
    )

    model_path = Path(
        "models/basma_yolo.pt"
    )

    if model_path.exists():

        st.success(
            "YOLO model is ready."
        )

    else:

        st.warning(
            "YOLO model not found."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )