import streamlit as st
import pandas as pd
import plotly.express as px

from datetime import datetime
from zoneinfo import ZoneInfo

from utils.data_manager import (
    load_students,
    load_attendance,
    load_activity
)


# =========================================================
# BASMA THEME COLORS
# =========================================================

PRIMARY = "#6F9E87"
DARK_GREEN = "#4D7964"
LIGHT_GREEN = "#DDEDE3"

LAVENDER = "#B9AFD8"
LIGHT_BLUE = "#DCEBF0"
LIGHT_YELLOW = "#E8E6C7"

TEXT_PRIMARY = "#34433B"
TEXT_SECONDARY = "#8A948D"

BACKGROUND = "#F8F8F3"
CARD = "#FFFFFF"
BORDER = "#E9EDE8"

SUCCESS = "#8BC59C"
DANGER = "#D9A5A0"


# =========================================================
# TIMEZONE
# =========================================================

BAHRAIN_TIMEZONE = ZoneInfo(
    "Asia/Bahrain"
)


# =========================================================
# ACTIVITY CLASSES
# =========================================================

ACTIVITY_CLASSES = [
    "Clapping",
    "Facing-Forward",
    "Hand-Raising",
    "Reading",
    "Sleeping",
    "Talking",
    "Using-Phone",
    "Writing"
]


# =========================================================
# PLOTLY BASE STYLE
# =========================================================

def apply_chart_style(fig):

    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,

        font=dict(
            family="Manrope, sans-serif",
            color=TEXT_PRIMARY
        ),

        title_font=dict(
            size=15,
            color=TEXT_PRIMARY
        ),

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),

        xaxis=dict(
            title_font=dict(
                color=TEXT_SECONDARY
            ),

            tickfont=dict(
                color=TEXT_SECONDARY
            ),

            gridcolor=BORDER,

            linecolor=BORDER
        ),

        yaxis=dict(
            title_font=dict(
                color=TEXT_SECONDARY
            ),

            tickfont=dict(
                color=TEXT_SECONDARY
            ),

            gridcolor=BORDER,

            linecolor=BORDER
        ),

        legend=dict(
            font=dict(
                color=TEXT_PRIMARY
            )
        )
    )

    return fig


# =========================================================
# CLASS ACTIVITY CHART
# =========================================================

def render_class_activity_chart():

    activities = load_activity()

    st.markdown(
        "### 🎯 Class Activity"
    )

    if activities.empty:

        st.info(
            "No classroom activity data available."
        )

        return

    # -------------------------------------
    # Check activity column
    # -------------------------------------

    if "activity" not in activities.columns:

        st.warning(
            "Activity data is missing."
        )

        return

    # -------------------------------------
    # Count activities
    # -------------------------------------

    activity_counts = (
        activities["activity"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    activity_counts.columns = [
        "Activity",
        "Count"
    ]

    # -------------------------------------
    # Create chart
    # -------------------------------------

    fig = px.bar(
        activity_counts,
        x="Activity",
        y="Count",
        title="Classroom Activity Detection",
        text="Count"
    )

    # -------------------------------------
    # BASMA colors
    # -------------------------------------

    fig.update_traces(
        marker_color=PRIMARY,
        textposition="outside",
        textfont=dict(
            color=TEXT_PRIMARY
        )
    )

    apply_chart_style(fig)

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# ATTENDANCE CHART
# =========================================================

def render_attendance_chart():

    attendance = load_attendance()

    st.markdown(
        "### 👥 Attendance"
    )

    if attendance.empty:

        st.info(
            "No attendance data available."
        )

        return

    # -------------------------------------
    # Check date column
    # -------------------------------------

    if "date" not in attendance.columns:

        st.warning(
            "Attendance date information is missing."
        )

        return

    # -------------------------------------
    # Bahrain today
    # -------------------------------------

    today = datetime.now(
        BAHRAIN_TIMEZONE
    ).strftime(
        "%Y-%m-%d"
    )

    attendance["date"] = (
        attendance["date"]
        .astype(str)
    )

    today_attendance = attendance[
        attendance["date"] == today
    ]

    # -------------------------------------
    # No attendance today
    # -------------------------------------

    if today_attendance.empty:

        st.info(
            "No attendance recorded today."
        )

        return

    # -------------------------------------
    # Present count
    # -------------------------------------

    present_count = len(
        today_attendance
    )

    st.metric(
        "Present Today",
        present_count
    )

    # -------------------------------------
    # Attendance chart
    # -------------------------------------

    chart_data = pd.DataFrame(
        {
            "Status": [
                "Present"
            ],

            "Students": [
                present_count
            ]
        }
    )

    fig = px.pie(
        chart_data,
        names="Status",
        values="Students",
        title="Today's Attendance"
    )

    # -------------------------------------
    # BASMA pie color
    # -------------------------------------

    fig.update_traces(
        marker=dict(
            colors=[
                PRIMARY
            ]
        ),

        textfont=dict(
            color=TEXT_PRIMARY
        )
    )

    apply_chart_style(fig)

    fig.update_layout(
        height=350
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PERFORMANCE CHART
# =========================================================

def render_performance_chart(
    selected_student_id=None
):

    attendance = load_attendance()

    activities = load_activity()

    students = load_students()

    st.markdown(
        "### 📈 Student Performance"
    )

    # -------------------------------------
    # No students
    # -------------------------------------

    if students.empty:

        st.info(
            "No students registered yet."
        )

        return

    # -------------------------------------
    # Filter selected student
    # -------------------------------------

    if selected_student_id is not None:

        student_id = str(
            selected_student_id
        )

        if "student_id" in attendance.columns:

            attendance = attendance[
                attendance[
                    "student_id"
                ]
                .astype(str)
                == student_id
            ]

        if "student_id" in activities.columns:

            activities = activities[
                activities[
                    "student_id"
                ]
                .astype(str)
                == student_id
            ]

    # -------------------------------------
    # Counts
    # -------------------------------------

    attendance_count = len(
        attendance
    )

    activity_count = len(
        activities
    )

    performance_data = pd.DataFrame(
        {
            "Metric": [
                "Attendance",
                "Activities"
            ],

            "Count": [
                attendance_count,
                activity_count
            ]
        }
    )

    # -------------------------------------
    # Create chart
    # -------------------------------------

    fig = px.bar(
        performance_data,
        x="Metric",
        y="Count",
        text="Count",
        title="Student Performance Overview"
    )

    # -------------------------------------
    # Different BASMA colors
    # -------------------------------------

    fig.update_traces(
        marker_color=[
            DARK_GREEN,
            LAVENDER
        ],

        textposition="outside",

        textfont=dict(
            color=TEXT_PRIMARY
        )
    )

    apply_chart_style(fig)

    fig.update_layout(
        height=350
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

