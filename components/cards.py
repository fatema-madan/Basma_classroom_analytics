import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.data_manager import (
    load_students,
    load_attendance,
    load_activity
)


def get_bahrain_today():

    return datetime.now(
        ZoneInfo("Asia/Bahrain")
    ).strftime("%Y-%m-%d")


def render_metric_cards():

    students = load_students()
    attendance = load_attendance()
    activities = load_activity()

    today = get_bahrain_today()

    total_students = len(students)

    today_attendance = attendance[
        attendance["date"].astype(str) == today
    ]

    present_students = today_attendance[
        today_attendance["status"] == "Present"
    ]["student_id"].astype(str).nunique()

    absent_students = max(
        total_students - present_students,
        0
    )

    if total_students > 0:

        attendance_rate = (
            present_students / total_students
        ) * 100

    else:

        attendance_rate = 0

    today_activities = activities[
        activities["date"].astype(str) == today
    ]

    activity_count = len(
        today_activities
    )

    metrics = [
        {
            "icon": "👥",
            "label": "Total Students",
            "value": total_students
        },
        {
            "icon": "✓",
            "label": "Present Today",
            "value": present_students
        },
        {
            "icon": "◔",
            "label": "Attendance Rate",
            "value": f"{attendance_rate:.1f}%"
        },
        {
            "icon": "⌁",
            "label": "Activity Detections",
            "value": activity_count
        }
    ]

    columns = st.columns(4)

    for column, metric in zip(
        columns,
        metrics
    ):

        with column:

            html = (
                f'<div class="metric-card">'
                f'<div class="metric-icon">'
                f'{metric["icon"]}'
                f'</div>'
                f'<div class="metric-label">'
                f'{metric["label"]}'
                f'</div>'
                f'<div class="metric-value">'
                f'{metric["value"]}'
                f'</div>'
                f'</div>'
            )

            st.markdown(
                html,
                unsafe_allow_html=True
            )
