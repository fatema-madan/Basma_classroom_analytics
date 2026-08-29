import streamlit as st
from datetime import datetime

from utils.data_manager import (
    load_students,
    load_attendance,
    load_activity
)


def render_metric_cards():

    students = load_students()
    attendance = load_attendance()
    activities = load_activity()

    today = datetime.now().strftime("%Y-%m-%d")

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

    activity_count = len(today_activities)

    metrics = [
        {
            "icon": "👥",
            "label": "Total Students",
            "value": total_students
        },
        {
            "icon": "✓",
            "label": "Present",
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

    for column, metric in zip(columns, metrics):

        with column:
            # IMPORTANT: no leading indentation before the HTML lines,
            # otherwise Markdown treats it as a code block.
            html = (
                f'<div class="metric-card">'
                f'<div class="metric-icon">{metric["icon"]}</div>'
                f'<div class="metric-label">{metric["label"]}</div>'
                f'<div class="metric-value">{metric["value"]}</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
