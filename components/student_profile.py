import streamlit as st
from pathlib import Path
from datetime import datetime

from utils.data_manager import (
    load_students,
    load_attendance,
    load_activity
)


def render_student_profile():

    students = load_students()

    if students.empty:
        st.info("No students registered yet.")
        return

    st.markdown(
        '<div class="page-title">Student Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        "View student information, attendance, and activities."
        '</div>',
        unsafe_allow_html=True
    )

    # Select student
    student_names = students["student_name"].tolist()

    selected_name = st.selectbox(
        "Select Student",
        student_names
    )

    student = students[
        students["student_name"] == selected_name
    ].iloc[0]

    student_id = str(student["student_id"])

    # Load data
    attendance = load_attendance()
    activities = load_activity()

    today = datetime.now().strftime("%Y-%m-%d")

    # Student attendance for today
    student_attendance = attendance[
        (attendance["student_id"].astype(str) == student_id)
        & (attendance["date"] == today)
    ]

    # Student activities for today
    student_activities = activities[
        (activities["student_id"].astype(str) == student_id)
        & (activities["date"] == today)
    ]

    # =====================================================
    # PROFILE
    # =====================================================

    profile_col, info_col = st.columns([1, 2])

    with profile_col:

        photo_path = Path(
            str(student["photo_path"])
        )

        if photo_path.exists():

            st.image(
                str(photo_path),
                width=180
            )

        else:

            st.info(
                "Student photo not found."
            )

    with info_col:

        st.markdown(
            '<div class="student-card">',
            unsafe_allow_html=True
        )

        # IMPORTANT: single-line HTML strings below — no leading
        # indentation, otherwise Markdown renders them as a raw
        # code block instead of parsing the HTML.

        st.markdown(
            f'<div class="student-name">{student["student_name"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="student-info">'
            f'Student ID: {student["student_id"]}<br>'
            f'Parent Email: {student["parent_email"]}<br>'
            f'Parent Phone: {student["parent_phone"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Attendance status
        if not student_attendance.empty:

            row = student_attendance.iloc[0]

            st.markdown(
                '<div class="present">✓ Present</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"First seen: {row['first_seen']}"
            )

            st.write(
                f"Last seen: {row['last_seen']}"
            )

        else:

            st.markdown(
                '<div style="display:inline-block;background:#F4E8E6;'
                'color:#A56F69;border-radius:20px;padding:5px 12px;'
                'font-size:11px;font-weight:700;">Absent</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # =====================================================
    # ACTIVITY SUMMARY
    # =====================================================

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        "Today's Activity"
        '</div>',
        unsafe_allow_html=True
    )

    if student_activities.empty:

        st.info(
            "No activity recorded today."
        )

    else:

        activity_counts = (
            student_activities["activity"]
            .value_counts()
        )

        for activity, count in activity_counts.items():

            st.write(
                f"**{activity}** — {count}"
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # ACTIVITY TIMELINE
    # =====================================================

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        "Activity Timeline"
        '</div>',
        unsafe_allow_html=True
    )

    if student_activities.empty:

        st.info(
            "No activity recorded today."
        )

    else:

        student_activities = student_activities.iloc[::-1]

        for _, row in student_activities.iterrows():

            # Single-line HTML again — same fix as above.
            st.markdown(
                f'<div class="timeline-item">'
                f'<span class="timeline-time">{row["time"]}</span>'
                f'<span class="timeline-activity">{row["activity"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )
