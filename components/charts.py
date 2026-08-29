import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_manager import (
    load_students,
    load_attendance,
    load_activity
)


def render_attendance_chart():
    students = load_students()
    attendance = load_attendance()

    total_students = len(students)

    if total_students == 0:
        st.info("No students registered yet.")
        return

    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    today_attendance = attendance[
        attendance["date"].astype(str) == today
    ]

    present = today_attendance[
        today_attendance["status"] == "Present"
    ]["student_id"].astype(str).nunique()

    absent = max(total_students - present, 0)

    data = pd.DataFrame({
        "Status": ["Present", "Absent"],
        "Students": [present, absent]
    })

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">Today\'s Attendance</div>',
        unsafe_allow_html=True
    )

    chart = px.pie(
        data,
        values="Students",
        names="Status",
        hole=0.72,
        color="Status",
        color_discrete_map={
            "Present": "#8BC59C",
            "Absent": "#E8E6C7"
        }
    )

    attendance_rate = (
        (present / total_students) * 100
        if total_students > 0
        else 0
    )

    chart.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        annotations=[
            dict(
                text=f"<b>{present}</b><br>"
                     f"<span style='font-size:10px'>Present</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    size=18,
                    color="#527B68"
                )
            )
        ]
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    st.markdown(
        f"🟢 Present: **{present}** &nbsp;&nbsp; "
        f"🟡 Absent: **{absent}**"
    )

    st.caption(
        f"Attendance Rate: {attendance_rate:.1f}%"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


def render_class_activity_chart():
    activities = load_activity()

    if activities.empty:
        st.info("No classroom activity recorded yet.")
        return

    activity_counts = (
        activities["activity"]
        .value_counts()
        .reset_index()
    )

    activity_counts.columns = [
        "Activity",
        "Count"
    ]

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">Class Activity</div>',
        unsafe_allow_html=True
    )

    chart = px.bar(
        activity_counts,
        x="Activity",
        y="Count",
        text="Count"
    )

    chart.update_layout(
        height=320,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(
            family="Manrope",
            size=10,
            color="#8A948D"
        ),
        xaxis=dict(
            title="",
            showgrid=False
        ),
        yaxis=dict(
            title="Detections",
            showgrid=True,
            gridcolor="#EEF2EE"
        )
    )

    chart.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


def render_performance_chart(student_id=None):
    activities = load_activity()

    if activities.empty:
        st.info("No activity recorded yet.")
        return

    if student_id is not None:
        activities = activities[
            activities["student_id"].astype(str)
            == str(student_id)
        ]

    if activities.empty:
        st.info("No activity recorded for this student.")
        return

    activity_counts = (
        activities["activity"]
        .value_counts()
        .reset_index()
    )

    activity_counts.columns = [
        "Activity",
        "Count"
    ]

    total = activity_counts["Count"].sum()

    activity_counts["Percentage"] = (
        activity_counts["Count"] / total * 100
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">Student Performance</div>',
        unsafe_allow_html=True
    )

    chart = px.bar(
        activity_counts,
        x="Percentage",
        y="Activity",
        orientation="h",
        text="Percentage"
    )

    chart.update_layout(
        height=280,
        margin=dict(
            l=0,
            r=10,
            t=0,
            b=0
        ),
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(
            family="Manrope",
            size=10,
            color="#8A948D"
        ),
        xaxis=dict(
            title="Percentage",
            range=[0, 100],
            showgrid=False
        ),
        yaxis=dict(
            title=""
        )
    )

    chart.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )