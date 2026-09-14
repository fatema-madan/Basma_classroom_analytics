import streamlit as st

from utils.data_manager import load_activity


def render_activity_timeline(student_id=None):

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

    activities = activities.iloc[::-1]

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">Activity Timeline</div>',
        unsafe_allow_html=True
    )

    for _, row in activities.iterrows():

        st.markdown(
            f"""
            <div class="timeline-item">

                <span class="timeline-time">
                    {row["time"]}
                </span>

                <span class="timeline-activity">
                    {row["activity"]}
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )