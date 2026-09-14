from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st


# =========================================
# Components
# =========================================

from components.live_classroom import (
    render_live_classroom
)

from components.sidebar import (
    render_sidebar
)

from components.student_registration import (
    render_student_registration
)

from components.cards import (
    render_metric_cards
)

from components.charts import (
    render_class_activity_chart,
    render_attendance_chart,
    render_performance_chart,
    ACTIVITY_CLASSES
)

from components.student_profile import (
    render_student_profile
)


# =========================================
# Data Manager
# =========================================

from utils.data_manager import (
    load_students,
    load_attendance,
    load_activity
)


# =========================================
# Reports
# =========================================

from utils.report_utils import (
    get_daily_attendance,
    get_daily_activity,
    get_weekly_attendance,
    get_weekly_activity,
    dataframe_to_csv,
    create_excel_report
)


# =========================================
# Google Sheets
# =========================================

from utils.google_sheets import (
    sync_all_data
)


# =========================================
# Email
# =========================================

from utils.email_utils import (
    send_attendance_report
)


# =========================================
# Page Configuration
# =========================================

st.set_page_config(
    page_title="BASMA — AI Classroom Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================
# Bahrain Timezone
# =========================================

BAHRAIN_TIMEZONE = ZoneInfo(
    "Asia/Bahrain"
)


def get_bahrain_now():

    return datetime.now(
        BAHRAIN_TIMEZONE
    )


def get_bahrain_date():

    return get_bahrain_now().date()


# =========================================
# Convert Student ID to Student Name
# =========================================

def add_student_names(dataframe, students):

    data = dataframe.copy()

    if data.empty:
        return data

    if "student_id" not in data.columns:
        return data

    if students.empty:
        return data

    student_map = dict(
        zip(
            students["student_id"].astype(str),
            students["student_name"]
        )
    )

    data["student_name"] = (
        data["student_id"]
        .astype(str)
        .map(student_map)
        .fillna("Unknown")
    )

    data = data.drop(
        columns=["student_id"]
    )

    return data


# =========================================
# Dashboard Settings
# =========================================

if "dashboard_settings" not in st.session_state:

    st.session_state.dashboard_settings = {

        "show_metrics": True,

        "show_activity": True,

        "show_attendance": True,

        "show_profile": True,

        "show_performance": True
    }


# =========================================
# Load Theme
# =========================================

theme_path = Path(
    "styles/basma_theme.css"
)

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


# =========================================
# Sidebar
# =========================================

selected_page = render_sidebar()


# =========================================
# Student Registration
# =========================================

if selected_page == "📝 Student Registration":

    render_student_registration()


# =========================================
# Dashboard
# =========================================

elif selected_page == "🏠 Dashboard":

    # -------------------------------------
    # Bahrain Time
    # -------------------------------------

    bahrain_time = get_bahrain_now()

    current_hour = bahrain_time.hour

    if current_hour < 12:

        greeting = "Good Morning"

    elif current_hour < 18:

        greeting = "Good Afternoon"

    else:

        greeting = "Good Evening"

    # -------------------------------------
    # Header
    # -------------------------------------

    st.markdown(
        f'<div class="page-title">'
        f'{greeting}, Teacher 👋'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Here's today's classroom overview."
        '</div>',
        unsafe_allow_html=True
    )

    # -------------------------------------
    # Load Data
    # -------------------------------------

    students = load_students()

    attendance = load_attendance()

    activities = load_activity()

    # -------------------------------------
    # Dashboard Filters
    # -------------------------------------

    st.markdown(
        '<div class="dashboard-filter-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="filter-title">'
        'Dashboard Filters'
        '</div>',
        unsafe_allow_html=True
    )

    filter_col1, filter_col2 = st.columns(2)

    # -------------------------------------
    # Student Filter
    # -------------------------------------

    with filter_col1:

        student_options = [
            "All Students"
        ]

        if not students.empty:

            student_options += (
                students[
                    "student_name"
                ]
                .dropna()
                .astype(str)
                .tolist()
            )

        selected_student = st.selectbox(
            "Student",
            student_options,
            key="dashboard_student"
        )

    # -------------------------------------
    # Activity Filter
    # -------------------------------------

    with filter_col2:

        activity_options = (
            ["All Activities"]
            + ACTIVITY_CLASSES
        )

        selected_activity = st.selectbox(
            "Activity",
            activity_options,
            key="dashboard_activity"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================
    # METRIC CARDS
    # =====================================

    if st.session_state.dashboard_settings[
        "show_metrics"
    ]:

        st.markdown(
            '<div class="section-label">'
            'Classroom Overview'
            '</div>',
            unsafe_allow_html=True
        )

        render_metric_cards()


    # =====================================
    # ACTIVITY + ATTENDANCE
    # =====================================

    show_activity = (
        st.session_state.dashboard_settings[
            "show_activity"
        ]
    )

    show_attendance = (
        st.session_state.dashboard_settings[
            "show_attendance"
        ]
    )


    # -------------------------------------
    # Both enabled
    # -------------------------------------

    if show_activity and show_attendance:

        chart_col, attendance_col = st.columns(
            [2.2, 1]
        )

        with chart_col:

            render_class_activity_chart()

        with attendance_col:

            render_attendance_chart()


    # -------------------------------------
    # Activity only
    # -------------------------------------

    elif show_activity:

        render_class_activity_chart()


    # -------------------------------------
    # Attendance only
    # -------------------------------------

    elif show_attendance:

        render_attendance_chart()


    # =====================================
    # STUDENT PROFILE + PERFORMANCE
    # =====================================

    show_profile = (
        st.session_state.dashboard_settings[
            "show_profile"
        ]
    )

    show_performance = (
        st.session_state.dashboard_settings[
            "show_performance"
        ]
    )


    # -------------------------------------
    # Find selected student ID
    # -------------------------------------

    selected_student_id = None

    if (
        selected_student != "All Students"
        and not students.empty
    ):

        selected_rows = students[
            students["student_name"]
            == selected_student
        ]

        if not selected_rows.empty:

            selected_student_id = (
                selected_rows.iloc[0][
                    "student_id"
                ]
            )


    # -------------------------------------
    # Both enabled
    # -------------------------------------

    if show_profile and show_performance:

        profile_col, performance_col = st.columns(
            [1.15, 1]
        )

        with profile_col:

            render_student_profile()

        with performance_col:

            render_performance_chart(
                selected_student_id
            )


    # -------------------------------------
    # Profile only
    # -------------------------------------

    elif show_profile:

        render_student_profile()


    # -------------------------------------
    # Performance only
    # -------------------------------------

    elif show_performance:

        render_performance_chart(
            selected_student_id
        )


    # =====================================
    # DOWNLOAD DASHBOARD DATA
    # =====================================

    st.markdown(
        '<div class="section-label">'
        'Download Dashboard Data'
        '</div>',
        unsafe_allow_html=True
    )

    download_col1, download_col2, download_col3 = (
        st.columns(3)
    )


    # -------------------------------------
    # Attendance
    # -------------------------------------

    with download_col1:

        attendance_download = add_student_names(
            attendance,
            students
        )

        attendance_csv = (
            attendance_download
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="📥 Download Attendance",
            data=attendance_csv,
            file_name="BASMA_Attendance.csv",
            mime="text/csv",
            use_container_width=True,
            key="dashboard_attendance_download"
        )


    # -------------------------------------
    # Activities
    # -------------------------------------

    with download_col2:

        activities_download = add_student_names(
            activities,
            students
        )

        activities_csv = (
            activities_download
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="📥 Download Activities",
            data=activities_csv,
            file_name="BASMA_Activities.csv",
            mime="text/csv",
            use_container_width=True,
            key="dashboard_activity_download"
        )


    # -------------------------------------
    # Students
    # -------------------------------------

    with download_col3:

        students_csv = (
            students
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="📥 Download Students",
            data=students_csv,
            file_name="BASMA_Students.csv",
            mime="text/csv",
            use_container_width=True,
            key="dashboard_students_download"
        )


# =========================================
# Live Classroom
# =========================================

elif selected_page == "📹 Live Classroom":

    render_live_classroom()


# =========================================
# Student Profile
# =========================================

elif selected_page == "👤 Student Profile":

    render_student_profile()


# =========================================
# Analytics
# =========================================

elif selected_page == "📊 Analytics":

    st.markdown(
        '<div class="page-title">'
        'Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Analyze classroom attendance and '
        'student activities.'
        '</div>',
        unsafe_allow_html=True
    )

    render_class_activity_chart()

    col1, col2 = st.columns(2)

    with col1:

        render_attendance_chart()

    with col2:

        render_performance_chart()


# =========================================
# Reports
# =========================================

elif selected_page == "📋 Reports":

    st.markdown(
        '<div class="page-title">'
        'Reports'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Export attendance and classroom activity reports.'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================
    # LOAD DATA
    # =====================================

    students = load_students()

    attendance = load_attendance()

    activities = load_activity()


    # =====================================
    # REPORT TYPE
    # =====================================

    report_type = st.radio(
        "Report Period",
        [
            "Daily Report",
            "Weekly Report"
        ],
        horizontal=True
    )


    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        'Report Settings'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================
    # DAILY
    # =====================================

    if report_type == "Daily Report":

        selected_date = st.date_input(
            "Select Date",
            value=get_bahrain_date()
        )

        daily_attendance = (
            get_daily_attendance(
                attendance,
                selected_date
            )
        )

        daily_activity = (
            get_daily_activity(
                activities,
                selected_date
            )
        )

        report_name = (
            f"BASMA_Daily_Report_"
            f"{selected_date}"
        )


    # =====================================
    # WEEKLY
    # =====================================

    else:

        today = get_bahrain_date()

        start_date = st.date_input(
            "Start Date",
            value=today - timedelta(days=6)
        )

        end_date = st.date_input(
            "End Date",
            value=today
        )

        daily_attendance = (
            get_weekly_attendance(
                attendance,
                start_date,
                end_date
            )
        )

        daily_activity = (
            get_weekly_activity(
                activities,
                start_date,
                end_date
            )
        )

        report_name = (
            f"BASMA_Weekly_Report_"
            f"{start_date}_"
            f"to_"
            f"{end_date}"
        )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================
    # CONVERT IDs TO NAMES
    # =====================================

    attendance_display = add_student_names(
        daily_attendance,
        students
    )

    activity_display = add_student_names(
        daily_activity,
        students
    )


    # =====================================
    # REPORT PREVIEW
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Attendance Records",
            len(daily_attendance)
        )

    with col2:

        st.metric(
            "Activity Detections",
            len(daily_activity)
        )


    st.markdown(
        "### Attendance Preview"
    )

    if attendance_display.empty:

        st.info(
            "No attendance records found."
        )

    else:

        st.dataframe(
            attendance_display,
            use_container_width=True,
            hide_index=True
        )


    st.markdown(
        "### Activity Preview"
    )

    if activity_display.empty:

        st.info(
            "No activity records found."
        )

    else:

        st.dataframe(
            activity_display,
            use_container_width=True,
            hide_index=True
        )


    # =====================================
    # DOWNLOAD
    # =====================================

    st.markdown(
        "### 📥 Download Report"
    )

    download_col1, download_col2 = st.columns(2)


    # -------------------------------------
    # Attendance CSV
    # -------------------------------------

    with download_col1:

        csv_data = dataframe_to_csv(
            attendance_display
        )

        st.download_button(
            label="Download Attendance CSV",
            data=csv_data,
            file_name=(
                f"{report_name}_attendance.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


    # -------------------------------------
    # Excel
    # -------------------------------------

    with download_col2:

        excel_data = create_excel_report(
            attendance_display,
            activity_display
        )

        st.download_button(
            label="Download Excel Report",
            data=excel_data,
            file_name=(
                f"{report_name}.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )


    # =====================================
    # EMAIL REPORT
    # =====================================

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        '📧 Email Attendance Report'
        '</div>',
        unsafe_allow_html=True
    )

    recipient_email = st.text_input(
        "Recipient Email",
        placeholder="teacher@example.com"
    )


    if st.button(
        "Send Attendance Report",
        type="primary",
        use_container_width=True
    ):

        if not recipient_email.strip():

            st.warning(
                "Please enter a recipient email."
            )

        else:

            if report_type == "Daily Report":

                email_date = str(
                    selected_date
                )

            else:

                email_date = (
                    f"{start_date} to {end_date}"
                )

            sent = send_attendance_report(
                recipient=recipient_email.strip(),
                attendance=daily_attendance,
                students=students,
                report_date=email_date
            )

            if sent:

                st.success(
                    "Attendance report sent successfully."
                )

            else:

                st.error(
                    "The report could not be sent. "
                    "Please check your email settings."
                )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================
    # GOOGLE SHEETS
    # =====================================

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        '📊 Google Sheets'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Sync attendance and activity records "
        "to your Google Sheet."
    )


    if st.button(
        "Sync Data to Google Sheets",
        use_container_width=True
    ):

        with st.spinner(
            "Syncing data..."
        ):

            synced = sync_all_data(
                attendance=attendance,
                activities=activities
            )

        if synced:

            st.success(
                "Attendance and activity data "
                "synced successfully."
            )

        else:

            st.error(
                "Google Sheets sync failed. "
                "Check your Google credentials "
                "and spreadsheet settings."
            )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================
# Settings
# =========================================

elif selected_page == "⚙️ Settings":

    st.markdown(
        '<div class="page-title">'
        'Settings'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Customize your BASMA dashboard.'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================
    # Dashboard Customization
    # =====================================

    st.markdown(
        '<div class="settings-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        'Dashboard Customization'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Choose which sections you want "
        "to display on your dashboard."
    )


    settings = (
        st.session_state.dashboard_settings
    )


    settings["show_metrics"] = st.checkbox(
        "Show Metric Cards",
        value=settings["show_metrics"],
        key="setting_metrics"
    )

    settings["show_activity"] = st.checkbox(
        "Show Class Activity",
        value=settings["show_activity"],
        key="setting_activity"
    )

    settings["show_attendance"] = st.checkbox(
        "Show Attendance",
        value=settings["show_attendance"],
        key="setting_attendance"
    )

    settings["show_profile"] = st.checkbox(
        "Show Student Profile",
        value=settings["show_profile"],
        key="setting_profile"
    )

    settings["show_performance"] = st.checkbox(
        "Show Student Performance",
        value=settings["show_performance"],
        key="setting_performance"
    )


    st.session_state.dashboard_settings = (
        settings
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================
    # System Information
    # =====================================

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        'System Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "BASMA AI Classroom Analytics"
    )

    st.write(
        "YOLO classroom activity detection "
        "is enabled."
    )

    st.write(
        "Timezone: Bahrain (UTC+3)"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )
