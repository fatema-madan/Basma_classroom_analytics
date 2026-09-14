import json

import gspread
import streamlit as st

from google.oauth2.service_account import Credentials


# =========================================================
# GOOGLE SHEETS SETTINGS
# =========================================================

SHEET_ID = st.secrets.get(
    "GOOGLE_SHEET_ID",
    ""
)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


# =========================================================
# CHECK GOOGLE SHEETS CONFIGURATION
# =========================================================

def is_google_sheets_configured():

    if not SHEET_ID:
        return False

    if "GOOGLE_SERVICE_ACCOUNT" not in st.secrets:
        return False

    return True


# =========================================================
# CONNECT TO GOOGLE SHEETS
# =========================================================

def get_google_sheet():

    if not is_google_sheets_configured():

        raise RuntimeError(
            "Google Sheets is not configured. "
            "Please add GOOGLE_SHEET_ID and "
            "GOOGLE_SERVICE_ACCOUNT to Streamlit Secrets."
        )

    service_account_value = st.secrets[
        "GOOGLE_SERVICE_ACCOUNT"
    ]

    # -----------------------------------------------------
    # Convert Secret to dictionary
    # -----------------------------------------------------

    if isinstance(
        service_account_value,
        str
    ):

        service_account_data = json.loads(
            service_account_value
        )

    else:

        service_account_data = dict(
            service_account_value
        )

    # -----------------------------------------------------
    # Create credentials
    # -----------------------------------------------------

    credentials = (
        Credentials.from_service_account_info(
            service_account_data,
            scopes=SCOPES
        )
    )

    # -----------------------------------------------------
    # Connect
    # -----------------------------------------------------

    client = gspread.authorize(
        credentials
    )

    spreadsheet = client.open_by_key(
        SHEET_ID
    )

    return spreadsheet


# =========================================================
# GET / CREATE WORKSHEET
# =========================================================

def get_or_create_worksheet(
    spreadsheet,
    worksheet_name,
    headers
):

    try:

        worksheet = spreadsheet.worksheet(
            worksheet_name
        )

    except gspread.WorksheetNotFound:

        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=1000,
            cols=max(len(headers), 10)
        )

        worksheet.append_row(
            headers
        )

    # -----------------------------------------------------
    # Add headers if sheet is empty
    # -----------------------------------------------------

    if not worksheet.get_all_values():

        worksheet.append_row(
            headers
        )

    return worksheet


# =========================================================
# ATTENDANCE
# =========================================================

def sync_attendance(
    student_id,
    date,
    first_seen,
    last_seen,
    status
):

    try:

        spreadsheet = get_google_sheet()

        worksheet = get_or_create_worksheet(
            spreadsheet,
            "Attendance",
            [
                "Student ID",
                "Date",
                "First Seen",
                "Last Seen",
                "Status"
            ]
        )

        records = worksheet.get_all_records()

        existing_row = None

        # -------------------------------------------------
        # Find existing attendance
        # -------------------------------------------------

        for index, record in enumerate(
            records,
            start=2
        ):

            same_student = (
                str(
                    record.get(
                        "Student ID",
                        ""
                    )
                )
                == str(student_id)
            )

            same_date = (
                str(
                    record.get(
                        "Date",
                        ""
                    )
                )
                == str(date)
            )

            if same_student and same_date:

                existing_row = index

                break

        # -------------------------------------------------
        # Update existing row
        # -------------------------------------------------

        if existing_row:

            worksheet.update(
                range_name=(
                    f"A{existing_row}:E{existing_row}"
                ),
                values=[[
                    student_id,
                    date,
                    first_seen,
                    last_seen,
                    status
                ]]
            )

        # -------------------------------------------------
        # Add new row
        # -------------------------------------------------

        else:

            worksheet.append_row([
                student_id,
                date,
                first_seen,
                last_seen,
                status
            ])

        return True

    except Exception as error:

        print(
            "Google Sheets attendance sync failed:",
            error
        )

        return False


# =========================================================
# ACTIVITY
# =========================================================

def sync_activity(
    student_id,
    date,
    time,
    activity
):

    try:

        spreadsheet = get_google_sheet()

        worksheet = get_or_create_worksheet(
            spreadsheet,
            "Activities",
            [
                "Student ID",
                "Date",
                "Time",
                "Activity"
            ]
        )

        worksheet.append_row([
            student_id,
            date,
            time,
            activity
        ])

        return True

    except Exception as error:

        print(
            "Google Sheets activity sync failed:",
            error
        )

        return False


# =========================================================
# SYNC ALL DATA
# =========================================================

def sync_all_data(
    attendance,
    activities
):

    # -----------------------------------------------------
    # Google Sheets not configured
    # -----------------------------------------------------

    if not is_google_sheets_configured():

        print(
            "Google Sheets is not configured."
        )

        return False

    try:

        success = True

        # =================================================
        # ATTENDANCE
        # =================================================

        if (
            attendance is not None
            and not attendance.empty
        ):

            for _, row in attendance.iterrows():

                ok = sync_attendance(
                    student_id=str(
                        row.get(
                            "student_id",
                            ""
                        )
                    ),

                    date=str(
                        row.get(
                            "date",
                            ""
                        )
                    ),

                    first_seen=str(
                        row.get(
                            "first_seen",
                            ""
                        )
                    ),

                    last_seen=str(
                        row.get(
                            "last_seen",
                            ""
                        )
                    ),

                    status=str(
                        row.get(
                            "status",
                            ""
                        )
                    )
                )

                if not ok:

                    success = False

        # =================================================
        # ACTIVITIES
        # =================================================

        if (
            activities is not None
            and not activities.empty
        ):

            for _, row in activities.iterrows():

                ok = sync_activity(
                    student_id=str(
                        row.get(
                            "student_id",
                            ""
                        )
                    ),

                    date=str(
                        row.get(
                            "date",
                            ""
                        )
                    ),

                    time=str(
                        row.get(
                            "time",
                            ""
                        )
                    ),

                    activity=str(
                        row.get(
                            "activity",
                            ""
                        )
                    )
                )

                if not ok:

                    success = False

        return success

    except Exception as error:

        print(
            "Google Sheets full sync failed:",
            error
        )

        return False
