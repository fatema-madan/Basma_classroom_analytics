from io import BytesIO

import pandas as pd


# =========================================================
# DAILY DATA
# =========================================================

def get_daily_attendance(
    attendance,
    date
):

    if attendance.empty:

        return attendance.copy()

    return attendance[
        attendance["date"].astype(str)
        == str(date)
    ].copy()


def get_daily_activity(
    activities,
    date
):

    if activities.empty:

        return activities.copy()

    return activities[
        activities["date"].astype(str)
        == str(date)
    ].copy()


# =========================================================
# WEEKLY DATA
# =========================================================

def get_weekly_attendance(
    attendance,
    start_date,
    end_date
):

    if attendance.empty:

        return attendance.copy()

    data = attendance.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    start_date = pd.Timestamp(
        start_date
    )

    end_date = pd.Timestamp(
        end_date
    )

    return data[
        (data["date"] >= start_date)
        & (data["date"] <= end_date)
    ].copy()


def get_weekly_activity(
    activities,
    start_date,
    end_date
):

    if activities.empty:

        return activities.copy()

    data = activities.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    start_date = pd.Timestamp(
        start_date
    )

    end_date = pd.Timestamp(
        end_date
    )

    return data[
        (data["date"] >= start_date)
        & (data["date"] <= end_date)
    ].copy()


# =========================================================
# CSV
# =========================================================

def dataframe_to_csv(
    dataframe
):

    return dataframe.to_csv(
        index=False
    ).encode("utf-8")


# =========================================================
# EXCEL
# =========================================================

def create_excel_report(
    attendance,
    activities
):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        attendance.to_excel(
            writer,
            sheet_name="Attendance",
            index=False
        )

        activities.to_excel(
            writer,
            sheet_name="Activities",
            index=False
        )

    output.seek(0)

    return output.getvalue()

