from pathlib import Path

import pandas as pd


# =========================================================
# DATA FILES
# =========================================================

DATA_FOLDER = Path("data")

STUDENTS_FILE = DATA_FOLDER / "students.csv"
ATTENDANCE_FILE = DATA_FOLDER / "attendance.csv"
ACTIVITY_FILE = DATA_FOLDER / "activity_log.csv"


# =========================================================
# STUDENTS
# =========================================================

def load_students():

    if not STUDENTS_FILE.exists():
        return pd.DataFrame(
            columns=[
                "student_id",
                "student_name",
                "parent_email",
                "parent_phone",
                "photo_path"
            ]
        )

    try:
        students = pd.read_csv(STUDENTS_FILE)

    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=[
                "student_id",
                "student_name",
                "parent_email",
                "parent_phone",
                "photo_path"
            ]
        )

    return students


def save_student(
    student_id,
    student_name,
    parent_email,
    parent_phone,
    photo_path
):

    students = load_students()

    new_student = {
        "student_id": student_id,
        "student_name": student_name,
        "parent_email": parent_email,
        "parent_phone": parent_phone,
        "photo_path": photo_path
    }

    students.loc[len(students)] = new_student

    students.to_csv(
        STUDENTS_FILE,
        index=False
    )


# =========================================================
# ATTENDANCE
# =========================================================

def load_attendance():

    if not ATTENDANCE_FILE.exists():
        return pd.DataFrame(
            columns=[
                "student_id",
                "date",
                "first_seen",
                "last_seen",
                "status"
            ]
        )

    try:
        attendance = pd.read_csv(
            ATTENDANCE_FILE
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=[
                "student_id",
                "date",
                "first_seen",
                "last_seen",
                "status"
            ]
        )

    return attendance


def save_attendance(student_id, date, time):
    """
    Record a student's attendance for today.
    Returns True if this is the FIRST time the student was seen
    today (so the caller knows whether to send a parent email),
    or False if we're just updating their "last seen" time.
    """

    attendance = load_attendance()

    already_marked = (
        (attendance["student_id"].astype(str) == str(student_id))
        & (attendance["date"] == date)
    ).any()

    if already_marked:

        mask = (
            (attendance["student_id"].astype(str) == str(student_id))
            & (attendance["date"] == date)
        )

        attendance.loc[mask, "last_seen"] = time

        attendance.to_csv(
            ATTENDANCE_FILE,
            index=False
        )

        return False

    else:

        new_record = {
            "student_id": student_id,
            "date": date,
            "first_seen": time,
            "last_seen": time,
            "status": "Present"
        }

        attendance.loc[len(attendance)] = new_record

        attendance.to_csv(
            ATTENDANCE_FILE,
            index=False
        )

        return True


# =========================================================
# ACTIVITY
# =========================================================

def load_activity():

    if not ACTIVITY_FILE.exists():
        return pd.DataFrame(
            columns=[
                "student_id",
                "date",
                "time",
                "activity"
            ]
        )

    try:
        activity = pd.read_csv(
            ACTIVITY_FILE
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=[
                "student_id",
                "date",
                "time",
                "activity"
            ]
        )

    return activity


def save_activity(
    student_id,
    date,
    time,
    activity
):

    activities = load_activity()

    new_activity = {
        "student_id": student_id,
        "date": date,
        "time": time,
        "activity": activity
    }

    activities.loc[len(activities)] = new_activity

    activities.to_csv(
        ACTIVITY_FILE,
        index=False
    )
