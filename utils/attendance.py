from pathlib import Path
from datetime import datetime
import pandas as pd


ATTENDANCE_FILE = Path("data/attendance.csv")


def mark_attendance(student_id):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%I:%M %p")

    if ATTENDANCE_FILE.exists():
        attendance = pd.read_csv(ATTENDANCE_FILE)
    else:
        attendance = pd.DataFrame(
            columns=[
                "student_id",
                "date",
                "first_seen",
                "last_seen",
                "status"
            ]
        )

    student_id = str(student_id)

    today_record = (
        (attendance["student_id"].astype(str) == student_id)
        & (attendance["date"].astype(str) == today)
    )

    if today_record.any():

        index = attendance[today_record].index[0]

        attendance.loc[index, "last_seen"] = current_time
        attendance.loc[index, "status"] = "Present"

    else:

        new_record = pd.DataFrame([{
            "student_id": student_id,
            "date": today,
            "first_seen": current_time,
            "last_seen": current_time,
            "status": "Present"
        }])

        attendance = pd.concat(
            [attendance, new_record],
            ignore_index=True
        )

    attendance.to_csv(
        ATTENDANCE_FILE,
        index=False
    )


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

    return pd.read_csv(
        ATTENDANCE_FILE
    )