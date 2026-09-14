from pathlib import Path
from datetime import datetime
import pandas as pd


ACTIVITY_FILE = Path("data/activity_log.csv")


def save_activity(student_id, activity):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%I:%M %p")

    if ACTIVITY_FILE.exists():
        activities = pd.read_csv(ACTIVITY_FILE)
    else:
        activities = pd.DataFrame(
            columns=[
                "student_id",
                "date",
                "time",
                "activity"
            ]
        )

    new_activity = pd.DataFrame([{
        "student_id": str(student_id),
        "date": today,
        "time": current_time,
        "activity": activity
    }])

    activities = pd.concat(
        [activities, new_activity],
        ignore_index=True
    )

    activities.to_csv(
        ACTIVITY_FILE,
        index=False
    )


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

    return pd.read_csv(
        ACTIVITY_FILE
    )