import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st


def send_attendance_email(parent_email, student_name, time_str):
    """
    Sends a "your child has arrived" email to the parent.
    Requires SMTP credentials set in Streamlit secrets:

        [email]
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "your_school_account@gmail.com"
        sender_password = "your_16_char_app_password"

    Returns True on success, False otherwise (never raises, so a
    failed email never crashes the camera page).
    """

    try:
        smtp_server = st.secrets["email"]["smtp_server"]
        smtp_port = int(st.secrets["email"]["smtp_port"])
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]
    except Exception:
        print("Email secrets are not configured — skipping email.")
        return False

    if not parent_email:
        return False

    subject = f"{student_name} has arrived at school"

    body = (
        f"Dear Parent,\n\n"
        f"This is to let you know that {student_name} was detected "
        f"in the classroom today at {time_str}.\n\n"
        f"— BASMA AI Classroom Analytics"
    )

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = parent_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(
                sender_email,
                parent_email,
                message.as_string()
            )
        return True

    except Exception as e:
        print(f"Failed to send attendance email: {e}")
        return False
