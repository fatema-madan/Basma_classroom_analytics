import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st


# =========================================================
# EMAIL SETTINGS
# =========================================================

def get_email_settings():

    try:

        email_settings = st.secrets["email"]

        smtp_server = str(
            email_settings["smtp_server"]
        ).strip()

        smtp_port = int(
            email_settings["smtp_port"]
        )

        sender_email = str(
            email_settings["sender_email"]
        ).strip()

        sender_password = str(
            email_settings["sender_password"]
        ).strip()

        return (
            smtp_server,
            smtp_port,
            sender_email,
            sender_password
        )

    except Exception as e:

        raise RuntimeError(
            f"Email settings are missing or incorrect: {e}"
        )


# =========================================================
# SEND EMAIL
# =========================================================

def send_email(
    recipient,
    subject,
    body
):

    if not recipient:

        return False

    recipient = str(
        recipient
    ).strip()

    if not recipient:

        return False

    try:

        (
            smtp_server,
            smtp_port,
            sender_email,
            sender_password
        ) = get_email_settings()

    except Exception as e:

        st.error(
            f"❌ Email configuration error: {e}"
        )

        return False


    message = MIMEMultipart()

    message["From"] = sender_email

    message["To"] = recipient

    message["Subject"] = subject

    message.attach(
        MIMEText(
            body,
            "plain",
            "utf-8"
        )
    )


    try:

        with smtplib.SMTP(
            smtp_server,
            smtp_port,
            timeout=20
        ) as server:

            server.ehlo()

            server.starttls()

            server.ehlo()

            server.login(
                sender_email,
                sender_password
            )

            server.sendmail(
                sender_email,
                [recipient],
                message.as_string()
            )

        return True


    except smtplib.SMTPAuthenticationError:

        st.error(
            "❌ Email login failed. "
            "Check your Gmail address and App Password."
        )

        return False


    except smtplib.SMTPRecipientsRefused:

        st.error(
            f"❌ The recipient email was rejected: "
            f"{recipient}"
        )

        return False


    except smtplib.SMTPConnectError:

        st.error(
            "❌ Could not connect to the email server."
        )

        return False


    except Exception as e:

        st.error(
            f"❌ Email sending failed: {e}"
        )

        return False


# =========================================================
# ATTENDANCE EMAIL TO PARENT
# =========================================================

def send_attendance_email(
    parent_email,
    student_name,
    time_str
):

    subject = (
        f"BASMA Attendance Alert - {student_name}"
    )

    body = (
        f"Dear Parent,\n\n"
        f"This is to inform you that "
        f"{student_name} has been detected "
        f"in the classroom today.\n\n"
        f"Attendance Time: {time_str}\n"
        f"Status: Present\n\n"
        f"Best regards,\n"
        f"BASMA AI Classroom Analytics"
    )

    return send_email(
        recipient=parent_email,
        subject=subject,
        body=body
    )


# =========================================================
# ATTENDANCE REPORT
# =========================================================

def send_attendance_report(
    recipient,
    attendance,
    students,
    report_date
):

    if attendance is None:

        return False

    # -------------------------------------
    # Build report
    # -------------------------------------

    if attendance.empty:

        report_text = (
            "No attendance records were found "
            f"for {report_date}."
        )

    else:

        lines = []

        lines.append(
            f"BASMA Attendance Report"
        )

        lines.append(
            f"Date: {report_date}"
        )

        lines.append(
            ""
        )

        for _, row in attendance.iterrows():

            student_id = row.get(
                "student_id",
                ""
            )

            student_name = row.get(
                "student_name",
                ""
            )

            status = row.get(
                "status",
                "Present"
            )

            time_value = row.get(
                "time",
                ""
            )

            if not student_name and students is not None:

                if not students.empty:

                    matching_students = students[
                        students[
                            "student_id"
                        ].astype(str)
                        == str(student_id)
                    ]

                    if not matching_students.empty:

                        student_name = (
                            matching_students.iloc[0][
                                "student_name"
                            ]
                        )

            lines.append(
                f"Student: {student_name}"
            )

            lines.append(
                f"Status: {status}"
            )

            if time_value:

                lines.append(
                    f"Time: {time_value}"
                )

            lines.append("")

        report_text = "\n".join(
            lines
        )


    # -------------------------------------
    # Send
    # -------------------------------------

    subject = (
        f"BASMA Attendance Report - "
        f"{report_date}"
    )

    return send_email(
        recipient=recipient,
        subject=subject,
        body=report_text
    )
