#!/usr/bin/env python3
"""Send room reservation email to Eric Pineda for Lean AI Club at Gates."""
from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SIGNATURE = """\
-----
Brando Miranda
Ph.D. Student
Computer Science, Stanford University
EDGE Scholar, Stanford University
brando9@stanford.edu
website: https://brando90.github.io/brandomiranda/
"""

TO = "ericpineda@cs.stanford.edu"
CC = ["brando9@stanford.edu", "brando9@cs.stanford.edu", "brando.science@gmail.com"]
FROM = "brandojazz@gmail.com"
SUBJECT = "Room Reservation Request – Lean AI Club – Monday, April 20"

BODY = f"""\
Hi Eric,

I hope this email finds you well. I'm reaching out to request a room reservation in the Gates Building for our Lean AI Club meeting.

Details:
  - Date: Monday, April 20, 2026
  - Time: 6:30 PM – 9:30 PM
  - Preferred Room: Gates 358
  - Alternative: Any available room in the Gates Building would work as well

Please let me know if Gates 358 or another room is available for that time. Thank you so much for your help!

Best,

{SIGNATURE}"""


def send() -> None:
    # Read Gmail app password
    pass_file = Path("~/keys/gmail_app_password.txt").expanduser()
    if pass_file.is_file():
        password = pass_file.read_text().strip()
    else:
        import os
        password = os.environ.get("SMTP_PASS", "")
    if not password:
        raise ValueError(
            "No SMTP password found. Either:\n"
            "  1. Save your Gmail app password to ~/keys/gmail_app_password.txt\n"
            "  2. Set the SMTP_PASS environment variable\n"
            "See: https://myaccount.google.com/apppasswords"
        )

    msg = MIMEMultipart()
    msg["Subject"] = SUBJECT
    msg["From"] = FROM
    msg["To"] = TO
    msg["Cc"] = ", ".join(CC)
    msg.attach(MIMEText(BODY, "plain"))

    all_recipients = [TO] + CC
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls()
        server.login(FROM, password)
        server.sendmail(FROM, all_recipients, msg.as_string())

    log.info("Email sent to %s (CC: %s): %s", TO, ", ".join(CC), SUBJECT)
    print(f"Email sent successfully to {TO}")
    print(f"CC: {', '.join(CC)}")


if __name__ == "__main__":
    send()
