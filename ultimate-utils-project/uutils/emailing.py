# import pathlib
# from pathlib import Path

def send_email_old(subject, message, destination):
    """ Send an e-mail from with message to destination email.

    NOTE: if you get an error with google gmails you might need to do this: 
    https://stackoverflow.com/questions/16512592/login-credentials-not-working-with-gmail-smtp
    
    Arguments:
        message {str} -- message string to send.
        destination {str} -- destination email (as string)
    """
    from socket import gethostname
    from email.message import EmailMessage
    import smtplib

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # not a real email account nor password, its all ok!
    server.login('slurm.miranda@gmail.com', 'dummy123!@#$321')

    # craft message
    msg = EmailMessage()

    message = f'{message}\nSend from Hostname: {gethostname()}'
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = 'slurm.miranda@gmail.com'
    msg['To'] = destination
    # send msg
    server.send_message(msg)


def send_email(subject, message, destination, password_path=None):
    """ Send an e-mail from with message to destination email.

    NOTE: if you get an error with google gmails you might need to do this: 
    https://stackoverflow.com/questions/16512592/login-credentials-not-working-with-gmail-smtp
    To use an app password (RECOMMENDED):
    https://stackoverflow.com/questions/60975490/how-does-one-send-an-e-mail-from-python-not-using-gmail

    Arguments:
        message {str} -- message string to send.
        destination {str} -- destination email (as string)
    """
    from socket import gethostname
    from email.message import EmailMessage
    import smtplib
    import json

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
    except:
        server = smtplib.SMTP('smtp.intel-research.net', 25)
    with open(password_path) as f:
        config = json.load(f)
        server.login('slurm.miranda@gmail.com', config['password'])
        # craft message
        msg = EmailMessage()

        message = f'{message}\nSend from Hostname: {gethostname()}'
        msg.set_content(message)
        msg['Subject'] = subject
        msg['From'] = 'slurm.miranda@gmail.com'
        msg['To'] = destination
        # send msg
        server.send_message(msg)


def send_email_pdf_figs(path_to_pdf, subject, message, destination, password_path=None):
    # credits: http://linuxcursor.com/python-programming/06-how-to-send-pdf-ppt-attachment-with-html-body-in-python-script
    from socket import gethostname
    # import email
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib
    import json

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
    except:
        server = smtplib.SMTP('smtp.intel-research.net', 25)
    with open(password_path) as f:
        config = json.load(f)
        server.login('slurm.miranda@gmail.com', config['password'])
        # Craft message (obj)
        msg = MIMEMultipart()

        message = f'{message}\nSend from Hostname: {gethostname()}'
        msg['Subject'] = subject
        msg['From'] = 'slurm.miranda@gmail.com'
        msg['To'] = destination
        # Insert the text to the msg going by e-mail
        msg.attach(MIMEText(message, "plain"))
        # Attach the pdf to the msg going by e-mail
        if path_to_pdf.exists():
            with open(path_to_pdf, "rb") as f:
                # attach = email.mime.application.MIMEApplication(f.read(),_subtype="pdf")
                attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header('Content-Disposition', 'attachment', filename=str(path_to_pdf))
            msg.attach(attach)
        # send msg
        server.send_message(msg)
