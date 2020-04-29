'''
Utils class with useful helper functions

utils: https://www.quora.com/What-do-utils-files-tend-to-be-in-computer-programming-documentation
'''

import numpy
import math
import random

import time
import logging
import argparse

import os
import shutil
import sys

import time

import smtplib
from socket import gethostname
from email.message import EmailMessage

import pathlib
from pathlib import Path

from pdb import set_trace as st

def parse_args():
    """
        Parses command line arguments
    """
    parser = argparse.ArgumentParser(description="DiMO")
    parser.add_argument(
        "--nodes",
        metavar="B",
        type=int,
        help="number of nodes per cell",
        default=5
    )

    return parser.parse_args()


def get_logger(log_path, log_filename):
    """
        Initializes and returns a standard logger
    """
    logger = logging.getLogger(log_filename)
    file_handler = logging.FileHandler(log_filename + ".log")
    logger.addHandler(file_handler)

    return logger

def HelloWorld():
    return 'HelloWorld in Utils!'

def remove_folders_recursively(path):
    print('WARNING: HAS NOT BEEN TESTED')
    path.expanduser()
    try:
        shutil.rmtree(str(path))
    except OSError:
        # do nothing if removing fails
        pass

def oslist_for_path(path):
    return [f for f in path.iterdir() if f.is_dir()]

def make_and_check_dir(path):
    '''
    tries to make dir/file, if it exists already does nothing else creates it.

    https://docs.python.org/3/library/pathlib.html

    :param path object path: path where the data will be saved
    '''
    path = os.path.expanduser(path)
    print(path)
    st()
    try:
        os.makedirs(path)
    except OSError:
        print(OSError)
        return OSError
        pass

def timeSince(start):
    '''
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    '''
    now = time.time()
    s = now - start
    ## compute how long it took in hours
    h = s/3600
    ## compute numbers only for displaying how long it took
    m = math.floor(s / 60) # compute amount of whole integer minutes it took
    s -= m * 60 # compute how much time remaining time was spent in seconds
    ##
    msg = f'time passed: hours:{h}, minutes={m}, seconds={s}'
    return msg, h

def report_times(start, verbose=False):
    '''
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    '''
    meta_str=''
    ## REPORT TIMES
    start_time = start
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    if verbose:
        print(f"--- {seconds} {'seconds '+meta_str} ---")
        print(f"--- {minutes} {'minutes '+meta_str} ---")
        print(f"--- {hours} {'hours '+meta_str} ---")
        print('\a')
    ##
    msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
    return msg, seconds, minutes, hours

def is_NaN(value):
    '''
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    '''
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)

##

def make_and_check_dir2(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        pass

####

'''
Greater than 4 I get this error:

ValueError: Seed must be between 0 and 2**32 - 1
'''

RAND_SIZE = 4

def get_random_seed():
    '''

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    '''
    random_data = os.urandom(RAND_SIZE) # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def seed_everything(seed=42):
    '''
    https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch
    '''
    import torch
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

####

def send_email_old(subject, message, destination):
    """ Send an e-mail from with message to destination email.

    NOTE: if you get an error with google gmails you might need to do this: 
    https://stackoverflow.com/questions/16512592/login-credentials-not-working-with-gmail-smtp
    
    Arguments:
        message {str} -- message string to send.
        destination {str} -- destination email (as string)
    """
    #server = smtplib.SMTP('smtp.gmail.com', 587)
    #server.starttls()
    # not a real email account nor password, its all ok!
    #server.login('slurm.miranda@gmail.com', 'dummy123!@#$321')

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

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
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
    ## credits: http://linuxcursor.com/python-programming/06-how-to-send-pdf-ppt-attachment-with-html-body-in-python-script
    from socket import gethostname
    #import email
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib
    import json

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
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
        with open(path_to_pdf, "rb") as f:
            #attach = email.mime.application.MIMEApplication(f.read(),_subtype="pdf")
            attach = MIMEApplication(f.read(),_subtype="pdf")
        attach.add_header('Content-Disposition','attachment',filename=str(path_to_pdf))
        msg.attach(attach)
        # send msg
        server.send_message(msg)

def make_dirpath_current_datetime_hostname(path=None, comment=''):
    """Creates directory name for tensorboard experiments.

    Keyword Arguments:
        path {str} -- [path to the runs directory] (default: {None})
        comment {str} -- [comment to add at the end of the file of the experiment] (default: {''})

    Returns:
        [PosixPath] -- [nice object interface for manipulating paths easily]
    """
    # makedirpath with current date time and hostname
    import socket
    import os
    from datetime import datetime
    # check if path is a PosixPath object
    if type(path) != pathlib.PosixPath and path is not None:
        path = Path(path)
    # get current time
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # make runs logdir path
    runs_log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
    # append path to front of runs_log_dir
    log_dir = Path(runs_log_dir)
    if path is not None:
        log_dir = path / runs_log_dir
    return log_dir

def host_local_machine(local_hosts=None):
    """ Returns True if its a recognized local host
    
    Keyword Arguments:
        local_hosts {list str} -- List of namaes of local hosts (default: {None})
    
    Returns:
        [bool] -- True if its a recognized local host False otherwise.
    """
    if local_hosts is None:
        local_hosts = ['Sarahs-iMac.local','Brandos-MacBook-Pro.local']
    hostname = gethostname()
    if hostname in local_hosts:
        return True
    else: # not known local host
        return False

def my_print(*args, filepath='~/my_stdout.txt'):
    """Modified print statement that prints to terminal/scree AND to a given file (or default).

    Note: import it as follows:

    from utils.utils import my_print as print

    to overwrite builtin print function
    
    Keyword Arguments:
        filepath {str} -- where to save contents of printing (default: {'~/my_stdout.txt'})
    """
    filepath = Path(filepath).expanduser()
    # do normal print
    __builtins__['print'](*args, file=sys.__stdout__) #prints to terminal
    # open my stdout file in update mode
    with open(filepath, "a+") as f:
        # save the content we are trying to print
        __builtins__['print'](*args, file=f) #saves to file

def collect_content_from_file(filepath):
    filepath = Path(filepath).expanduser()
    contents = ''
    with open(filepath,'r') as f:
        for line in f.readlines():
            contents = contents + line
    return contents

if __name__ == '__main__':
    #send_email('msg','miranda9@illinois.edu')
    print('sending email test')
    p = Path('~/automl-meta-learning/automl/experiments/pw_app.config.json').expanduser()
    send_email(subject='TEST: send_email2', message='MESSAGE', destination='brando.science@gmail.com', password_path=p)
    print(f'EMAIL SENT\a')
    print('Done \n\a')