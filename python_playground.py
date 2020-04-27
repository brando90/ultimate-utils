#!/usr/bin/env python
#SBATfCH --mail-type=ALL
#SBATCH --mail-user=miranda9@illinois.edu
#SBATCH --array=1-1
#SBATCH --partition=gpux1

import sys

print(sys.version)
print(sys.path)

def helloworld():
    print('helloworld')
    print('hello12345')

def union_dicts():
    d1 = {'x':1}
    d2 = {'y':2, 'z':3}
    d_union = {**d1, **d2}
    print(d_union)

def get_stdout_old():
    import sys

    # contents = ""
    # #with open('some_file.txt') as f:
    # #with open(sys.stdout,'r') as f:
    # # sys.stdout.mode = 'r'
    # for line in sys.stdout.readlines():
    #     contents += line
    # print(contents)

    # print(sys.stdout)
    # with open(sys.stdout.buffer) as f:
    #     print(f.readline())

    # import subprocess

    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # stdout = []
    # while True:
    #     line = p.stdout.readline()
    #     stdout.append(line)
    #     print( line )
    #     if line == '' and p.poll() != None:
    #         break
    # print( ''.join(stdout) )
    
    import sys
    myfile = "input.txt"
    def print(*args):
        __builtins__.print(*args, file=sys.__stdout__)
        with open(myfile, "a+") as f:
            __builtins__.print(*args, file=f)
    
    print('a')
    print('b')
    print('c')

    repr(sys.stdout)


def get_stdout():
    import sys
    myfile = "my_stdout.txt"
    # redefine print
    def print(*args):
        __builtins__.print(*args, file=sys.__stdout__)    #prints to terminal
        with open(myfile, "a+") as f:
            __builtins__.print(*args, file=f)    #saves in a file

    print('a')
    print('b')
    print('c')

def logging_basic():
    import logging
    logging.warning('Watch out!')  # will print a message to the console
    logging.info('I told you so')  # will not print anything

def logging_to_file():
    import logging
    logging.basicConfig(filename='example.log',level=logging.DEBUG)
    #logging.
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('And this, too')

def logging_to_file_INFO_LEVEL():
    import logging
    import sys
    format = '{asctime}:{levelname}:{name}:lineno {lineno}:{message}'
    logging.basicConfig(filename='example.log',level=logging.INFO,format=format,style='{')
    #logging.basicConfig(stream=sys.stdout,level=logging.INFO,format=format,style='{')
    #logging.
    logging.debug('This message should NOT go to the log file')
    logging.info('This message should go to log file')
    logging.warning('This, too')

def logger_SO_print_and_write_to_my_stdout():
    """My sample logger code to print to screen and write to file (the same thing).

    Note: trying to replace this old answer of mine using a logger: 
    - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced

    Credit: 
    - https://www.youtube.com/watch?v=jxmzY9soFXg&t=468s
    - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced
    - https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716

    Other resources:
    - https://docs.python-guide.org/writing/logging/
    - https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
    - https://stackoverflow.com/questions/61084916/how-does-one-make-an-already-opened-file-readable-e-g-sys-stdout/61255375#61255375
    """
    from pathlib import Path
    import logging
    import os
    import sys
    from datetime import datetime

    ## create directory (& its parents) if it does not exist otherwise do nothing :)
    # get current time
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') 
    logs_dirpath = Path(f'~/logs/python_playground_logs_{current_time}/').expanduser()
    logs_dirpath.mkdir(parents=True, exist_ok=True)
    my_stdout_filename = logs_dirpath / Path('my_stdout.log')
    # remove my_stdout if it exists (note you can also just create a new log dir/file each time or append to the end of the log file your using)
    #os.remove(my_stdout_filename) if os.path.isfile(my_stdout_filename) else None

    ## create top logger
    logger = logging.getLogger(__name__) # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
    logger.setLevel(logging.DEBUG) # note: use logging.DEBUG, CAREFUL with logging.UNSET: https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716

    ## log to my_stdout.log file
    file_handler = logging.FileHandler(filename=my_stdout_filename)
    #file_handler.setLevel(logging.INFO) # not setting it means it inherits the logger. It will log everything from DEBUG upwards in severity to this handler.
    log_format = "{asctime}:{levelname}:{lineno}:{name}:{message}" # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(fmt=log_format, style='{') # set the logging format at for this handler
    file_handler.setFormatter(fmt=formatter)

    ## log to stdout/screen
    stdout_stream_handler = logging.StreamHandler(stream=sys.stdout) # default stderr, though not sure the advatages of logging to one or the other
    #stdout_stream_handler.setLevel(logging.INFO) # Note: having different set levels means that we can route using a threshold what gets logged to this handler
    log_format = "{name}:{levelname}:-> {message}" # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(fmt=log_format, style='{') # set the logging format at for this handler
    stdout_stream_handler.setFormatter(fmt=formatter)

    logger.addHandler(hdlr=file_handler) # add this file handler to top logger
    logger.addHandler(hdlr=stdout_stream_handler) # add this file handler to top logger

    logger.log(logging.NOTSET, 'notset')
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')

def logging_unset_level():
    """My sample logger explaining UNSET level

    Resources: 
    - https://stackoverflow.com/questions/21494468/about-notset-in-python-logging
    - https://www.youtube.com/watch?v=jxmzY9soFXg&t=468s
    - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced
    """
    import logging

    logger = logging.getLogger(__name__) # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
    print(f'DEFAULT VALUE: logger.level = {logger.level}')

    file_handler = logging.FileHandler(filename='my_log.log')
    log_format = "{asctime}:{levelname}:{lineno}:{name}:{message}" # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(fmt=log_format, style='{') 
    file_handler.setFormatter(fmt=formatter) 

    stdout_stream_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_stream_handler.setLevel(logging.INFO) 
    log_format = "{name}:{levelname}:-> {message}" # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(fmt=log_format, style='{')
    stdout_stream_handler.setFormatter(fmt=formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=stdout_stream_handler)

    logger.log(logging.NOTSET, 'notset')
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')

def logger():
    from pathlib import Path
    import logging

    # create directory (& its parents) if it does not exist otherwise do nothing :)
    logs_dirpath = Path('~/automl-meta-learning/logs/python_playground_logs/').expanduser()
    logs_dirpath.mkdir(parents=True, exist_ok=True)
    my_stdout_filename = logs_dirpath / Path('my_stdout.log')
    # remove my_stdout if it exists (used to have this but now I decided to create a new log & file each)
    #os.remove(my_stdout_filename) if os.path.isfile(my_stdout_filename) else None

    logger = logging.getLogger(__name__) # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
    logger.setLevel(logging.INFO)

    log_format = "{asctime}:{levelname}:{name}:{message}"
    formatter = logging.Formatter(fmt=log_format, style='{')

    file_handler = logging.FileHandler(filename=my_stdout_filename)
    file_handler.setFormatter(fmt=formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=logging.StreamHandler())

    for i in range(3):
        logger.info(f'i = {i}')
    
    logger.info(f'logger DONE')

def logging_example_from_youtube():
    """https://github.com/CoreyMSchafer/code_snippets/blob/master/Logging-Advanced/employee.py
    """
    import logging
    import pytorch_playground # has employee class & code
    import sys

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler('sample.log')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.critical('not really critical :P')

    def add(x, y):
        """Add Function"""
        return x + y

    def subtract(x, y):
        """Subtract Function"""
        return x - y

    def multiply(x, y):
        """Multiply Function"""
        return x * y

    def divide(x, y):
        """Divide Function"""
        try:
            result = x / y
        except ZeroDivisionError:
            logger.exception('Tried to divide by zero')
        else:
            return result


    logger.info('testing if log info is going to print to screen. it should because everything with debug or above is printed since that stream has that level.')

    num_1 = 10
    num_2 = 0

    add_result = add(num_1, num_2)
    logger.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))

    sub_result = subtract(num_1, num_2)
    logger.debug('Sub: {} - {} = {}'.format(num_1, num_2, sub_result))

    mul_result = multiply(num_1, num_2)
    logger.debug('Mul: {} * {} = {}'.format(num_1, num_2, mul_result))

    div_result = divide(num_1, num_2)
    logger.debug('Div: {} / {} = {}'.format(num_1, num_2, div_result))

def plot():
    """
    source:
        - https://www.youtube.com/watch?v=UO98lJQ3QGI
        - https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/Matplotlib/01-Introduction/finished_code.py
    """
    from matplotlib import pyplot as plt

    plt.xkcd()

    ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

    py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640, 84666,
                84392, 78254, 85000, 87038, 91991, 100000, 94796, 97962, 93302, 99240, 102736, 112285, 100771, 104708, 108423, 101407, 112542, 122870, 120000]
    plt.plot(ages_x, py_dev_y, label='Python')

    js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000, 34372, 37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674, 68745, 68746, 74583, 79000,
                78508, 79996, 80403, 83820, 88833, 91660, 87892, 96243, 90000, 99313, 91660, 102264, 100000, 100000, 91660, 99240, 108000, 105000, 104000]
    plt.plot(ages_x, js_dev_y, label='JavaScript')

    dev_y = [17784, 16500, 18012, 20628, 25206, 30252, 34368, 38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752, 77232,
            78000, 78508, 79536, 82488, 88935, 90000, 90056, 95000, 90000, 91633, 91660, 98150, 98964, 100000, 98988, 100000, 108923, 105000, 103117]
    plt.plot(ages_x, dev_y, color='#444444', linestyle='--', label='All Devs')

    plt.xlabel('Ages')
    plt.ylabel('Median Salary (USD)')
    plt.title('Median Salary (USD) by Age')

    plt.legend()

    plt.tight_layout()

    plt.savefig('plot.png')

    plt.show()

def subplot():
    """https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/Matplotlib/10-Subplots/finished_code.py
    """

    import pandas as pd
    from matplotlib import pyplot as plt

    plt.style.use('seaborn')

    data = pd.read_csv('data.csv')
    ages = data['Age']
    dev_salaries = data['All_Devs']
    py_salaries = data['Python']
    js_salaries = data['JavaScript']

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.plot(ages, dev_salaries, color='#444444',
            linestyle='--', label='All Devs')

    ax2.plot(ages, py_salaries, label='Python')
    ax2.plot(ages, js_salaries, label='JavaScript')

    ax1.legend()
    ax1.set_title('Median Salary (USD) by Age')
    ax1.set_ylabel('Median Salary (USD)')

    ax2.legend()
    ax2.set_xlabel('Ages')
    ax2.set_ylabel('Median Salary (USD)')

    plt.tight_layout()

    plt.show()

    fig1.savefig('fig1.png')
    fig2.savefig('fig2.png')

def import_utils_test():
    import utils
    from utils.utils import logger

    print(utils)

    print()

if __name__ == '__main__':
    print()
    #union_dicts()
    #get_stdout()
    #logger()
    #logger_SO_print_and_write_to_my_stdout()
    #logging_basic()
    #logging_to_file()
    #logging_to_file_INFO_LEVEL()
    #logging_example_from_youtube()
    #logging_unset_level()
    import_utils_test()
    print('\n---> DONE\a\n\n')