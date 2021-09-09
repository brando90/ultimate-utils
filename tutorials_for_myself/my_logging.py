
# %%

# https://realpython.com/python-logging/

import logging

logging.debug('This is a debug message')
logging.info('This is an info message')

logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

"""
prints

WARNING:root:This is a warning message
ERROR:root:This is an error message
CRITICAL:root:This is a critical message


Notice that the debug() and info() messages didn’t get logged. This is because, by default, the logging module logs the 
messages with a severity level of WARNING or above. You can change that by configuring the logging module to log events
 of all levels if you want. You can also define your own severity levels by changing configurations, but it is 
 generally not recommended as it can cause confusion with logs of some third-party libraries that you might be using.
"""

#%%

import logging

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.debug('This will get logged')

""" also didn't work...this is why I will stick with printing..."""

#%%

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.warning('Admin logged out')


""" also didn't work...this is why I will stick with printing..."""
