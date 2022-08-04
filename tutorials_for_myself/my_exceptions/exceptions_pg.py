#%%
# https://stackoverflow.com/questions/4564559/get-exception-description-and-stack-trace-which-caused-an-exception-all-as-a-st
import traceback

print(traceback.format_exc())

#%%
# catch all exceptions https://stackoverflow.com/questions/4990718/how-can-i-write-a-try-except-block-that-catches-all-exceptions
"""
import traceback
import logging

try:
    whatever()
except Exception as e:
    logging.error(traceback.format_exc())
    # Logs the error appropriately.
"""
# avoid just except
"""
The advantage of except Exception over the bare except is that there are a few exceptions that it wont catch,
most obviously KeyboardInterrupt and SystemExit: if you caught and swallowed those then you could make it hard for
anyone to exit your script.
"""