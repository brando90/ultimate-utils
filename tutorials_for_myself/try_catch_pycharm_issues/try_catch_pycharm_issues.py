"""
trying to resolve:
- https://intellij-support.jetbrains.com/hc/en-us/requests/3764538

"""

def invoke_handled_exception():
    try:
        1 / 0
    except ZeroDivisionError:
        print('exception caught')

def invoke_unhandled_exception():
    1 / 0

invoke_handled_exception()
invoke_unhandled_exception()