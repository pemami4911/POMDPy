"""
CONSOLE LOGGING VERBOSITY LEVELS
---------------------------------
0 - FATAL
1 - CRITICAL
2 - INFO
3 - LOUD
4 - DEBUG
"""
from __future__ import print_function

VERBOSITY = 3


def print_divider(size):
    if size == "large":
        print("======================================================================")
    elif size == "medium":
        print("==========================================")
    else:
        print("========")


def console(verbosity_level, module, msg):
    if verbosity_level > VERBOSITY:
        return
    else:
        print(module + ' - ' + msg)


# Pass a function that handles printing
def console_no_print(verbosity_level, func):
    if verbosity_level > VERBOSITY:
        return
    else:
        assert callable(func)
        func()
