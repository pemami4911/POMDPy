__author__ = 'patrickemami'

'''
CONSOLE LOGGING VERBOSITY LEVELS
---------------------------------
0 - FATAL
1 - CRITICAL
2 - INFO
3 - LOUD
4 - DEBUG
'''
VERBOSITY = 3

def console(verbosity_level, source, msg):
    if verbosity_level > VERBOSITY:
        return
    else:
        print source + ' - ' + msg

# Pass a function that handles printing
def console_no_print(verbosity_level, func):
    if verbosity_level > VERBOSITY:
        return
    else:
        assert callable(func)
        func()

