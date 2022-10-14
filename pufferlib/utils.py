from pdb import set_trace as T

import time


def current_datetime():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

def myprint(d):
    stack = d.items()
    while stack:
        k, v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.iteritems())
        else:
            print("%s: %s" % (k, v))
