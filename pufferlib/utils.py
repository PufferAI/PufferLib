from pdb import set_trace as T

import time


def current_datetime():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

