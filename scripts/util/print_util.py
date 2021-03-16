"""
Utility methods for displaying data.
"""
import os
import sys
import shutil
import logging
import matplotlib.pyplot as plt


# private
class Tee(object):
    """
    Class to control where output is printed to.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()   # output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


# public
def get_logger(filename=''):
    """
    Return a logger object.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def remove_logger(logger):
    """
    Remove handlers from logger.
    """
    logger.handlers = []


def clear_dir(in_dir):
    """
    Clear contents of directory.
    """
    if not os.path.exists(in_dir):
        return -1

    # remove contents of the directory
    for fn in os.listdir(in_dir):
        fp = os.path.join(in_dir, fn)

        # directory
        if os.path.isdir(fp):
            shutil.rmtree(fp)

        # file
        else:
            os.remove(fp)

    return 0


def stdout_stderr_to_log(filename):
    """
    Log everything printed to stdout or
    stderr to this specified `filename`.
    """
    logfile = open(filename, 'w')

    stderr = sys.stderr
    stdout = sys.stdout

    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = sys.stdout

    return logfile, stdout, stderr


def reset_stdout_stderr(logfile, stdout, stderr):
    """
    Restore original stdout and stderr
    """
    sys.stdout = stdout
    sys.stderr = stderr
    logfile.close()


def plot_settings(family='serif', fontsize=11,
                  markersize=5, linewidth=None):
    """
    Matplotlib settings.
    """
    plt.rc('font', family=family)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('axes', labelsize=fontsize)
    plt.rc('axes', titlesize=fontsize)
    plt.rc('legend', fontsize=fontsize)
    plt.rc('legend', title_fontsize=fontsize)
    plt.rc('lines', markersize=markersize)
    if linewidth is not None:
        plt.rc('lines', linewidth=linewidth)
