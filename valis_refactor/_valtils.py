"""Minimal vendored valtils functions needed by warp_tools."""

import os
import multiprocessing
import warnings
from colorama import Fore, Style


def print_warning(msg, warning_type=UserWarning, rgb=Fore.YELLOW, traceback_msg=None):
    warning_msg = f"{rgb}{msg}{Style.RESET_ALL}"
    if warning_type is None:
        print(warning_msg)
    else:
        warnings.simplefilter("always", warning_type)
        warnings.warn(warning_msg, warning_type)
    if traceback_msg is not None:
        traceback_msg_rgb = f"{rgb}{traceback_msg}{Style.RESET_ALL}"
        print(traceback_msg_rgb)


def get_name(f):
    """Return basename without extension."""
    fonly = os.path.split(f)[1]
    # split on last dot
    if "." in fonly:
        return fonly.rsplit(".", 1)[0]
    return fonly


def levenshtein_d(str1, str2):
    m = len(str1)
    n = len(str2)
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)
    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(curr_row[j - 1], prev_row[j], prev_row[j - 1])
        prev_row = curr_row.copy()
    return curr_row[n]


def get_ncpus_available():
    if hasattr(os, "sched_getaffinity"):
        return int(len(os.sched_getaffinity(0)))
    if hasattr(multiprocessing, "cpu_count"):
        return int(multiprocessing.cpu_count())
    return 2
