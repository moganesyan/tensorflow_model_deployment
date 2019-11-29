import logging
import time
from logging.handlers import TimedRotatingFileHandler
import os
import sys

ROOT_DIR = os.path.abspath(".")

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")

LOG_DIR = os.path.join(ROOT_DIR, 'logs')

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
    print("Directory " , LOG_DIR ,  " Created ")
else:    
    print("Directory " , LOG_DIR ,  " already exists")


timestr = time.strftime("%Y%m%d-%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f'logs_api_test_{timestr}.log')

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.DEBUG)
    return file_handler


def get_logger(*, logger_name):
    """Get logger with prepared handlers."""

    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False

    return logger
