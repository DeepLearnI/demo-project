import logging
import os
from datetime import datetime as dt

log_path = 'logs' #/home/rm/lottery_ticket/logs/

try:
    os.mkdir(log_path)
except:
    pass

def get_logger(name):
    logger_path = log_path

    # add logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging
    if not logger.handlers:
        # create a file handler
        current_time = dt.now().strftime('%m-%d')
        file_handler = logging.FileHandler(os.path.join(logger_path, '{}.log'.format(current_time)))
        file_handler.setLevel(logging.INFO)
        # create a logging format
        formats = '[%(asctime)s - %(name)s-%(lineno)d - %(funcName)s - %(levelname)s] %(message)s'
        file_formatter = logging.Formatter(formats, '%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        # add the handlers to the logger
        logger.addHandler(file_handler)

        # console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_formatter = logging.Formatter(formats, '%m-%d %H:%M:%S')
        c_handler.setFormatter(c_formatter)
        logger.addHandler(c_handler)
    return logger

