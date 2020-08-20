import logging
import os
from datetime import datetime
import sys

log_path = "./logs"
file_name = "training_logs"
log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
curr_time = datetime.now()
year, month, day, hour, min = curr_time.year, curr_time.month, curr_time.day, curr_time.hour, curr_time.minute

def setup_custom_logger(name):
    logFormatter = logging.Formatter(log_format)
    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.INFO)

    log_filename = "{}/{}_{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.log".format(log_path, file_name, year, month, day, hour, min)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.INFO)
    root_logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    root_logger.addHandler(consoleHandler)
    return root_logger

