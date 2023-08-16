"""
reading in the config file 
"""
import configparser
import logging
import os

from utils.file_utils import make_unique_directory


# import os
# from sys import argv

# set up the parser
parser = configparser.ConfigParser(allow_no_value=True)
parser.read("classification.conf")  # sys.argv[1])

# read general
run_name = parser.get("general", "run_name")
# create a directory to save run data:
abs_path = os.path.abspath(".")
run_path = abs_path + "/" + run_name

if not os.path.exists(run_path):
    pass
else:
    # ask the user if they want to overwrite or create a new one TODO
    run_path, run_name = make_unique_directory(abs_path, run_name)
    # print(f"Created directory: {run_path}")


# Set up the logger only when config_reader.py is imported for the first time
if not hasattr(logging, "logger_is_configured"):
    #print("conf logger ")
    # Create the logger, add handlers, and set the flag to avoid duplicate setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(run_name + ".log")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.logger_is_configured = True  # Flag to indicate logger setup


# read synthetic data
CREATE_PLOTS = parser.getboolean("synthetic data", "create_plots")
data_set = parser.get("synthetic data", "data_set")
number_of_events = parser.getint("synthetic data", "number_of_events")
imbalance = parser.getfloat("synthetic data", "imbalance")

# set up sampling
SAMPLING = parser.getboolean("sampling", "sampling")
sampler = parser.get("sampling", "sampler")
