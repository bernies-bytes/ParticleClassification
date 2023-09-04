"""
reading in the config file 
"""
import configparser
import logging
import os
import json

# import ast
import re

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
    # print("conf logger ")
    # Create the logger, add handlers, and set the flag to avoid duplicate setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
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

# set up test/train split
test_size = parser.getfloat("data split", "test_size")

# set up sampling
enable_sampling = parser.getboolean("sampling", "sampling")
sampler_value = parser.get("sampling", "sampler")

if sampler_value.startswith("[") and sampler_value.endswith("]"):
    sampler = re.findall(r"\b\w+\b", sampler_value)
else:
    sampler = sampler_value

undersamp_ratio = parser.getfloat("sampling", "undersamp_ratio")

# set up hyper-parameter tuning
enable_hypertune = parser.getboolean("hyper-parameter tuning", "hyper_tune")
hyper_para_tune_method = parser.get("hyper-parameter tuning", "method")

bayes_opt_init_pts = parser.getint("hyper-parameter tuning", "init_points")
bayes_opt_n_iter = parser.getint("hyper-parameter tuning", "n_iter")
bayes_opt_acq = parser.get("hyper-parameter tuning", "acq")
bayes_opt_xi = parser.getfloat("hyper-parameter tuning", "xi")

json_string = parser.get("hyper-parameter tuning", "para_to_tune")
# Parse the JSON string into a Python dictionary
para_to_tune = json.loads(json_string)

# Now, para_to_tune is a dictionary in your Python code
# print(type(para_to_tune))

# exit(0)

