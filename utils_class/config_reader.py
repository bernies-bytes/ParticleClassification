"""
reading in the config file 
"""
import configparser
import logging

# import os
# from sys import argv

# set up the parser
parser = configparser.ConfigParser(allow_no_value=True)
parser.read("classification.conf")  # sys.argv[1])

# read general
run_name = parser.get("general", "run_name")
logging_level = parser.get("general", "logging_level")

# set up the logger
logger = logging.getLogger(run_name)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.FileHandler(run_name + ".log")
handler.setFormatter(formatter)
logger.addHandler(handler)

# read synthetic data
CREATE_PLOTS = parser.getboolean("synthetic data", "create_plots")
data_set = parser.get("synthetic data", "data_set")
