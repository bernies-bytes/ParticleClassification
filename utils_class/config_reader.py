"""
reading in the config file 
"""
import configparser
import os
from sys import argv

# set up the parser
parser = configparser.ConfigParser(allow_no_value=True)
parser.read(argv[1])  # sys.argv[1])

# getting absolute path in python:
abs_path = os.path.abspath(".")
run_name = parser.get("general", "run_name")
logging_level = parser.get("general", "logging_level")
DEBUG_MODE = parser.getboolean("general", "debug_mode")
