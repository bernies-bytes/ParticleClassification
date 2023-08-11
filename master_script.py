"""
master script that controls what is called and what not
"""
import os

# import logging
from utils_class.file_utils import create_unique_directory
from utils_class.config_reader import run_name, data_set  # , DEBUG_MODE, logging_level
from synth_data.create_data import create_with_make_class

###############################################################################################

###############################################################################################
# General
###############################################################################################

######################################################

# create a directory to save run data:
abs_path = os.path.abspath(".")
run_path = abs_path + "/" + run_name
if not os.path.exists(run_path):
    os.mkdir(run_path)
else:
    # ask the user if they want to overwrite or create a new one TODO
    unique_path = create_unique_directory(abs_path, run_name)
    print(f"Created unique directory: {unique_path}")

######################################################
if data_set == "synthetic":
    create_with_make_class(100000, 0.70)
######################################################
