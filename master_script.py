#
# master script that controls what is called and what not
#
import os
import configparser
import logging
from utils_class.file_utils import create_unique_directory

###############################################################################################
# set up the parser
parser = configparser.ConfigParser(allow_no_value=True)
parser.read('classification.conf') # sys.argv[1])
###############################################################################################
# General
###############################################################################################
#getting absolute path in python:
abs_path = os.path.abspath(".")
run_name   = parser.get("general", "run_name")
logging_level = parser.get("general", "logging_level")
######################################################

# create a directory to save run data:
run_path = abs_path + "/" + run_name
if not os.path.exists(run_path):
    os.mkdir(run_path)
else:
    unique_path = create_unique_directory(abs_path, run_name)
    print(f"Created unique directory: {unique_path}")

######################################################
logger = logging.getLogger(run_name)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler(run_name + '.log')
handler.setFormatter(formatter)
logger.addHandler(handler)
######################################################
