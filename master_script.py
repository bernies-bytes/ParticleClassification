"""
master script that controls what is called and what not
"""

# import logging
import os

from utils.config_setup import (
    # run_name,
    run_path,
    data_set,
    logger,
)  # , DEBUG_MODE, logging_level
from synth_data.create_data import create_synth_data_with_make_classif


###############################################################################################

######################################################
logger.info("create run directory: %s", run_path)
os.mkdir(run_path)
os.mkdir(run_path + "/debug_plots")
######################################################
if data_set == "synthetic":
    create_synth_data_with_make_classif(100000, 0.70)
######################################################
