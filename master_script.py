"""
master script that controls what is called and what not
"""

# import logging
import os
import random
import time
import numpy as np
import pandas as pd

from utils.config_setup import (
    # run_name,
    run_path,
    data_set,
    logger,
    number_of_events,
    imbalance,
    SAMPLING,
    sampler
)  # , DEBUG_MODE, logging_level
from synth_data.create_data import create_synth_data_with_make_classif
from sampling import oversample_data

###############################################################################################
SEED = random.seed(time.time())
######################################################
logger.info("create run directory: %s", run_path)
os.mkdir(run_path)
os.mkdir(run_path + "/debug_plots")
######################################################
if data_set == "synthetic":
    data_df = create_synth_data_with_make_classif(number_of_events, imbalance)
######################################################
if SAMPLING:
    if sampler in ["SMOTE"]:
        features_samp, target_samp = oversample_data(data_df, sampler, SEED)
    elif sampler in ["RandomUnder"]:
        pass


#target_samp[:,:-1] = features_samp
data_samp_df = pd.DataFrame(np.c_[target_samp, features_samp ] , columns=data_df.columns)
#print(data_samp_df)
