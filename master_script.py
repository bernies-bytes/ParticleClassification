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
    sampler,
    undersamp_ratio,
)  # , DEBUG_MODE, logging_level
from synth_data.create_data import create_synth_data_with_make_classif
from sampling import oversample_data, undersample_data, combined_underover_sample_data

###############################################################################################
SEED = random.seed(time.time())
######################################################
logger.info("create run directory: %s", run_path)
os.mkdir(run_path)
os.mkdir(run_path + "/debug_plots")
######################################################
data_df = None
if data_set == "synthetic":
    data_df = create_synth_data_with_make_classif(number_of_events, imbalance)

# TODO possibility to read in a data set

if not isinstance(data_df, pd.DataFrame):
    raise ValueError("No data-set has been set.")
######################################################

if SAMPLING:
    if isinstance(sampler, list):
        features_samp, target_samp = combined_underover_sample_data(
        data_df, sampler, undersamp_ratio, SEED)
    elif sampler in ["RandomOverSampler", "SMOTE", "SVMSMOTE", "BoderlineSMOTE"]:
        features_samp, target_samp = oversample_data(data_df, sampler, "minority", SEED)
    elif sampler in ["TomekLinks", "RandomUnderSampler", "ClusterCentroids"]:
        features_samp, target_samp = undersample_data(
            data_df, sampler, "majority", SEED
        )
    else:
        raise ValueError("Invalid sampler selected.")

    data_orig_df = data_df
    data_df = pd.DataFrame(
        np.c_[target_samp, features_samp], columns=data_df.columns
    )
    # print(data_samp_df)
