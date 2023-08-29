"""
functions for data sampling - over, under & combinations
"""

import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids

from utils.config_setup import logger


def oversample_data(pd_data, whichone, strategy, SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the given datafrmae pd_data with the chosen sampler whichone
    Parameters:
        pd_data (pandas.DataFrame): The input DataFrame containing features and target.
        whichone (str): The name of the undersampling technique to use.
        SEED (int): The random seed for reproducibility.
        strategy (str): sampling strategy.

    Returns:
        tuple: A tuple containing two elements:
            - features_resampled (numpy.ndarray): Resampled features after undersampling.
            - target_resampled (numpy.ndarray): Resampled target labels after undersampling.
    """
    logger.info("Oversampling the training data with %s sampling.", whichone)
    sampler = None

    features = pd_data.drop(["target"], axis=1).values
    target = pd_data["target"].values

    if whichone == "RandomOverSampler":
        sampler = RandomOverSampler(random_state=SEED, sampling_strategy=strategy)
    elif whichone == "SMOTE":
        sampler = SMOTE(random_state=SEED, sampling_strategy=strategy)
    elif whichone == "SVMSMOTE":
        sampler = SVMSMOTE(random_state=SEED, sampling_strategy=strategy)
    elif whichone == "BorderlineSMOTE":
        sampler = BorderlineSMOTE(random_state=SEED, sampling_strategy=strategy)

    features_resampled, target_resampled = sampler.fit_resample(features, target)

    logger.debug(
        "Number of signal events before oversampling: %i", len(features[target == 1])
    )
    logger.debug(
        "Number of background events before oversampling: %i",
        len(features[target == 0]),
    )
    logger.debug(
        "Number of signal events after oversampling: %i",
        len(features_resampled[target_resampled == 1]),
    )
    logger.debug(
        "Number of background events after oversampling: %i",
        len(features_resampled[target_resampled == 0]),
    )

    return features_resampled, target_resampled


def undersample_data(
    pd_data, whichone, strategy, SEED
) -> tuple[np.ndarray, np.ndarray]:
    """
    Undersample the given datafrmae pd_data with the chosen sampler whichone
    Parameters:
        pd_data (pandas.DataFrame): The input DataFrame containing features and target.
        whichone (str): The name of the undersampling technique to use.
        random_state (int): The random seed for reproducibility.
        strategy (str): sampling strategy.

    Returns:
        tuple: A tuple containing two elements:
            - features_resampled (numpy.ndarray): Resampled features after undersampling.
            - target_resampled (numpy.ndarray): Resampled target labels after undersampling.
    """
    logger.info("Undersampling the training data with %s sampling.", whichone)
    sampler = None

    features = pd_data.drop(["target"], axis=1).values
    target = pd_data["target"].values

    if whichone == "TomekLinks":
        sampler = TomekLinks(sampling_strategy=strategy)
    elif whichone == "RandomUnderSampler":
        sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=strategy)
    elif whichone == "ClusterCentroids":
        sampler = ClusterCentroids(
            voting="auto", random_state=SEED, sampling_strategy=strategy
        )

    features_resampled, target_resampled = sampler.fit_resample(features, target)

    logger.debug(
        "Number of signal events before undersampling: %i", len(features[target == 1])
    )
    logger.debug(
        "Number of background events before undersampling: %i",
        len(features[target == 0]),
    )
    logger.debug(
        "Number of signal events after undersampling: %i",
        len(features_resampled[target_resampled == 1]),
    )
    logger.debug(
        "Number of background events after undersampling: %i",
        len(features_resampled[target_resampled == 0]),
    )

    return features_resampled, target_resampled


def combined_underover_sample_data(
    pd_data, whichone, undersamp_ratio, SEED
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use a combination of under and oversampling to the given
    datafrmae pd_data with the chosen sampler whichone
    Parameters:
        pd_data (pandas.DataFrame): The input DataFrame containing features and target.
        whichone (str): The name of the undersampling technique to use.
        undersamp_ratio (float): percentage of under-sampling the majority class
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing two elements:
            - features_resampled (numpy.ndarray): Resampled features after undersampling.
            - target_resampled (numpy.ndarray): Resampled target labels after undersampling.
    """

    logger.info(
        "Combination of over and under sampling the training data "
        "with %s for under-sampling and %s for over-sampling.",
        whichone[0],
        whichone[1],
    )

    num_majority_class = len(
        pd_data[pd_data["target"] == 0]
    )  # Number of majority class samples
    num_minority_class = len(
        pd_data[pd_data["target"] == 1]
    )  # Number of majority class samples
    undersample_majority_samples = int(num_majority_class * undersamp_ratio)
    under_strategy = {0: undersample_majority_samples, 1: num_minority_class}

    under_feat, under_tar = undersample_data(pd_data, whichone[0], under_strategy, SEED)
    data_df_for_over = pd.DataFrame(
        np.c_[under_tar, under_feat], columns=pd_data.columns
    )
    over_feat, over_tar = oversample_data(
        data_df_for_over, whichone[1], "minority", SEED
    )

    return over_feat, over_tar
