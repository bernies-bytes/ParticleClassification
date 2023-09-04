# pylint: disable=import-error
"""
train test splitting
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config_setup import logger


def train_test_splitting(
    df_data, test_size: float = 0.33
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    split the data frame into training and test samples
    '''
    train, test = train_test_split(df_data, test_size=test_size)
    logger.info(
        "Number of background training events: %i", len(train[train["target"] == 0])
    )
    logger.info(
        "Number of signal training events: %i", len(train[train["target"] == 1])
    )

    logger.info(
        "Number of background test events: %i", len(test[test["target"] == 0])
    )
    logger.info(
        "Number of signal test sevents: %i", len(test[test["target"] == 1])
    )

    return train, test
