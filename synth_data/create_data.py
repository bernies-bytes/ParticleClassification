"""
create synthetic data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from utils_class.config_reader import (
    DEBUG_MODE,
    logging_level,
)  # abs_path, run_name  # , DEBUG_MODE, logging_level


def create_with_make_class(sample_size=100000, percentage_bg=0.9):
    """
    this function uses sklearn's make_classification to create
    a synthetic data set
    """
    # The total number of features. These comprise n_informative informative features,
    # n_redundant redundant features, n_repeated duplicated features and
    # n_features-n_informative-n_redundant-n_repeated useless features drawn at random.
    features, target = make_classification(
        n_samples=sample_size,
        n_classes=2,
        # Set label 0 for ... and 1 for rest of observations
        weights=[percentage_bg],
        n_features=12,
        n_informative=8,
        n_redundant=3,
        n_clusters_per_class=1,
        class_sep=2,
        random_state=11,
    )
    columns = ["target"]
    feat_cols = ["feat" + str(i) for i in range(0, 12)]
    columns.extend(feat_cols)
    df_sample = pd.DataFrame(
        np.hstack(((target).reshape((len(target), 1)), features)), columns=columns
    )
    # print(df_sample)

    if DEBUG_MODE:
        figure, axes = plt.subplots(3, 4, figsize=(10, 6))
        for col in range(0, 4):
            for row in range(0, 3):
                print(col + 4 * row)
                sns.histplot(
                    data=df_sample,
                    x=f"feat{col + 4*row}",
                    hue="target",
                    stat="density",
                    ax=axes[row][col],
                )
        plt.show()

    return df_sample


def create_data_set(bg_size=10000, sig_size=500):
    """
    this function creates the feature distributions inspired by
    real data but without respecting the real correlations between features
    UNDER CONSTRUCTION
    """
    # TODO bg_size and sig_size must be int - add test or warning or so
    # create the data frame to save the features in
    # df_final = pd.DataFrame()

    # create the track distributions
    # background
    n_binom, p_binom = 5, 0.5
    bg_tracks = np.random.binomial(n_binom, p_binom, int(bg_size - bg_size * 0.5))
    bg_tracks_append = np.ones(int(bg_size * 0.5)) * 2
    bg_tracks = np.append(bg_tracks, bg_tracks_append)
    sns.histplot(data=bg_tracks, stat="density")  # , x="bg_tracks")

    # signal
    n_binom, p_binom = 8, 0.45
    sig_tracks = np.random.binomial(n_binom, p_binom, sig_size)
    sns.histplot(data=sig_tracks, stat="density")
    plt.show()


if __name__ == "__main__":
    # Code to run when the script is executed directly
    # create_data_set(10000,500)
    feat, tar = create_with_make_class(100000, 0.70)
