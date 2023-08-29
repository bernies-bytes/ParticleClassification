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

    undersamp_ratio (float): percentage of under-sampling the majority class
    """
    logger.info(
        "Combination of over and under sampling the training data "
        "with %s for under-sampling and %s for over-sampling.",
        whichone[0],
        whichone[1]
    )

    num_majority_class = len(pd_data[pd_data["target"]==0])  # Number of majority class samples
    num_minority_class = len(pd_data[pd_data["target"]==1])  # Number of majority class samples

    undersample_majority_samples = int(num_majority_class * undersamp_ratio)

    # 'auto' to leave the minority class unchanged
    under_strategy = {0: undersample_majority_samples, 1: num_minority_class }
    #under_strategy = undersamp_ratio

    under_feat, under_tar = undersample_data(pd_data, whichone[0], under_strategy, SEED)
    data_df_for_over = pd.DataFrame(
        np.c_[under_tar, under_feat], columns=pd_data.columns
    )
    over_feat, over_tar = oversample_data(
        data_df_for_over, whichone[1], 'minority', SEED
    )

    return over_feat, over_tar

    # feat_0 = pd_data[pd_data[["target"]] == 0]
    # tar_0 = feat_0["target"].values
    # feat_0 = feat_0.drop(["target"], axis=1).values

    # feat_1 = pd_data[pd_data[["target"]] == 1]
    # tar_1 = feat_1["target"].values
    # feat_1 = feat_1.drop(["target"], axis=1).values

    # feat_under, tar_under, feat_over_tar_over = 0, 0, 0, 0

    # if len(tar_0) > len(tar_1):
    #    feat_under, tar_under = feat_0, tar_0
    #    feat_over, tar_over = feat_1, tar_1
    # else:
    #    feat_under, tar_under = feat_1, tar_1
    #    feat_over, tar_over = feat_0, tar_0


"""
def combinedsample_custom_data(X, y, whichone_over, whichone_under, SEED):
    print( "----------------------------------------------------------------------------" )
    print( "Custom combination of Over- and Undersampling test...")
    sampler = None
    
    X_cosmics = X[y == 0]
    X_pbars   = X[y == 1]
    y_pbars = np.ones(len(X_pbars))
      
    print( X.shape)
    print( X_cosmics.shape)
    print( X_pbars.shape)
    
    print( " ")

    if whichone_under == "Random":
        print( "random choice .........")
        pick_from = np.arange(0, len(X_cosmics))
        #print pick_from
        inds = np.random.choice(pick_from, size = int(len(X_cosmics)*0.30))
        
        #print "samp!", inds.shape
        
        X_cosmics_under = X_cosmics[inds]
        
        
        
        #print "samp!", X_cosmics.shape
        y_cosmics = np.zeros(len(X_cosmics_under))
        print( X_cosmics_under.shape)
        
        print( " ")

    if whichone_over == "SMOTE": 
        print( "SMOTE........."      ) 
        sampler = SMOTE(kind = 'regular', random_state =RANDOM)


    X = np.vstack((X_pbars, X_cosmics_under))
    y = np.hstack((y_pbars, y_cosmics))
    
    # TODO shuffle?
    
    print( X.shape)
        
    X_resampled, y_resampled = sampler.fit_sample(X, y)

    print( X_resampled.shape, y_resampled.shape)

    pick_from2 = np.arange(0, len(X_resampled))
    #print pick_from2
    np.random.shuffle(pick_from2)

    #print shuffled_inds
    #print shuffled_inds.shape

    X_resampled = X_resampled[pick_from2]
    y_resampled = y_resampled[pick_from2]


    print( X_resampled.shape, y_resampled.shape)

    print( "Number of pbar events before oversampling: ", len(X[y == 1]))
    print( "Number of cosmic events before oversampling: ", len(X[y == 0]))
    print( " ")
    print( "Number of pbar events after oversampling: ", len(X_resampled[y_resampled == 1]))
    print( "Number of cosmic events after oversampling: ", len(X_resampled[y_resampled == 0]))
    
    return X_resampled, y_resampled

#########################################################################################################  
#########################################################################################################  

def combinedsample_data(X, y, whichone, RANDOM):
    print( "----------------------------------------------------------------------------" )
    print( "Combination of Over- and Undersampling test...")
    sampler = None

    if whichone == "SMOTETomek":
        print( "SMOTE + Tomek.........")
        sampler = SMOTETomek(random_state = RANDOM)

    if whichone == "SMOTEENN": 
        print( "SMOTE + ENN........."   )    
        sampler = SMOTEENN(random_state = RANDOM)



    X_resampled, y_resampled = sampler.fit_sample(X, y)


    print( "Number of pbar events before oversampling: ", len(X[y == 1]))
    print( "Number of cosmic events before oversampling: ", len(X[y == 0]))
    print( " ")
    print( "Number of pbar events after oversampling: ", len(X_resampled[y_resampled == 1]))
    print( "Number of cosmic events after oversampling: ", len(X_resampled[y_resampled == 0]))
    
    return X_resampled, y_resampled
    
#########################################################################################################  
#########################################################################################################  
    
#########################################################################################################    
#########################################################################################################     

def undersample_data(X, y, whichone, RANDOM):

    print( "----------------------------------------------------------------------------" )
    print( "Undersampling test...")
    sampler = None
    
    #X_cos_help = X[y==0]
    #y_cos_help = y[y==0]
    
    #X_help = X[y==1]
    #y_help = y[y==1]
    
    if whichone == "tomek":
        print( "Tomek.........")
        sampler = TomekLinks(return_indices=False, random_state=RANDOM)
        
    if whichone == "under_random":
        print( "random.........")
        sampler = RandomUnderSampler(return_indices=False, random_state=RANDOM)
        
    if whichone == "clustercentroids":
        print( "cluster cetroids.........")
        sampler = ClusterCentroids(ratio = 'majority', voting = 'auto', random_state=RANDOM)
        
        
    X_resampled, y_resampled = sampler.fit_sample(X, y)


    print( "Number of pbar events before undersampling: ", len(X[y == 1]))
    print( "Number of cosmic events before undersampling: ", len(X[y == 0]))
    print( " ")
    print( "Number of pbar events after undersampling: ", len(X_resampled[y_resampled == 1]))
    print( "Number of cosmic events after undersampling: ", len(X_resampled[y_resampled == 0]))

    
    return X_resampled, y_resampled
    

"""
