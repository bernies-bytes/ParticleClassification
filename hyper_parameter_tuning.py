# pylint: disable=import-error
import xgboost as xgb
import pandas as pd
from bayes_opt import BayesianOptimization

# from bayes_opt.util import UtilityFunction
from sklearn import metrics

from utils.config_setup import (
    logger,
    bayes_opt_init_pts,
    bayes_opt_n_iter,
    #    bayes_opt_acq,
    #    bayes_opt_xi,
    para_to_tune,
)


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np


def XGB_CV(
    dtrain,
    n_estimators,
    eta,
    max_depth,
    gamma,
    min_child_weight,
    max_delta_step,
    subsample,
    colsample_bytree,
    best_metric_glob,
    # best_iter_glob,
):
    param_current = {
        "booster": "gbtree",
        # "n_estimators": n_estimators,  # I got the warning that this is unused.
        # this is because n_boost_rounds
        # below is set und it is the same!
        "max_depth": max_depth.astype(int),
        "gamma": gamma,
        "eta": eta,
        "objective": "binary:logistic",
        "nthread": 10,
        # "silent": True, # I got the warning that this is unused
        #'eval_metric': 'logloss',
        "eval_metric": "auc",
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "max_delta_step": max_delta_step.astype(int),
        "seed": 1001,
    }

    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    auc_scores = []

    for train_index, val_index in skf.split(dtrain.get_data(), dtrain.get_label()):
        X_train, y_train = (
            dtrain.get_data()[train_index],
            dtrain.get_label()[train_index],
        )
        X_val, y_val = dtrain.get_data()[val_index], dtrain.get_label()[val_index]

        # Oversample the minority class using RandomOverSampler
        sampler = RandomOverSampler(random_state=1001)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

        dtrain_resampled = xgb.DMatrix(
            X_train_resampled, label=y_train_resampled, missing=999
        )
        dval = xgb.DMatrix(X_val, label=y_val, missing=999)

        xgbr = xgb.train(
            param_current,
            dtrain_resampled,
            num_boost_round=100000,
            evals=[(dval, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred = xgbr.predict(
            dval
        )  # this uses automatically the best iteration because early_stopping_rounds is set
        auc_score = roc_auc_score(y_val, y_pred)
        auc_scores.append(auc_score)

    avg_auc = np.mean(auc_scores)
    if avg_auc > best_metric_glob:
        best_metric_glob = avg_auc
        # best_iter_glob = len(xgbr)

    # for logloss we return negative
    # return (-1.0 * avg_auc)  # needs to be negative cos bayes maximizes,
    # but we want to minimize logloss
    return 1.0 * avg_auc  # for auc


def bayes_hyper_opt(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    folds = 5
    best_metric_glob = -1.0
    best_iter_glob = 0

    #######################################################################################################
    # nrrun = 1  # this used to be the index of a for loop
    # samp = "SMOTE"  # this also was an index of a loop
    features_train = df_train.drop(["target"], axis=1).values
    target_train = df_train["target"].values

    features_test = df_test.drop(["target"], axis=1).values
    target_test = df_test["target"].values

    ##############################################################################################

    dtrain = xgb.DMatrix(features_train, label=target_train, missing=999)
    dtest = xgb.DMatrix(features_test, label=target_test, missing=999)

    # Create a lambda function that wraps XGB_CV and sets dtrain as a fixed parameter
    objective_function = lambda n_estimators, eta, max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree, dtrain=dtrain: XGB_CV(
        dtrain,
        n_estimators,
        eta,
        max_depth,
        gamma,
        min_child_weight,
        max_delta_step,
        subsample,
        colsample_bytree,
        best_metric_glob,
        # best_iter_glob,
    )

    # Create an instance of BayesianOptimization
    XGB_BO = BayesianOptimization(
        f=objective_function,  # Use the lambda function as the objective function
        pbounds=para_to_tune,  # Bounds for the hyperparameters
    )
    # n_iter: How many steps of bayesian optimization you want to perform.
    # The more steps the more likely to find a good maximum you are.
    # init_points: How many steps of random exploration you want to perform.
    # Random exploration can help by diversifying the exploration space.
    # Maximize the Bayesian optimization
    XGB_BO.maximize(
        init_points=bayes_opt_init_pts,
        n_iter=bayes_opt_n_iter,
    )

    logger.info(
        "Hyper-parameter tuning with Bayes-optimisation done (%i initial points and %i iterations )",
        bayes_opt_init_pts,
        bayes_opt_n_iter,
    )

    logger.info("Results:")

    logger.info("Best metric (max AUC): %f", XGB_BO.max["target"])
    logger.info("Best parameters:")

    for key, value in XGB_BO.max["params"].items():
        logger.info("%s: %s",key, value)

    logger.info("Making Predictions on the test set:")

    best_param_for_test = {
        "booster": "gbtree",
        #"n_estimators": XGB_BO.max["params"]["n_estimators"],
        "max_depth": XGB_BO.max["params"]["max_depth"].astype(int),
        "gamma": XGB_BO.max["params"]["gamma"],
        "eta": XGB_BO.max["params"]["eta"],
        "objective": "binary:logistic",
        "nthread": 3,
        #"silent": True,
        #'eval_metric': 'logloss',
        "eval_metric": "auc",
        "subsample": XGB_BO.max["params"]["subsample"],
        "colsample_bytree": XGB_BO.max["params"]["colsample_bytree"],
        "min_child_weight": XGB_BO.max["params"]["min_child_weight"],
        "max_delta_step": XGB_BO.max["params"]["max_delta_step"].astype(int),
        "seed": 1001,
    }

    logger.info("Training with best parameters of Bayes-optimisation and full training set:")

    xgb_bayesresult = xgb.train(
        best_param_for_test, dtrain,
        num_boost_round=int(XGB_BO.max["params"]["n_estimators"]),
        #early_stopping_rounds=50,
    )

    test_prediction = xgb_bayesresult.predict(
        xgb.DMatrix(dtest.get_data(), missing=999), iteration_range=xgb_bayesresult.best_iteration + 1
    )
    test_prediction2 = test_prediction.round()

    exit(0)

   
#############################################################
if __name__ == "__main__":
    pass
