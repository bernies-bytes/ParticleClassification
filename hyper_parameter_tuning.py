# pylint: disable=unnecessary-lambda-assignment
"""
Module containing functions for optimising XGBoost hyperparameter via bayesian optimisation
"""
# pylint: disable=import-error
import xgboost as xgb
import pandas as pd
from bayes_opt import BayesianOptimization

# from bayes_opt.util import UtilityFunction
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np


from utils.config_setup import (
    logger,
    bayes_opt_init_pts,
    bayes_opt_n_iter,
    #    bayes_opt_acq,
    #    bayes_opt_xi,
    para_to_tune,
)


def xgb_cv(
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
    """
    Function that performs cross validation for a given set
    of XGBoost parameters
    """
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

    # stratified split into 5 folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)

    auc_scores = []

    for train_index, val_index in skf.split(dtrain.get_data(), dtrain.get_label()):
        feat_train, target_train = (
            dtrain.get_data()[train_index],
            dtrain.get_label()[train_index],
        )
        feat_val, target_val = (
            dtrain.get_data()[val_index],
            dtrain.get_label()[val_index],
        )

        # Oversample the minority class using RandomOverSampler
        sampler = RandomOverSampler(random_state=1001)
        feat_train_resampled, target_train_resampled = sampler.fit_resample(
            feat_train, target_train
        )

        dtrain_resampled = xgb.DMatrix(
            feat_train_resampled, label=target_train_resampled, missing=999
        )
        dval = xgb.DMatrix(feat_val, label=target_val, missing=999)

        xgbr = xgb.train(
            param_current,
            dtrain_resampled,
            num_boost_round=100000,
            evals=[(dval, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        target_pred = xgbr.predict(
            dval
        )  # this uses automatically the best iteration because early_stopping_rounds is set
        auc_score = roc_auc_score(target_val, target_pred)
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
    """
    function that carries out the hyper parameter tuning.
    parameter:
    df_train is a data frame that contains the training data (unsampled)
    df_test contains the test data samples to test the trained model with the
    tuned parameters. It is used to calculate the performances scores at the
    end of the function.
    """
    # folds = 5
    best_metric_glob = -1.0
    # best_iter_glob = 0

    ######################################################
    # nrrun = 1  # this used to be the index of a for loop
    # samp = "SMOTE"  # this also was an index of a loop
    features_train = df_train.drop(["target"], axis=1).values
    target_train = df_train["target"].values

    features_test = df_test.drop(["target"], axis=1).values
    target_test = df_test["target"].values

    ##############################################################################################

    dtrain = xgb.DMatrix(features_train, label=target_train, missing=999)
    dtest = xgb.DMatrix(features_test, label=target_test, missing=999)

    # Create a lambda function that wraps xgb_cv and sets dtrain as a fixed parameter
    objective_function = (
        lambda n_estimators, eta, max_depth, gamma, min_child_weight, max_delta_step,
        subsample, colsample_bytree, dtrain=dtrain: xgb_cv(
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
    )

    # Create an instance of BayesianOptimization
    xgb_bo = BayesianOptimization(
        f=objective_function,  # Use the lambda function as the objective function
        pbounds=para_to_tune,  # Bounds for the hyperparameters
    )
    # n_iter: How many steps of bayesian optimization you want to perform.
    # The more steps the more likely to find a good maximum you are.
    # init_points: How many steps of random exploration you want to perform.
    # Random exploration can help by diversifying the exploration space.
    # Maximize the Bayesian optimization
    xgb_bo.maximize(
        init_points=bayes_opt_init_pts,
        n_iter=bayes_opt_n_iter,
    )

    logger.info(
        "Hyper-parameter tuning with Bayes-optimisation done"
        "(%i initial points and %i iterations )",
        bayes_opt_init_pts,
        bayes_opt_n_iter,
    )

    logger.info("Results:")

    logger.info("Best metric (max AUC): %f", xgb_bo.max["target"])
    logger.info("Best parameters:")

    for key, value in xgb_bo.max["params"].items():
        logger.info("%s: %s", key, value)

    logger.info("Making Predictions on the test set:")

    best_param_for_test = {
        "booster": "gbtree",
        # "n_estimators": xgb_bo.max["params"]["n_estimators"],
        "max_depth": xgb_bo.max["params"]["max_depth"].astype(int),
        "gamma": xgb_bo.max["params"]["gamma"],
        "eta": xgb_bo.max["params"]["eta"],
        "objective": "binary:logistic",
        "nthread": 3,
        # "silent": True,
        #'eval_metric': 'logloss',
        "eval_metric": "auc",
        "subsample": xgb_bo.max["params"]["subsample"],
        "colsample_bytree": xgb_bo.max["params"]["colsample_bytree"],
        "min_child_weight": xgb_bo.max["params"]["min_child_weight"],
        "max_delta_step": xgb_bo.max["params"]["max_delta_step"].astype(int),
        "seed": 1001,
    }

    logger.info(
        "Training with best parameters of Bayes-optimisation and full training set:"
    )

    xgb_bayesresult = xgb.train(
        best_param_for_test,
        dtrain,
        num_boost_round=int(xgb_bo.max["params"]["n_estimators"]),
        # early_stopping_rounds=50,
    )

    test_prediction = xgb_bayesresult.predict(
        xgb.DMatrix(dtest.get_data(), missing=999),
        # iteration_range=xgb_bayesresult.best_iteration + 1,
    )
    test_prediction_round = test_prediction.round()

    dtest_sig = df_test[df_test["target"] == 1]
    dtest_bg = df_test[df_test["target"] == 0]
    features_test_sig = dtest_sig.drop(["target"], axis=1).values
    features_test_bg = dtest_bg.drop(["target"], axis=1).values

    test_sig_prediction = xgb_bayesresult.predict(
        xgb.DMatrix(features_test_sig, missing=999)
    )
    test_sig_prediction_round = test_sig_prediction.round()

    test_bg_prediction = xgb_bayesresult.predict(
        xgb.DMatrix(features_test_bg, missing=999)
    )
    test_bg_prediction_round = test_bg_prediction.round()

    # CREATE a loop for the stuff below to print the reports (DEBUG logger)
    samples = [
        [df_test, test_prediction, test_prediction_round],
        [dtest_sig, test_sig_prediction, test_sig_prediction_round],
        [dtest_bg, test_bg_prediction, test_bg_prediction_round],
    ]

    samples_info = [
        "complete test set",
        "signal events of test set",
        "background events of test set",
    ]

    for samp_, samp_info_ in zip(samples, samples_info):
        accuracy_score = metrics.accuracy_score(samp_[0]["target"].values, samp_[2])
        # roc_auc_score = metrics.roc_auc_score(samp_[0]["target"].values, samp_[1])
        average_precision_score = metrics.average_precision_score(
            samp_[0]["target"].values, samp_[1]
        )
        precision_score = metrics.precision_score(samp_[0]["target"].values, samp_[2])
        recall_score = metrics.recall_score(samp_[0]["target"].values, samp_[2])
        f1_score = metrics.f1_score(
            samp_[0]["target"].values, samp_[2], average="weighted"
        )
        logger.info("results on following data: %s", samp_info_)
        logger.info("accuracy score: %f", accuracy_score)
        logger.info("average precision score: %f", average_precision_score)
        logger.info("precision score: %f", precision_score)
        logger.info("recall score: %f", recall_score)
        logger.info("f1 score: %f", f1_score)


#############################################################
if __name__ == "__main__":
    pass
