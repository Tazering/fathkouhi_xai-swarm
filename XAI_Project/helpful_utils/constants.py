"""
This python function lists a bunch of constants that will be used for the experiment.
"""

import pandas as pd
import numpy as np

import sklearn
import sklearn.ensemble
import sklearn.linear_model
import xgboost as xgb


# dictionary list of models
models = {
    "logistic_regression": sklearn.linear_model.LogisticRegression(),
    "support_vector_machine": sklearn.svm.SVC(probability = True),
    "random_forests": sklearn.ensemble.RandomForestClassifier()
    # "gradient_boosted_machine": xgb.XGBClassifier(use_label_encoder = False, eval_metrics = "logloss", n_jobs = 1),
}

datasets = {
    # 1 features
    "reading_hydro_downstream": 44267,
    "balloon": 512,
    "humandevel": 924,
    "reading_hydro_upstream": 44221,
    "SquareF": 45617,

    # 2 features
    "transplant": 544,
    "prnn_synth": 464,
    "analcatdata_neavote": 523,
    "vinnie": 519,
    "analcatdata_hiroshima": 494,
    "rmftsa_ctoarrivals": 686,
    "chscase_vine2": 689,
    "UNIX_user_data": 373,
    "rmftsa_sleepdata": 679,
    "rabe_266": 679,
    "banana": 1460,
    "wall-robot-navigation": 1525,
    "chscase_geyser1": 895,
    "mbagrade": 887,
    "visualizing_ethanol": 711,
    "chscase_geyser1712": 712,
    "vineyard": 713,
    "vineyard_192": 192,
    "RAM_price": 40601,
    "ELE-1": 42361,
    "olindda_outliers": 42793,

    # 4 features
    "iris": 61
}
