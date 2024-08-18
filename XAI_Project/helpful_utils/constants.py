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
    "random_forests": sklearn.ensemble.RandomForestClassifier(),
    "gradient_boosted_machine": xgb.XGBClassifier(use_label_encoder = False, eval_metrics = "logloss", n_jobs = 1)
}
