"""
This python file is for testing various methods. Particularly,
the evaluation_metrics.md file.
"""

import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import sklearn
import time

from lime import lime_tabular
import shap

# local from other python files
import utils.data_tools as data_tools
import base_xai
import constants
import data_preprocessing as dp

import coalitional_methods as coal

import XAI_Swarm_Opt

# iris
def test_iris():
    # grab data and split it
    X_preprocessed, y_preprocessed = dp.process_openml_dataset(61, "class")

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_preprocessed, y_preprocessed, test_size = .2)

    # run all models: logistic regression, support vector machines, random forests, and gradient boosted machines
    lr_model = constants.models["logistic_regression"].fit(X_train, y_train)
    svm_model = constants.models["support_vector_machine"].fit(X_train, y_train)
    rf_model = constants.models["random_forests"].fit(X_train, y_train)
    gb_model = constants.models["gradient_boosted_machine"].fit(X_train, y_train)

    # print the scores for testing purposes
    # data_tools.print_generic("Logistic Regression Score", lr_model.score(X_test, y_test))
    # data_tools.print_generic("SVM Score", svm_model.score(X_test, y_test))
    # data_tools.print_generic("RF Score", rf_model.score(X_test, y_test))
    # data_tools.print_generic("GB Score", gb_model.score(X_test, y_test))
    
    # get list of feature names
    features_names = X_train.columns

    # run swarm explanations
    # grab a sample 
    sample_number = 5

    # data_tools.print_variable("x_val", x_val)

    # details of a single datapoint
    sample = X_test.iloc[sample_number]
    sample_y = y_test.iloc[sample_number]
    sample_y = 1 if sample_y == True else 0
    sample = sample.values.reshape(1, -1)

    # converts sample to a single list
    sample_size = np.size(sample[0])

    sample_list = sample[0]
    temp_categorical = {"Begin_Categorical": 3, "Categorical_Index": [1, 2]}

    data_tools.print_generic("svm_model output", svm_model.predict(sample))

    results = XAI_Swarm_Opt.XAI(svm_model.predict(sample)[0], sample_list, np.size(sample_list), 50, 20,30, -1, 1, X_preprocessed, temp_categorical, False).XAI_swarm_Invoke()

    contribute_dict = {}
    contributes = results["contribute"]

    for i in range(len(contributes)):
        contribute_dict[features_names[i]] = contributes[i]
    

    results["contribute"] = contribute_dict

    data_tools.print_generic("results", results)

    # get metrics

    # store results

    return None

# SCRATCH WORK for testing functions

# run test here
test_iris()