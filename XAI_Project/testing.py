"""
This python file is for testing various methods. Particularly,
the evaluation_metrics.md file.
"""

import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import data_preprocessing as dp
import constants
import sklearn

import utils.data_tools as data_tools

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
    data_tools.print_scalar("Logistic Regression Score", lr_model.score(X_test, y_test))
    data_tools.print_scalar("SVM Score", svm_model.score(X_test, y_test))
    data_tools.print_scalar("RF Score", rf_model.score(X_test, y_test))
    data_tools.print_scalar("GB Score", gb_model.score(X_test, y_test))

    # run base explanations

    # run swarm explanations

    # get metrics

    # store results

    return None

# run test here
test_iris()