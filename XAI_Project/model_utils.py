"""
This python file has all the functions related to the models of interest.
"""
import pandas as pd
import numpy as np
import sklearn

# local from other python files
import utils.data_tools as data_tools
import utils.constants as constants


"""
Run the models on a given dataset
@Parameters:
    X_train: the input training data
    y_train: the output training data
    X_test: the input testing data
    y_test: the output testing data

@Returns
    lr_model: fitted logistic regression model
    svm_model: fitted support vector machine model
    rf_model: fitted random forest model
    gb_model: fitted gradient-boosted machine model
"""
def run_models(X_train, y_train, X_test, y_test):
    # run all models: logistic regression, support vector machines, random forests, and gradient boosted machines
    lr_model = constants.models["logistic_regression"].fit(X_train, y_train)
    svm_model = constants.models["support_vector_machine"].fit(X_train, y_train)
    rf_model = constants.models["random_forests"].fit(X_train, y_train)
    gb_model = constants.models["gradient_boosted_machine"].fit(X_train, y_train)

    # prints the scores of the models
    data_tools.print_generic("Logistic Regression Score", lr_model.score(X_test, y_test))
    data_tools.print_generic("SVM Score", svm_model.score(X_test, y_test))
    data_tools.print_generic("RF Score", rf_model.score(X_test, y_test))
    data_tools.print_generic("GB Score", gb_model.score(X_test, y_test))

    return lr_model, svm_model, rf_model, gb_model
