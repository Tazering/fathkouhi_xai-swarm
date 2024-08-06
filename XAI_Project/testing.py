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



# iris
def test_iris():
    # grab data and split it
    X_preprocessed, y_preprocessed = dp.process_openml_dataset(61, "class")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_preprocessed, y_preprocessed, test_size = .2)

    print(X_preprocessed.shape)

    data_tools.print_variable("X_preprocessed", X_preprocessed)

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

    # run base explanations and their shap values
    # lime_explanations, lime_shap_values, lime_elapsed_time = base_xai.explanation_values_lime(X = X_preprocessed, clf = lr_model, mode = "classification")
    # data_tools.print_variable("lime_explanations", lime_explanations)
    # data_tools.print_variable("lime_shap_values", lime_shap_values)
    # data_tools.print_generic("lime_elapsed_time", lime_elapsed_time)


    # tree_shap_explanation, tree_shap_values, tree_shap_elapsed_time = base_xai.explanation_values_treeSHAP(X = X_preprocessed, clf = rf_model)
    # data_tools.print_variable("tree_shap_explanation", tree_shap_explanation)
    # data_tools.print_variable("tree_shap_values", tree_shap_values)
    # data_tools.print_generic("tree_shap_elapsed_time", tree_shap_elapsed_time)

    # X_preprocessed.reset_index(drop = True, inplace = True)
    # kernelSHAP_explanations, kernelSHAP_values, kernelSHAP_elapsed_time = base_xai.explanation_values_kernelSHAP(X = X_preprocessed, clf = svm_model, n_background_samples = 10)
    # data_tools.print_variable("kernelSHAP_values", kernelSHAP_values)

    _, spearman_values, __ = base_xai.explanation_values_spearman(X = X_preprocessed, y = y_preprocessed, clf = svm_model, rate = 0.25,
                                                                  problem_type = "Classification", complexity = True)
    # data_tools.print_variable(spearman_values)

    # complete_explanation, complete_values, complete_elapsed_time = base_xai.explanation_values_complete(X = X_preprocessed, y = y_preprocessed, clf = svm_model, problem_type = "Classification")

    # tree_approx_explanations, tree_approx_values, tree_approx_time = base_xai.explanation_values_treeSHAP_approx(X = X_preprocessed, clf = rf_model)
    # data_tools.print_variable("tree_approx_values", tree_approx_values)

    # run swarm explanations

    # get metrics

    # store results

    return None

# run test here
test_iris()