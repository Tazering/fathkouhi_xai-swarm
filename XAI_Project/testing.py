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

# iris
def test_iris():
    # grab data and split it
    X_preprocessed, y_preprocessed = dp.process_openml_dataset(61, "class")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_preprocessed, y_preprocessed, test_size = .2)

    # print(X_preprocessed.shape)

    # data_tools.print_variable("X_preprocessed", X_preprocessed)

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

    spearman_explanations, spearman_inf, spearman_time_difference = base_xai.explanation_values_spearman(X = X_preprocessed, y = y_preprocessed, clf = svm_model, rate = 0.25,
                                                                  problem_type = "Classification", complexity = True)
    
    data_tools.print_generic("spearman_explanations", spearman_explanations)
    
    # data_tools.print_variable("spearman_explanations", spearman_explanations)
    # data_tools.print_variable(spearman_values)

    # complete_explanation, complete_values, complete_elapsed_time = base_xai.explanation_values_complete(X = X_preprocessed, y = y_preprocessed, clf = svm_model, problem_type = "Classification")

    # tree_approx_explanations, tree_approx_values, tree_approx_time = base_xai.explanation_values_treeSHAP_approx(X = X_preprocessed, clf = rf_model)
    # data_tools.print_variable("tree_approx_values", tree_approx_values)

    # run swarm explanations

    # get metrics

    # store results

    return None

# SCRATCH WORK for testing functions

# Spearman Test
# def explanation_values_spearman(X, y, clf, rate, problem_type, complexity=False, fvoid=None, look_at=1, progression_bar=True):
#     t0 = time.time()
#     spearman_inf = coal.coalitional_method(X, y, clf, rate, problem_type, fvoid=fvoid, complexity=complexity, method='spearman', look_at=look_at, progression_bar=progression_bar)
#     print("==========\n\n", spearman_inf, "\n\n=============")

#     t1 = time.time()
    
#     if fvoid is None:
#         if problem_type == "Classification":
#             fvoid = (
#                 y.value_counts(normalize=True).sort_index().values
#             )
#         elif problem_type == "Regression":
#             fvoid = y.mean()
    

#     explanation = generate_correct_explanation(shap_values=spearman_inf.values,
#                                                base_values=fvoid[look_at]*np.ones(X.shape[0]),
#                                                X=X,
#                                                look_at=look_at)
    
#     return explanation, spearman_inf, t1-t0

# def explanation_values_spearman(X, y, clf, rate, problem_type, complexity = False, fvoid = None, look_at = 1, progression_bar = True):
#     t0 = time.time()
    
#     spearman_inf = coal.coalitional_method(X, y, clf, rate, problem_type, fvoid = fvoid, complexity = complexity, 
#                                            method = "spearman", look_at = look_at, progression_bar = progression_bar)

#     ##########

#     # f = open('file.txt', 'w')
#     # f.write(str(spearman_inf))
#     # f.close()

#     # data_tools.print_generic("spearman_inf[0]", spearman_inf[1])
#     ##########

#     t1 = time.time()

#     if fvoid is None:
#         if problem_type == "Classification":
#             fvoid = (
#                 y.value_counts(normalize=True).sort_index().values
#             )
#         elif problem_type == "Regression":
#             fvoid = y.mean()

#     explanation = generate_correct_explanation(shap_values=spearman_inf.values,
#                                                base_values=fvoid[look_at]*np.ones(X.shape[0]),
#                                                X=X,
#                                                look_at=look_at)

#     return explanation, spearman_inf, t1-t0

# # testing purposes
# def generate_correct_explanation(explanation=None,shap_values=None,base_values=None,X=None,look_at=1):
#     # Recreate a shap explanation object that can be used for Waterfall plots
    
#     if explanation is None:
#         if (shap_values is None) | (base_values is None) | (X is None):
#             raise Exception("If you pass no explanation, you need to pass shap_values, base_values and X to construct an Explanation object")
            
#     if shap_values is None:
#         if len(np.array(explanation.values).shape) == 3:
#             shap_values = explanation.values[:,:,look_at]
#         else:
#             shap_values = explanation.values
#     if base_values is None:
#         if len(np.array(explanation.base_values).shape) == 2:
#             base_values = explanation.base_values[:,look_at]
#         else:
#             base_values = explanation.base_values
#     if X is None:
#         X = pd.DataFrame(explanation.data,columns=explanation.feature_names)
        
#     correct_explanation = shap.Explanation(shap_values,
#                                          base_values=base_values,
#                                          data=X.values,
#                                          feature_names=X.columns.to_list())

#     return correct_explanation

# run test here
test_iris()