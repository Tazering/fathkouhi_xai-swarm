"""
This python file is the main driver for the experiment.
A signficant portion of this code was made to be similar to Emmanuel Doumard et al's 
experiment found in:

github.com/EmmanuelDoumard/local_explanation_comparative_study

The largest difference is the implementation of our XAI algorithms that use SWARM 
optimizers as also another approach for explainability.
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import swarm_xai

import data_preprocessing
import helpful_utils.data_tools as data_tools
import helpful_utils.constants as constants
import model_utils
import base_xai
import xai_utils
import metrics

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_selector as selector
from colorama import Style, Fore


"""
The main function that calls each of the needed steps for the experiment.
"""
def main():

    # swarm parameters
    all_swarm_parameters = {}
    all_swarm_parameters["pso"] = {}

    # pso parameters
    pso_swarm_parameters = {
        "optimizer_number": 1,
        "num_agents": 10,
        "upper_bound": 1,
        "lower_bound": -1,
        "dimension": 4,
        "num_iterations": 10,
        "w": .5,
        "c1": 1,
        "c2": 1

    }

    # bat parameters
    bat_swarm_parameters = {
        "optimizer_number": 2,
        "num_agents": 10,
        "upper_bound": 1,
        "lower_bound": -1,
        "dimension": 4,
        "num_iterations": 10,
        "r0": .9,
        "V0": .5,
        "fmin": 0,
        "fmax": .02,
        "alpha": .9,
        "csi": .9
    }

    # abc parameters
    abc_swarm_parameters = {
        "optimizer_number": 3,
        "num_agents": 10,
        "upper_bound": 1,
        "lower_bound": -1,
        "dimension": 4,
        "num_iterations": 10
    }

    all_swarm_parameters["pso"] = pso_swarm_parameters
    all_swarm_parameters["bat"] = bat_swarm_parameters
    all_swarm_parameters["abc"] = abc_swarm_parameters

    experiment_dataset(61, all_swarm_parameters = all_swarm_parameters)

    return None


"""
This function runs the pipeline on a single dataset
"""
def experiment_dataset(dataset_id, all_swarm_parameters):
    # Step 1: grab the dataset and preprocess it
    # Step 2: run the four machine learning models
    # Step 3: run the base xai functions on the dataset
    # Step 4: run the swarm approach on the dataset
    # Step 5: get values with the six metrics

    ##########
    #   Step 1: Grab the dataset
    ##########
    dataset_id = 61

    X_preprocessed, y_preprocessed = data_preprocessing.process_openml_dataset(dataset_index = dataset_id)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_preprocessed, test_size = .2)

    ##########
    #   Step 2: Run the models
    ##########
    lr_model, svm_model, rf_model = model_utils.run_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

    ##########
    #   Step 3: Run the Base XAI functions on the dataset and Run the Swarm Approach on the dataset
    ##########

    # dictionary in the form: {(string) name of approach: (tuple) (explanation, shap_values, time_consumption)}
    base_xai_dict = base_xai_study(X_preprocessed = X_test, y_preprocessed = y_test, clf = svm_model)

    # data_tools.print_generic("base_xai_dict[lime]", base_xai_dict["lime"][1].abs().mean().sort_values(ascending = False).cumsum())
    # data_tools.print_generic("base_xai_dict[lime] total", base_xai_dict["lime"][1].abs().mean())

    ##########
    #   Step 4: 
    ##########
    
    # dictionary in the form: {(string) name of approach: (dictionary) {(string) measurement: (numerical) value}}
    # swarm_xai.run_swarm_approach(X_test = X_test, y_test = y_test, num_trials = 10)

    swarm_xai_dict = swarm_xai_study(X_test = X_test, y_test = y_test, model = svm_model, num_trials = 10, all_swarm_parameters = all_swarm_parameters)


    # ##########
    # #   Step 5: Get the Metrics of both base and optimized
    # ##########

    # # dictionary that stores metrics of base_xai approaches
    experiment_results = metrics.calculate_metrics_of_model(X = X_test, base_xai_dict = base_xai_dict, swarm_xai_dict = swarm_xai_dict)


    data_tools.print_generic("experiment_results", experiment_results)

"""
Run all base_xai approaches on a single dataset and single model
"""
def base_xai_study(X_preprocessed, y_preprocessed, clf):
    # basic dictionary that stores results of base_xai results
    base_xai_dict = {}
    base_xai_dict["lime"] = {}
    base_xai_dict["spearman"] = {}
    base_xai_dict["complete"] = {}

    base_xai_dict["treeshap"] = {}
    base_xai_dict["treeshap_approx"] = {}
    base_xai_dict["kernelshap"] = {}

    if clf is not None: # in the case with multiclass xgboost
        # run all the base_xai approaches
        lime_tuple = base_xai.explanation_values_lime(X = X_preprocessed, clf = clf, mode = "classification")
        spearman_tuple = base_xai.explanation_values_spearman(X = X_preprocessed, y = y_preprocessed, clf = clf, rate = .25, problem_type = "Classification", complexity = True)
        complete_tuple = base_xai.explanation_values_complete(X = X_preprocessed, y = y_preprocessed, clf = clf, problem_type = "Classification")

        # treeshap, treeshap_approx, and kernelshap work on a case-by-case scenario

        try: # kernelshap
            kernelshap_tuple = base_xai.explanation_values_kernelSHAP(X = X_preprocessed, clf = clf, n_background_samples = 1)
        except:
            kernelshap_tuple = ()

        try: # treeshap
            treeshap_tuple = base_xai.explanation_values_treeSHAP(X = X_preprocessed, clf = clf)
        except:
            treeshap_tuple = ()

        try: # treeshap_approx
            treeshap_approx_tuple = base_xai.explanation_values_treeSHAP_approx(X = X_preprocessed, clf = clf)
        except:
            treeshap_approx_tuple = ()
        

        base_xai_dict["lime"] = lime_tuple
        base_xai_dict["spearman"] = spearman_tuple
        base_xai_dict["complete"] = complete_tuple
        base_xai_dict["treeshap"] = treeshap_tuple
        base_xai_dict["treeshap_approx"] = treeshap_approx_tuple
        base_xai_dict["kernelshap"] = kernelshap_tuple

    else: # gb is detected and is None
        base_xai_dict["treeshap"] = base_xai.treeSHAP_multi_to_binary_class(X = X_preprocessed, y = y_preprocessed)
        base_xai_dict["treeshap_approx"] = base_xai.treeSHAPapprox_multi_to_binary_class(X = X_preprocessed, y = y_preprocessed)

    return base_xai_dict

"""
Run the XAI on a single model of a single dataset.
The swarm models of interest are pso, bat, and abc swarm optimizers.
"""
def swarm_xai_study(X_test, y_test, model, num_trials, all_swarm_parameters):

    # grabs a sample
    swarm_optimizers = ["pso", "bat", "abc"]

    # dictionary to hold outputs
    output_dict = {}

    for swarm_optimizer in swarm_optimizers:
        output_dict[swarm_optimizer] = {}
        output_dict[swarm_optimizer] = swarm_xai.run_swarm_approach(X_test = X_test, y_test = y_test, model = model, num_trials = num_trials, swarm_parameters = all_swarm_parameters[swarm_optimizer])

    
    return output_dict


# def swarm_xai_study(X_preprocessed, X_test, y_test, model, categorical = {"Begin_Categorical": 5, "Categorical_Index": [1, 2]}, predict_prob = False):
#     n = X_preprocessed.shape[0]

#     # grabs a sample
#     # sample, sample_list = xai_utils.grab_sample(X_test, y_test, sample_number)

#     # dictionary to hold outputs
#     output_dict = {}
#     output_dict["pso"] = {}
#     output_dict["bat"] = {}
#     output_dict["abc"] = {}

#     if model is not None:
#         # dictionaries that list outputs of swarm approach
#         if predict_prob:
#             output_dict["pso"] = swarm_xai.XAI(max(model.predict_proba(sample)[0]), sample_list, np.size(sample_list), 50, 20, 30, -1, 1, X_preprocessed, categorical, False, 1).XAI_swarm_Invoke()
#             output_dict["bat"] = swarm_xai.XAI(max(model.predict_proba(sample)[0]), sample_list, np.size(sample_list), 50, 20, 30, -1, 1, X_preprocessed, categorical, False, 2).XAI_swarm_Invoke()
#             output_dict["abc"] = swarm_xai.XAI(max(model.predict_proba(sample)[0]), sample_list, np.size(sample_list), 50, 20, 30, -1, 1, X_preprocessed, categorical, False, 3).XAI_swarm_Invoke()
#         else:
#             output_dict["pso"] = swarm_xai.XAI(model.predict(sample), sample_list, np.size(sample_list), 50, 20, 30, -1, 1, X_preprocessed, categorical, False, 1).XAI_swarm_Invoke()
#             output_dict["bat"] = swarm_xai.XAI(model.predict(sample), sample_list, np.size(sample_list), 50, 20, 30, -1, 1, X_preprocessed, categorical, False, 2).XAI_swarm_Invoke()
#             output_dict["abc"] = swarm_xai.XAI(model.predict(sample), sample_list, np.size(sample_list), 50, 20, 30, -1, 1, X_preprocessed, categorical, False, 3).XAI_swarm_Invoke()

    
#     return output_dict

main()



