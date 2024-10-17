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
import random

import swarm_xai

import data_preprocessing
import helpful_utils.data_tools as data_tools
import helpful_utils.constants as constants
import model_utils
import base_xai
import xai_utils
import metrics
import helpful_utils.constants as helpful_constants

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

    experiment_dataset("olindda_outliers", all_swarm_parameters = all_swarm_parameters)

    return None


"""
This function runs the pipeline on a single dataset
"""
def experiment_dataset(name, all_swarm_parameters, random_seed = 17):
    # Step 1: grab the dataset and preprocess it
    # Step 2: run the four machine learning models
    # Step 3: run the base xai functions on the dataset
    # Step 4: run the swarm approach on the dataset
    # Step 5: get values with the six metrics

    ##########
    #   Step 1: Grab the dataset
    ##########
    dataset_info = {}
    dataset_id = constants.datasets[name]
    random.seed(random_seed)


    X_preprocessed, y_preprocessed = data_preprocessing.process_openml_dataset(dataset_index = dataset_id)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_preprocessed, test_size = .2)

    dimension = X_train.shape[1]

    all_swarm_parameters["pso"]["dimension"] = dimension
    all_swarm_parameters["bat"]["dimension"] = dimension
    all_swarm_parameters["abc"]["dimension"] = dimension


    X_preprocessed_shape = X_preprocessed.shape

    dataset_info["id"] = dataset_id
    dataset_info["name"] = name
    dataset_info["num_features"] = X_preprocessed_shape[1]
    dataset_info["num_instances"] = X_preprocessed_shape[0]


    ##########
    #   Step 2: Run the models
    ##########
    lr_model, svm_model, rf_model = model_utils.run_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

    model_names = ["logistic_regression", "support_vector_machine", "random_forest"] # names for dictionary
    models = [lr_model, svm_model, rf_model] # list of models
    experiment_results = {} # empty results


    # loop through all the models
    for model_num in range(len(models)):

        ##########
        #   Step 3: Run the Base XAI functions on the dataset and Run the Swarm Approach on the dataset
        ##########

        # dictionary in the form: {(string) name of approach: (tuple) (explanation, shap_values, time_consumption)}
        base_xai_dict = base_xai_study(X_preprocessed = X_test, y_preprocessed = y_test, clf = models[model_num])

        ##########
        #   Step 4: 
        ##########
        
        # dictionary in the form: {(string) name of approach: (dictionary) {(string) measurement: (numerical) value}}
        # swarm_xai.run_swarm_approach(X_test = X_test, y_test = y_test, num_trials = 10)

        swarm_xai_dict = swarm_xai_study(X_test = X_test, y_test = y_test, model = models[model_num], num_trials = 10, all_swarm_parameters = all_swarm_parameters)


        # ##########
        # #   Step 5: Get the Metrics of both base and optimized
        # ##########

        # # dictionary that stores metrics of base_xai approaches

        experiment_results[model_names[model_num]] = metrics.calculate_metrics_of_model(X = X_test, base_xai_dict = base_xai_dict, swarm_xai_dict = swarm_xai_dict)



    data_tools.display_dataset_information(dataset_info = dataset_info)
    # data_tools.print_generic("experiment_results", experiment_results)

    ########
    #    Step 6: Add to Excel File for Efficiency
    ########

    data_tools.create_excel_sheet(dataset_info = dataset_info, experiment_results = experiment_results)

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

main()



