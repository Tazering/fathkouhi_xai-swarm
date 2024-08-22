"""
This python file simply has all the metrics in a single file.

There are six metrics and all are described as separate functions.
Some may have more supplemental functions than others.
"""
import math
import pandas as pd
import numpy as np

import scipy.stats
import helpful_utils.data_tools as data_tools

##################################################################
#   Methods
##################################################################


"""
Metric 1: Mean per Instance
   Parameters:
       computation_time: the time for the xai to compute local dataset
       n: number of instances of the dataset
"""
def mean_per_instance(computation_time, n):
    return computation_time/n


"""
Metric 2: Error with Complete Method
@Parameters
    n: int 
        number of examples
    p: int
        number of features (for averaging)
    d: int 
        number of features (from the dataset)
    exp_influence: list
        influence of a feature of a particular instance by some explanation method
    comp_influence: list 
        influence of a feature of a particular instance by the complete method
"""
def complete_method_error(n, p, d, exp_influence, comp_influence):

    # needed variables
    instance_sum = 0

    # convert to numpy arrays
    if not isinstance(exp_influence, np.ndarray) and not isinstance(exp_influence, list):
        exp_influence = exp_influence.to_numpy()
    
    if not isinstance(comp_influence, np.ndarray):
        comp_influence = comp_influence.to_numpy()

    # data_tools.print_generic("exp_influence", exp_influence[0][3])
    # data_tools.print_and_end(comp_influence[0][3])

    # loop through each instance
    for instance_num in range(n): # loop through instances

        feature_sum = 0

        for feature_num in range(d):
            # print("\n\n", instance_num, feature_num, "\n\n")

            feature_sum += abs(exp_influence[instance_num][feature_num] - comp_influence[instance_num][feature_num]) # get the error
        
        instance_sum += feature_sum # add the feature sum
    

    return instance_sum / (n * p) # average error out



"""
Metric 3: Area under curve
d = number of features
C = cumulative imoprtance proportion vector
"""
def auc(d, C):

    total = 0

    for feature_num in range(d - 1): # loop through d - 1 features
        total += (C[feature_num] + C[feature_num + 1])

    return total / (2 * d)

"""
Metric 4: Robustness
data = the actual data
influences = the influences of the features of a data
epsilon = threshold for set N
"""
def robustness(data, influences, epsilon):
    max = -99999999999

    for feature_num in range(data.shape[0] - 1): # loop through all instances
        if abs(data[feature_num] - data[feature_num + 1]) <= epsilon: # check that it is within the set of numbers
            temp_max = np.linalg.norm(influences[feature_num] - influences[feature_num + 1]) / np.linalg.norm(data[feature_num] - data[feature_num + 1]) # run the equation

            if temp_max > max: # check for largest values
                max = temp_max

    return max


"""
Metric 5: Readability
data = the original/preprocessed dataset
explanation = the explanations tensor
d = the number of features
"""
def readability(data, explanation, d):
    total = 0

    for feature_num in range(d): # loop through each feature
        total += abs(scipy.stats.spearmanr(data[:, feature_num] - explanation[:, feature_num])) # calculate spearman correlation

    return total / d

"""
Metric 6: Clusterability 
dataset = the actual dataset preprocessed
n = number of instances
d = number of features
K = clustering function
S = evaluation function
explanations = list of explanations
"""
def clusterability(dataset, d, K, S, explanations):
    total = 0

    for feature_num in range(d - 1): # loop through all features
        total += S(K(explanations[:, feature_num], explanations[:, feature_num + 1])) # summations
    
    scaling_factor = 2 / (d * (d - 1))

    return scaling_factor * total

##################################################################
#   Functions that are used for the experiment directly
##################################################################

"""
Calculates the value of the metrics described in the paper for a single model of a single dataset
for either all the base xai approaches or the swarm optimizer approaches.
@parameters
    X: pandas dataframe
        dataset of interest
    xai_dict: {(string) name of approach: (tuple) (explanation, shap_values, time_consumption)} 
              OR {(string) name of approach: (dictionary) {(string) measurement: (numerical) value}}
        dictionary that has all the base_xai information
    is_swarm: boolean
        is the dictionary a swarm or base dictionary format
        
@returns:

"""
def calculate_metrics_of_model(X, base_xai_dict, swarm_xai_dict):
    # common variables
    n = X.shape[0] # number of datapoints
    d = X.shape[1] # number of features
    output_dict = {}
    output_dict["base_xai"] = {}
    output_dict["swarm_xai"] = {}

    # base approaches
    approach_names = ["lime", "complete", "kernelshap", "spearman", "treeshap", "treeshap_approx"]
    for base_xai_name in approach_names: # go through various base_xai algorithms
        # initialize to empty dictionary
        output_dict["base_xai"][base_xai_name] = {}

        # in case of empty results of a method
        if len(base_xai_dict[base_xai_name]) == 0:
            print(f"Empty tuple for {base_xai_name}")
            continue

        # mean per instance
        computation_time = base_xai_dict[base_xai_name][2] # grabs time consumption
        output_dict["base_xai"][base_xai_name]["mean_per_instance"] = mean_per_instance(computation_time = computation_time, n = n)

        print("\n\n", base_xai_name, "\n\n")

        # complete error 
        try:
            output_dict["base_xai"][base_xai_name]["complete_error"] = complete_method_error(n = n, p = d, d = d, 
                                                                                            exp_influence = base_xai_dict[base_xai_name][1], 
                                                                                            comp_influence = base_xai_dict["complete"][1]) # grabs the shap values or influences
        except:
            output_dict["base_xai"][base_xai_name]["complete_error"] = None


    # swarm approaches
    approach_names = ["pso", "bat", "abc"]
    for swarm_xai_name in approach_names:
        # initiate to empty dictionary
        output_dict["swarm_xai"][swarm_xai_name] = {}

        # mean per instance
        computation_time = swarm_xai_dict[swarm_xai_name]["average_time_value"]
        output_dict["swarm_xai"][swarm_xai_name]["mean_per_instance"] = mean_per_instance(computation_time = computation_time, n = n)

        # complete method
        output_dict["swarm_xai"][swarm_xai_dict]["complete"] = complete_method_error(n = n, p = d, d = d, 
                                                                                    exp_influence = swarm_xai_dict[swarm_xai_name]["contribute"], 
                                                                                    comp_influence = base_xai_dict["complete"][1])


    return output_dict