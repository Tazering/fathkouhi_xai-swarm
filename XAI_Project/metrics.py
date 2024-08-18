"""
This python file simply has all the metrics in a single file.

There are six metrics and all are described as separate functions.
Some may have more supplemental functions than others.
"""
import math
import pandas as pd
import numpy as np

import scipy.stats

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
n = number of samples
p = number of features (for averaging)
d = number of features (from the dataset)
exp_influence = influence of a feature of a particular instance by some explanation method
comp_influence = influence of a feature of a particular instance by the complete method
"""
def complete_method_error(n, p, d, exp_influence, comp_influence):

    # needed variables
    n = 10
    p = 5
    d = 5
    instance_sum = 0
    
    # loop through each instance
    for instance_num in range(n): # loop through instances

        feature_sum = 0

        for feature_num in range(d):
            feature_sum += abs(exp_influence[feature_num] - comp_influence[feature_num]) # get the error
        
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
Calculates the value of the metrics described in the paper for a single dataset of single model
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
def calculate_metrics_of_model(X, xai_dict, is_swarm = False):
    # common variables
    n = X.shape[0]
    output_dict = {}


    # actual algorithm
    if is_swarm: # swarm approach
        # initial setup of the dictionary
        output_dict["pso"] = {}
        output_dict["bat"] = {}
        output_dict["abc"] = {}

        computation_time_pso = xai_dict["pso"]["average_time_value"]

        output_dict["pso"]["mean_per_instance"] = mean_per_instance(computation_time = computation_time_pso, n = n)


        pass

    else: # base approach
        # initial setup of the dictionary
        output_dict["lime"] = {}
        output_dict["kernelshap"] = {}
        output_dict["spearman"] = {}
        output_dict["treeshap"] = {}
        output_dict["treeshap_approx"] = {}

        # lime
        computation_time_lime = xai_dict["lime"][2]

        output_dict["lime"]["mean_per_instance"] = mean_per_instance(computation_time = computation_time_lime, n = n)

    
    return output_dict

"""
This functions aggregates the base_xai and swarm_xai dictionaries.
"""
def aggregate_base_and_swarm_dictionaries(base_xai_dict, swarm_xai_dict):
    aggregated_dict = {}

    aggregated_dict["base_xai"] = base_xai_dict
    aggregated_dict["swarm_xai"] = swarm_xai_dict

    return aggregated_dict