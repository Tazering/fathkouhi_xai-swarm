"""
This python file simply has all the metrics in a single file.

There are six metrics and all are described as separate functions.
Some may have more supplemental functions than others.
"""
import math
import pandas as pd
import numpy as np
from numpy import linalg as LA

import scipy.stats
import helpful_utils.data_tools as data_tools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    
    if isinstance(exp_influence, list):
        exp_influence = np.array(exp_influence)
    
    if not isinstance(comp_influence, np.ndarray):
        comp_influence = comp_influence.to_numpy()

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
def auc(exp_influence):

    if isinstance(exp_influence, pd.DataFrame):
        exp_influence = exp_influence.to_numpy()

    # grab cumulative importance
    cum_importance_abs = np.abs(exp_influence)
    cum_importance_mean = np.mean(cum_importance_abs, axis = 0)
    cum_importance = -np.sort(-cum_importance_mean).cumsum()

    cum_total = cum_importance_mean.sum()

    importance = np.concatenate(([0], (cum_importance / cum_total)), axis = 0)

    # perform trapezoidal integration with dx being 1/(d-1)
    auc_value = np.trapz(importance, dx = 1 / (len(importance) - 1))

    # for feature_num in range(d - 1): # loop through d - 1 features
    #     total += (C[feature_num] + C[feature_num + 1])

    return auc_value




"""
Metric 4: Robustness
data = the actual data
influences = the influences of the features of a data
epsilon = threshold for set N
"""
def robustness(data, influences, epsilon):
    max_value = -99999999999

    m = data.shape[0] # number of instances

    if isinstance(data, pd.DataFrame): # convert to np array if needed
        data = data.to_numpy()

    if isinstance(influences, pd.DataFrame): # convert to np array if needed
        influences = influences.to_numpy()
    
    if isinstance(data, list):
        data = np.array(data)
    
    if isinstance(influences, list):
        influences = np.array(influences)

    for instance_num in range(m): # loop through instances
        
        for second_instance_num in range(m): # loop through instances again

            vector_difference = data[instance_num] - data[second_instance_num] # calculates the difference between two vectors
            if (instance_num != second_instance_num) and (LA.norm(vector_difference) <= epsilon): # if the value is within the N set
                instance_to_instance_value = (LA.norm(influences[instance_num] - influences[second_instance_num])) / (LA.norm(vector_difference))

                # update the max value
                if instance_to_instance_value > max_value:
                    max_value = instance_to_instance_value

                # print(f"{np_data[second_instance_num]} works with value of {LA.norm(np_data[instance_num] - np_data[second_instance_num])}")

    return max_value


"""
Metric 5: Readability
data = the original/preprocessed dataset
explanation = the explanations tensor
d = the number of features
"""
def readability(data, explanations):
    n = data.shape[1] # number of features

    # convert things to numpy
    if isinstance(data, pd.DataFrame): 
        data = data.to_numpy()
    
    if isinstance(explanations, pd.DataFrame):
        explanations = explanations.to_numpy()
    
    if isinstance(data, list):
        data = np.array(data)
    
    if isinstance(explanations, list):
        explanations = np.array(explanations)

    # calculate the metric
    spearman_total = 0

    for feature_num in range(n): # loop through features
        feature_vector = grab_feature_vector(data = data, feature_num = feature_num)
        explainable_feature_vector = grab_feature_vector(data = explanations, feature_num = feature_num)

        spearman_coefficient = abs(scipy.stats.spearmanr(feature_vector, explainable_feature_vector).statistic)
        spearman_total += spearman_coefficient

    return spearman_total / n


"""
Metric 6: Clusterability 
explanations = list of explanations
"""
def clusterability(explanations, num_clusters = 2):

    # convert things to numpy
    if isinstance(explanations, pd.DataFrame):
        explanations = explanations.to_numpy()

    if isinstance(explanations, list):
        explanations = np.array(explanations)

    d = explanations.shape[1] # number of features
    total = 0
    
    # calculate clusterability
    for feature_num in range(d - 1): # loop through all feature
        first_feature_vector = grab_feature_vector(data = explanations, feature_num = feature_num)
        second_feature_vector = grab_feature_vector(data = explanations, feature_num = feature_num + 1)

        coord_pair = np.column_stack((first_feature_vector, second_feature_vector)) # grab the coordinates by stacking the feature vectors

        kmeans = KMeans(n_clusters = num_clusters, n_init = "auto").fit(coord_pair) # run a clustering algorithms: k-means

        score = silhouette_score(coord_pair, kmeans.labels_) # calculate the silhouette score

        total += score # accumulate score

        # data_tools.print_generic("labels", kmeans.labels_)
        # data_tools.print_generic("first_feature_vector", first_feature_vector)
        # data_tools.print_generic("second_feature_vector", second_feature_vector)
        # data_tools.print_generic("coord_pair", coord_pair)
    
    # data_tools.print_generic("total", total)



    # for feature_num in range(d - 1): # loop through all features
    #     total += S(K(explanations[:, feature_num], explanations[:, feature_num + 1])) # summations
    
    scaling_factor = 2 / (d * (d - 1))

    return scaling_factor * total

# grabs a feature vector
def grab_feature_vector(data, feature_num):
    feature_vector = []

    m = data.shape[0]

    for instance_num in range(m): # loop through instances
        datapoint = data[instance_num]

        feature_vector.append(datapoint[feature_num]) # add to feature vector

    return feature_vector

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
def calculate_metrics_of_model(X, base_xai_dict, swarm_xai_dict, epsilon = .3):
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

        # complete error 
        try:
            output_dict["base_xai"][base_xai_name]["complete_error"] = complete_method_error(n = n, p = d, d = d, 
                                                                                            exp_influence = base_xai_dict[base_xai_name][1], 
                                                                                            comp_influence = base_xai_dict["complete"][1]) # grabs the shap values or influences
        except:
            output_dict["base_xai"][base_xai_name]["complete_error"] = None

        # auc
        output_dict["base_xai"][base_xai_name]["auc"] = auc(exp_influence = base_xai_dict[base_xai_name][1])

        # robustness
        # output_dict["base_xai"][base_xai_name]["robustness"] = robustness(data = X, influences = base_xai_dict[base_xai_name][1], epsilon = epsilon)

        # readability
        output_dict["base_xai"][base_xai_name]["readability"] = readability(data = X, explanations = base_xai_dict[base_xai_name][1])

        # clusterability
        output_dict["base_xai"][base_xai_name]["clusterability"] = clusterability(explanations = base_xai_dict[base_xai_name][1])


    # swarm approaches
    approach_names = ["pso", "bat", "abc"]
    for swarm_xai_name in approach_names:
        # initiate to empty dictionary
        output_dict["swarm_xai"][swarm_xai_name] = {}

        # mean per instance
        computation_time = swarm_xai_dict[swarm_xai_name]["average_time_value"]
        output_dict["swarm_xai"][swarm_xai_name]["mean_per_instance"] = mean_per_instance(computation_time = computation_time, n = n)

        # complete method
        output_dict["swarm_xai"][swarm_xai_name]["complete_error"] = complete_method_error(n = n, p = d, d = d, 
                                                                                    exp_influence = swarm_xai_dict[swarm_xai_name]["contribute"], 
                                                                                    comp_influence = base_xai_dict["complete"][1])
        
        # auc
        output_dict["swarm_xai"][swarm_xai_name]["auc"] = auc(exp_influence = swarm_xai_dict[swarm_xai_name]["contribute"])

        # robustness
        output_dict["swarm_xai"][swarm_xai_name]["robustness"] = robustness(data = X, influences = swarm_xai_dict[swarm_xai_name]["contribute"], epsilon = epsilon)

        # readability
        output_dict["swarm_xai"][swarm_xai_name]["readability"] = readability(data = X, explanations = swarm_xai_dict[swarm_xai_name]["contribute"])

        # clusterability
        output_dict["swarm_xai"][swarm_xai_name]["clusterability"] = clusterability(explanations = swarm_xai_dict[swarm_xai_name]["contribute"])

    return output_dict