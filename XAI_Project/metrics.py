"""
This python file simply has all the metrics in a single file.

There are six metrics and all are described as separate functions.
Some may have more supplemental functions than others.
"""

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
"""
def complete_method_error(n, p, d, exp_influence, comp_influence):

    # needed variables
    n = 10
    p = 5
    d = 5
    
    # loop through each instance

    return None



"""
Metric 3: Area under curve
"""
def auc():
    return None

"""
Metric 4: Robustness
"""
def robustness():
    return None

"""
Metric 5: Readability
"""
def readability():
    return None

"""
Metric 6: Clusterability 
"""
def clusterability():
    return None