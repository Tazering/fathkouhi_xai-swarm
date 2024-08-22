"""
This python file holds all the XAI swarm approaches code.
"""

import time as time
import pandas as pd
import numpy as np

import xai_utils 
import helpful_utils.data_tools as data_tools

import SwarmPackagePy


"""
This function invokes the xai swarm approach.
@Parameters
    X_test: pandas.dataframe
        the testing input data
    Y_test: pandas.dataframe
        the testing output data
    num_trials: int
        number of trials to run
    swarm_parameters: dictionary
        dictionary that holds parameters for running an optimizer:

        Key: Value
        optimizer_number: int
            indicate which swarm optimizer to use; 1 - PSO, 2 - BAT, 3 - ABC
        num_agents: int
            the number of agents/particles/bats/bees
        upper_bound: int
            indicates the upperbound of the results
        lower_bound: int
            indicates the lowerbound of the results
        dimension: int
            dimensions of the space
        num_iterations: int
            the number of iterations of optimizing

        For PSO
    ------------------ 
        w: float
            mostly for pso; balance between range of research and consideration for suboptimal decisions
        c1: float
            ratio between cognitive and social component
        c2: float
            ratio between cognitive and social component
    ------------------
        OR for bat
    ------------------
        r0: float
            level of impulse emission
        V0: float
            volume of sound
        fmin: float
            min wave frequency
        fmax: float
            max wave frequency
        alpha: float
            constant for change a volume of sound
        csi: float
            constant for change a level of impulse emission
    ------------------
 """
def run_swarm_approach(X_test, y_test, model = None, num_trials = 10, swarm_parameters = {}):

    experiment_results = run_xai_swarm(X_test, y_test, model = model, num_trials = num_trials, swarm_parameters = swarm_parameters)
    return experiment_results

"""
TEST FUNCTION to test optimizer
"""
def test_function(solution):
    x = solution[0]
    y = solution[1]
    return pow(x, 2) + pow(y, 2) - x + 2 * y 

"""
This function runs that actual swarm optimization approach.
@Parameters
    X_test: pandas.dataframe
        the input testing data
    y_test: pandas.dataframe
        the output testing data
    num_trials: int
        the number of trials to run for the experiment
    swarm_parameters: dict
        the parameters needed to run the swarm optimization algorithms

        Key:Value
        optimizer_number: int
            indicate which swarm optimizer to use; 1 - PSO, 2 - BAT, 3 - ABC
        num_agents: int
            the number of agents/particles/bats/bees
        upper_bound: int
            indicates the upperbound of the results
        lower_bound: int
            indicates the lowerbound of the results
        dimension: int
            dimensions of the space
        num_iterations: int
            the number of iterations of optimizing
        w: float
            mostly for pso; balance between range of research and consideration for suboptimal decisions
        c1: float
            ratio between cognitive and social component
        c2: float
            ratio between cognitive and social component

@Returns
    output_dict: dict
        The results of the experiment in a dictionary format

        Key : Value
        average_time_value : float
            the average time for the explainer to explain all instances
        minimum_cost_value : float
            the lowest cost from all trials
        average_cost_value: float
            average cost from all trials
        explainer_model_prediction: float
            prediction of the surrogate model
        local_fidelity_measure: float
        contribute: list
            list of contributions of all features for each instance
"""
def run_xai_swarm(X_test, y_test, model, num_trials, swarm_parameters):
    # initial variables
    n = X_test.shape[0] # number of instances
    output_dict = {}
    min_cost = np.inf
    total_times_of_all_trials = 0
    best_pos = np.inf
    total_cost_of_all_trials = 0
    explainer_best_predictions = []

    model_pred = model.predict(X_test) # predictions

    # output_dict = {
    #         "average_time_value": np.mean(time_consumption) * 10**-9,
    #         "minimum_cost_value": min_cost,
    #         "average_cost_value": np.mean(Avg_cost),
    #         "explainer_model_prediction": np.array(best_pos).dot(np.array(self.sample).T),
    #         "local_fidelity_measure": np.abs(self.model_predict - np.array(best_pos).dot(np.array(self.sample).T)),
    #         "contribute": contribute
    #     }

    # loop through trials
    for trial in range(num_trials):

        # initialize variables
        total_time_of_all_instances = 0
        total_cost = 0
        solutions = []


        for instance_num in range(n): # loop through instances
            t1 = t2 = 0
            sample, sample_list = xai_utils.grab_sample(X_test = X_test, y_test = y_test, sample_number = instance_num) # grabs a sample

            # objective function
            def explainer_func(solution):
                return sample_list.T.dot(solution)

            def cost_eval(solution):
                return (model_pred[instance_num] - explainer_func(solution))**2


            # run the optimizers
            if swarm_parameters["optimizer_number"] == 1: # run the pso algorithm
                t1 = time.time()
                optimizer = SwarmPackagePy.pso(n = swarm_parameters["num_agents"], function = cost_eval, 
                                               lb = swarm_parameters["lower_bound"], ub = swarm_parameters["upper_bound"],
                                               dimension = swarm_parameters["dimension"], iteration = swarm_parameters["num_iterations"], w = swarm_parameters["w"], c1 = swarm_parameters["c1"], 
                                               c2 = swarm_parameters["c2"])
                t2 = time.time()
            elif swarm_parameters["optimizer_number"] == 2: # run the bat algorithm
                t1 = time.time()
                optimizer = SwarmPackagePy.ba(n = swarm_parameters["num_agents"], function = cost_eval,
                                              lb = swarm_parameters["lower_bound"], ub = swarm_parameters["upper_bound"],
                                              dimension = swarm_parameters["dimension"], iteration = swarm_parameters["num_iterations"], r0 = swarm_parameters["r0"],
                                              V0 = swarm_parameters["V0"], fmin = swarm_parameters["fmin"], fmax = swarm_parameters["fmax"],
                                              alpha = swarm_parameters["alpha"], csi = swarm_parameters["csi"])
                t2 = time.time()
            elif swarm_parameters["optimizer_number"] == 3: # run the abc algorithm
                t1 = time.time()
                optimizer = SwarmPackagePy.aba(n = swarm_parameters["num_agents"], function = cost_eval,
                                               lb = swarm_parameters["lower_bound"], ub = swarm_parameters["upper_bound"],
                                               dimension = swarm_parameters["dimension"], iteration = swarm_parameters["num_iterations"])
                t2 = time.time()

            total_time_of_all_instances += (t2 - t1) # add to the total time
            solution = optimizer.get_Gbest() # grabs the best solution of the optimizer
            total_cost += cost_eval(solution = solution) # get the cost for all instances
            solutions.append(solution)
        
        total_cost_of_all_trials += total_cost

        trial_cost = total_cost / n # cost for the particular trial

        # update the minimum cost
        min_cost, best_pos = update_min_cost_and_solution(trial_cost = trial_cost, best_pos = best_pos, min_cost = min_cost, solution = solutions)
        
        total_times_of_all_trials += total_time_of_all_instances # add to total time

    # get the prediction results from surrogate
    results = best_surrogate_pred(X_test = X_test, best_pos = best_pos)

    # shap values

    # create the output dictionary
    output_dict = {
            "average_time_value": total_times_of_all_trials / num_trials,
            "minimum_cost_value": min_cost,
            "average_cost_value": total_cost_of_all_trials / num_trials,
            "explainer_model_prediction": results,
            "local_fidelity_measure": np.abs((model_pred - results).mean()),
            "contribute": 0
        }


    return output_dict

"""
Helpful functions
"""
# updates the min_cost and the solution
def update_min_cost_and_solution(trial_cost, min_cost, best_pos, solution):
    # updates the lowest cost value of trial

    if trial_cost < min_cost:
        min_cost = trial_cost
        best_pos = solution
    
    return min_cost, best_pos

# grabs the prediction results
def best_surrogate_pred(X_test, best_pos):

    output = []

    # change X_test to a numpy array
    X_test_numpy = X_test.to_numpy()
    best_pos_numpy = np.asarray(best_pos)

    for i in range(X_test_numpy.shape[0]): # loop through each instance
        class_raw_label = np.dot(X_test_numpy[i], best_pos_numpy[i])
        class_label = round(class_raw_label + .5)
        output.append(class_label)

    return output

# get the shap values
def get_shap_values(pred, X_test):
    return None