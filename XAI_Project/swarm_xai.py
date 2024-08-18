import time

import numpy as np
from colorama import Fore, Style
import SwarmPackagePy
import matplotlib.pyplot as plt
import helpful_utils.data_tools as data_tools

"""
size: (int) 
model_predict: the complex model in need of being predicted
loss: the loss of running the model
sample:
optimizer_param:
features_list:
categorical:
categorical_status:
opt
"""

class XAI:

    """
    The init function for the class.
    model_predict: the complex model of interest
    sample: the particular datapoint of interest
    size:
    num_pso: number of particles
    num_iteration: the number of iterations for the swarm algorithm
    num_trials: the number of trials
    lower_bound: lower bound
    upper_bound: upper bound
    numerical_features: numerical features list
    Categorical: categorical features
    Categorical_Status: boolean for whether to use categorical features or not
    """
    def __init__(self, model_predict, sample, size, num_pso, num_iteration, num_trials, lower_bound, upper_bound, numerical_features, Categorical, Categorical_Status, optimizer_type):
        self.size = size + 1 
        self.model_predict = model_predict
        self.loss = 0
        self.sample = np.array(np.copy(sample).tolist() + [1])
        self.optimizer_param = {'lb': lower_bound, 'upper_bound': upper_bound, 'num_pso': num_pso, 'num_iteration': num_iteration, 'num_trials': num_trials}
        self.numerical_features = numerical_features
        self.Categorical = Categorical
        self.Categorical_Status = Categorical_Status
        self.optimizer_type = optimizer_type
        self.Beta = 0

    """
    This function allows the user to choose optimizer, number of iterations, and number of trials desired
    to run the experiment.
    """
    def Select_Optimizer(self):

        # print the optimizer choice
        print('Select an Optimizer: ')
        print('1- PSO Algorithm')
        print('2- Bat Algorithm')
        print('3- Artificial Bee Algorithm')
        choice = input()

        # variables to store the message prompts
        num_iterations_msg = "Enter the number of iterations: "
        num_trials_msg = "Enter the number of trials: "

        # prompt for number of iterations and number of trials
        if choice == '1':
            return 1, int(input(num_iterations_msg)), int(input(num_trials_msg))
        elif choice == '2':
            return 2, int(input(num_iterations_msg)), int(input(num_trials_msg))
        elif choice == '3':
            return 3, int(input(num_iterations_msg)), int(input(num_trials_msg))
        else:
            exit()


    """
    This function computes the dot product of a sample and the solution that was found.

    f = sample.T <dot> solution
    """
    def explainer_func(self, solutions):
        return self.sample.T.dot(solutions)

    """
    This function calculates the cost between the explainer_function and the
    predicted function. 
    
    cost = (predicted - explained)^2
    """
    def cost_eval(self, solutions):
        return (self.model_predict - self.explainer_func(solutions))**2


    """
    This functions actually invokes the swarm and xai for computation.

    """
    def XAI_swarm_Invoke(self):

        # creates an array that stores the upperbound of the x values
        x_max = self.optimizer_param['upper_bound'] * np.ones(self.size)

        # stores other important parameters   
        x_min = -1 * x_max
        optimizer = None
        time_consumption = []
        Avg_cost = []
        min_cost = np.inf
        best_pos = None

        # iterate through trials
        for i in range(self.optimizer_param['num_trials']): 
            t1 = t2 = 0

            # the different optimizer algorithms
            if self.optimizer_type == 1: # PSO
                t1 = time.time_ns()
                optimizer = SwarmPackagePy.pso(self.optimizer_param['num_pso'], self.cost_eval, x_min, x_max, self.size,
                                               self.optimizer_param['num_iteration'])
                t2 = time.time_ns()
            elif self.optimizer_type == 2: # bat
                t1 = time.time_ns()
                optimizer = SwarmPackagePy.ba(self.optimizer_param['num_pso'], self.cost_eval, x_min, x_max, self.size,
                                               self.optimizer_param['num_iteration'])
                t2 = time.time_ns()
            elif self.optimizer_type == 3: # abc
                t1 = time.time_ns()
                # print(self.sample)
                optimizer = SwarmPackagePy.aba(self.optimizer_param['num_pso'], self.cost_eval, x_min, x_max, self.size,
                                               self.optimizer_param['num_iteration'])
                t2 = time.time_ns()
            
            # grabs the best value from the optimizer algorithm
            pos = optimizer.get_Gbest()
            # data_tools.print_variable("pos", pos)
            cost = self.cost_eval(pos)

            # updates the lowest cost value
            if cost < min_cost:
                min_cost = cost
                best_pos = pos
            time_consumption.append(t2 - t1) # grabs the overlapsed time
            Avg_cost.append(cost) # append the costs

        # account for categorical data
        if self.Categorical_Status:
            contribute = self.Interpret(best_pos, True)
        else:
            contribute = self.Interpret(best_pos, False)
        
        output_dict = {
            "average_time_value": np.mean(time_consumption) * 10**-9,
            "minimum_cost_value": min_cost,
            "average_cost_value": np.mean(Avg_cost),
            "explainer_model_prediction": np.array(best_pos).dot(np.array(self.sample).T),
            "local_fidelity_measure": np.abs(self.model_predict - np.array(best_pos).dot(np.array(self.sample).T)),
            "contribute": contribute
        }

        return output_dict

    """
    Interprets
    best_pos: the best position found
    Categorical_Auth: (boolean) whether to account for categorical variables or not

    """
    def Interpret(self, best_pos, Categorical_Auth):
        Contribute = []
        if Categorical_Auth:
            begin_feature = self.Categorical['Begin_Categorical'] # int
            for i in range(begin_feature): # loops from 0 to begin_feature
                Contribute.append(best_pos[i]) # appends to contribute variable

            for i in self.Categorical['Categorical_Index']: # loop through all categorical indices
                Categorical_var = 0
                for j in range(begin_feature,begin_feature + i): # loops from begin_feature to begin_feature + i
                    Categorical_var += best_pos[j] # adds up values in best_pos
                begin_feature += i # increments begin_feature by i
                Contribute.append(Categorical_var) # appends summed up positions to contribute
        else:
            Contribute = list(best_pos[0:-1]) # just gets everything
        self.Beta = best_pos[-1]

        # does the negative/positive distinguisher
        Negative, Positive = self.Neg_Positive_Distinguisher(Contribute)

        return Contribute

    """
    This function distinguishes between negative and positive values. This is done by checking if the 
    value is greater than or less than 0
    """
    def Neg_Positive_Distinguisher(self, Contribute):
        Negative = []
        Positive = []
        temp_index = 0

        # store whether contributions are negative or positive
        for i in Contribute:
            if i < 0:
                Negative.append(temp_index)
            else:
                Positive.append(temp_index)
            temp_index += 1
        return Negative, Positive