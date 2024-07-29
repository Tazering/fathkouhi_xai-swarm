import pyswarms as ps
import numpy as np
from colorama import init, Fore, Back, Style

"""
This class is the XAI class that uses particle swarm optimization instead of
the traditional approaches.
"""

class XAI:

    def __init__(self, model_predict, sample, size, no_pso, no_iteration, lb, up, features_list):
        self.size = size
        self.model_predict = model_predict
        self.loss = 0
        self.sample = sample
        self.optimizer_param = {'lb': lb, 'up': up, 'no_pso': no_pso, 'no_iteration': no_iteration}
        self.options = {'c1': 2, 'c2': 2, 'w': 1}
        self.features_list = features_list

    def explainer_func(self, solutions):
        return np.sum(self.sample * solutions, axis=1)

    # calculates the difference between the predicted model and the
    # explainer model
    def cost_eval(self, solutions):
        return (self.model_predict - self.explainer_func(solutions))**2

    # actual algorithm
    def XAI_swarm_Invoke(self):
        # sets the max and min x sizes
        x_max = self.optimizer_param['up'] * np.ones(self.size)
        x_min = -1 * x_max
        bounds = (x_min, x_max)

        # the swarm optimizer
        optimizer = ps.single.GlobalBestPSO(n_particles= self.optimizer_param['no_pso'],
                                            dimensions = self.size,
                                            options = self.options,
                                            bounds = bounds)

        # optimizes the parameters using swarm
        cost, pos = optimizer.optimize(self.cost_eval, iters= self.optimizer_param['no_iteration'],verbose=False)

        # prints the costs and positions
        print(Style.BRIGHT + Fore.CYAN + 'Cost value: ', Style.BRIGHT + Fore.YELLOW + str(cost))
        print(Style.BRIGHT + Fore.CYAN + 'Explainer model prediction: ', Style.BRIGHT + Fore.YELLOW + str(pos.dot(self.sample.T)),'\n')
        print(Style.BRIGHT + Fore.RED + 'Contribution of features: ')
        for i in range(len(self.features_list)):
            print(Style.BRIGHT + Fore.BLUE + self.features_list[i],' : ', Style.BRIGHT + Fore.GREEN + str(pos[i]))