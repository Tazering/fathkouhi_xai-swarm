import time

import numpy as np
from colorama import Fore, Style
import SwarmPackagePy
import matplotlib.pyplot as plt

"""
size: (int) 
model_predict:
loss:
sample:
optimizer_param:
features_list:
categorical:
categorical_status:
opt
"""

class XAI:

    # initialization
    def __init__(self, model_predict, sample, size, no_pso, no_iteration, no_generation, lb, up, features_list, Categorical, Categorical_Status):
        self.size = size + 1 
        self.model_predict = model_predict
        self.loss = 0
        self.sample = np.array(np.copy(sample).tolist() + [1])
        self.optimizer_param = {'lb': lb, 'up': up, 'no_pso': no_pso, 'no_iteration': no_iteration, 'no_generation': no_generation}
        self.features_list = features_list
        self.Categorical = Categorical
        self.Categorical_Status = Categorical_Status
        self.optimizer_type, self.optimizer_param['no_iteration'], self.optimizer_param['no_generation'] = self.Select_Optimizer()
        self.Beta = 0

    # prints for use

    def Select_Optimizer(self):
        print('plz select the optimizer: ')
        print('1- Firefly Algorithm')
        print('2- Bat Algorithm')
        print('3- Artificial Bee Algorithm')
        print('4- Cat Algorithm')
        print('5- Chicken Swarm Optimization')
        choice = input()

        if choice == '1':
            return 1, int(input('plz enter number of iteration: ')), int(input('plz enter number of generation: '))
        elif choice == '2':
            return 2, int(input('plz enter number of iteration: ')), int(input('plz enter number of generation: '))
        elif choice == '3':
            return 3, int(input('plz enter number of iteration: ')), int(input('plz enter number of generation: '))
        elif choice == '4':
            return 4, int(input('plz enter number of iteration: ')), int(input('plz enter number of generation: '))
        elif choice == '5':
            return 5, int(input('plz enter number of iteration: ')), int(input('plz enter number of generation: '))
        else:
            return -1


    # run the explainer function
    def explainer_func(self, solutions):
        return self.sample.T.dot(solutions)

    # calculate the cost, J
    def cost_eval(self, solutions):
        return (self.model_predict - self.explainer_func(solutions))**2


    # actual process
    def XAI_swarm_Invoke(self):

        # creates an array that stores the upperbound of the x values
        x_max = self.optimizer_param['up'] * np.ones(self.size)
        x_min = -1 * x_max
        optimizer = None
        time_consumption = []
        Avg_cost = []
        min_cost = np.inf
        best_pos = None

        # iterate through generations
        for i in range(self.optimizer_param['no_generation']): 
            t1 = t2 = 0

            # the different optimizer algorithms
            if self.optimizer_type == 1: # firefly
                t1 = time.time_ns()
                optimizer = SwarmPackagePy.fa(self.optimizer_param['no_pso'], self.cost_eval, x_min, x_max, self.size,
                                               self.optimizer_param['no_iteration'])
                t2 = time.time_ns()
            elif self.optimizer_type == 2: # bat
                t1 = time.time_ns()
                optimizer = SwarmPackagePy.ba(self.optimizer_param['no_pso'], self.cost_eval, x_min, x_max, self.size,
                                               self.optimizer_param['no_iteration'])
                t2 = time.time_ns()
            elif self.optimizer_type == 3: # abc
                t1 = time.time_ns()
                # print(self.sample)
                optimizer = SwarmPackagePy.aba(self.optimizer_param['no_pso'], self.cost_eval, x_min, x_max, self.size,
                                               self.optimizer_param['no_iteration'])
                t2 = time.time_ns()
            elif self.optimizer_type == 4: # cat
                t1 = time.time_ns()
                optimizer = SwarmPackagePy.ca(self.optimizer_param['no_pso'], self.cost_eval, x_min, x_max, self.size,
                                               self.optimizer_param['no_iteration'])
                t2 = time.time_ns()
            elif self.optimizer_type == 5: # chicken swarm
                t1 = time.time_ns()
                optimizer = SwarmPackagePy.chso(self.optimizer_param['no_pso'], self.cost_eval, x_min, x_max, self.size,
                                               self.optimizer_param['no_iteration'])
                t2 = time.time_ns()
            
            # grabs the best value from the optimizer algorithm
            pos = optimizer.get_Gbest()
            cost = self.cost_eval(pos)

            # updates the lowest cost value
            if cost < min_cost:
                min_cost = cost
                best_pos = pos
            time_consumption.append(t2 - t1) # grabs the overlapsed time
            Avg_cost.append(cost) # append the costs

        # print the best_pos
        print("====Best Position Variable====\n" + str(best_pos))
        print("Size of best_pos:", len(best_pos))


        if self.Categorical_Status:
            self.Interpret(best_pos, True)
        else:
            self.Interpret(best_pos, False)
        
        # prints the resulting values
        print(Style.BRIGHT + Fore.CYAN + 'Average time value: ', Style.BRIGHT + Fore.YELLOW + str(np.mean(time_consumption) * 10**-9))
        print(Style.BRIGHT + Fore.CYAN + 'minimum Cost value: ', Style.BRIGHT + Fore.YELLOW + str(min_cost))
        print(Style.BRIGHT + Fore.CYAN + 'Average Cost value: ', Style.BRIGHT + Fore.YELLOW + str(np.mean(Avg_cost)))
        print(Style.BRIGHT + Fore.CYAN + 'Explainer model prediction: ', Style.BRIGHT + Fore.YELLOW + str(np.array(best_pos).dot(np.array(self.sample).T)),'\n')
        print(Style.BRIGHT + Fore.CYAN + 'Local fidelity measure: ', Style.BRIGHT + Fore.YELLOW + str(np.abs(self.model_predict - np.array(best_pos).dot(np.array(self.sample).T))),'\n')
        #
        # print(Style.BRIGHT + Fore.RED + 'Feature contribution of best solution: ')
        #
        # print(Style.BRIGHT + Fore.BLUE + 'Beta0', ' : ', Style.BRIGHT + Fore.GREEN + str(best_pos[-1]))
        #
        # for i in range(len(self.features_list)):
        #     print(Style.BRIGHT + Fore.BLUE + self.features_list[i],' : ', Style.BRIGHT + Fore.GREEN + str(best_pos[i] * self.sample[i]))


    """
    Interprets

    """
    def Interpret(self, best_pos, Categorical_Auth):
        Contribute = []
        if Categorical_Auth:
            begin_feature = self.Categorical['Begin_Categorical'] # int
            for i in range(begin_feature):
                Contribute.append(best_pos[i])

            for i in self.Categorical['Categorical_Index']:
                Categorical_var = 0
                for j in range(begin_feature,begin_feature + i):
                    Categorical_var += best_pos[j]
                begin_feature += i
                Contribute.append(Categorical_var)
        else:
            Contribute = list(best_pos[0:-1])
        self.Beta = best_pos[-1]
        # self.features_list.append('Beta 0')

        # does the negative/positive distinguisher
        Negative, Positive = self.Neg_Positive_Distinguisher(Contribute)

        # plots into a donut graph
        fig, axes = plt.subplots(2)
        self.Donut([float(abs(Contribute[i])) for i in Positive],[self.features_list[i] + ': ' + str(abs(Contribute[i])) for i in Positive],'Positive', axes[0])
        self.Donut([float(abs(Contribute[i])) for i in Negative],[self.features_list[i] + ': -' + str(abs(Contribute[i])) for i in Negative],'Negative', axes[1])
        plt.text(x = 2.2,y = 3.4,s='Actual prediction: ' + str(self.model_predict) + '\n' + 'approximate prediction: ' +
                   str(np.array(best_pos).dot(np.array(self.sample).T)) + '\n' + 'local fidelity: ' +
                   str(np.abs(self.model_predict - np.array(best_pos).dot(np.array(self.sample).T))),size=12,
                    bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       ))
        plt.show()

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

    # I think this creates a donut plot
    def Donut(self,Contribute, Labels, Effect_Type, plt):
        # data = Contribute
        # recipe = Labels
        x = Labels
        y = np.array(Contribute)

        patches, texts = plt.pie(y, startangle=90, radius=1.2)
        labels = x
        print(labels)
        plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),
                   fontsize=8)
        plt.set_title(Effect_Type + ' effect on prediction')
        # plt.show()


        # fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        # wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
        #
        # bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        # kw = dict(arrowprops=dict(arrowstyle="-"),
        #           bbox=bbox_props, zorder=0, va="center")
        #
        # for i, p in enumerate(wedges):
        #     ang = (p.theta2 - p.theta1) / 2. + p.theta1
        #     y = np.sin(np.deg2rad(ang))
        #     x = np.cos(np.deg2rad(ang))
        #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        #     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
        #     ax.annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
        #                 horizontalalignment=horizontalalignment, **kw)
        #
        # ax.set_title('Features which has ' + Effect_Type + ' effects on prediction')
        # plt.show()