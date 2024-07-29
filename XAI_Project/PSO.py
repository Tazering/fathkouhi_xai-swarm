import numpy as np

"""
This python class stores the PSO object states and behaviors. The positions
and velocities are recorded as separate arrays. The fitness are also recorded
another array.
"""

class PSO:

    """
    no_bat: the number of bats/particles
    dimension: the dimensions or number of coordinates of a bat/particle
    no_generations:
    lb: lowerbound
    ub: upperbound

    """
    def __init__(self, no_bat, dimension, no_generation, lb, ub, model_predict, sample):
        self.loss = 0
        self.sample = sample
        self.model_predict = model_predict
        self.no_bat = no_bat
        self.fitness = [0.0] * self.no_bat # fitness for each bat/particle
        self.p_fitness = [0.0] * self.no_bat
        self.best_index = 0 # best particle
        self.dimension = dimension
        self.best_history = []

        # list of velocities
        self.v = [[0.0 for i in range(self.dimension)] for j in range(self.no_bat)]

        # particle is a 2D array with the dimensions: dimension x no_bat
        self.particle = [[0.0 for i in range(self.dimension)] for j in range(self.no_bat)]
        self.p_best_particle = [[0.0 for i in range(self.dimension)] for j in range(self.no_bat)]

        # upperbound and lowerbound are 2D arrays of the dimension:
        # dimension x no_bat
        self.upperbound = [[0.0 for i in range(self.dimension)] for j in range(self.no_bat)]
        self.lowerbound = [[0.0 for i in range(self.dimension)] for j in range(self.no_bat)]

        self.fitness_minimum = 0.0 # set a lowerbound for fitness value
        self.best = [0.0] * self.dimension # the best locations
        self.no_generation = no_generation # the number of iterations

        # bounds
        self.lb = lb
        self.ub = ub

        # cognitive and social coefficients (explore vs exploit)
        self.c1 = 2
        self.c2 = 2

        self.w = 1 # inertia weight constants

        # velocity min and max
        self.v_min = -2
        self.v_max = 2


        self.itr = 0
        self.p_no = 0


    # grabs the best particle
    def best_particle(self):
        i = 0
        j = 0

        # iterate through bats
        for i in range(self.no_bat):
            if self.fitness[i] < self.fitness[j]:
                j = i

        for i in range(self.dimension):
            self.best[i] = self.particle[j][i]

        self.fitness_minimum = self.fitness[j]
        self.best_index = j


    """
    This function initializes the process.
    """
    def process_init(self):
        for i in range(self.no_bat): # loops through each bat/particle
            for j in range(self.dimension): # loops through the dimensions
                # initializes the upper and lower bound matrices
                self.lowerbound[i][j] = self.lb
                self.upperbound[i][j] = self.ub

        for i in range(self.no_bat): # loops through each bat/particle
            print('initialization: ',i)
            for j in range(self.dimension): # loops through dimensions of
                # each particle
                random = np.random.uniform(0, 1)

                # initiates velocities
                self.v[i][j] = self.v_min + (self.v_max - self.v_min) * random

                # initiates positions
                self.particle[i][j] = self.lowerbound[i][j] + (self.upperbound[i][j] - self.lowerbound[i][j]) * random

            # calculates initial cost
            cost_eval = self.Cost_eval(self.particle[i])
            self.fitness[i] = cost_eval

        self.p_best_particle = np.copy(self.particle)
        self.p_fitness = np.copy(self.fitness)
        self.best_particle()


    """
    This function sets the bounds of a given value X. In other words,
    the value X will always be between the upperbound and lowerbound. 
    In this particular function, if the X value exceeds the upperbound, then 
    the X value is set to the upper bound. If the X value subceeds the 
    lowerbound, then X will be set to the lowerbound.
    """
    def normalization_particle(self, X):
        if X > self.ub:
            X = self.ub
        if X < self.lb:
            X = self.lb
        return X

    """
    This function is similar to the one above except it bounds the v (velocity)
    value.
    """
    def normalization_velocity(self, v):
        if v > self.v_max:
            v = self.v_max
        if v < self.v_min:
            v = self.v_min
        return v

    """
    This function actually calculates the optimal value via PSO approach.
    """
    def process(self):
        # initialize the process
        self.process_init()

        # for every iteration
        for n in range(self.itr, self.no_generation):
            print('iteration: ', n)

            # for each bat
            for i in range(self.p_no, self.no_bat):
                for j in range(self.dimension):

                    # updates the velocity
                    self.v[i][j] = self.w * self.v[i][j] + (self.best[j] - self.particle[i][j]) * self.c2 * np.random.rand(1) + \
                                   (self.p_best_particle[i][j] - self.particle[i][j]) * self.c1 * np.random.rand(1)

                    # updates the position using the new velocity and
                    # normalizes both the position and velocity
                    self.particle[i][j] = np.float(self.particle[i][j] + self.v[i][j])
                    self.v[i][j] = self.normalization_velocity(self.v[i][j])
                    self.particle[i][j] = self.normalization_particle(self.particle[i][j])

                    # calculate the fitness value using cost evaluation
                    self.fitness[i] = self.Cost_eval(self.particle[i])

                    # update the fitness values
                    if self.fitness[i] < self.p_fitness[i]:
                        for j in range(self.dimension):
                            self.p_best_particle[i][j] = self.particle[i][j]
                        self.p_fitness[i] = self.fitness[i]

                    # double checks the fitness minimum
                    if self.fitness[i] < self.fitness_minimum:
                        self.fitness_minimum = self.fitness[i]
                        self.best_index = i
                        for j in range(self.dimension):
                            self.best[j] = self.particle[i][j]

            self.best_history.append(self.fitness_minimum)
            # print('iter: ' + str(n) + ' minimum : ', self.fitness_minimum, 'predict sample: ', np.dot(self.best, self.sample.T))
            self.p_no = 0

        return self.best

    """
    This function calculates the loss of the solution.   
    """
    def Cost_eval(self, solution):
        self.loss = -1 * np.sum(solution * np.sin(np.sqrt(np.abs(solution))))
        # self.loss = (self.model_predict - np.dot(solution, self.sample.T)) ** 2
        print('loss is: ', self.loss)
        return self.loss