import numpy as np
import random
from Main import Fit

class GreyLogGooseOptimization:
    def __init__(self, population_size, dimensions, max_iterations, objective_function):
        self.population_size = population_size
        self.dimensions = dimensions
        self.max_iterations = max_iterations
        self.objective_function = objective_function
        self.population = np.random.rand(population_size, dimensions)  # Initialize population
        self.best_solution = None
        self.best_fitness = float('inf')

    def update_position_exploration(self, X_p, A, C, t,lamda):

        #-------- EWMA _GGO eqn update
        return X_p *(1-A*C) +A/lamda *(X_p[t+1]-(1-lamda)*X_p[t-1])

    def update_position_exploration_alternative(self, paddles, w1, w2, w3, z):
        # Equation (4): X(t + 1) = w1 * X''Paddle1 + z * w2 * (X''Paddle2 - X''Paddle3) + (1 - z) * w3 * (X - X''Paddle1)
        return (w1 * paddles[0] + z * w2 * (paddles[1] - paddles[2]) +
                (1 - z) * w3 * (self.best_solution - paddles[0]))

    def generate_trajectory(self,n_d,m):
        return [(random.uniform(0, m), random.uniform(0, m)) for _ in range(n_d)]

    def update_position_exploitation(self, X_p, w4, b, l, r4, r5):
        # Equation (5): X(t + 1) = w4 * |X*(t) - X(t)| * e^(b * α * cos(2π * α)) + [2 * w1 * (r4 + r5)] * X*(t)
        alpha = np.random.uniform(-1, 1)
        return (w4 * np.abs(self.best_solution - X_p) *
                np.exp(b * alpha * np.cos(2 * np.pi * alpha)) +
                2 * (r4 + r5) * self.best_solution)

    def run(self,robots,n_d,m,obs):
        # --------- Genenrate Trajectory
        for uav in robots:
            uav.path = self.generate_trajectory(n_d, m)
            # -------- fitness calculation
        for t in range(1,self.max_iterations):
            F,  PL, S = Fit.func(self.population, robots[t], obs)

            for i in range(1,self.population_size-1):
                # Calculate fitness
                fitness = self.objective_function(self.population[i])

                # Update best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = self.population[i]

                # Exploration group (n1)
                if t % 2 == 0:
                    r3 = np.random.rand()
                    A = 2 * np.random.rand() - 1  # Update A
                    C = 2 * np.random.rand()  # Update C
                    lamda = 0.3

                    if r3 < 0.5:
                        if np.abs(A) < 1:
                            self.population[i] = self.update_position_exploration(self.population[i], A, C, t,lamda)
                        else:
                            paddles = [self.population[np.random.randint(self.population_size)] for _ in range(3)]
                            z = 1 - (t / self.max_iterations) ** 2
                            self.population[i] = self.update_position_exploration_alternative(paddles,
                                                                                              np.random.rand() * 2,
                                                                                              np.random.rand() * 2,
                                                                                              np.random.rand() * 2, z)
                    else:
                        self.population[i] = self.update_position_exploitation(self.population[i], np.random.rand() * 2,
                                                                               0.1, np.random.uniform(-1, 1),
                                                                               np.random.rand(), np.random.rand())

                # Exploitation group (n2)
                else:
                    # Assuming sentry solutions are determined elsewhere
                    sentries = [self.population[np.random.randint(self.population_size)] for _ in range(3)]
                    A1 = 2 * np.random.rand() - 1
                    C1 = 2 * np.random.rand()

                    X1 = sentries[0] - A1 * np.abs(C1 * sentries[0] - self.best_solution)
                    X2 = sentries[1] - A1 * np.abs(C1 * sentries[1] - self.best_solution)
                    X3 = sentries[2] - A1 * np.abs(C1 * sentries[2] - self.best_solution)

                    self.population[i] = (X1 + X2 + X3) / 3  # Equation (6): X(t + 1) = (X1 + X2 + X3) / 3

            # After each iteration, update best fitness and solution
            # print(f'Iteration {t + 1}/{self.max_iterations}, Best Fitness: {self.best_fitness}')

        return self.population,PL,S,np.mean(F)


# Example usage
def objective_function(x):
    return np.sum(x ** 2)  # Simple objective function: Sphere function

def opt(robots,n_robots,n_d,obs):
    T = n_robots  # Max iterations
    N = n_d  # Number of candidate solutions
    m = 20
    ggo = GreyLogGooseOptimization(population_size=N, dimensions=m, max_iterations=T,
                                   objective_function=objective_function)
    path,PL,S,F = ggo.run(robots,n_d,m,obs)
    return path,PL,S,F
