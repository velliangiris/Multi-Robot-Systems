import numpy as np
from  Main import Fit
import random
def fitness_function(position):
    # Example fitness function: Sphere function
    return np.sum(position ** 2)


def initialize_population(pop_size, dim, bounds):
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    return population


def evaluate_population(population):
    fitness_values = np.array([fitness_function(ind) for ind in population])
    return fitness_values


def divide_into_groups(population, fitness, PS):
    pop_size = len(population)
    num_stallions = int(np.ceil(pop_size * PS))  # Eq. (G = ⌈N × PS⌉)
    stallions_indices = np.random.choice(pop_size, num_stallions, replace=False)
    stallions = population[stallions_indices]
    remaining_indices = list(set(range(pop_size)) - set(stallions_indices))
    foals = population[remaining_indices]

    groups = []
    for i in range(num_stallions):
        group_size = len(remaining_indices) // num_stallions
        if i == num_stallions - 1:
            group = foals[i * group_size:]
        else:
            group = foals[i * group_size:(i + 1) * group_size]
        groups.append(group)

    return stallions, groups


def update_position_grazing(foal, stallion, Z, R):
    # Eq. (1): X̄ j i,G = 2Z cos(2πRZ) × (Stallionj − Xj i,G) + Stallionj
    return 2 * Z * np.cos(2 * np.pi * R * Z) * (stallion - foal) + stallion


def update_position_mating(parent1, parent2):
    # Eq. (4): Crossover = Mean
    return (parent1 + parent2) / 2


def update_position_leader(stallion, WH, Z, R):
    # Eq. (5)
    if R > 0.5:
        return 2 * Z * np.cos(2 * np.pi * R * Z) * (WH - stallion) + WH
    else:
        return 2 * Z * np.cos(2 * np.pi * R * Z) * (WH - stallion) - WH

def generate_trajectory(n_d):
    return [(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(n_d)]
def select_best_leader(stallion, group, fitness):
    group_fitness = np.array([fitness_function(member) for member in group])
    best_index = np.argmin(group_fitness)
    if fitness_function(stallion) > group_fitness[best_index]:
        return group[best_index]
    return stallion


def optimizer(dim, bounds,robots,n_r,n_d,obs):
    pop_size = n_d
    PS = 0.2
    PC = 0.13
    max_iter = n_r
    population = initialize_population(pop_size, dim, bounds)
    # fitness = evaluate_population(population)

    for iter in range(max_iter):

        fitness, PL, S = Fit.func(population, robots[iter], obs)
        stallions, groups = divide_into_groups(population, fitness, PS)
        best_solution = population[np.argmin(fitness)]

        TDR = 1 - iter / max_iter  # Eq. (3): TDR = 1 - iter / max_iter

        for i, stallion in enumerate(stallions):
            Z = np.random.rand(dim)
            for j, foal in enumerate(groups[i]):
                R = np.random.uniform(-2, 2)
                if np.random.rand() > PC:
                    # Eq. (1)
                    new_position = update_position_grazing(foal, stallion, Z, R)
                else:
                    other_group_index = (i + 1) % len(groups)
                    other_foal = groups[other_group_index][np.random.randint(len(groups[other_group_index]))]
                    # Eq. (4)
                    new_position = update_position_mating(foal, other_foal)

                groups[i][j] = np.clip(new_position, bounds[0], bounds[1])

            WH = best_solution
            R = np.random.uniform(-2, 2)
            # Eq. (5)
            new_stallion_position = update_position_leader(stallion, WH, Z, R)
            stallions[i] = np.clip(new_stallion_position, bounds[0], bounds[1])

        population = np.vstack([stallions] + groups)
        fitness = evaluate_population(population)

        # Select new best leader if better
        for i, group in enumerate(groups):
            stallions[i] = select_best_leader(stallions[i], group, fitness)

        new_best_solution = population[np.argmin(fitness)]
        if fitness_function(new_best_solution) < fitness_function(best_solution):
            best_solution = new_best_solution

    return best_solution,PL, S,np.mean(fitness)

def algm(robots,n_r,n_d,obs):
    # Example usage
    dim = 10  # Number of dimensions
    bounds = (-5.12, 5.12)  # Bounds for each dimension
    path,PL, S, fitness= optimizer(dim, bounds,robots,n_r,n_d,obs)

    return path,PL, S, fitness
