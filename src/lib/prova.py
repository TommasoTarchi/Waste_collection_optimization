import numpy as np
from deap import creator, base, tools

# Example population with four objectives to be minimized
population = [
    np.array([1.0, 2.0, 3.0, 4.0]),  # Solution 1
    #np.array([2.0, 3.0, 4.0, 1.0]),  # Solution 2
    #np.array([3.0, 1.0, 2.0, 3.0]),  # Solution 3
    #np.array([4.0, 4.0, 1.0, 2.0]),  # Solution 4
    np.array([0.3, 0.1, 0.2, 0.1]),  # Solution 5
    #np.array([2.5, 1.5, 4.5, 3.5]),  # Solution 6
    #np.array([3.5, 4.5, 1.5, 2.5]),  # Solution 7
    #np.array([2.0, 2.0, 2.0, 2.0]),  # Solution 8
]

creator.create("FitnessMinMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMinMulti)

individuals = [creator.Individual(objectives) for objectives in population]

print(individuals)

fronts = tools.sortNondominated(individuals, len(individuals), first_front_only=False)

print(fronts)
print(len(fronts))
