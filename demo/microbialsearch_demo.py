'''
Demo of MicrobialSearch
evolving for 10 dim vectors with each element in [0,1], maximizing their means
i.e. best solution is [1,1,1,1,1,1,1,1,1,1]
'''

import numpy as np
from stochsearch import MicrobialSearch
import matplotlib.pyplot as plt

def fitness_function(individual):
    '''
    sample fitness function
    '''
    return np.mean(individual)

# defining the parameters for the evolutionary search
evol_params = {
    'num_processes' : 12, # (optional) number of proccesses for multiprocessing.Pool
    'pop_size' : 100,    # population size
    'genotype_size': 10, # dimensionality of solution
    'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
    'recomb_prob': 0.1, # fraction of population retained as is between generations
    'mutation_variance': 0.01, # mutation noise added to offspring.
}

# create evolutionary search object
ms = MicrobialSearch(evol_params)

'''OPTION 1'''
# execute the search for 100 generations
num_gens = 100
hist = np.zeros((100,100))
best = []
avg = []

for i in range(num_gens):
    ms.step_generation()
    hist[i,:] = ms.get_fitnesses()
    best.append(ms.get_best_individual_fitness())
    avg.append(ms.get_mean_fitness())

plt.pcolormesh(np.asarray(hist))
plt.xlabel('Fitness of each individual in population')
plt.ylabel('Generations')
plt.show()
plt.plot(best,label='Best')
plt.plot(avg,label='Average')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
