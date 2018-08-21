'''
Demo of EvolSearch
evolving for 10 dim vectors with each element in [0,1], maximizing their means
i.e. best solution is [1,1,1,1,1,1,1,1,1,1]
'''

import numpy as np
import matplotlib.pyplot as plt
from stochsearch import EvolSearch

def fitness_function(individual):
    '''
    sample fitness function
    '''
    return np.mean(individual)

# defining the parameters for the evolutionary search
evol_params = {
    'num_processes' : 4, # (optional) number of proccesses for multiprocessing.Pool
    'pop_size' : 100,    # population size
    'genotype_size': 50, # dimensionality of solution
    'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
    'elitist_fraction': 0.04, # fraction of population retained as is between generations
    'mutation_variance': 0.2 # mutation noise added to offspring.
}

# create evolutionary search object
es = EvolSearch(evol_params)

'''OPTION 1
# execute the search for 100 generations
num_gens = 100
es.execute_search(num_gens)
'''

'''OPTION 2'''
# keep searching till a stopping condition is reached
best_fit = []
mean_fit = []
num_gen = 0
max_num_gens = 100
desired_fitness = 0.98
#while es.get_best_individual_fitness() < desired_fitness and num_gen < max_num_gens:
while num_gen < max_num_gens:
    print('Gen #'+str(num_gen)+' Best Fitness = '+str(es.get_best_individual_fitness()))
    es.step_generation()
    best_fit.append(es.get_best_individual_fitness())
    mean_fit.append(es.get_mean_fitness())
    num_gen += 1

# print results
print('Max fitness of population = ',es.get_best_individual_fitness())
print('Best individual in population = ',es.get_best_individual())

# plot results
plt.figure()
plt.plot(best_fit)
plt.plot(mean_fit)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['best fitness', 'avg. fitness'])
plt.show()
