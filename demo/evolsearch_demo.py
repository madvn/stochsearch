'''
Demo of EvolSearch
evolving for 10 dim vectors with each element in [0,1], maximizing their means
i.e. best solution is [1,1,1,1,1,1,1,1,1,1]
'''

import numpy as np
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
    'genotype_size': 10, # dimensionality of solution
    'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
    'elitist_fraction': 0.04, # fraction of population retained as is between generations
    'mutation_variance': 0.05 # mutation noise added to offspring.
}

# create evolutionary search object
es = EvolSearch(evol_params)

'''OPTION 1'''
# execute the search for 100 generations
num_gens = 100
es.execute_search(num_gens)

'''OPTION 2
# keep searching till a stopping condition is reached
num_gen = 0
max_num_gens = 100
desired_fitness = 0.75
while es.get_best_individual_fitness() < desired_fitness and num_gen < max_num_gens:
    print('Gen #'+str(num_gen)+' Best Fitness = '+str(es.get_best_individual_fitness()))
    es.step_generation()
    num_gen += 1
'''

# print results
print('Max fitness of population = ',es.get_best_individual_fitness())
print('Best individual in population = ',es.get_best_individual())
