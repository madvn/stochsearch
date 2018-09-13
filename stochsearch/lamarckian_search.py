'''
Contains the multiprocessing Lamarckian search class

Madhavun Candadai
Sep, 2018
'''
#from multiprocessing import Pool
import time
import numpy as np
from pathos.multiprocessing import ProcessPool

__search_process_pool = None

class LamarckianSearch:
    def __init__(self,evol_params):
        '''
        Initialize evolutionary search
        ARGS:
        evol_params: dict
            required keys -
                pop_size: int - population size,
                genotype_size: int - genotype_size,
                fitness_function: function - a user-defined function that takes a genotype as arg and returns updated genotype and float fitness value
                elitist_fraction: float - fraction of top performing individuals to retain for next generation
                mutation_variance: float - variance of the gaussian distribution used for mutation noise
            optional keys -
                fitness_args: list-like - optional additional arguments to pass while calling fitness function
                                           list such that len(list) == 1 or len(list) == pop_size
                num_processes: int -  pool size for multiprocessing.pool.Pool - defaults to os.cpu_count()
        '''
        # check for required keys
        required_keys = ['pop_size','genotype_size','fitness_function','elitist_fraction','mutation_variance']
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception('Argument evol_params does not contain the following required key: {}'.format(key))

        # checked for all required keys
        self.pop_size = evol_params['pop_size']
        self.genotype_size = evol_params['genotype_size']
        self.fitness_function = evol_params['fitness_function']
        self.elitist_fraction = int(np.ceil(evol_params['elitist_fraction']*self.pop_size))
        self.mutation_variance = evol_params['mutation_variance']

        # validating fitness function
        assert self.fitness_function,"Invalid fitness_function"
        rand_genotype = np.random.rand(self.genotype_size)
        fitness_return = self.fitness_function(rand_genotype)
        assert len(fitness_return) == 2, "Fitness function must return 2 items - updated_genotype and fitness"
        updated_genotype = fitness_return[0]
        rand_genotype_fitness = fitness_return[1]
        assert type(rand_genotype_fitness) == type(0.) or type(rand_genotype_fitness) in np.sctypes['float'],\
               "Invalid return type for second return of fitness_function. Should be float or np.dtype('np.float*')"
        assert len(updated_genotype) == self.genotype_size, \
                "Invalid length for first return type of fitness function: length should be equal to genotype_size={}".format(self.genotype_size)

        # create other required data
        self.num_processes = evol_params.get('num_processes',None)
        self.pop = np.random.rand(self.pop_size,self.genotype_size)
        self.fitness = np.zeros(self.pop_size)
        self.num_batches = int(self.pop_size/self.num_processes)
        self.num_remainder = int(self.pop_size%self.num_processes)

        # check for fitness function kwargs
        if 'fitness_args' in evol_params.keys():
            optional_args = evol_params['fitness_args']
            assert len(optional_args) == 1 or len(optional_args) == self.pop_size,\
                    "fitness args should be length 1 or pop_size."
            self.optional_args = optional_args
        else:
            self.optional_args = None

        # creating the global process pool to be used across all generations
        global __search_process_pool
        __search_process_pool = ProcessPool(self.num_processes)
        time.sleep(0.5)

    def evaluate_fitness(self,individual_index):
        '''
        Call user defined fitness function and pass genotype
        '''
        if self.optional_args:
            if len(self.optional_args) == 1:
                individual, fitness = self.fitness_function(self.pop[individual_index,:], self.optional_args[0])
            else:
                individual, fitness = self.fitness_function(self.pop[individual_index,:], self.optional_args[individual_index])
        else:
            individual, fitness = self.fitness_function(self.pop[individual_index,:])

        # inserting updated genotype back into population
        self.pop[individual_index] = individual
        return fitness

    def elitist_selection(self):
        '''
        from fitness select top performing individuals based on elitist_fraction
        '''
        self.pop = self.pop[np.argsort(self.fitness)[-self.elitist_fraction:],:]

    def mutation(self):
        '''
        create new pop by repeating mutated copies of elitist individuals
        '''
        # number of copies of elitists required
        num_reps = int((self.pop_size-self.elitist_fraction)/self.elitist_fraction)+1

        # creating copies and adding noise
        mutated_elites = np.tile(self.pop,[num_reps,1])
        mutated_elites += np.random.normal(loc=0.,scale=self.mutation_variance,
                                                size=[num_reps*self.elitist_fraction,self.genotype_size])

        # concatenating elites with their mutated versions
        self.pop = np.vstack((self.pop,mutated_elites))

        # clipping to pop_size
        self.pop = self.pop[:self.pop_size,:]

        # clipping to genotype range
        self.pop = np.clip(self.pop,0,1)

    def step_generation(self):
        '''
        evaluate fitness of pop, and create new pop after elitist_selection and mutation
        '''
        global __search_process_pool

        # estimate fitness using multiprocessing pool
        if __search_process_pool:
            # pool exists
            self.fitness = np.asarray(__search_process_pool.map(self.evaluate_fitness,np.arange(self.pop_size)))
        else:
            # re-create pool
            __search_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(__search_process_pool.map(self.evaluate_fitness,np.arange(self.pop_size)))

        # elitist_selection
        self.elitist_selection()

        # mutation
        self.mutation()

    def execute_search(self,num_gens):
        '''
        runs the evolutionary algorithm for given number of generations, num_gens
        '''
        # step generation num_gens times
        for gen in np.arange(num_gens):
            self.step_generation()

    def get_fitnesses(self):
        '''
        simply return all fitness values of current population
        '''
        return self.fitness

    def get_best_individual(self):
        '''
        returns 1D array of the genotype that has max fitness
        '''
        return self.pop[np.argmax(self.fitness),:]

    def get_best_individual_fitness(self):
        '''
        return the fitness value of the best individual
        '''
        return np.max(self.fitness)

    def get_mean_fitness(self):
        '''
        returns the mean fitness of the population
        '''
        return np.mean(self.fitness)

    def get_fitness_variance(self):
        '''
        returns variance of the population's fitness
        '''
        return np.std(self.fitness)**2

if __name__ == "__main__":
    def fitness_function(individual):
        '''
        sample fitness function
        '''
        return individual, np.mean(individual)

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
    es = LamarckianSearch(evol_params)

    '''OPTION 1
    # execute the search for 100 generations
    num_gens = 100
    es.execute_search(num_gens)
    '''

    '''OPTION 2'''
    # keep searching till a stopping condition is reached
    num_gen = 0
    max_num_gens = 100
    desired_fitness = 0.75
    while es.get_best_individual_fitness() < desired_fitness and num_gen < max_num_gens:
        print('Gen #'+str(num_gen)+' Best Fitness = '+str(es.get_best_individual_fitness()))
        es.step_generation()
        num_gen += 1

    # print results
    print('Max fitness of population = ',es.get_best_individual_fitness())
    print('Best individual in population = ',es.get_best_individual())
