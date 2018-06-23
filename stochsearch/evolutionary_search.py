'''
Contains the multiprocessing evolutionary search class

Madhavun Candadai
Jan, 2018
'''
from multiprocessing import Pool
import numpy as np

__evolsearch_process_pool = None

class EvolSearch:
    def __init__(self,evol_params):
        '''
        Initialize evolutionary search
        ARGS:
        evol_params: dict
            required keys -
                pop_size: int - population size,
                genotype_size: int - genotype_size,
                fitness_function: function - a user-defined function that takes a genotype as arg and returns a float fitness value
                elitist_fraction: float - fraction of top performing individuals to retain for next generation
                mutation_variance: float - variance of the gaussian distribution used for mutation noise
            optional keys -
                num_processes: int -  pool size for multiprocessing.pool.Pool - defaults to os.cpu_count()
        '''
        # check for required keys
        required_keys = ['pop_size','genotype_size','fitness_function','elitist_fraction','mutation_variance']
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception('Argument evol_params does not contain the following required key: '+key)

        # checked for all required keys
        self.pop_size = evol_params['pop_size']
        self.genotype_size = evol_params['genotype_size']
        self.fitness_function = evol_params['fitness_function']
        self.elitist_fraction = int(np.ceil(evol_params['elitist_fraction']*self.pop_size))
        self.mutation_variance = evol_params['mutation_variance']

        # validating fitness function
        assert self.fitness_function,"Invalid fitness_function"
        rand_genotype = np.random.rand(self.genotype_size)
        rand_genotype_fitness = self.fitness_function(rand_genotype)
        assert type(rand_genotype_fitness) == type(0.) or type(rand_genotype_fitness) in np.sctypes['float'],"Invalid return type for fitness_function. Should be float or np.dtype('np.float*')"

        # create other required data
        self.num_processes = evol_params.get('num_processes',None)
        self.pop = np.random.rand(self.pop_size,self.genotype_size)
        self.fitness = np.zeros(self.pop_size)
        self.num_batches = int(self.pop_size/self.num_processes)
        self.num_remainder = int(self.pop_size%self.num_processes)

        # creating the global process pool to be used across all generations
        global __evolsearch_process_pool
        __evolsearch_process_pool = Pool(self.num_processes)

    def evaluate_fitness(self,individual_index):
        '''
        Call user defined fitness function and pass genotype
        '''
        return self.fitness_function(self.pop[individual_index,:]) #np.sum(self.pop[indvidual_index,:])

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
        global __evolsearch_process_pool

        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            self.fitness = np.asarray(__evolsearch_process_pool.map(self.evaluate_fitness,np.arange(self.pop_size)))
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(__evolsearch_process_pool.map(self.evaluate_fitness,np.arange(self.pop_size)))

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
