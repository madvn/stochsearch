'''
Multiprocessor Spatial Microbial GA

Eduardo Izquierdo
July, 2018
'''
from pathos.multiprocessing import ProcessPool
import random
import numpy as np

_pool = None

class MicrobialSearch():
    def __init__(self, evol_params): #generations, pop_size, genotype_size, recomb_prob, mutation_variance, num_processes):
        '''
        Initialize evolutionary search
        ARGS:
        evol_params: dict
            required keys -
                pop_size: int - population size,
                genotype_size: int - genotype_size,
                fitness_function: function - a user-defined function that takes a genotype as arg and returns a float fitness value
                mutation_variance: float - variance of the gaussian distribution used for mutation noise
                recomb_prob: float between [0,1] -- proportion of genotype transfected from winner to loser
                num_processes: int -  pool size for multiprocessing.pool.Pool - defaults to os.cpu_count()
        '''
        # check for required keys
        required_keys = ['pop_size','genotype_size','fitness_function','recomb_prob','mutation_variance','num_processes']
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception('Argument evol_params does not contain the following required key: '+key)

        # checked for all required keys
        self.pop_size = evol_params['pop_size']
        self.genotype_size = evol_params['genotype_size']
        self.fitness_function = evol_params['fitness_function']
        self.mutation_variance = evol_params['mutation_variance']
        self.recomb_prob = evol_params['recomb_prob']
        self.num_processes = evol_params['num_processes']

        # validating fitness function
        assert self.fitness_function,"Invalid fitness_function"
        rand_genotype = np.random.rand(self.genotype_size)
        rand_genotype_fitness = self.fitness_function(rand_genotype)
        assert type(rand_genotype_fitness) == type(0.) or type(rand_genotype_fitness) in np.sctypes['float'],\
                 "Invalid return type for fitness_function. Should be float or np.dtype('np.float*')"

        # Search parameters
        self.group_size = int(self.pop_size/3)

        # Keep track of individuals to be mutated
        self.mutlist = np.zeros((self.group_size), dtype=int)
        self.mutfit = np.zeros((self.group_size))

        # Creating the global process pool to be used across all generations
        global _pool
        _pool = ProcessPool(self.num_processes)

        # check for fitness function kwargs
        if 'fitness_args' in evol_params.keys():
            optional_args = evol_params['fitness_args']
            assert len(optional_args) == 1 or len(optional_args) == pop_size,\
                    "fitness args should be length 1 or pop_size."
            self.optional_args = optional_args
        else:
            self.optional_args = None

        # Create population and evaluate everyone once
        self.pop = np.random.random((self.pop_size,self.genotype_size))
        self.fitness = np.asarray(_pool.map(self.evaluate_fitness,np.arange(self.pop_size)))

    def evaluate_fitness(self,individual_index):
        '''
        Call user defined fitness function and pass genotype
        '''
        if self.optional_args:
            if len(self.optional_args) == 1:
                return self.fitness_function(self.pop[individual_index,:], self.optional_args[0])
            else:
                return self.fitness_function(self.pop[individual_index,:], self.optional_args[individual_index])
        else:
            return self.fitness_function(self.pop[individual_index,:])

    def step_generation(self):
        '''
        evaluate fitness and step on generation
        '''
        global _pool
        # Perform tournament for every individual in population
        for j in range(3):
            k = 0
            for a in range(j,self.pop_size-2,3):
                # Step 1: Pick 2nd individual as left or right hand side neighbor of first
                b = (a+random.choice([-1,1]))%self.pop_size
                # Step 2: Compare their fitness
                if (self.fitness[a] > self.fitness[b]):
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
                # Step 3: Transfect loser with winner
                for l in range(self.genotype_size):
                    if (random.random() < self.recomb_prob):
                        self.pop[loser][l] = self.pop[winner][l]
                # Step 4: Mutate loser
                m = np.random.normal(0.0, self.mutation_variance, self.genotype_size)
                self.pop[loser] = np.clip(np.add(self.pop[loser],m),0.0,1.0)
                # Step 5: Add to mutated list (which will be re-evaluated)
                self.mutlist[k]=loser
                k+=1
            # Step 6: Recalculate fitness of list of mutated losers
            self.mutfit = list(_pool.map(self.evaluate_fitness, self.mutlist))
            for k in range(self.group_size):
                self.fitness[self.mutlist[k]] = self.mutfit[k]

    def execute_search(self, num_gens):
        '''
        runs the evolutionary algorithm for given number of generations, num_gens
        '''
        for _ in range(num_gens):
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
