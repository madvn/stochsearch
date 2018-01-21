Stochastic Search - Evolutionary Optimization
=========================
An evolutionary algorithm is a stochastic search based optimization technique. It is a population based method, where the optimization starts with a population of random solutions (individuals or genotypes). Each individual is assigned a fitness score based on how well they perform in the task at hand. Based on this fitness, a fraction of the best performing individuals are retained for the next iteration (generation). a new population of solutions is then created for the next generation with these 'elite' individuals and copies of them that have been subjected to mutation noise. This process is repeated either for a fixed number of generations, or until a desired fitness value is reached by the best individual in the population. In a non-stochastic system, this procedure will cause the fitness to be non-decreasing over generations. For those familiar with hill climbing, this approach can be seen as multiple hill climbers searching in parallel, where the number of hill climbers would be given by the elitist fraction of the population that are retained generation after generation. This same implementation can be used to perform hill-climbing if the elitist fraction is set such that elitist_fraction*population_size = 1.

This Python stochastic search package, stochsearch, includes an implementation of evolutionary algorithms in a class called EvolSearch using the Python multiprocessing framework. Fitness evaluation of individuals in a population is carried out in parallel across CPUs in a multiprocessing.pool.Pool with the number of processes defined by the user or by os.cpu_count() of the system. This package can be imported as follows "from stochsearch import EvolSearch".

Installation
---------------
        $ pip install stochsearch
               
Requirements: numpy

Usage
---------------
In order to use this package
#### Importing evolutionary search
        from stochsearch import EvolSearch
        
#### Setup parameters for evolutionary search using a dictionary as follows 
        evol_params = {
            'num_processes' : 4, # (optional) number of proccesses for multiprocessing.Pool
            'pop_size' : 100,    # population size
            'genotype_size': 10, # dimensionality of solution
            'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
            'elitist_fraction': 0.04, # fraction of population retained as is between generations
            'mutation_variance': 0.05 # mutation noise added to offspring.
        }
 
Define a function that takes a genotype as argument and returns the fitness value for that genotype - passed as the 'fitness_function' key in the evol_params dictionary. 

#### Create an evolutionary search object
        es = EvolSearch(evol_params)

#### Executing the search
Option 1: Run the search for a certain number of generations

        num_gens = 100
        es.execute_search(num_gens)
        
Option 2: Step through the generations based on a condition

        max_num_gens = 100
        gen = 0
        desired_fitness = 0.9
        while es.get_best_individual_fitness() < desired_fitness and gen < max_num_gens:
                print('Gen #'+str(gen)+' Best Fitness = '+str(es.get_best_individual_fitness()))
                es.step_generation()
                gen += 1
                
#### Accessing results
        print('Max fitness of population = ',es.get_best_individual_fitness())
        print('Best individual in population = ',es.get_best_individual())

See [demos] folder for a sample program.

[demos]: https://github.com/madvn/stochsearch/blob/master/demo/evolsearch_demo.py
