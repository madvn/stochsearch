### Search algorithms and their parameters

#### Evolutionary Search (from stochsearch import EvolSearch)
An evolutionary algorithm is a stochastic search based optimization technique. It is a population based method, where optimization starts with a population of random solutions (individuals or genotypes). Each individual is assigned a fitness score based on how well they perform in the task at hand. Based on this fitness, a fraction of the best performing individuals are retained for the next iteration (generation). a new population of solutions is then created for the next generation with these 'elite' individuals and copies of them that have been subjected to mutation noise. This process is repeated either for a fixed number of generations, or until a desired fitness value is reached by the best individual in the population. In a non-stochastic system, this procedure will cause the fitness to be non-decreasing over generations. For those familiar with hill climbing, this approach can be seen as multiple hill climbers searching in parallel, where the number of hill climbers would be given by the elitist fraction of the population that are retained generation after generation. This same implementation can be used to perform hill-climbing if the elitist fraction is set such that elitist_fraction*population_size = 1.

    evol_params = {
        'num_processes' : 4, # (optional) number of processes for multiprocessing.Pool
        'pop_size' : 100,    # population size
        'genotype_size': 10, # dimensionality of solution
        'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
        'elitist_fraction': 0.04, # fraction of population retained as is between generations
        'mutation_variance': 0.05, # mutation noise added to offspring.
        'fitness_args': np.arange(100), # (optional) fitness_function \*argv, len(list) should be 1 or pop_size
    }


#### Microbial Search (from stochsearch import MicrobialSearch)
This search method is quite similar to evolutionary except in its selection. This algorithm involves a tournament style selection. From the population list, in each generation, each individual competes against one of its neighbors based on a coin toss. The winner is put back in the population as is. The winner gets to "corrupt" the loser with its own genes based on a recombination probability (recomb_prob). The loser is then mutated based on a 0-mean gaussian noise with variance defined by mutation_variance and finally put back in its own position into the population. This process is repeated over several generations.

    evol_params = {
        'num_processes' : 12, # (optional) number of processes for multiprocessing.Pool
        'pop_size' : 100,    # population size
        'genotype_size': 10, # dimensionality of solution
        'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
        'recomb_prob': 0.1, # probability of winner genes transfecting loser
        'mutation_variance': 0.01, # mutation noise added to offspring.
        'generations' : 100
        'fitness_args': np.arange(100), # (optional) fitness_function \*argv, len(list) should be 1 or pop_size
    }

#### Lamarckian evolution (from stochsearch import LamarckianSearch)
This type of search involves some kind of learning during each generation. The fitness function that the user writes would receive a genotype, creates a phenotype (e.g. a neural network), train the phenotype (updated the weights of the neural network) and then evaluates its fitness. Once this is done, the fitness function returns the genotype remapped from the new trained phenotype and the fitness. The updated genotype is then put into the population. Once all individuals are evaluated like this, the population goes through the same elitist selection and mutation process as described in evolutionary search.

    evol_params = {
        'num_processes' : 4, # (optional) number of processes for multiprocessing.Pool
        'pop_size' : 100,    # population size
        'genotype_size': 10, # dimensionality of solution
        'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
        'elitist_fraction': 0.04, # fraction of population retained as is between generations
        'mutation_variance': 0.05, # mutation noise added to offspring.
        'fitness_args': np.arange(100), # (optional) fitness_function \*argv, len(list) should be 1 or pop_size
    }
