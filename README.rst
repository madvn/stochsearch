Stochastic Search
=================

This Python stochastic search package, stochsearch, includes an
implementation of algorithms such as evolutionary algorithm, microbial
genetic algorithm, and lamarckian evolutionary algorithm, using the
Python pathos multiprocessing framework. Fitness evaluation of
individuals in a population is carried out in parallel across CPUs in a
multiprocessing pool with the number of processes defined by the user or
by os.cpu_count() of the system. Read below for installation and usage
instructions.

Installation
------------

::

       $ pip install stochsearch

Requirements: numpy, pathos

Usage
-----

This section illustrates how to use this package for evolutionary
search. It is similar for other search methods. The only items that may
change are the parameters of the search. See `this`_ for a description
and list of parameters for each search method.

Importing evolutionary search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

       from stochsearch import EvolSearch

Setup parameters for evolutionary search using a dictionary as follows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

       evol_params = {
           'num_processes' : 4, # (optional) number of proccesses for multiprocessing.Pool
           'pop_size' : 100,    # population size
           'genotype_size': 10, # dimensionality of solution
           'fitness_function': fitness_function, # custom function defined to evaluate fitness of a solution
           'elitist_fraction': 0.04, # fraction of population retained as is between generations
           'mutation_variance': 0.05, # mutation noise added to offspring.
           'fitness_args': np.arange(100), # (optional) fitness_function *argv, len(list) should be 1 or pop_size
       }

Define a function that takes a genotype as argument and returns the
fitness value for that genotype - passed as the ‘fitness_function’ key
in the evol_params dictionary.

Create an evolutionary search object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

       es = EvolSearch(evol_params)

Executing the search
^^^^^^^^^^^^^^^^^^^^

Option 1: Run the search for a certain number of generations

::

       num_gens = 100
       es.execute_search(num_gens)

Option 2: Step through the generations based on a condition

::

       max_num_gens = 100
       gen = 0
       desired_fitness = 0.9
       while es.get_best_individual_fitness() < desired_fitness and gen < max_num_gens:
               print('Gen #'+str(gen)+' Best Fitness = '+str(es.get_best_individual_fitness()))
               es.step_generation()
               gen += 1

Accessing results
^^^^^^^^^^^^^^^^^

::

       print('Max fitness of population = ',es.get_best_individual_fitness())
       print('Best individual in population = ',es.get_best_individual())

See `demos`_ folder for a sample script.

.. _this: https://github.com/madvn/stochsearch/blob/master/stochsearch/README.md
.. _demos: https://github.com/madvn/stochsearch/blob/master/demo/evolsearch_demo.py
