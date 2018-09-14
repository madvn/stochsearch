from setuptools import setup
from stochsearch.__init__ import __version__

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='stochsearch',
    version=__version__,
    description='A package that implements anumber of stochastic search algorithms using the pathos multiprocessing framework for parallelization',
    long_description=readme,
    author='Madhavun Candadai',
    author_email='madvncv@gmail.com',
    url='https://github.com/madvn/stochsearch',
    license=license,
    packages=['stochsearch'],
    install_requires=['numpy','pathos','dill','ppft']
)
