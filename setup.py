import numpy
from setuptools import setup, Extension

PairwiseModule = Extension('Pairwise', sources = ['PairwiseDistances.cpp', 'PairwiseIndexer.cpp'],
						   include_dirs = [numpy.get_include()],
						   extra_compile_args = ["-fopenmp", "-lm", "-O2", "-march=native"],
						   extra_link_args = ["-fopenmp", "-lm"])

setup(name = 'Pairwise', version = '0.1', ext_modules = [PairwiseModule])