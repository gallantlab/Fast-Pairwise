from setuptools import setup, Extension

PairwiseModule = Extension('Pairwise', sources = ['PairwiseDistances.cpp'],
						   extra_compile_args = ["-fopenmp", "-lm"],
						   extra_link_args = ["-fopenmp", "-lm"])

setup(name = 'Pairwise', version = '0.1', ext_modules = [PairwiseModule])