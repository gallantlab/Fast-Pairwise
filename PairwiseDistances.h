#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <omp.h>
#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "PairwiseIndexer.h"

PyMODINIT_FUNC PyInit_Pairwise(void);

/**
 * Python-facing function calls -
 * All have the signature of
 	* nullptr FUNCTION(input items, output array)
 * where the input items is a 2D(index, feature) array and the output is a pre-allocated
 * 1D condensed distance matrix into which the distances are written.
 * Both arrays are numpy arrays. The output array should be pre-allocated
 * in python because combinatorics easily exceed the size of a npy_intp
 * and allocating in python makes typing easier
 */

/**
 * Calculates distances given the results of RandomForest.apply()
 * @param self 	ignored
 * @param args 	the first value is the array, 2nd value is a preallocated output array
 * @return
 */
static PyObject* GetPairwiseRandomForestDistance(PyObject *self, PyObject *args);

static PyObject* GetPairwiseEuclideanDistance(PyObject *self, PyObject *args);

static PyObject* GetPairwiseCorrelationDistance(PyObject *self, PyObject *args);

static PyObject* GetClusteringDistance(PyObject *self, PyObject *args);

static PyObject* GetClusteringDistances(PyObject *self, PyObject *args);

static PyObject* GetClusteringDistancesExplicitAVX(PyObject *self, PyObject *args);


/**
 * Python module boilerplate
 */

static PyMethodDef PythonDistanceMethods[] =
	{
		{"GetPairwiseRandomForestDistance", &GetPairwiseRandomForestDistance, METH_VARARGS,
		 "GetPairwiseRandomForestDistance(items, out)\n"
		 "Gets the pairwise random forest distance between items, i.e. the fraction of the forest in which the pair of\n"
		 "items fell into the same leaf node\n"
		 "@param items:	a 2d [nItems, nFeatures] numpy array\n"
		 "@param out:	an empty 1d array of size (nItems * (nItems - 1) / 2) for the output condensed distance matrix\n"},

		{"GetPairwiseEuclideanDistance", &GetPairwiseEuclideanDistance,       METH_VARARGS,
			"GetPairwiseEuclideanDistance(items, out)\n"
			"Gets the pairwise euclidean distance between items\n"
			"items fell into the same leaf node\n"
			"@param items:	a 2d [nItems, nFeatures] numpy array\n"
			"@param out:	an empty 1d array of size (nItems * (nItems - 1) / 2) for the output condensed distance matrix\n"},

		{"GetPairwiseCorrelationDistance", &GetPairwiseCorrelationDistance, METH_VARARGS,
			"GetPairwiseCorrelationDistance(items, out)\n"
			"Gets the pairwise correlation distance between items\n"
			"items fell into the same leaf node\n"
			"@param items:	a 2d [nItems, nFeatures] numpy array\n"
			"@param out:	an empty 1d array of size (nItems * (nItems - 1) / 2) for the output condensed distance matrix\n"},

		{"GetClusteringDistance", &GetClusteringDistance, METH_VARARGS,
			"GetClusteringDistance(solution1, solution2, normalize: bool = True)\n"
			"Gets the clustering distance between solutions as defined in \n"
			"Wang Biometrika 2010 and Fang & Wang Comp Stat & Data Analysis. 2012\n"
			"@param solution1:	one clustering solution of the set of items in which each entry is an int for the cluster ID\n"
			"@param solution2:	another colustering solution\n"
			"@param normalize:	normalize the return value by the number of pairs?"
			"@return:	clustering distance\n"},

		{"GetClusteringDistances", &GetClusteringDistances, METH_VARARGS,
			"GetClusteringDistance(solutions, out, normalize: bool = True)\n"
			"Gets the clustering distance between all pairs of solutions. Uses the same method as GetClusteringDistance\n"
			"but this is for many solutions instead of a just a single pair\n"
			"@param solutions:	[nSolutions, nitems] many different clustering solutions on the same number of items\n"
			"@param out:		empty array of number of clustering pairs to output condensed distance matrix\n"
			"@param normalize:	normalize the clustering distances by the number of pairs of items?"},

		{"GetClusteringDistancesExplicitAVX", &GetClusteringDistancesExplicitAVX, METH_VARARGS,
		 "GetClusteringDistanceExplicitAVX(solutions, out, normalize: bool = True)\n"
		 "Like GetClusteringDistances but explicitly uses AVX512 intrinsics\n"
		 "but this is for many solutions instead of a just a single pair\n"
		 "@param solutions:	[nSolutions, nitems] many different clustering solutions on the same number of items\n"
		 "@param out:		empty array of number of clustering pairs to output condensed distance matrix\n"
		 "@param normalize:	normalize the clustering distances by the number of pairs of items?"},
		{NULL, NULL, 0,     NULL}
	};


static struct PyModuleDef moduleDefinition = {PyModuleDef_HEAD_INIT,
											  "Pairwise",
											  "",
											  -1,
											  PythonDistanceMethods,
											  NULL,NULL,NULL,NULL};

/**
 * Internal functions not exposed to python
 */

/**
 * Generic parallelized pairwise distance function
 * @param args 				input python arguments
 * @param DistanceFunction 	function pointer to distance metric to use
 * @param arrayType 		expected python input array
 * @return nullptr
 */

static PyObject* GetPairwiseDistance(PyObject *args, double (*DistanceFunction)(PyArrayObject*, int, int, int, const double*),
									 NPY_TYPES arrayType);

/**
 * Distance functions with the following signature
 * @param items			numpy array from the actual objects
 * @param i				index of first item in pair
 * @param j				index of second item in pair
 * @param length		length of these vectors
 * @param meanSqaures	mean square of each entry, only used by Correlation, but shared for signature consistency in function pointer
 * @return	distance between the pairs
 */


static double Euclidean(PyArrayObject* items, int i, int j, int length, const double* meanSqaures);

static double RandomForest(PyArrayObject* items, int i, int j, int length, const double* meanSqaures);

static double Correlation(PyArrayObject* items, int i, int j, int length, const double* meanSqaures);

enum SIMD
{
	UINT8,
	UINT16,
	NONE
};

/**
 * Functions for AVX stuff that deals with different data types that necessitate the use of different vector sizes
 * but has the same signature
 * @param solutions			cluster assignments in ints in [solution, item]
 * @param distances 		array to write the distances back out to
 * @param numSolutions 		number of solutions (solutions dim 1)
 * @param numItems 			numer of items (solutions dim 2)
 * @param numSolutionPairs 	number of solution pairs
 * @param numItemPairs 		number of item pairs
 * @param itemIndexer 		an indexer for backwards indexing into item pairs
 * @param solutionIndexer 	an indexer for backwards idnexing into solution pairs
 */
static void ClusteringDistanceUINT8(PyArrayObject* solutions, PyArrayObject* distances, unsigned long long numSolutions, unsigned long long numItems,
									unsigned long long numSolutionPairs, unsigned long long numItemPairs, Indexer *itemIndexer, Indexer *solutionIndexer);
static void ClusteringDistanceUINT16(PyArrayObject* solutions, PyArrayObject* distances, unsigned long long numSolutions, unsigned long long numItems,
									 unsigned long long numSolutionPairs, unsigned long long numItemPairs, Indexer *itemIndexer, Indexer *solutionIndexer);


/**
* Shorthand macros for getting doubles from arrays
*/

#define GET_1D_DOUBLE(arr, i) *(double*)PyArray_GETPTR1(arr, i)
#define GET_2D_DOUBLE(arr, i, k) *(double*)PyArray_GETPTR2(arr, i, k)
#define GET_2D_INT(arr, i, k) *(int*)PyArray_GETPTR2(arr, i, k)
#define GET_1D_INT(arr, i) *(int*)PyArray_GETPTR1(arr, i)
