#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <omp.h>
#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

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


/**
* Shorthand macros for getting doubles from arrays
*/

#define GET_ARRAY1D(arr, i) *(double*)PyArray_GETPTR1(arr, i)
#define GET_ARRAY2D(arr, i, k) *(double*)PyArray_GETPTR2(arr, i, k)
#define GET_1D_INT(arr, i) *(int*)PyArray_GETPTR1(arr, i)

/**
* Pairwise distance indexing functions
 * see https://stackoverflow.com/a/36867493
*/
#define  REVERSE_INDEX

#ifdef REVERSE_INDEX

static int RowIndex(unsigned long long condensed, unsigned long long nItems)
{
	return int(ceil(0.5 * (- sqrt(-8 * condensed + 4 * nItems * nItems - 4 * nItems - 7) + 2 * nItems - 1) - 1));
}

/**
 * Number of items up to and including the nth row
 * @param rowIndex
 * @param nItems
 * @return
 */
static int NumItemsToRow(unsigned long long rowIndex, unsigned long long nItems)
{
	return rowIndex * (nItems - 1 - rowIndex) + (rowIndex * (rowIndex + 1)) / 2;
}

static int ColIndex(unsigned long long condensed, unsigned int rowIndex, unsigned long long nItems)
{
	return int(nItems - NumItemsToRow(rowIndex + 1, nItems) + condensed);
}

#endif