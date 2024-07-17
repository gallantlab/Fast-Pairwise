#include "PairwiseDistances.h"

#include <iostream>
using namespace std;


PyMODINIT_FUNC PyInit_Pairwise(void)
{
	import_array();
	return PyModule_Create(&moduleDefinition);
}

static double Euclidean(PyArrayObject* items, int i, int j, int length)
{
	double dist = 0.0;
	for (int k = 0; k < length; k++)
	{
		double d1 = *(double*)PyArray_GETPTR2(items, i, k),
			d2 = *(double*)PyArray_GETPTR2(items, j, k);
		dist += ((d1 - d2) * (d1 - d2));
	}
	return sqrt(dist);
}

static double RandomForest(PyArrayObject* items, int i, int j, int length)
{
	double count = 0.0;
	for (int c = 0; c < length; c++)
	{
		if (*(long*)PyArray_GETPTR2(items, i, c) == *(long*)PyArray_GETPTR2(items, j, c))
			count++;
	}
	// compute index in the output array - this way we don't need to lock and also have concurrency indexing issues
	return 1 - count / length;
}

static double Correlation(PyArrayObject* items, int i, int j, int length)
{
	// two-pass algo as used in scipy.spatial.distance
	double Xmean = 0.0, Ymean = 0.0;
	for (int k = 0; k < length; k++)
	{
		Xmean += GET_ARRAY2D(items, i, k);
		Ymean += GET_ARRAY2D(items, j, k);
	}
	Xmean /= (double)length; Ymean /= (double)length;

	double sumXY = 0.0,
			sumX2 = 0.0,
			sumY2 = 0.0;

	for (int k = 0; k < length; k++)
	{
		double x = GET_ARRAY2D(items, i, k) - Xmean;
		double y = GET_ARRAY2D(items, j, k) - Ymean;

		sumX2 += x * x;
		sumY2 += y * y;
		sumXY += x * y;
	}
	sumXY /= (double)length; sumX2 /= (double)length; sumY2 /= (double)length;

	return 1.0 - sumXY / sqrt(sumX2 * sumY2);
}


static PyObject* GetPairwiseRandomForestDistance(PyObject *self, PyObject *args)
{
	return GetPairwiseDistance(args, &RandomForest, NPY_INT64);
}


static PyObject* GetPairwiseEuclideanDistance(PyObject *self, PyObject *args)
{
	return GetPairwiseDistance(args, &Euclidean, NPY_DOUBLE);
}

static PyObject* GetPairwiseCorrelationDistance(PyObject *self, PyObject *args)
{
	return GetPairwiseDistance(args, &Correlation, NPY_DOUBLE);
}

static PyObject* GetPairwiseDistance(PyObject *args, double (*DistanceFunction)(PyArrayObject*, int i, int j, int length),
									 NPY_TYPES arrayType)
{
	// == Read in the arguments as a generic python object and an int
	PyObject *items;	// this should be the array that is passed to us
	PyObject *out;			// number of values in the condensed distance array should be computed in numpy because large factorial

	if (!PyArg_ParseTuple(args, "OO", &items, &out))	// "O" means object, 'i' means integer
		return nullptr;

	// == Cast the generic python object to a Numpy array object
	PyArrayObject *itemArray = (PyArrayObject*)PyArray_FROM_OTF(items, arrayType, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *outArray = (PyArrayObject*)PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

	if (itemArray == nullptr || outArray == nullptr) // || sizeArray == nullptr)
		return nullptr;
	// get dimensions of the input array
	npy_intp *dims = PyArray_DIMS(itemArray);
	unsigned long long numItems = dims[0];
	unsigned int numFeatures = dims[1];

#ifdef REVERSE_INDEX
	unsigned long long numCondensed = numItems * (numItems - 1) / 2;
#endif

#ifndef REVERSE_INDEX
	#pragma omp parallel for
	for (unsigned long long k = 0; k < numItems * numItems; k++)
	{
		unsigned long long i = k / numItems, j = k % numItems;
		if (i < j)
		{
			unsigned long long index = numItems * i - (i * (i + 1)) / 2 + j - 1 - i;
			double d = DistanceFunction(itemArray, i, j, numFeatures);
			*((double*)PyArray_GETPTR1(outArray, index)) = d;
		}
	}
#else
	#pragma omp parallel for
	for (unsigned long i = 0; i < numCondensed; i ++)
	{
		int row = RowIndex(i, numItems);
		int col = ColIndex(i, row, numItems);
		*((double*)PyArray_GETPTR1(outArray, i)) = DistanceFunction(itemArray, row, col, numFeatures);
	}
#endif

	Py_RETURN_NONE;
}