#include "PairwiseDistances.h"

#include <iostream>
using namespace std;


PyMODINIT_FUNC PyInit_Pairwise(void)
{
	import_array();
	return PyModule_Create(&moduleDefinition);
}

static double Euclidean(PyArrayObject* items, int i, int j, int length, const double*)
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

static double RandomForest(PyArrayObject* items, int i, int j, int length, const double*)
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

static double Correlation(PyArrayObject* items, int i, int j, int length, const double* meanSquares)
{
	// two-pass algo as used in scipy.spatial.distance
	// the commented out stuff are factored out to reduce complexity
	// but kept here to show the algorithm

	/*
	double Xmean = 0.0, Ymean = 0.0;
	for (int k = 0; k < length; k++)
	{
		Xmean += GET_ARRAY2D(items, i, k);
		Ymean += GET_ARRAY2D(items, j, k);
	}
	Xmean /= (double)length; Ymean /= (double)length;
	 */

	double sumXY = 0.0,
			meanX2 = meanSquares[i],
			meanY2 = meanSquares[j];

	for (int k = 0; k < length; k++)
	{
		double x = GET_ARRAY2D(items, i, k) /*- Xmean*/;
		double y = GET_ARRAY2D(items, j, k) /*- Ymean*/;

	/*	meanX2 += x * x;
		meanY2 += y * y;
	*/	sumXY += x * y;
	}
	sumXY /= (double)length; /*meanX2 /= (double)length; meanY2 /= (double)length;*/

	return 1.0 - sumXY / sqrt(meanX2 * meanY2);
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

static PyObject* GetPairwiseDistance(PyObject *args, double (*DistanceFunction)(PyArrayObject*, int, int, int, const double*),
									 NPY_TYPES arrayType)
{
	// == Read in the arguments as a generic python object and an int
	PyObject *items;	// this should be the array that is passed to us
	PyObject *out;		// output array because the number of values in the condensed distance array
						// is a large factorial that can exceed npy_intp, and thus it's easier to
						// allocate the output in python first

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

	// de-mean correlation - doing it within the main loop would be O(n^2) demean ops, whereas
	// doing it here is a O(n) demean ops
	double *meanSquares = nullptr;
	double *rawDemeaned = nullptr;
	if (DistanceFunction == &Correlation)
	{
		rawDemeaned = new double[numItems * numFeatures];	// numpy stores these arrays as 1D
		meanSquares = new double[numItems];

		#pragma omp parallel for
		for (unsigned long i = 0; i < numItems; i++)
		{
			double mean = 0.0;
			meanSquares[i] = 0.0;
			for (unsigned int j = 0; j < numFeatures; j++)
				mean += GET_ARRAY2D(itemArray, i, j);
			mean /= (double)numFeatures;

			for (unsigned int j = 0; j < numFeatures; j++)
			{
				double val = GET_ARRAY2D(itemArray, i, j) - mean;
				rawDemeaned[i * numFeatures + j] = val;
				meanSquares[i] += val * val;
			}
			meanSquares[i] /= (double)numFeatures;
		}

		// and then we operate on this demeaned array
		itemArray = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, rawDemeaned);
	}

#ifndef REVERSE_INDEX
	#pragma omp parallel for
	for (unsigned long long k = 0; k < numItems * numItems; k++)
	{
		unsigned long long i = k / numItems, j = k % numItems;
		if (i < j)
		{
			unsigned long long index = numItems * i - (i * (i + 1)) / 2 + j - 1 - i;
			double d = DistanceFunction(itemArray, i, j, numFeatures, meanSquares);
			*((double*)PyArray_GETPTR1(outArray, index)) = d;
		}
	}
#else
	unsigned long long numCondensed = numItems * (numItems - 1) / 2;
	Indexer *indexer = new Indexer(numItems);

	#pragma omp parallel for
	for (unsigned long i = 0; i < numCondensed; i ++)
	{
		unsigned int row = indexer->RowIndex(i);
		unsigned int col = indexer->ColIndex(i, row);
		*((double*)PyArray_GETPTR1(outArray, i)) = DistanceFunction(itemArray, row, col, numFeatures, meanSquares);
	}

	delete indexer;
#endif

	// clean up stuff used for correlation
	if (DistanceFunction == &Correlation)
	{
		delete [] meanSquares;
		itemArray->~tagPyArrayObject();
		delete [] rawDemeaned;
	}


	Py_RETURN_NONE;
}

#ifdef  REVERSE_INDEX
Indexer::Indexer(unsigned long nItems)
{
	this->nItems = nItems;
	this->itemsToRow = new unsigned long[nItems];
	this->itemsToRow[0] = 0;

	// compute LUT
	unsigned int thisRow;
	for (unsigned long i = 1; i < nItems; i++)
	{
		thisRow = nItems - i;
		this->itemsToRow[i] = this->itemsToRow[i - 1] + thisRow;
	}
}

Indexer::~Indexer()
{
	delete [] this->itemsToRow;
}


unsigned long Indexer::RowIndex(unsigned long long condensed) const
{
	long center;
	long left = 0;
	long right = nItems - 2;
	while (left < right)
	{
		center = (left + right) / 2;
		if (itemsToRow[center + 1] <= condensed)
			left = center + 1;
		else
			right = center;
	}
	return left;
}

unsigned long Indexer::ColIndex(unsigned long long condensed, unsigned int row) const
{
	return (row + 1) + (condensed - this->itemsToRow[row]);
}

#endif