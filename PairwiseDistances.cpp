#include "PairwiseDistances.h"
#include "PairwiseIndexer.h"

#include <iostream>
#include <time.h>
#include <boost/dynamic_bitset.hpp>
#include <immintrin.h>
using namespace std;
using namespace boost;


PyMODINIT_FUNC PyInit_Pairwise(void)
{
	import_array();
	return PyModule_Create(&moduleDefinition);
}

static double Euclidean(PyArrayObject* items, int i, int j, int length, const double*)
{
	double dist = 0.0;
	if (PyArray_ISCARRAY(items)) // we can directly do pointer arithmetic
	{
		double* v1 = (double*)PyArray_GETPTR2(items, i, 0);
		double* v2 = (double*)PyArray_GETPTR2(items, j, 0);
		#pragma omp simd reduction(+:dist)
		for (int k = 0; k < length; k++)
			dist += ((v1[k] - v2[k]) * (v1[k] - v2[k]));
	}
	else
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
	if (PyArray_ISCARRAY(items))
	{
		long* v1 = (long*)PyArray_GETPTR2(items, i, 0);
		long* v2 = (long*)PyArray_GETPTR2(items, j, 0);
		#pragma omp simd reduction(+:count)
		for (int c = 0; c < length; c++)
		{
			if (v1[c] == v2[c])
				count++;
		}
	}
	else
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
	// see line 116 in PairwiseDistances.cpp
	// but kept here to show the algorithm

	/*
	double Xmean = 0.0, Ymean = 0.0;
	for (int k = 0; k < length; k++)
	{
		Xmean += GET_2D_DOUBLE(items, i, k);
		Ymean += GET_2D_DOUBLE(items, j, k);
	}
	Xmean /= (double)length; Ymean /= (double)length;
	 */

	double sumXY = 0.0,
			meanX2 = meanSquares[i],
			meanY2 = meanSquares[j];
	if (PyArray_ISCARRAY(items))
	{
		double* v1 = (double*)PyArray_GETPTR2(items, i, 0);
		double* v2 = (double*)PyArray_GETPTR2(items, j, 0);
		#pragma omp simd reduction(+:sumXY)
		for (int k = 0; k < length; k++)
		{
			double x = v1[k] /*- Xmean*/;
			double y = v2[k] /*- Ymean*/;

			/*	meanX2 += x * x;
				meanY2 += y * y;
			*/	sumXY += x * y;
		}
	}
	else
		for (int k = 0; k < length; k++)
		{
			double x = GET_2D_DOUBLE(items, i, k) /*- Xmean*/;
			double y = GET_2D_DOUBLE(items, j, k) /*- Ymean*/;

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
			if (PyArray_ISCARRAY(itemArray))
			{
				double* v = (double*)PyArray_GETPTR2(itemArray, i, 0);

				#pragma omp simd reduction(+:mean)
				for (unsigned int j = 0; j < numFeatures; j++)
					mean += v[j];
				mean /= (double)numFeatures;

				#pragma omp simd reduction(+:meanSquares[i])
				for (unsigned int j = 0; j < numFeatures; j++)
				{
					double val = v[j] - mean;
					rawDemeaned[i * numFeatures + j] = val;
					meanSquares[i] += val * val;
				}
			}
			else
			{
				for (unsigned int j = 0; j < numFeatures; j++)
					mean += GET_2D_DOUBLE(itemArray, i, j);
				mean /= (double)numFeatures;

				for (unsigned int j = 0; j < numFeatures; j++)
				{
					double val = GET_2D_DOUBLE(itemArray, i, j) - mean;
					rawDemeaned[i * numFeatures + j] = val;
					meanSquares[i] += val * val;
				}
			}
			meanSquares[i] /= (double)numFeatures;
		}

		// and then we operate on this demeaned array
		itemArray = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, rawDemeaned);
	}

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

	// clean up stuff used for correlation
	if (DistanceFunction == &Correlation)
	{
		delete [] meanSquares;
		itemArray->~tagPyArrayObject();
		delete [] rawDemeaned;
	}


	Py_RETURN_NONE;
}

static PyObject* GetClusteringDistance(PyObject *self, PyObject *args)
{
	// == Read in the arguments as a generic python objects first
	PyObject *s1;	// this should be the array that is passed to us
	PyObject *s2;	// output array because the number of values in the condensed distance array
	bool normalize = true;
	// is a large factorial that can exceed npy_intp, and thus it's easier to
	// allocate the output in python first

	if (!PyArg_ParseTuple(args, "OO|p", &s1, &s2, &normalize))	// "O" means object, 'p' means bool
		return nullptr;

	// == Cast the generic python objects to Numpy array objects
	PyArrayObject *solution1 = (PyArrayObject*)PyArray_FROM_OTF(s1, NPY_INT64, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *solution2 = (PyArrayObject*)PyArray_FROM_OTF(s2, NPY_INT64, NPY_ARRAY_IN_ARRAY);

	// get dimensions
	npy_intp *dims = PyArray_DIMS(solution1);
	unsigned long long numItems = dims[0];


	unsigned long long numPairs = numItems * (numItems - 1) / 2;

	Indexer* indexer = new Indexer(numItems);

	double dist = 0.0;

	// calculate pairwise relationships between items in each solution
	// and calc distance
#pragma omp parallel for
	for (unsigned long long i = 0; i < numPairs; i++)
	{
		unsigned long row = indexer->RowIndex(i);
		unsigned long col = indexer->ColIndex(i, row);

		bool sol1Pair = GET_1D_INT(solution1, row) == GET_1D_INT(solution1, col);
		bool sol2Pair = GET_1D_INT(solution2, row) == GET_1D_INT(solution2, col);

		if (sol1Pair != sol2Pair)
		{
			#pragma omp atomic
			dist += 1;
		}
	}

	delete indexer;

	if (normalize)
		dist /= (double)numPairs;

	return PyFloat_FromDouble(dist);
}

static PyObject* GetClusteringDistances(PyObject *self, PyObject *args)
{
	// == Read in the arguments as a generic python objects first
	PyObject *arg1;	// this should be the array that is passed to us
	PyObject *arg2;	// output array because the number of values in the condensed distance array
	bool normalize = true;
	// is a large factorial that can exceed npy_intp, and thus it's easier to
	// allocate the output in python first

	if (!PyArg_ParseTuple(args, "OO|p", &arg1, &arg2, &normalize))	// "O" means object, 'p' means bool
		return nullptr;

	// == Cast the generic python objects to Numpy array objects
	PyArrayObject *clusterSolutions = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_INT64, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *clusterDistances = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

	// get dimensions
	npy_intp *dims = PyArray_DIMS(clusterSolutions);
	unsigned long long numSolutions = dims[0];
	unsigned long long numItems = dims[1];
	unsigned long long numSolutionPairs = numSolutions * (numSolutions - 1) / 2;
	unsigned long long numItemPairs = numItems * (numItems - 1) / 2;

	// pairwise indexers
	Indexer* itemIndexer = new Indexer(numItems);
	Indexer* solutionIndexer = new Indexer(numSolutions);


	cout << "item distances" << endl;
	// pre-compute within-solution pairwise indicator variables
	dynamic_bitset<> *solutionPairs = new dynamic_bitset<>[numSolutionPairs];

	time_t start, end;

	start = time(nullptr);

	#pragma omp parallel for
	for (unsigned long long solution = 0; solution < numSolutions; solution++)
	{
		solutionPairs[solution] = dynamic_bitset<>(numItemPairs);
		for (unsigned long long j = 0; j < numItemPairs; j++)
		{
			unsigned long item1 = itemIndexer->RowIndex(j);
			unsigned long item2 = itemIndexer->ColIndex(j, item1);
			solutionPairs[solution].set(j, GET_2D_INT(clusterSolutions, solution, item1) == GET_2D_INT(clusterSolutions, solution, item2));
		}
	}
	end = time(nullptr);
	double diff = difftime(end, start);
	cout << "With-solution vertex distances " << diff << " seconds; average " << diff / (double)numSolutions << " each" << endl;

	// for pairs of solutions, compute their distances
	start = time(nullptr);
	#pragma omp parallel for
	for (unsigned long long solutionPair = 0; solutionPair < numSolutionPairs; solutionPair++)
	{
		unsigned long solution1 = solutionIndexer->RowIndex(solutionPair);
		unsigned long solution2 = solutionIndexer->ColIndex(solutionPair, solution1);

		// this is a hamming distance, no?
		GET_1D_DOUBLE(clusterDistances, solutionPair) = (double)(solutionPairs[solution1] ^ solutionPairs[solution2]).count() / (double)(solutionPairs[solution1] | solutionPairs[solution2]).count();
	}
	end = time(nullptr);
	diff = difftime(end, start);
	cout << "Solution distances " << diff << " seconds; average " << diff / (double)numSolutionPairs << " each" << endl;


	delete [] solutionPairs;

	Py_RETURN_NONE;
}


static PyObject* GetClusteringDistancesAVX(PyObject *self, PyObject *args)
{
	// == Read in the arguments as a generic python objects first
	PyObject* arg1;    // this should be the array that is passed to us
	PyObject* arg2;    // output array because the number of values in the condensed distance array
	// is a large factorial that can exceed npy_intp, and thus it's easier to
	// allocate the output in python first

	if (!PyArg_ParseTuple(args, "OO|pp", &arg1, &arg2))    // "O" means object, 'p' means bool
		return nullptr;

	// == Cast the generic python objects to Numpy array objects
	PyArrayObject* clusterSolutions;
	int arrayType = PyArray_TYPE((PyArrayObject*)PyArray_FROM_O(arg1));
	switch (arrayType)
	{
		case NPY_UINT8:
			clusterSolutions = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
			break;
		case NPY_UINT16:
			clusterSolutions = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
			break;
		default:
			clusterSolutions = nullptr;
	}

	PyArrayObject* clusterDistances = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);


	if (clusterSolutions == nullptr)
	{
		PyErr_SetString(PyExc_ValueError, "Cluster solutions need to be in uint8 or uint16");
		Py_RETURN_NONE;
	}

	// get dimensions
	npy_intp* dims = PyArray_DIMS(clusterSolutions);
	unsigned long long numSolutions = dims[0];
	unsigned long long numItems = dims[1];
	unsigned long long numSolutionPairs = numSolutions * (numSolutions - 1) / 2;
	unsigned long long numItemPairs = numItems * (numItems - 1) / 2;

	// pairwise indexers
	Indexer* itemIndexer = new Indexer(numItems);
	Indexer* solutionIndexer = new Indexer(numSolutions);

	void(*DistanceFunction)(PyArrayObject*, PyArrayObject*,
		unsigned long long, unsigned long long,
		unsigned long long, unsigned long long,
		Indexer *, Indexer *)  = nullptr;

	switch (arrayType)
	{
		case NPY_UINT8:
			DistanceFunction = &ClusteringDistanceUINT8;
			break;
		case NPY_UINT16:
			DistanceFunction = &ClusteringDistanceUINT16;
			break;
	}
	(*DistanceFunction)(clusterSolutions, clusterDistances,
		numSolutions, numItems,
		numSolutionPairs, numItemPairs,
		itemIndexer, solutionIndexer);


	#pragma omp parallel for simd
	for (unsigned long long i = 0; i < numSolutionPairs; i++)
		GET_1D_DOUBLE(clusterDistances, i) /= (double)numItemPairs;

	Py_RETURN_NONE;
}

static PyObject* GetClusteringDistancesJaccardAVX(PyObject *self, PyObject *args)
{
	// == Read in the arguments as a generic python objects first
	PyObject* arg1;    // this should be the array that is passed to us
	PyObject* arg2;    // output array because the number of values in the condensed distance array
	// is a large factorial that can exceed npy_intp, and thus it's easier to
	// allocate the output in python first

	if (!PyArg_ParseTuple(args, "OO|pp", &arg1, &arg2))    // "O" means object, 'p' means bool
		return nullptr;

	// == Cast the generic python objects to Numpy array objects
	PyArrayObject* clusterSolutions;
	int arrayType = PyArray_TYPE((PyArrayObject*)PyArray_FROM_O(arg1));
	switch (arrayType)
	{
		case NPY_UINT8:
			clusterSolutions = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
			break;
		case NPY_UINT16:
			clusterSolutions = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
			break;
		default:
			clusterSolutions = nullptr;
	}

	PyArrayObject* clusterDistances = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);


	if (clusterSolutions == nullptr)
	{
		PyErr_SetString(PyExc_ValueError, "Cluster solutions need to be in uint8 or uint16");
		Py_RETURN_NONE;
	}

	// get dimensions
	npy_intp* dims = PyArray_DIMS(clusterSolutions);
	unsigned long long numSolutions = dims[0];
	unsigned long long numItems = dims[1];
	unsigned long long numSolutionPairs = numSolutions * (numSolutions - 1) / 2;
	unsigned long long numItemPairs = numItems * (numItems - 1) / 2;

	// pairwise indexers
	Indexer* itemIndexer = new Indexer(numItems);
	Indexer* solutionIndexer = new Indexer(numSolutions);

	void(*DistanceFunction)(PyArrayObject*, PyArrayObject*,
							unsigned long long, unsigned long long,
							unsigned long long, unsigned long long,
							Indexer *, Indexer *)  = nullptr;

	switch (arrayType)
	{
		case NPY_UINT8:
			DistanceFunction = &ClusteringDistanceUINT8Jaccard;
			break;
		case NPY_UINT16:
			DistanceFunction = &ClusteringDistanceUINT16Jaccard;
			break;
	}
	(*DistanceFunction)(clusterSolutions, clusterDistances,
						numSolutions, numItems,
						numSolutionPairs, numItemPairs,
						itemIndexer, solutionIndexer);



	#pragma omp parallel for simd
	for (unsigned long long i = 0; i < numSolutionPairs; i++)
		GET_1D_DOUBLE(clusterDistances, i) /= (double)numItemPairs;

	Py_RETURN_NONE;
}


static void ClusteringDistanceUINT8(PyArrayObject* clusterSolutions, PyArrayObject* clusterDistances,
									unsigned long long numSolutions, unsigned long long numItems,
									unsigned long long numSolutionPairs, unsigned long long numItemPairs,
									Indexer *itemIndexer, Indexer *solutionIndexer)
{
	// we use a raw bit array to store things because it's faster and we can use AVX intrinsics on them
	u_int64_t ** bitArrays = new u_int64_t*[numSolutions];
	unsigned long long nFullInts = numItemPairs / 64;
	unsigned long long nOverflowBits = numItemPairs % 64;
	unsigned long long nTotalInts = nFullInts + (nOverflowBits > 0 ? 1 : 0);

	// For each solution, compute whether each pair of items ended up in the same solution
#pragma omp parallel for
	for (unsigned long long solution = 0; solution < numSolutions; solution++)
	{
		bitArrays[solution] = new u_int64_t[nTotalInts];

		unsigned long count = 0;

		__m512i top, bottom;
		__mmask64 result;
		uint8_t topNums[64];
		uint8_t bottomNums[64];

		for (unsigned long long j = 0; j < numItemPairs; j += 64)
		{
			unsigned long item1 = itemIndexer->RowIndex(j);
			unsigned long item2 = itemIndexer->ColIndex(j, item1);
			if (item1 == itemIndexer->RowIndex(j + 64))
			{
				// top value is the same and bottom value doesn't wrap
#pragma omp simd
				for(int k = 0; k < 64; k++)
					topNums[k] = *(u_int8_t *)PyArray_GETPTR2(clusterSolutions, solution, item1);
				top = _mm512_loadu_si512(topNums);
//				top = _mm512_set1_epi8(*(char*)PyArray_GETPTR2(clusterSolutions, solution, item1));
				bottom = _mm512_loadu_si512(PyArray_GETPTR2(clusterSolutions, solution, item2));
			}
			else
			{
				for (int c = 0; c < 64; c++)
				{
					if ((c + j) < numItemPairs)
					{
						item1 = itemIndexer->RowIndex(j + c);
						item2 = itemIndexer->ColIndex(j + c, item1);
						topNums[c] = *(uint8_t*)PyArray_GETPTR2(clusterSolutions, solution, item1);
						bottomNums[c] = *(uint8_t*)PyArray_GETPTR2(clusterSolutions, solution, item2);
					}
					else
					{
						topNums[c] = 0;
						bottomNums[c] = 0;
					}
				}

				top = _mm512_loadu_si512(topNums);
				bottom = _mm512_loadu_si512(bottomNums);
			}
			result = _mm512_cmpeq_epu8_mask(top, bottom);
			_store_mask64(((__mmask64*)(&bitArrays[solution][count++])), result);
		}

	}

	// for pairs of solutions, compute their distances
	#pragma omp parallel for
	for (unsigned long long solutionPair = 0; solutionPair < numSolutionPairs; solutionPair++)
	{
		unsigned long solution1 = solutionIndexer->RowIndex(solutionPair);
		unsigned long solution2 = solutionIndexer->ColIndex(solutionPair, solution1);

		unsigned long long count = 0;
		#pragma omp simd reduction(+:count)
		for(unsigned long i = 0; i < nTotalInts; i++)
		{ // we don't have to think about the last half-full int because the extra bits are all 0 and guaranteed to XOR to false and don't count
			count += _mm_popcnt_u64(bitArrays[solution1][i] ^ bitArrays[solution2][i]);
		}
		GET_1D_DOUBLE(clusterDistances, solutionPair) = (double)count;// / (double)numSolutionPairs;
	}

	for (int i = 0; i < numSolutions; i++)
		delete [] bitArrays[i];
	delete [] bitArrays;
}

static void ClusteringDistanceUINT8Jaccard(PyArrayObject* clusterSolutions, PyArrayObject* clusterDistances,
									unsigned long long numSolutions, unsigned long long numItems,
									unsigned long long numSolutionPairs, unsigned long long numItemPairs,
									Indexer *itemIndexer, Indexer *solutionIndexer)
{
	// we use a raw bit array to store things because it's faster and we can use AVX intrinsics on them
	u_int64_t *** bitArrays = new u_int64_t**[numSolutions];
	unsigned long long nFullInts = numItems / 64;
	unsigned long long nOverflowBits = numItems % 64;
	unsigned long long nTotalInts = nFullInts + (nOverflowBits > 0 ? 1 : 0);

#pragma omp parallel for
	// For each solution, compute whether each pair of items ended up in the same solution
	// and store the full matrix instead of the pairwise matrix
	for (unsigned long long solution = 0; solution < numSolutions; solution++)
	{
		bitArrays[solution] = new u_int64_t*[numItems];

		__m512i top, bottom;
		__mmask64 result;
		uint8_t topNums[64];
		uint8_t bottomNums[64];

		for (unsigned long long i = 0; i < numItems; i ++)
		{
			unsigned long count = 0;
			bitArrays[solution][i] = new u_int64_t[nTotalInts];

			for (int k = 0; k < 64; k++)
				topNums[k] = *(u_int8_t *)PyArray_GETPTR2(clusterSolutions, solution, i);
			top = _mm512_loadu_si512(topNums);

			unsigned long long j = 0;
			for (; j < numItems - 64; j += 64)
			{
				bottom = _mm512_loadu_si512(PyArray_GETPTR2(clusterSolutions, solution, j));
				result = _mm512_cmpeq_epu8_mask(top, bottom);
				_store_mask64(((__mmask64*)(&bitArrays[solution][i][count++])), result);
			}
			for (int k = 0; k < 64; k++)
			{
				if ((k + j) < numItems)
				{
					topNums[k] = *(u_int8_t *)PyArray_GETPTR2(clusterSolutions, solution, i);
					bottomNums[k] = *(u_int8_t *)PyArray_GETPTR2(clusterSolutions, solution, j);
				}
				else
				{
					topNums[k] = 0;
					bottomNums[k] = 0;
				}
			}
			top = _mm512_loadu_si512(topNums);
			bottom = _mm512_loadu_si512(bottomNums);
			result = _mm512_cmpeq_epu8_mask(top, bottom);
			_store_mask64(((__mmask64*)(&bitArrays[solution][i][count++])), result);
		}

	}

	// for pairs of solutions, compute their distances
#pragma omp parallel for
	for (unsigned long long solutionPair = 0; solutionPair < numSolutionPairs; solutionPair++)
	{
		unsigned long solution1 = solutionIndexer->RowIndex(solutionPair);
		unsigned long solution2 = solutionIndexer->ColIndex(solutionPair, solution1);

		double val = 0;
		for(unsigned long i = 0; i < numItems; i++)
		{
			unsigned long long count = 0;
			unsigned long long bottom = 0;
			#pragma omp simd reduction(+:count, bottom)
			for (unsigned long j = 0; j < nTotalInts;  j++)
			{
				count += _mm_popcnt_u64(bitArrays[solution1][i][j] ^ bitArrays[solution2][i][j]);
				bottom += _mm_popcnt_u64(bitArrays[solution1][i][j] | bitArrays[solution2][i][j]);
			}
			val += (double)count / (double)bottom;
		}
		GET_1D_DOUBLE(clusterDistances, solutionPair) = val / (double)numSolutionPairs;
	}

	for (int i = 0; i < numSolutions; i++)
	{
		for (int j = 0; j < numItems; j++)
			delete[] bitArrays[i][j];
		delete[] bitArrays[i];
	}
	delete [] bitArrays;
}

static void ClusteringDistanceUINT16(PyArrayObject* clusterSolutions, PyArrayObject* clusterDistances,
									 unsigned long long numSolutions, unsigned long long numItems,
									unsigned long long numSolutionPairs, unsigned long long numItemPairs,
									Indexer *itemIndexer, Indexer *solutionIndexer)
{
	// we use a raw bit array to store things because it's faster and we can use AVX intrinsics on them
	u_int32_t ** bitArrays = new u_int32_t*[numSolutions];
	unsigned long long nFullInts = numItemPairs / 32;
	unsigned long long nOverflowBits = numItemPairs % 32;
	unsigned long long nTotalInts = nFullInts + (nOverflowBits > 0 ? 1 : 0);

#pragma omp parallel for
	for (unsigned long long solution = 0; solution < numSolutions; solution++)
	{
		bitArrays[solution] = new u_int32_t[nTotalInts];

		unsigned long count = 0;

		__m512i top, bottom;
		__mmask32 result;
		uint16_t topNums[32];
		uint16_t bottomNums[32];

		for (unsigned long long j = 0; j < numItemPairs; j += 32)
		{
			unsigned long item1 = itemIndexer->RowIndex(j);
			unsigned long item2 = itemIndexer->ColIndex(j, item1);
			if (item1 == itemIndexer->RowIndex(j + 32))
			{
				// top value is the same and bottom value doesn't wrap
				#pragma omp simd
				for(int k = 0; k < 32; k++)
					topNums[k] = *(u_int16_t *)PyArray_GETPTR2(clusterSolutions, solution, item1);
				top = _mm512_loadu_si512(topNums);
//				top = _mm512_set1_epi16(*(u_int16_t*)PyArray_GETPTR2(clusterSolutions, solution, item1));
				bottom = _mm512_loadu_si512(PyArray_GETPTR2(clusterSolutions, solution, item2));
			}
			else
			{
				for (int c = 0; c < 32; c++)
				{
					if ((c + j) < numItemPairs)
					{
						item1 = itemIndexer->RowIndex(j + c);
						item2 = itemIndexer->ColIndex(j + c, item1);
						topNums[c] = *(uint16_t*)PyArray_GETPTR2(clusterSolutions, solution, item1);
						bottomNums[c] = *(uint16_t*)PyArray_GETPTR2(clusterSolutions, solution, item2);
					}
					else
					{
						topNums[c] = 0;
						bottomNums[c] = 0;
					}
				}

				top = _mm512_loadu_si512(topNums);
				bottom = _mm512_loadu_si512(bottomNums);
			}
			result = _mm512_cmpeq_epu16_mask(top, bottom);
			_store_mask32(((__mmask32*)(&bitArrays[solution][count++])), result);
		}
	}

	// for pairs of solutions, compute their distances
	#pragma omp parallel for
	for (unsigned long long solutionPair = 0; solutionPair < numSolutionPairs; solutionPair++)
	{
		unsigned long solution1 = solutionIndexer->RowIndex(solutionPair);
		unsigned long solution2 = solutionIndexer->ColIndex(solutionPair, solution1);

		unsigned long long count = 0;
		#pragma omp simd reduction(+:count)
		for(unsigned long i = 0; i < nTotalInts; i++)
		{ // we don't have to think about the last half-full int because the extra bits are all 0 and guaranteed to XOR to false and don't count
			count += _mm_popcnt_u32(bitArrays[solution1][i] ^ bitArrays[solution2][i]);
		}
		GET_1D_DOUBLE(clusterDistances, solutionPair) = (double)count;// / (double)numSolutionPairs;
	}

	for (int i = 0; i < numSolutions; i++)
		delete [] bitArrays[i];
	delete [] bitArrays;
}


static void ClusteringDistanceUINT16Jaccard(PyArrayObject* clusterSolutions, PyArrayObject* clusterDistances,
									 unsigned long long numSolutions, unsigned long long numItems,
									 unsigned long long numSolutionPairs, unsigned long long numItemPairs,
									 Indexer *itemIndexer, Indexer *solutionIndexer)
{
	u_int32_t *** bitArrays = new u_int32_t**[numSolutions];
	unsigned long long nFullInts = numItems / 32;
	unsigned long long nOverflowBits = numItems % 32;
	unsigned long long nTotalInts = nFullInts + (nOverflowBits > 0 ? 1 : 0);

	// For each solution, compute whether each pair of items ended up in the same solution
	// and store the full matrix instead of the pairwise matrix

#pragma omp parallel for
	for (unsigned long long solution = 0; solution < numSolutions; solution++)
	{
		bitArrays[solution] = new u_int32_t*[numItems];


		__m512i top, bottom;
		__mmask32 result;
		uint8_t topNums[32];
		uint8_t bottomNums[32];
		for (unsigned long long i = 0; i < numItems; i ++)
		{
			unsigned long count = 0;
			bitArrays[solution][i] = new u_int32_t[nTotalInts];

			for (int k = 0; k < 32; k++)
				topNums[k] = *(u_int8_t *)PyArray_GETPTR2(clusterSolutions, solution, i);
			top = _mm512_loadu_si512(topNums);

			unsigned long long j = 0;
			for (; j < numItems - 32; j += 32)
			{
				bottom = _mm512_loadu_si512(PyArray_GETPTR2(clusterSolutions, solution, j));
				result = _mm512_cmpeq_epu16_mask(top, bottom);
				_store_mask32(((__mmask32*)(&bitArrays[solution][i][count++])), result);
			}
			for (int k = 0; k < 32; k++)
			{
				if ((k + j) < numItems)
				{
					topNums[k] = *(u_int16_t *)PyArray_GETPTR2(clusterSolutions, solution, i);
					bottomNums[k] = *(u_int16_t *)PyArray_GETPTR2(clusterSolutions, solution, j);
				}
				else
				{
					topNums[k] = 0;
					bottomNums[k] = 0;
				}
			}
			top = _mm512_loadu_si512(topNums);
			bottom = _mm512_loadu_si512(bottomNums);
			result = _mm512_cmpeq_epu16_mask(top, bottom);
			_store_mask32(((__mmask32*)(&bitArrays[solution][i][count++])), result);
		}

	}

	// for pairs of solutions, compute their distances
#pragma omp parallel for
	for (unsigned long long solutionPair = 0; solutionPair < numSolutionPairs; solutionPair++)
	{
		unsigned long solution1 = solutionIndexer->RowIndex(solutionPair);
		unsigned long solution2 = solutionIndexer->ColIndex(solutionPair, solution1);

		double val = 0;
		for(unsigned long i = 0; i < numItems; i++)
		{
			unsigned long long count = 0;
			unsigned long long bottom = 0;
#pragma omp simd reduction(+:count, bottom)
			for (unsigned long j = 0; j < nFullInts; j++)
			{
				count += _mm_popcnt_u32(bitArrays[solution1][i][j] ^ bitArrays[solution2][i][j]);
				bottom += _mm_popcnt_u32(bitArrays[solution1][i][j] | bitArrays[solution2][i][j]);
			}
			val += (double)count / (double)bottom;
		}
		GET_1D_DOUBLE(clusterDistances, solutionPair) = val / (double)numSolutionPairs;
	}

	for (int i = 0; i < numSolutions; i++)
	{
		for (int j = 0; j < numItems; j++)
			delete[] bitArrays[i][j];
		delete[] bitArrays[i];
	}
	delete [] bitArrays;
}

