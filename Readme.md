A dirty and barebones parallelized pairwise distance calculator as a replacement for `scipy.spatial.distance.pdist`, which is single-threaded.

To install:
``python3 setup.py install --user``

This *requires* a processor that supports AVX512 instructions, specifically AVX512F and AVX512BW sets.

Currently provides correlation, euclidean, and random forest distances. 

To use:

assuming `data` is a 2D numpy array in which the first index is samples and the 2nd index is features
````
out = Pairwise.pdist(data, metric = "euclidean")
````
`Pairwise.pdist` has the same signature as `scipy.spatial.distance.pdist`, `pdist(X, metric = "euclidean", *, out = None)`, and returns the condensed distance matrix. `metric` is one of `euclidean`, `correlation`, `randomforest`, `clustering`, `clusteringavx`, or `jaccard`. If `out` is given, it must be a C-contiguous float64 array of length `n * (n - 1) / 2` and the distances are written into it.

The older functions are still available:
````
n = data.shape[0]
out = numpy.zeros(int(n * (n - 1) / 2))
Pairwise.GetPairwise[Euclidean|Correlation|RandomForest]Distance(data, out)
````
`out` will then contain the same data as returned by `pdist`

The actual math is slower than numpy, but thanks to mulithreading, this is faster on larger arrays. 
All and all the complexity is still O(n^2), but this gives you a better multiplier.
