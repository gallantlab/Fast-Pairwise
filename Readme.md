A dirty and barebones parallelized pairwise distance calculator as a replacement for `scipy.spatial.distance.pdist`, which is single-threaded.

To install:
``python3 setup.py install --user``

Currently provides correlation, euclidean, and random forest distances. 

To use:

assuming `data` is a 2D numpy array in which the first index is samples and the 2nd index is features
````
n = data.shape[0]
out = numpy.zeros(int(n * (n - 1) / 2))
Pairwise.GetPairwise[Euclidean|Correlation|RandomForest]Distance(data, out)
````
`out` will then contain the same data as returned by `pdist`

The actual math is slower than numpy, but thanks to mulithreading, this is faster on larger arrays. 
All and all the complexity is still O(n^2), but this gives you a better multiplier.