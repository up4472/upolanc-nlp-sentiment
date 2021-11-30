import itertools
import numpy

def UPGMA (distances : numpy.ndarray) -> numpy.ndarray :
	"""Unweighted pair group method with arithmetic mean (UPGMA) agglomerative clustering.

	Parameters
	----------
	distances: np.ndarray
		A two dimensional, square, symmetric matrix containing distances between data
		points. The diagonal is zeros.

	Returns
	-------
	np.ndarray
		The linkage matrix, as specified in scipy. Briefly, this should be a 2d matrix
		each row containing 4 elements. The first and second element should denote the
		cluster IDs being merged, the third element should be the distance, and the
		fourth element should be the number of elements within this new cluster. Any
		new cluster should be assigned an incrementing ID, e.g. after the first step
		where the first two points are merged into a cluster, the new cluster ID should
		be N, then N+1, N+2, ... in subsequent steps.

	Notes
	-----
	You can validate your implementation to scipy's `cluster.hierarchy.linkage`
	function using average linkage. UPGMA and average linkage agglomerative clustering
	are the same thing. Your function should return identical results in all cases.

	"""

	# Number of initial clusters
	n = distances.shape[0]

	# Create copy of the distanc matrix, since we will modify inline
	distances = distances.astype(dtype = float, copy = True)

	# Create empty cluster buffer
	clusters = numpy.zeros(shape = (n - 1, 4), dtype = float)

	# Crete cluster information data
	cluster_done = numpy.full(shape = n, fill_value = False, dtype = bool)
	cluster_size = numpy.ones(shape = n, dtype = int)
	cluster_name = numpy.arange(start = 0, stop = n)

	# Infinite loop starting from 'n'
	for cluster_id in itertools.count(start = n) :
		# Find minimum row and column indices
		min_dist = numpy.inf
		min_x = -1
		min_y = -1

		for x in range(n) :
			if cluster_done[x] :
				continue

			for y in range(x) :
				if cluster_done[y] :
					continue

				dist = distances[x, y]

				if min_dist > dist :
					min_dist = dist
					min_x = x
					min_y = y

		# If every cluster has been proceesed break the loop
		if min_x < 0 or min_y < 0 :
			break

		# Compute the new distance vector
		x_dist = distances[min_x]
		y_dist = distances[min_y]

		# Get the cluster names (ids)
		x_id = cluster_name[min_x]
		y_id = cluster_name[min_y]

		cluster_done[min_y] = True
		cluster_name[min_x] = cluster_id

		for x in range(distances.shape[0]) :
			if not cluster_done[x] and x != min_x :
				mean = (
					(
						distances[min_x, x] * cluster_size[min_x] +
						distances[min_y, x] * cluster_size[min_y]
					) / (
						cluster_size[min_x] + cluster_size[min_y]
					)
				)

				distances[min_x, x] = mean
				distances[x, min_x] = mean

		# Update the cluster information data
		cluster_size[min_x] = cluster_size[min_x] + cluster_size[min_y]

		# Create new cluster
		clusters[cluster_id - n] = [x_id, y_id, distances[min_x, min_y], cluster_size[min_x]]

		# Append cluster to clusters
		#clusters = numpy.vstack([clusters, cluster])

	# Return clusters
	return clusters

def jukes_cantor (p : float) -> float:
	"""The Jukes-Cantor correction for estimating genetic distances.

	Parameters
	----------
	p: float
		The proportional distance, i.e. the number of of mismatching symbols (Hamming
		distance) divided by the total sequence length.

	Returns
	-------
	float
		The corrected genetic distance.
	"""

	# Check proportioanl distance (log is underfined if p is greater or equal 0.75)
	if p < 3.0 / 4.0 :
		p = -3.0 / 4.0 * numpy.log(1.0 - 4.0 / 3.0 * p)

		# Fix negative zero
		if p == -0.0 :
			p = p * (-1.0)
	else :
		p = numpy.inf

	# Return corrected proportioanl distance
	return p

if __name__ == '__main__' :
	data = numpy.array([
		[ 0, 22, 39, 39, 41],
		[22,  0, 41, 41, 43],
		[39, 41,  0, 18, 20],
		[39, 41, 18,  0, 10],
		[41, 43, 20, 10,  0]], dtype = float)

	print(UPGMA(distances = data))
