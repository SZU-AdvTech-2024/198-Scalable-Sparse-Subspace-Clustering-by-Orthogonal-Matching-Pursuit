import time

from sklearn import cluster

from cluster.selfrepresentation import SparseSubspaceClusteringOMP, ElasticNetSubspaceClustering, \
    LeastSquaresSubspaceClustering
from gen_union_of_subspaces import gen_union_of_subspaces
from metrics.cluster.accuracy import clustering_accuracy, self_representation_loss, self_representation_sparsity, \
    self_representation_connectivity

# =================================================
# Generate dataset where data is drawn from a union of subspaces
# =================================================
ambient_dim = 9
subspace_dim = 6
num_subspaces = 5
num_points_per_subspace = [50, 100, 150, 250, 400, 600, 1000, 1500, 2000]
# [50, 100, 150, 250, 400, 600, 1000, 1500, 2000, 4000]
# [9000, 13000, 20000]

# =================================================
# Create cluster objects
# =================================================

# Baseline: non-subspace clustering methods
# model_kmeans = cluster.KMeans(n_clusters=num_subspaces)  # k-means as baseline
# model_spectral = cluster.SpectralClustering(n_clusters=num_subspaces, affinity='nearest_neighbors',
#                                             n_neighbors=6)  # spectral clustering as baseline

# model_lsr = ElasticNetSubspaceClustering(n_clusters=num_subspaces, algorithm='spams', gamma_nz=False, gamma=500, tau=0.0)
#
model_ssc_bp = ElasticNetSubspaceClustering(n_clusters=num_subspaces, algorithm='spams', active_support=True, gamma=500, tau=1.0)

# Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=num_subspaces, thr=1e-5)

clustering_algorithms = (
    # ('KMeans', model_kmeans),
    # ('Spectral Clustering', model_spectral),
    # ('LSR', model_lsr),
    ('SSC-BP', model_ssc_bp),
    ('SSC-OMP', model_ssc_omp),
)

res = list()
for num_points in num_points_per_subspace:
    data, label = gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points, 0.00)
    for name, algorithm in clustering_algorithms:
        t_begin = time.time()
        algorithm.fit(data)
        t_end = time.time()
        accuracy = clustering_accuracy(label, algorithm.labels_)
        if not (name == "KMeans" or name == "Spectral Clustering"):
            loss = self_representation_loss(label, algorithm.representation_matrix_)
            sparsity = self_representation_sparsity(algorithm.representation_matrix_)
            conn = self_representation_connectivity(label, algorithm.representation_matrix_)
            str = 'Algorithm: {}. datasize: {}. lablesize: {}. loss: {}. spa: {}. conn: {}. Clustering accuracy: {}. Running time: {}'.format(name, len(data), len(label), loss,
                                                                                                             sparsity,
                                                                                                             conn,
                                                                                                             accuracy,
                                                                                                             t_end - t_begin)
            print(str)
        else:
            str = 'Algorithm: {}. datasize: {}. lablesize: {}. Clustering accuracy: {}. Running time: {}'.format(name, len(data), len(label), accuracy, t_end - t_begin)
            print(str)
        res.append(str)
        # print('Algorithm: {}. Clustering accuracy: {}. Running time: {}'.format(name, accuracy, t_end - t_begin))
for str in res:
    print(str)
