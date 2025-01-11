import time

import numpy as np
import torch
from kymatio.torch import Scattering2D
from sklearn import cluster
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torchvision import datasets, transforms

from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP, \
    LeastSquaresSubspaceClustering
from decomposition.dim_reduction import dim_reduction
from metrics.cluster.accuracy import clustering_accuracy, self_representation_loss, self_representation_sparsity, \
    self_representation_connectivity

# =================================================
# Prepare MNIST dataset
# =================================================

print('Prepare scattering...')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

scattering = Scattering2D(J=3, shape=(32, 32))
if use_cuda:
    scattering = scattering.cuda()

print('Prepare MNIST...')
transforms_MNIST = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])
MNIST_train = datasets.MNIST('./data', train=True, download=True, transform=transforms_MNIST)
MNIST_test = datasets.MNIST('./data', train=False, download=True, transform=transforms_MNIST)
MNIST_train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=4000, shuffle=True)
MNIST_test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=4000, shuffle=False)

raw_train_data, label_train = next(iter(MNIST_train_loader))  # data shape: torch.Size([60000, 1, 28, 28])
raw_test_data, label_test = next(iter(MNIST_test_loader))  # data shape: torch.Size([10000, 1, 28, 28])
label = torch.cat((label_train, label_test), 0)

print('Computing scattering on MNIST...')
if use_cuda:
    raw_train_data = raw_train_data.cuda()
    raw_test_data = raw_test_data.cuda()

train_data = torch.tensor(scattering(raw_train_data))
test_data = torch.tensor(scattering(raw_test_data))
print(type(train_data), type(test_data))
print(train_data.shape, test_data.shape)
data = torch.cat((train_data, test_data), 0)

print('Data preprocessing....')
n_sample = data.shape[0]

# scattering transform normalization
data = data.cpu().numpy().reshape(n_sample, data.shape[2], -1)
image_norm = np.linalg.norm(data, ord=np.inf, axis=2, keepdims=True)  # infinity norm of each transform
data = data / image_norm  # normalize each scattering transform to the range [-1, 1]
data = data.reshape(n_sample, -1)  # fatten and concatenate all transforms

# dimension reduction
data = dim_reduction(data, 500)  # dimension reduction by PCA

label = label.numpy()

# =================================================
# Create cluster objects
# =================================================
print('Begin clustering...')

# Baseline: non-subspace clustering methods
# model_kmeans = cluster.KMeans(n_clusters=10)  # k-means as baseline
#
# model_spectral = cluster.SpectralClustering(n_clusters=10, affinity='nearest_neighbors',
#                                             n_neighbors=5)  # spectral clustering as baseline

# model_lsr = LeastSquaresSubspaceClustering(n_clusters=10, gamma=200)

# Our work: elastic net subspace clustering (EnSC)
# You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
# model_ssc_bp = ElasticNetSubspaceClustering(n_clusters=10, affinity='nearest_neighbors', algorithm='spams', n_jobs=-1,
#                                             active_support=True, gamma=200, tau=1.0)

# Our work: sparse subspace clustering by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=10, affinity='symmetrize', n_nonzero=5, thr=1.0e-5)

clustering_algorithms = (
    # ('KMeans', model_kmeans),
    # ('Spectral Clustering', model_spectral),
    # ('LSR', model_lsr),
    # ('SSC-BP', model_ssc_bp),
    ('SSC-OMP', model_ssc_omp),
)

res = list()
for name, algorithm in clustering_algorithms:
    t_begin = time.time()
    algorithm.fit(data)
    t_end = time.time()
    acc = clustering_accuracy(label, algorithm.labels_)
    nmi = normalized_mutual_info_score(label, algorithm.labels_, average_method='geometric')
    ari = adjusted_rand_score(label, algorithm.labels_)
    loss = self_representation_loss(label, algorithm.representation_matrix_)
    str = "Algorithm: {}. acc: {}, nmi: {}, ari: {}, loss: {}, Running time: {}".format(name, acc, nmi, ari, loss, t_end - t_begin)
    res.append(str)
    print('Algorithm: {}. acc: {}, nmi: {}, ari: {}, loss: {}, Running time: {}'.format(name, acc, nmi, ari, loss, t_end - t_begin))
for str in res:
    print(str)
