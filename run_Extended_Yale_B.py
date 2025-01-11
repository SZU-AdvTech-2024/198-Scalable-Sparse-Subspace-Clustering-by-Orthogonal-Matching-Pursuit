import os
import numpy as np
import time
import torch
from sklearn import cluster
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.io import loadmat
import random

from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from metrics.cluster.accuracy import clustering_accuracy

# 设置数据集路径
data_path = 'data/EYB_fc.mat'  # https://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html


# =================================================
# 加载和预处理数据
# =================================================
def load_yale_b_data(mat_file, n_subjects=38, image_size=(48, 42)):
    """
    加载并预处理 Extended Yale B 数据集（从.mat 文件中读取）。

    参数：
    mat_file --.mat 文件路径
    n_subjects -- 选择的个体数量
    image_size -- 图像的缩放大小
    """
    data = loadmat(mat_file)  # 读取.mat 文件
    print(data['fea'].shape)
    X = data['fea']  # 图像数据保存在 'fea' 中
    Y = data['Label'].flatten()  # 标签保存在 'gnd' 中，并转换为一维数组

    # 随机选择 n 个个体
    subjects = np.unique(Y)[:n_subjects]  # 获取唯一的个体标签并选择 n 个
    selected_indices = np.isin(Y, subjects)  # 获取与所选个体标签匹配的图像索引

    X_selected = X[selected_indices]  # 选择符合条件的图像数据
    Y_selected = Y[selected_indices]  # 选择相应的标签

    # 对每张图像进行缩放，并展平
    images = []
    for img in X_selected:
        # 将图像重塑为 192x168，并进行缩放到 48x42
        img = img.reshape(192, 168)
        img_resized = np.resize(img, image_size)
        images.append(img_resized.flatten())

    # 转换为 NumPy 数组
    images = np.array(images)

    return images, Y_selected


# 加载数据
n_subjects = random.choice([2, 10, 20, 30, 38])
print(f'Loading data for {n_subjects} subjects...')
images, labels = load_yale_b_data(mat_file=data_path, n_subjects=n_subjects, image_size=(48, 42))

print(f'Loaded data shape: {images.shape}, labels shape: {labels.shape}')

# =================================================
# 数据预处理
# =================================================
# 数据归一化
images = images / 255.0  # 将像素值归一化到 [0, 1] 范围内

# 进行 PCA 降维
print('Applying PCA for dimension reduction...')
pca = PCA(n_components=500)
reduced_images = pca.fit_transform(images)

# =================================================
# 聚类算法
# =================================================
print('Begin clustering...')

# KMeans
model_kmeans = cluster.KMeans(n_clusters=n_subjects)

# Spectral Clustering
model_spectral = cluster.SpectralClustering(n_clusters=n_subjects, affinity='nearest_neighbors', n_neighbors=5)

# model_lsr = ElasticNetSubspaceClustering(n_clusters=n_subjects, algorithm='spams', gamma_nz=False, gamma=500, tau=0.0)
#
model_ssc_bp = ElasticNetSubspaceClustering(n_clusters=n_subjects, algorithm='spams', active_support=True, gamma=500, tau=1.0)

# Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=n_subjects, thr=1e-5)


clustering_algorithms = [
    ('KMeans', model_kmeans),
    ('Spectral Clustering', model_spectral),
    # ('LSR', model_lsr),
    ('SSC-BP', model_ssc_bp),
    ('SSC-OMP', model_ssc_omp),
]

# 计算聚类性能
for name, algorithm in clustering_algorithms:
    t_begin = time.time()
    algorithm.fit(reduced_images)
    t_end = time.time()
    acc = clustering_accuracy(labels, algorithm.labels_)
    nmi = normalized_mutual_info_score(labels, algorithm.labels_, average_method='geometric')
    ari = adjusted_rand_score(labels, algorithm.labels_)

    print(f'Algorithm: {name}. acc: {acc}, nmi: {nmi}, ari: {ari}, Running time: {t_end - t_begin:.4f}s')
