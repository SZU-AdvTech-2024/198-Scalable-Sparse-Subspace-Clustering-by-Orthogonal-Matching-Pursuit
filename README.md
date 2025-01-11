# 基于正交匹配追踪的可扩展稀疏子空间聚类复现代码
论文：C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016

聚类算法作为类*SparseSubspaceClusteringOMP* 实现，具有用于学习聚类的 fit 函数。使用方式与 *sklearn.cluster* 中的 *KMeans*、*SpectralClustering* 和其他工具相同。

# 依赖
```
kymatio==0.3.0
matplotlib==3.10.0
numpy==2.2.1
progressbar33==2.4
scikit_learn==1.6.1
scipy==1.15.0
spams==2.6.5.4
torch==2.4.1
torchvision==0.19.1
```

该项目来源于https://github.com/ChongYou/subspace-clustering

