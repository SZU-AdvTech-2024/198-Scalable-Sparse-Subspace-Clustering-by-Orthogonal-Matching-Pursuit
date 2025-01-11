from matplotlib import pyplot as plt

algorithms = (
    ('KMeans', 'KMeans'),
    ('Spectral Clustering', 'Spectral'),
    ('SSC-BP', 'SSC-BP'),
    ('SSC-OMP', 'SSC-OMP')
)
# 用于存储每个算法的准确度和执行时间
accuracies = {name: [] for name, _ in algorithms}
times = {name: [] for name, _ in algorithms}
algorithm_names = [name for name, _ in algorithms]
data_sizes = [500, 1000, 5000, 10000, 50000, 100000, 500000]

accuracies['KMeans'] = [
    0.254,
    0.232173913,
    0.220582524,
    0.214679803,
    0.211744766,
    0.21704,
    0.22511
]
accuracies['Spectral Clustering'] = [
    0.284,
    0.300869565,
    0.560582524,
    0.690738916,
    0.967696909,
    0.98349,
    None
]
accuracies['SSC-OMP'] = [
    0.738,
    0.84,
    0.950291262,
    0.967586207,
    0.986141575,
    0.99191,
    0.996408
]
accuracies['SSC-BP'] = [
    0.94,
    0.975,
    0.9952,
    0.997,
    0.99912,
    0.99966,
    None
]

times['KMeans'] = [
    0.075809956,
    0.005505562,
    0.009048939,
    0.01001668,
    0.05099988,
    0.147060871,
    0.314495564
]
times['Spectral Clustering'] = [
    0.063757658,
    0.139601946,
    2.352295876,
    9.977127552,
    412.0423908,
    3182.154161,
    None, ]

times['SSC-OMP'] = [
    0.135050774,
    0.21713686,
    0.760440588,
    1.819101334,
    29.90525913,
    63.8990984,
    3480.74
]
times['SSC-BP'] = [
    0.26026535,
    0.3320539,
    1.409105062,
    3.062453747,
    46.71814227,
    136.8928695,
    7365.3
]

plt.figure(figsize=(20, 6))

plt.subplot(1, 1, 1)
for name in algorithm_names:
    plt.plot(data_sizes, accuracies[name], marker='o', label=f"{name}")  # 绘制每个算法的折线
plt.title('Clustering Accuracy')  # 标题
plt.xlabel('data points')  # x轴标签
plt.ylabel('Accuracy')  # y轴标签
plt.grid(True)
plt.legend()  # 显示图例
# 设置 x 轴刻度，仅显示指定的数字，并使用 MaxNLocator 来控制刻度数量
# ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=6))

# 调整150到500区间的间隔
plt.xticks([500, 10000, 50000, 100000, 500000])

# plt.subplot(1, 2, 2)  # 选中第二个子图
# for name in algorithm_names:
#     plt.plot(data_sizes, times[name], marker='o', label=name)  # 绘制每个算法的执行时间plt.title('Running Time')  # 标题
# plt.xlabel('Algorithm')  # x轴标签
# plt.ylabel('Time (seconds)')  # y轴标签
# plt.grid(True)

plt.tight_layout()
plt.show()
