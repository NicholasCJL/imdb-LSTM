from reviewunique import ReviewUnique
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np

path = "D:/Thesis/IMDb LSTM/Results/hyperband500_small_NoL2_1000_5-4/unique_points_24900"
all_points = []
for i in range(0, 15000, 100):
    with open(f"{path}/{i}_{i+99}_120000.pkl", 'rb') as file:
        all_points.extend(pickle.load(file))

opt_points = [points[0] for points in all_points]
late_points = [points[1] for points in all_points]

intervals = opt_points[0].intervals

interval_nums = {interval: [] for interval in intervals}
print(interval_nums)

for i in range(15000):
    if opt_points[i].rev_len >= 500:
    # if True:
        for interval in intervals:
            interval_nums[interval].append(late_points[i].get_data(interval))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nbins = 500
cmap = matplotlib.cm.get_cmap('viridis')

for interval in intervals:
    hist, bins = np.histogram(interval_nums[interval], bins=nbins)
    for i in range(len(hist)):
        if hist[i] == 0:
            hist[i] = 1
    hist = np.log10(hist)
    xs = (bins[:-1] + bins[1:]) / 2
    c = cmap((interval-intervals[0]) / (intervals[-1] - intervals[0]))
    ax.bar(xs, hist, zs=interval, zdir='y', color=c, ec=c, alpha=0.8)

plt.tight_layout()
fig.set_size_inches(11, 8)

ax.set_xlabel('Number of unique points')
ax.xaxis.labelpad = 8
ax.set_ylabel('Timestep')
ax.yaxis.labelpad = 10
ax.set_zlabel('log Count')
ax.zaxis.labelpad = 8

plt.show()