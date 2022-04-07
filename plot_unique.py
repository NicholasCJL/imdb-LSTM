from reviewunique import ReviewUnique
import matplotlib.pyplot as plt
import pickle
import numpy as np

path = "D:/Thesis/IMDb LSTM/Results/hyperband500_small_NoL2_1000_5-4/unique_points_24900_trained"
all_points = []
for i in range(0, 15000, 100):
    with open(f"{path}/{i}_{i+99}_120000.pkl", 'rb') as file:
        all_points.extend(pickle.load(file))

opt_points = [points[0] for points in all_points]
late_points = [points[1] for points in all_points]

fig, ax = plt.subplots()

intervals = opt_points[0].intervals
print(intervals)

x_opt = []
y_opt = []
x_late = []
y_late = []
opt_cmap = [review.rev_len if review.rev_len <= 500 else 500 for review in opt_points for _ in range(len(intervals))]
late_cmap = [review.rev_len if review.rev_len <= 500 else 500 for review in late_points for _ in range(len(intervals))]
for i in range(15000):
    if opt_points[i].rev_len >= 500:
    # if True:
        for interval in intervals:
            x_opt.append(interval)
            y_opt.append(opt_points[i].get_data(interval))
            x_late.append(interval)
            y_late.append(late_points[i].get_data(interval))

# im = ax.scatter(x_late, y_late, s=0.45, c=opt_cmap, cmap='viridis')
im = ax.scatter(x_late, y_late, s=0.45)
# cbar = fig.colorbar(im, orientation='vertical')
# cbar.ax.tick_params(labelsize=16)
# plt.title("Unique points (last 24900 points) vs length of poincare sequence (Late)")
plt.xlabel("Length of poincare sequence", fontsize=18)
plt.ylabel("Number of unique points", fontsize=18)
ax.tick_params(axis='both', labelsize=16)
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.show()