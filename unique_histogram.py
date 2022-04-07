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

intervals = opt_points[0].intervals

interval_nums = {interval: [] for interval in intervals}
print(interval_nums)

for i in range(15000):
    # if opt_points[i].rev_len >= 500:
    if True:
        for interval in intervals:
            interval_nums[interval].append(late_points[i].get_data(interval))

for interval in reversed(intervals):
    interval_nums[interval] = np.asarray(interval_nums[interval])

    q25, q75 = np.percentile(interval_nums[interval], [0.25, 0.75])
    print(q25, q75)

    bin_width = 2 * (q75 - q25) * len(interval_nums[interval]) ** (-1/3)
    bin_width = bin_width if bin_width > 0 else 40
    print(len(interval_nums[interval]))
    print(max(interval_nums[interval]))
    print(min(interval_nums[interval]))
    print(bin_width)
    bins = round((max(interval_nums[interval]) - min(interval_nums[interval])) / bin_width)
    mean = np.mean(interval_nums[interval])
    median = np.median(interval_nums[interval])
    stddev = np.std(interval_nums[interval])
    print(f"Number of bins: {bins}")

    fig, ax = plt.subplots()

    # plt.axvline(median, color='r', linestyle='-', linewidth=0.5)
    # plt.axvline(mean-stddev, color = 'r', linestyle = '-.', linewidth=0.5)
    # plt.axvline(mean+stddev, color = 'r', linestyle = '-.', linewidth=0.5)
    plt.hist(interval_nums[interval], bins=bins, color='b', histtype='stepfilled')
    # log scale on y axis, do not plot 0 counts
    plt.yscale('log', nonposy='clip')
    # plt.title(f"Histogram at {interval if interval != intervals[-1] else interval + 1} timesteps (Late epoch)")
    plt.xlabel("Number of unique points", fontsize=22)
    plt.ylabel("log Count", fontsize=22)
    ax.tick_params(axis='both', labelsize=21)
    fig.set_size_inches(10, 8)
    plt.tight_layout()
    plt.show()
