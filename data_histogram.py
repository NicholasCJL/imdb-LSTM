import matplotlib.pyplot as plt
import numpy as np
import pickle
from normalise_activations_individual import RegressionData

with open("Results/timesteps150_embed32_hidden45_vocab4000/norm_activations_by_type_reg_stats/reg_data_all.pkl", 'rb') as file:
    reg_data = pickle.load(file)

for data in reg_data:
    x = np.asarray(data.slope)
    print(len(x))
    validation = data.iscorrect
    x_correct = np.asarray([x[i] for i in range(len(x)) if validation[i]])
    x_wrong = np.asarray([x[i] for i in range(len(x)) if not validation[i]])
    print(len(x_correct), len(x_wrong))

    # make this more streamlined if using more

    # all data
    q25, q75 = np.percentile(x, [0.25, 0.75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3) # Freedman-Diaconis rule
    print(max(x), min(x), bin_width)
    bins = round((max(x) - min(x)) / bin_width)
    mean = np.mean(x)
    median = np.median(x)
    stddev = np.std(x)
    print(f"Number of bins: {bins}")

    fig, ax = plt.subplots()

    plt.axvline(mean, color='r', linestyle='-', linewidth=0.5)
    plt.axvline(mean-stddev, color = 'r', linestyle = '-.', linewidth=0.5)
    plt.axvline(mean+stddev, color = 'r', linestyle = '-.', linewidth=0.5)
    plt.hist(x, bins=bins, color='b')
    plt.xlim([-2, 2])
    plt.xlabel("Gradient")
    plt.ylabel("Count")
    plt.title(f'Spread of gradients of {data.type} across dataset')
    plt.text(0.02, 0.9, f'Mean = {round(mean, 3)}\nMedian = {round(median, 3)}\nStddev = {round(stddev, 3)}',
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    plt.show()

    # correct data
    q25, q75 = np.percentile(x_correct, [0.25, 0.75])
    bin_width = 2 * (q75 - q25) * len(x_correct) ** (-1/3) # Freedman-Diaconis rule
    print(max(x_correct), min(x_correct), bin_width)
    bins = round((max(x_correct) - min(x_correct)) / bin_width)
    mean = np.mean(x_correct)
    median = np.median(x_correct)
    stddev = np.std(x_correct)
    print(f"Number of bins: {bins}")

    fig, ax = plt.subplots()

    plt.axvline(mean, color='r', linestyle='-', linewidth=0.5)
    plt.axvline(mean-stddev, color = 'r', linestyle = '-.', linewidth=0.5)
    plt.axvline(mean+stddev, color = 'r', linestyle = '-.', linewidth=0.5)
    plt.hist(x_correct, bins=bins, color='g')
    plt.xlim([-2, 2])
    plt.xlabel("Gradient")
    plt.ylabel("Count")
    plt.title(f'Spread of gradients of {data.type} across dataset (correct predictions)')
    plt.text(0.02, 0.9, f'Mean = {round(mean, 3)}\nMedian = {round(median, 3)}\nStddev = {round(stddev, 3)}',
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    plt.show()

    # wrong data
    q25, q75 = np.percentile(x_wrong, [0.25, 0.75])
    bin_width = 2 * (q75 - q25) * len(x_wrong) ** (-1 / 3)  # Freedman-Diaconis rule
    print(max(x_wrong), min(x_wrong), bin_width)
    bins = round((max(x_wrong) - min(x_wrong)) / bin_width)
    mean = np.mean(x_wrong)
    median = np.median(x_wrong)
    stddev = np.std(x_wrong)
    print(f"Number of bins: {bins}")

    fig, ax = plt.subplots()

    plt.axvline(mean, color='r', linestyle='-', linewidth=0.5)
    plt.axvline(mean - stddev, color='r', linestyle='-.', linewidth=0.5)
    plt.axvline(mean + stddev, color='r', linestyle='-.', linewidth=0.5)
    plt.hist(x_wrong, bins=bins, color='orange')
    plt.xlim([-2, 2])
    plt.xlabel("Gradient")
    plt.ylabel("Count")
    plt.title(f'Spread of gradients of {data.type} across dataset (wrong predictions)')
    plt.text(0.02, 0.9, f'Mean = {round(mean, 3)}\nMedian = {round(median, 3)}\nStddev = {round(stddev, 3)}',
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    plt.show()