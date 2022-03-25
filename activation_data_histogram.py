# histogram of activation power spectrum gradients

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from normalise_activations_individual import RegressionData

min_length = 0
max_length = 9999
type = "trimmed_train"

path = "timesteps500_embed32_hidden60_vocab4000_4"

with open(f"Results/{path}/norm_activation_by_type_length/trimmed_train/power_fft_reg_data_train_trimmed_009.pkl", 'rb') as file:
    reg_data = pickle.load(file)

# os.mkdir(f'Results/{path}/norm_activation_by_type_length/{type}')
os.mkdir(f'Results/{path}/norm_activation_by_type_length/{type}/{min_length}-{max_length}')

for data in reg_data:
    x = data.slope
    x_length = data.input_length
    x = np.asarray([x[i] for i in range(len(x)) if (x_length[i] >= min_length and x_length[i] <= max_length)])
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
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Count")
    plt.title(f'Spread of exponents of {data.type} across dataset')
    plt.text(0.72, 0.9, f'Mean = {round(mean, 3)}\nMedian = {round(median, 3)}\nStddev = {round(stddev, 3)}',
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    fig.savefig(f"Results/{path}/norm_activation_by_type_length/{type}/{min_length}-{max_length}/{data.type}.png", dpi=fig.dpi)

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
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Count")
    plt.title(f'Spread of gradients of {data.type} across dataset (correct predictions)')
    plt.text(0.72, 0.9, f'Mean = {round(mean, 3)}\nMedian = {round(median, 3)}\nStddev = {round(stddev, 3)}',
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    fig.savefig(f"Results/{path}/norm_activation_by_type_length/{type}/{min_length}-{max_length}/{data.type}_c.png", dpi=fig.dpi)


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
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Count")
    plt.title(f'Spread of gradients of {data.type} across dataset (wrong predictions)')
    plt.text(0.72, 0.9, f'Mean = {round(mean, 3)}\nMedian = {round(median, 3)}\nStddev = {round(stddev, 3)}',
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    fig.savefig(f"Results/{path}/norm_activation_by_type_length/{type}/{min_length}-{max_length}/{data.type}_w.png", dpi=fig.dpi)
