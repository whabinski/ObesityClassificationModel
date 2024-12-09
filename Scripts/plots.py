# Plot Metric function
#
# This function plots a training metric (loss, accuracy, etc.) over epochs for validation purposes
# Parameters:
# - metric_for_epoch: y values to plot
# - metric_name: metric name as string for labeling 
# - plot_as_log: boolean to plot function as log values

import math
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metric_for_epoch, metric_name, plot_as_log=False):

    plt.figure(figsize=(8, 6))
    epochs = np.arange(len(metric_for_epoch)) # x values

    # log metric
    if plot_as_log:
        metric_for_epoch = [math.log(x) for x in metric_for_epoch]

    # plot metric over epoch
    plt.plot(epochs, metric_for_epoch, label=f'Train {metric_name}', color='blue')

    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()