import math
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(train_metrics, metric_name, plot_as_log=False):

    plt.figure(figsize=(8, 6))
    epochs = np.arange(len(train_metrics))

    if plot_as_log:
        train_metrics = [math.log(x) for x in train_metrics]

    plt.plot(epochs, train_metrics, label=f'Train {metric_name}', color='blue')

    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()