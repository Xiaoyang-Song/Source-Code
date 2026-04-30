import numpy as np
from matplotlib import pyplot as plt
import os
from collections import defaultdict


PCA = defaultdict(list)
PLS = defaultdict(list)
DCCA = defaultdict(list)
CLS = defaultdict(list)

SIGMAS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

for sigma in SIGMAS:

    log_file_path = os.path.join('checkpoint', 'simulation-circle-crack', f'results-sigma{sigma}.log')
    with open(log_file_path, "r") as file:
        lines = []
        for line in file:
            lines.append(line.strip())
            # print(line.strip())  # Use .strip() to remove newline characters

        # Get results
        PLS['F1'].append(float(lines[-1]))
        PLS['II'].append(float(lines[-3]))
        PLS['I'].append(float(lines[-5]))

        PCA['F1'].append(float(lines[74]))
        PCA['II'].append(float(lines[72]))
        PCA['I'].append(float(lines[70]))

        CLS['F1'].append(float(lines[49]))
        CLS['II'].append(float(lines[47]))
        CLS['I'].append(float(lines[45]))

        # DCCA
        dcca_t1 = float(lines[29][len("False Alarm Rate:"):].strip())
        dcca_t2 = float(lines[30][len("Mis-Detection Rate:"):].strip())
        dcca_f1 = float(lines[31][len("F1:"):].strip())

        DCCA['F1'].append(dcca_f1)
        DCCA['II'].append(dcca_t2)
        DCCA['I'].append(dcca_t1)

METRICS = ['F1', 'I', 'II']
METRIC_NAME = {
    'F1': 'F1 Score',
    'I': 'Type-I Error',
    'II': 'Type-II Error'
}

for metric in METRICS:
    # plt.figure(figsize=(10, 6))
    plt.plot(SIGMAS, DCCA[metric], label='OURS', marker='s')
    plt.plot(SIGMAS, PCA[metric], label=r'PCA+$T^2$', marker='o')
    plt.plot(SIGMAS, PLS[metric], label='PLS+$T^2$', marker='x')
    plt.plot(SIGMAS, CLS[metric], label='CLS', marker='v')
    

    plt.title(rf'{METRIC_NAME[metric]} vs. Noise Level ($\delta$)', fontdict={'fontsize': 16})
    plt.xlabel(r'Noise Level ($\delta$)', fontdict={'fontsize': 14})
    plt.ylabel(f'{METRIC_NAME[metric]} (%)', fontdict={'fontsize': 14})
    plt.xticks(SIGMAS)
    plt.grid()
    plt.legend()
    
    # Save the plot
    save_dir = os.path.join('Document', 'simulation-circle-crack')
    os.makedirs(save_dir, exist_ok=True)
    plot_file_path = os.path.join('Document', 'simulation-circle-crack', f'plot-{metric}.png')
    plt.savefig(plot_file_path)
    plt.close()
    # Show the plot
    # plt.show()
