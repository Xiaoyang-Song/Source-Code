import numpy as np
from matplotlib import pyplot as plt
import os
from collections import defaultdict

SIGMAS = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

WINDOW_SIZES = [20,25,30,35]
HS = [4,6,8,10]

METRICS = ['F1','I','II']
METRIC_NAME = {
    'F1':'F1 Score',
    'I':'Type-I Error',
    'II':'Type-II Error'
}

for window_size in WINDOW_SIZES:
    for h in HS:

        PCA = defaultdict(list)
        PLS = defaultdict(list)
        DCCA = defaultdict(list)
        CLS = defaultdict(list)

        base_dir = os.path.join(
            'checkpoint',
            'simulation-circle-crack',
            'sensitivity_analysis',
            f'{h}-{window_size}'
        )

        for sigma in SIGMAS:

            log_file_path = os.path.join(
                base_dir,
                f'results-sigma{sigma}.log'
            )

            if not os.path.exists(log_file_path):
                print(f"Missing file: {log_file_path}")
                continue

            with open(log_file_path,"r") as file:
                lines=[line.strip() for line in file]

            # PLS
            PLS['F1'].append(float(lines[-1]))
            PLS['II'].append(float(lines[-3]))
            PLS['I'].append(float(lines[-5]))

            # PCA
            PCA['F1'].append(float(lines[-14]))
            PCA['II'].append(float(lines[-16]))
            PCA['I'].append(float(lines[-18]))

            # CLS
            CLS['F1'].append(float(lines[49]))
            CLS['II'].append(float(lines[47]))
            CLS['I'].append(float(lines[45]))

            # DCCA
            dcca_t1=float(lines[29][len("False Alarm Rate:"):].strip())
            dcca_t2=float(lines[30][len("Mis-Detection Rate:"):].strip())
            dcca_f1=float(lines[31][len("F1:"):].strip())

            DCCA['F1'].append(dcca_f1)
            DCCA['II'].append(dcca_t2)
            DCCA['I'].append(dcca_t1)


        # Plot
        for metric in METRICS:

            plt.figure()

            plt.plot(SIGMAS,DCCA[metric],label='OURS',marker='s')
            plt.plot(SIGMAS,PCA[metric],label=r'PCA+$T^2$',marker='o')
            plt.plot(SIGMAS,PLS[metric],label='PLS+$T^2$',marker='x')
            plt.plot(SIGMAS,CLS[metric],label='CLS',marker='v')

            plt.title(
                rf'{METRIC_NAME[metric]} vs Noise ($\delta$) (h={h}, w={window_size})',
                fontsize=16
            )

            plt.xlabel(r'Noise Level ($\delta$)',fontsize=14)
            plt.ylabel(f'{METRIC_NAME[metric]} (%)',fontsize=14)

            plt.xticks(SIGMAS)
            plt.grid()
            plt.legend()

            save_dir=os.path.join(
                'Document',
                'simulation-circle-crack',
                'sensitivity_analysis',
                f'{h}-{window_size}'
            )

            os.makedirs(save_dir,exist_ok=True)

            plot_file_path=os.path.join(
                save_dir,
                f'plot-{metric}.png'
            )

            plt.savefig(plot_file_path)
            plt.close()