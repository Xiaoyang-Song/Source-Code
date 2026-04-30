# Quality-Ensured In-Situ Process Monitoring Using Deep Canonical Correlation Analysis

This repository contains official implementation for the paper _"Quality-Ensured In-Situ Process Monitoring Using Deep Canonical Correlation Analysis"_. In particular, this repository provides source code and commands for reproducing results for our simulation experiments and also sensitivity analysis. For the case study in the paper, we provide the source code for model architecture and training but not the actual dataset we used.

## Environment Configuration

The development OS and key cloud server environments for this codebase are outlined below. It is **highly recommended** to reproduce the results on HPCs or Linux-based system because the experiments are extensive and the experiment automation is more compatible with those systems.

```
Red Hat Enterprise Linux (RHEL) 8.10
python>=3.9.7
pytorch>=1.12.1
CUDA>=12.8.0
CUDA Driver Version=570.124.06
```

Although the code is developed on a High-Performance Cluster (HPC) with RHEL system, it should work smoothly without any difficulties on a Linux-based OS. For Windows system, we recommend running on a **WSL system like Ubuntu** in order to do full automation of the experiments. Here we provide detailed instructions on setting up the environments.

### Prerequisite

Before setting up the environment, please make sure you have the following prerequisite software or package management libraries downloaded.

- **Anaconda (or Miniconda)**: environment and package management system.
- **pip**: python package management library.
- **Ubuntu 22.04** (For Windows users only): open-source Linux operating system, required to use WSL (Windows Subsystem for Linux) in Windows computer.

Note that when downloading those packages, please download them such that they are in compatible with your systems and computer configurations.

### Environment Setup

For your ease, we provide the following setup commands (it is also wrapped into the `setup.sh` script that can be run on Linux system):

```
conda create -n DCCA-AM python=3.9.7
conda activate DCCA-AM

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tqdm
pip install tensorboard
pip install torchinfo
pip install PyYAML
```
Note that all other required packages are typically downloaded by default when downloading Python. If in any cases there is missing package, please install it using either `conda install <package_name>` or `pip install <package_name>`.

### Remarks, Tips, and Troubleshooting

For all users, please make sure that you activate the conda environment before running the code below. For Windows users, it is highly recommended to use WSL with Ubuntu 22.04 to set up and run all the code. If you are using native Windows Powershell and Git Bash, it may not be compatible with all the automation `.sh` code but should still work with all the python commands. The simplest fixing strategy will be copying and pasting all commands from the `.sh` automation script and run them individually in the terminal.

In addition, Linux and Windows systems have different default file formatting. For Windows users, if you encountered errors like `$'\r': command not found` or `No such file or directory: ...` because of the existing of hidden character `\r`, you can resolve this by modifying the `.sh` automation script in your WSL:

```
sudo apt install dos2unix # installation (only run once)
dos2unix ****.sh # Just replace **** with the `.sh` file you run
```

Users with our recommended Linux OS or HPCs should **NOT** encounter this issue as the files and code are developed on the same system.

For hardware, this code is primarily developed on Tesla V100-PCIE-16GB and NVIDIA A100 16GB. That said, any properly configured GPU with at least 12 GB memory should work smoothly.

## Simulation Study

The simulation experiments in this paper involves two stages: (1) data generation and (2) model training/evaluation (including baseline models). However, both stages are wrapped into single `.sh` file for automation. First, we provide all checkpoints so that the results produced in the paper can be easily reproduced.

### Sanity Check
Before proceeding to reproduce the experimental results, please make sure that the following has been done correctly:

1. The environment has been correctly setup following the previous instructions and also activated using `conda activate DCCA-AM`.
2. (For Windows users) Make sure you are using Ubuntu WSL system; otherwise the bash automation script may not run successfully and may need manual copy and paste of python commands.
3. Source your python path to be the root folder using ``export PYTHONPATH=$PYTHONPATH$:`pwd` ``. If your default python path is already sourced at the current working directory, you do not have to make this adjustment.

### Reproducing results from checkpoints

To reproduce the results from checkpoints, simply run the following command:

```
bash jobs/simulation/plot_main.sh
# OR simply run the following python commands
# python jobs/simulation/plot.py --p-values 6 --n-values 25 --plot-best-baseline
```
The results can be found in the folder `Document/simulation-circle-crack/`.
### Reproducing results from scratch

Reproducing results from scratch requires training of the proposed DCCA-based framework and also other baseline methods. The following command can be used to train from scratch:

```
bash jobs/simulation/main.sh
```

This scripts will produce everything you need for offline model training and online model evaluation. The results can be found in the folder `Document/simulation-circle-crack/`.

**(Optional)** In addition, our sensitivity analysis experiments in the Appendix can be run in a similar way, where we just loop over possible combinations of window size and hidden correlation dimension:
```
bash jobs/simulation/sensitivity_analysis.sh
```
However, you may expect that this script will take more than 24 hours to run on NVIDIA A100 GPU due to the large set of combinations that we examined. Hence, we would recommend first reproducing the results in the main paper using the command given before.

**Note:** If you are NOT running on an online HPC server or Linux system with the environment specification mentioned above, you may encounter issues of running automation scripts. Instead, the simplest fixing strategy is to copy and paste python command and run them individually in order.

## Case Study

### Data availability

Due to licensing restrictions, the dataset used in the case study is not intended to make publicly available. However, case data access permission can be requested via emails to authors for appropriate use. The folder `Data/raw/`, `Data/ft`, `Data/processed/`, and `Data/spectra/` need to be populated by real data after request is permitted.

### Source code structure & commands

Although the data is not made publicly available, we choose to release the source code so that the audience can have a better sense regarding the following aspects:

1. How we process the data based on the descriptions in the paper?
2. How we align data of different sampling frequencies and modalities together?
3. How we design different model architectures for different types of data?
4. How we conduct online monitoring in real applications?

These details are provided in `models/` and `preprocess/`.

After the data is requested with permissions, the case study results can be easily reproduced by running the following command:

```
bash jobs/real-data/run.sh
```

Again, this is an automation `.sh` script that unites offline training and online evaluation. Please refer to the script for more instructions of running them separately.

## Citation

This paper is under review.
