# Quality-Ensured In-Situ Process Monitoring Using Deep Canonical Correlation Analysis

This repository contains official implementation for the paper _"Quality-Ensured In-Situ Process Monitoring Using Deep Canonical Correlation Analysis"_.

## Environment Specification

Some key OS and environment specifications that this code is developed on are provided as follow.

```
Red Hat Enterprise Linux (RHEL) 8.10
python=3.9.7
pytorch=1.12.1
CUDA=12.8.0
CUDA Driver Version=570.124.06
```

For all other packages, there are no specific version requirements as long as they are compatiable with the above versions and can be downloaded from `pip` or `conda`, depending on the package management system used in your local devices. For OS, we recommend running this on a RHEL system, which is common for cloud server. However, a windows OS with version 10+ with the specified cuda driver should also work smoothly, although additional setup is required for file path specification.

For hardware, this code is primarily developed on Tesla V100-PCIE-16GB and NVIDIA A100 16GB. That said, any properly configured GPU with at least 12 GB memory (this is required for running case study experiments) should work smoothly.

## Simulation Study

The simulation experiments in this paper involves two stages: (1) data generation and (2) model training/evaluation. First, we provide all checkpoints so that the results produced in the paper can be easily reproduced.

### Reproducing results from checkpoints

The generated data for the simulation as well as checkpoints can always be retrieved from [this link](https://drive.google.com/file/d/1wIvAOnbRH_3nGX0dsZ10Ur5cXdG9kM-M/view?usp=sharing) and after you download and unzip it, please place it under the folder `Data/`. The correct structure should look like: `Data/simulation-circle-crack/`.

To reproduce the results from checkpoints, simply run the following command:

```
python jobs/simulation/plt.py
```

**Note:** If you are NOT running on an online server with the environment specification mentioned above, you may need to be careful about the python path and need to adjust those accordingly. More detailed instructions can be found in _lines 14 - 15_ in `gen_data.py`, `automate.py`, and `test_baseline.py`. That said, we recommend running the code on the specified requirements.

### Reproducing results from scratch

Reproducing results from scratch requires training of the proposed DCCA-based framework. To accelerate the process, we provide an automation `bash` script that unites both data generation process and model training. The following command can be used to train from scratch:

```
bash jobs/simulation/test.sh
```

This scripts will produce everything you need for offline model training and online model evaluation.

### Re-generating data by yourself (optional)

Here we note that the data generation process has a fixed seed to ensure consistency of experiments. However, you can always regenerate those data yourself if you are interested in. To do so, you may follow instructions on `notebook/simulation-dgp.ipynb`.

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
