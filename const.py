import numpy as np
import pandas as pd
import torch
import os
import sys
# path to directories
CKPT_LOG_DIR = os.path.join('checkpoint', 'log')
CKPT_SAVE_DIR = os.path.join(os.getcwd(), 'checkpoint', 'model')
FIG_SAVE_DIR = os.path.join('checkpoint', 'figure')
DATA_RAW_DIR = os.path.join('Data', 'raw')
DATA_PROCESSED_DIR = os.path.join('Data', 'processed')
DATA_SPEC_DIR = os.path.join('Data', 'spectra')
FT_DATA_DIR = os.path.join('Data', 'ft')

# SPECTRA MAP:
# TODO: Finish this map later
SPECTRA_NAME = {
    '1Dr': r'f15 1800w 1.00beamdia 3v 3 layer z-7 (1Dr).SOMS',
    '2D': r'f18 1400w 1.00beamdia 3v 3 layer (2D).SOMS',
    '2B': r'f18 1750w 1.00beamdia 6v 3 layer z-7 power1.25 (2B).SOMS',
    '2Dr': r'f18 1400w 1.00beamdia 3v 3 layer rep1 (2Dr).SOMS',
    '3D': r'f21 1800w 1.00beamdia 3v 3 layer (3D).SOMS',
    '3B': r'f21 2250w 1.00beamdia 6v 3 layer z-7 power1.25 (3B).SOMS',
    '4B': r'f21 1750w 1.00beamdia 6v 3 layer z-7 power1.25 (4B).SOMS',
    '4D': r'f21 1400w 1.00beamdia 3v 3 layer (4D).SOMS',
    '4Dr': r'f21 1400w 1.00beamdia 3v 3 layer rep1 (4Dr).SOMS'
}

SPECTRA_THRESHOLD = {
    '1Dr': 1020,
    '2D': 1020,
    '3D': 1020,
    '4B': 1001,
    '3B': 1002,
    '4D': 1020,
    '4Dr': 1015
}

SPECTRA_MIN = {
    '1Dr': 932.951,
    '2D': 934.523,
    '4B': 932.792,
    '4D': 933.431
}

SPECTRA_MAX = {
    '1Dr':  1796.286,
    '2D': 1196.829,
    '4B': 1346.782,
    '4D': 1662.698
}

STEP_SIZE = {
    '1Dr': 0.20,
    # '1Dr': 0.05,
    '4B': 0.27,
    '4D': 0.27
}

# Wavelength configuration
import platform

if platform.system() == "Linux":
    print("Likely running on a server")
    ON_GL = True
else:
    print("Running on a local machine")
    ON_GL = False
# ON_GL = True
# # ON_GL = False
if not ON_GL:
    N_WL = 2038
    raw = np.loadtxt(open(os.path.join(DATA_SPEC_DIR, SPECTRA_NAME['1Dr']), "rb"), delimiter="\t")
    WL = raw[1:, 0]
    # Target Wavelength range: 394nm - 398nm (88 indices)
    # Actual Wavelength range: 393.829nm - 398.168nm (96 indices)
    LB, UB = 393.829, 398.168
    W_MASK = np.logical_and(WL <= UB, WL >= LB)
    # print(WL[W_MASK])
    # print(f"Number of wavelengths: {sum(W_MASK)}")
    RESOLUTION_MASK = range(0, sum(W_MASK), 3)
    # RESOLUTION_MASK = range(0,sum(W_MASK)) # Only used for plots
    print(f"Resolution Mask dimension: {len(RESOLUTION_MASK)}")
    print(RESOLUTION_MASK)


# Device configuration & information
def device_information(f = None):
    if torch.cuda.is_available():
        print(f"-- Current Device: {torch.cuda.get_device_name(0)}")
        print(
            f"-- Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print("-- Let's use", torch.cuda.device_count(), "GPUs!")
        if f is not None:
            f.write("="*80 + "\n")
            f.write(f"-- Current Device: {torch.cuda.get_device_name(0)}\n")
            f.write(
                f"-- Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
            f.write("-- Let's use " + str(torch.cuda.device_count()) + " GPUs!\n")
            f.write("="*80 + "\n")
    else:
        print("-- Unfortunately, we are only using CPUs now.")
        if f is not None:
            f.write("-- Unfortunately, we are only using CPUs now.\n")
    # Global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


W, H, = 250, 730
WHITE_IMAGES_RAW = torch.ones((W, H)) * 255
WHITE_IMAGES_PROCESSED = torch.zeros((W, H))
