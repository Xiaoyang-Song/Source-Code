import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from models.simulation import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(2024)
torch.manual_seed(2024)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='data directory')
parser.add_argument('--n', help='number of samples')
parser.add_argument('--sigma', help='noise level')
parser.add_argument('--sp_type', help='spectra generation method')
parser.add_argument('--data_dir_2', default=None, help='data directory for existing weights')
args = parser.parse_args()

# Generate spectra signals
BETAS = np.array([0.3, 1, 2, 0.4, -0.5, -0.7])
# NOISE = np.random.normal(0, 0.0001, 6)
sigma = float(args.sigma)
n = int(args.n)
# Generate noise
NOISE = np.random.normal(0, 0.1**2, (n, 6))
print(NOISE.shape)

data_dir = os.path.join('Data', args.data_dir)
ctscans = torch.load(os.path.join(data_dir, 'ct-raw.pt')).to(DEVICE)
model = SimAutoEncoder(6).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(data_dir, 'ct_ae-dgp.pt'), map_location=DEVICE))

with torch.no_grad():
    U = model.encoder(ctscans)[0].detach().cpu().numpy()
    print(U.shape)
    V = U * BETAS + NOISE
    print(V.shape)

# Check CCA criterion
U = torch.tensor(U, dtype=torch.float32)
V = torch.tensor(V, dtype=torch.float32)
criterion = CCA(6, False, 'cpu')
loss, _ = criterion(U, V)
print(-loss)
# Sanity check
print(V[0])
print(U[0])
print(V[0] / U[0])
# Save U and V
torch.save(U, os.path.join(data_dir, 'U.pt'))
torch.save(V, os.path.join(data_dir, 'V.pt'))


# Generate from scratch (2FC)
if args.sp_type == '2fc':
    if args.data_dir_2 is None:
        spectra, M , M2 = generate_2fc_waves(V)
        torch.save(M, os.path.join(data_dir, 'M.pt'))
        torch.save(M2, os.path.join(data_dir, 'M2.pt'))
    else:
        data_dir2 = os.path.join('Data', args.data_dir_2)
        M = torch.load(os.path.join(data_dir2, 'M.pt'))
        M2 = torch.load(os.path.join(data_dir2, 'M2.pt'))
        spectra = torch.matmul(V, M)
        spectra = F.relu(spectra)
        spectra = torch.log(torch.abs(1 / (1 - torch.matmul(spectra, M2)))) + np.random.normal(0, sigma**2, (n, 16))
elif args.sp_type == 'sin':
    if args.data_dir_2 is None:
        spectra, MW, MA, MB = generate_simple_sinusoidal_waves(V)
        torch.save(MW, os.path.join(data_dir, 'MW.pt'))
        torch.save(MA, os.path.join(data_dir, 'MA.pt'))
        torch.save(MB, os.path.join(data_dir, 'MB.pt'))
    else:
        data_dir2 = os.path.join('Data', args.data_dir_2)
        MW = torch.load(os.path.join(data_dir2, 'MW.pt'))
        MA = torch.load(os.path.join(data_dir2, 'MA.pt'))
        MB = torch.load(os.path.join(data_dir2, 'MB.pt'))
        W, A, B = torch.matmul(V, MW), torch.matmul(V, MA), torch.matmul(V, MB)
        spectra = []
        # 2 * pi approx 6
        for x in np.linspace(0, 2 * np.pi, 16):
            spectra.append(A*torch.sin(W * x + B))
        spectra = torch.column_stack(spectra)
else:    
    raise ValueError('Spectra type not supported') 

sp_name = '2fc'
print(spectra.shape)
if n == 10000:
    torch.save(spectra[0:5000], os.path.join(data_dir, f'sp_{sp_name}.pt'))
    torch.save(spectra[5000:10000], os.path.join(data_dir, f'sp_{sp_name}-test.pt'))
else:
    torch.save(spectra[0:2500], os.path.join(data_dir, f'sp_{sp_name}.pt'))
    torch.save(spectra[2500:5000], os.path.join(data_dir, f'sp_{sp_name}-test.pt'))
