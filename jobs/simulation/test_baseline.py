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
from scipy import stats 
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(2024)
torch.manual_seed(2024)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='data directory')
parser.add_argument('--ckpt_dir', help='data directory')
parser.add_argument('--h', help="h", type=int)
parser.add_argument('--y_target', help="Target Y value as normal", type=int, default=0)
parser.add_argument('--n_abn', help="n_abn", type=int, default=0)
parser.add_argument('--window_size', help="n_window", type=int, default=25)
parser.add_argument('--alpha', help="alpha", type=float, default=0.05)
parser.add_argument('--test_reg', action='store_true')
parser.add_argument('--feature_level', action='store_true')
args = parser.parse_args()

ALPHA = args.alpha

# Load data
data_dir = os.path.join('Data', args.data_dir)
ckpt_dir = os.path.join('checkpoint', args.ckpt_dir)

# tri_set = torch.load(os.path.join(ckpt_dir, ckpt_name, 'tri_set.pt'))
if args.data_dir != 'simple':
    CT = torch.load(os.path.join(data_dir,  'ct.pt'))
    SP = torch.load(os.path.join(data_dir,  f'sp_2fc.pt'))
    SP = torch.tensor(SP.unsqueeze(1), dtype=torch.float32)
    Y_train = torch.load(os.path.join(data_dir,  'label.pt'))
    print(Counter(np.array(Y_train)))
    CT_test = torch.load(os.path.join(data_dir,  'ct-test.pt'))
    SP_test = torch.load(os.path.join(data_dir,  'sp_2fc-test.pt'))
    SP_test = torch.tensor(SP_test.unsqueeze(1), dtype=torch.float32)
    Y_test = torch.load(os.path.join(data_dir, 'label-test.pt'))
    print(Counter(np.array(Y_test)))
    tst_set = list(zip(CT_test, SP_test, Y_test))
else:
    CT = torch.load(os.path.join(data_dir,  'ct_train.pt'))
    SP = torch.load(os.path.join(data_dir,  f'sp_train.pt'))
    SP = torch.tensor(SP.unsqueeze(1), dtype=torch.float32)
    Y_train = torch.load(os.path.join(data_dir,  'y_train.pt'))
    print(Counter(np.array(Y_train)))
    # Y_train = Y_train.to(DEVICE)
    CT_test = torch.load(os.path.join(data_dir,  'ct_test.pt'))
    SP_test = torch.load(os.path.join(data_dir,  'sp_test.pt'))
    SP_test = torch.tensor(SP_test.unsqueeze(1), dtype=torch.float32)
    Y_test = torch.load(os.path.join(data_dir, 'y_test.pt'))
    print(Counter(np.array(Y_test)))
    # Y_test = Y_test.to(DEVICE)
    tst_set = list(zip(CT_test, SP_test, Y_test))


h=args.h
n_abn = args.n_abn
window_size = args.window_size

if not args.test_reg:
    ckpt_name = f'hypo-good-only-{h}D-window-{window_size}'
else:
    ckpt_name = f'hypo-good-only-{h}D-window-{window_size}-{args.n_abn}'

dcca_model = torch.load(os.path.join(ckpt_dir, ckpt_name, 'model.pt'), map_location='cpu')
ct_encoder, sp_encoder = dcca_model.ct_encoder, dcca_model.sp_encoder
tri_set = torch.load(os.path.join(ckpt_dir, ckpt_name, 'tri_set.pt')) # Training set
val_set = torch.load(os.path.join(ckpt_dir, ckpt_name, 'val_set.pt')) # Real validation set
print(f"Validation set size: {len(tri_set)}")

# Explore training / testing good parts
u_tri, v_tri, cts_tri, sps_tri, y_tri = extract_uv(tri_set, ct_encoder, sp_encoder)
u_val, v_val, cts_val, sps_val, y_val = extract_uv(val_set, ct_encoder, sp_encoder)
# Testing set
u_tst, v_tst, cts_tst, sps_tst, y_tst = extract_uv(tst_set, ct_encoder, sp_encoder)
# Set up masks
mask0_tri = y_tri == args.y_target
mask0_val = y_val == args.y_target
mask0_tst = y_tst == args.y_target

# SETUP Validation samples
sps_val0, sps_val1 = sps_val[mask0_val], sps_val[~mask0_val]
v_val0, v_val1 = v_val[mask0_val], v_val[~mask0_val]    
y_val0, y_val1 = y_val[mask0_val], y_val[~mask0_val]

# SETUP test samples
sps_tst0, sps_tst1 = sps_tst[mask0_tst], sps_tst[~mask0_tst]
v_tst0, v_tst1 = v_tst[mask0_tst], v_tst[~mask0_tst]
y_tst0, y_tst1 = y_tst[mask0_tst], y_tst[~mask0_tst]


# Training spectra pool (normal): for NN matching usage
sps_tri0 = sps_tri[mask0_tri]
u_tri0 = u_tri[mask0_tri]
v_tri0 = v_tri[mask0_tri]

# Test normal as bank, normal and abnormal as testing.
# Conclusion: exists false alarm but no missing detection.
n_test = 1000
n_samples = window_size
n_bootstrap_samples = window_size

# Validation samples
sps, vs, ys = generate_simulation_test_samples(sps_val0, v_val0, y_val0, n_bootstrap_samples, 2000)
validation_data = list(zip(sps, vs, ys))

# Normal as test samples
sps, vs, ys = generate_simulation_test_samples(sps_tst0, v_tst0, y_tst0, n_samples, n_test)
normal_data = list(zip(sps, vs, ys))
normal_label = np.zeros(1000)

# Abnormal as test samples
sps, vs, ys = generate_simulation_test_samples(sps_tst1, v_tst1, y_tst1, n_samples, n_test)
abnormal_data = list(zip(sps, vs, ys))
abnormal_label = np.ones(1000)

# Mixed as test samples (not useful)
# sps, vs, ys = generate_simulation_test_samples(sps_tst, v_tst, y_tst, n_samples, n_test)
# mixed_data = list(zip(sps, vs, ys))
# mixed_label = np.ones(1000)

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
prec = 5
TEST_SEPARATED = False
if h == 6 and TEST_SEPARATED:
    MERGE=False
    print("Separated...")
    # Bootstrap samples
    bootstrap_sample_size = window_size
    n_bootstrap_samples = 1000
    bootstrap_samples = randomized_corr_customized(u_tri[mask0_tri], v_tri[mask0_tri], bootstrap_sample_size, n_bootstrap_samples, h=h, merge=MERGE)

    # Threshold & alpha value (type i error)
    alpha = 0.01
    T = get_threshold(bootstrap_samples, alpha, h, MERGE)
    # Test not Merge
    print("Normal as test.")
    corrs0, normal_acc = get_test_results(normal_data, sps_tri0, u_tri0, n_test, h, T, normal_label, MERGE)
    print("\nAbormal as test.")
    corrs1, abnormal_acc = get_test_results(abnormal_data, sps_tri0, u_tri0, n_test, h, T, abnormal_label, MERGE)
    # Combined results
    print('\nFalse Alarm Rate: ', 100*round(1 - normal_acc, prec))
    print('Mis-Detection Rate: ', 100*round(1 - abnormal_acc, prec))
    f1 = (2 * abnormal_acc) / (2 * abnormal_acc + (1 - normal_acc) + (1 - abnormal_acc))
    print("F1: ", f1)
    print("\nMixed as test.")
    corrs2, mixed_acc = get_test_results(mixed_data, sps_tri0, u_tri0, n_test, h, T, mixed_label, MERGE)

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

MERGE=True
print("Merged Regime...")
# Threshold & alpha value (type i error) based on validation set
print("Bootstrap sample size: ", len(validation_data))
if not args.feature_level:
    v_tri0 = None
corrs_val = test(validation_data, sps_tri0, u_tri0, n_test=n_test, h=h, merge=MERGE,v_bank=v_tri0)
print(">> Mean correlations: ", np.round(np.mean(corrs_val, axis=0), 3))
print(">> Std correlations: ", np.round(np.std(corrs_val, axis=0), 3))
alpha = ALPHA
T = get_threshold(corrs_val, alpha, h, MERGE)
# Test
print("\nNormal as test.")
corrs0, normal_acc = get_test_results(normal_data, sps_tri0, u_tri0, n_test, h, T, normal_label, MERGE, v_tri0)
print("\nAbormal as test.")
corrs1, abnormal_acc = get_test_results(abnormal_data, sps_tri0, u_tri0, n_test, h, T, abnormal_label, MERGE, v_tri0)

# Combined results
print('\nFalse Alarm Rate: ', 100*round(1 - normal_acc, prec))
print('Mis-Detection Rate: ', 100*round(1 - abnormal_acc, prec))
f1 = (2 * abnormal_acc) / (2 * abnormal_acc + (1 - normal_acc) + (1 - abnormal_acc))
print("F1: ", 100*round(f1, prec))

# Test adjusted threshold results
target_alpha = 0.01
T_adjusted = np.quantile(corrs0, target_alpha)
# print(f"\nAdjusted threshold for Type I error {target_alpha}: ", T_adjusted)
# Verification of type I errors
normal_acc = sum(corrs0 >= T_adjusted) / len(corrs0)
# Test mis-detection rate
abnormal_acc = sum(corrs1 < T_adjusted) / len(corrs1)
f1 = (2 * abnormal_acc) / (2 * abnormal_acc + (1 - normal_acc) + (1 - abnormal_acc))
false_alarms = 1 - normal_acc
mis_detections = 1 - abnormal_acc
# print("False Alarms: ", 100*np.round(false_alarms, prec))
# print("Mis Detections: ", 100*np.round(mis_detections, prec))
# print("F1: ", 100*np.round(f1, prec))
    

# print("\nMixed as test.")
# corrs2, mixed_acc = get_test_results(mixed_data, sps_tri0, u_tri0, n_test, h, T, mixed_label, MERGE)

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
TEST_CLS = True
if TEST_CLS:
    print("Baseline...")
    encoder = SimSpecAutoEncoder(args.h)
    model = SimDecNet(h=args.h, encoder=encoder, op=False, freeze=False).to(DEVICE)
    ckpt_name = f'naive-cls-{n_abn}'
    model = torch.load(os.path.join(ckpt_dir, ckpt_name, 'model.pt'), map_location=DEVICE)
    print("Baseline model loaded successfully")

    # Run on test set
    SP_test = torch.load(os.path.join(data_dir,  'sp_2fc-test.pt'))
    SP_test = torch.tensor(SP_test.unsqueeze(1), dtype=torch.float32)
    if args.y_target == 1:
        Y_test = 1 - torch.load(os.path.join(data_dir, 'label-test.pt'))
    else:
        Y_test = torch.load(os.path.join(data_dir, 'label-test.pt'))
    print("Test set: ", Counter(np.array(Y_test)))
    tst_set = list(zip(SP_test, Y_test))
    tst_ldr = torch.utils.data.DataLoader(tst_set, batch_size=256, shuffle=True)

    test_cls_on_dset(tst_ldr, model)

    print("Start testing windowed data...")
    normal_preds = get_acc_cls_baseline(normal_data, model)
    preds_sum = np.sum(normal_preds, axis=1)

    abnormal_preds = get_acc_cls_baseline(abnormal_data, model)
    abnormal_preds_sum = np.sum(abnormal_preds, axis=1)

    false_alarms, mis_detections, F1 = [], [], []
    for i in range(n_samples):
        false_alarm = 1 - sum(preds_sum <= i) / len(normal_preds)
        mis_detection = 1 - sum(abnormal_preds_sum > i) / len(abnormal_preds)
        abnormal_acc = 1 - mis_detection
        normal_acc = 1 - false_alarm
        f1 = (2 * abnormal_acc) / (2 * abnormal_acc + (1 - normal_acc) + (1 - abnormal_acc))

        false_alarms.append(false_alarm)
        mis_detections.append(mis_detection)
        F1.append(f1)

        # print("\nThreshold: ", i+1)
        # print(">> False Alarm: ", false_alarm)
        # print(">> Mis Detection: ", mis_detection)
        # print(">> F1: ", f1)

    # print("False Alarms: ")
    # print(100*np.round(false_alarms, prec))
    # print("Mis Detections: ")
    # print(100*np.round(mis_detections, prec))
    # print("F1: ")
    # print(100*np.round(F1, prec))
    print("Mean False Alarms: ")
    print(100*np.round(np.mean(false_alarms), prec))
    print("Mean Mis Detections: ")
    print(100*np.round(np.mean(mis_detections), prec))
    print("Mean F1: ")
    print(100*np.round(np.mean(F1), prec))


def T2_control_chart(method, m=1000, n=25, p=5, alpha=0.05):
    if method == 'dcca':
        print("="*80)
        print("DCCA")
        v = v_tri
    elif method == 'pca':
        print("="*80)
        print("PCA")
        # PCA is trained using training data
        sp_training = SP[Y_train == args.y_target]
        print(f"PCA Training Shape: {sp_training.shape}")
        scaler = StandardScaler()
        sp_training = scaler.fit_transform(sp_training.squeeze().numpy())
        pca = PCA(n_components=p).fit(sp_training)
        explained_variance_ratio = pca.explained_variance_ratio_
        # Total variance explained by all components
        total_variance_explained = np.sum(explained_variance_ratio)
        print(f"Explained Variance Ratio: {explained_variance_ratio}")
        print(f"Total Variance Explained: {total_variance_explained}")
        # Standardize: Zero mean, unit variance
        sps_tri_standardized = scaler.fit_transform(sps_tri.squeeze().numpy())
        v_pca = pca.transform(sps_tri_standardized)
        v = v_pca
        # print(v.shape)

    elif method == 'pls':
        print("="*80)
        print("PLS")
        from sklearn.cross_decomposition import PLSRegression
        sp_training = SP[Y_train == args.y_target]
        ct_training = CT[Y_train == args.y_target]
        print(f"PLS CT Training Shape: {ct_training.shape}")
        print(f"PLS SP Training Shape: {sp_training.shape}")
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        sp_training_scaled = scaler_X.fit_transform(sp_training.squeeze().numpy())
        print(f"Reshaped CTs: {ct_training.reshape(len(ct_training), -1).shape}")
        ct_training_scaled = scaler_Y.fit_transform(ct_training.reshape(len(ct_training), -1).numpy())
        pls = PLSRegression(n_components=p).fit(sp_training_scaled, ct_training_scaled)
        # Evalute features on validation set
        sps_scaled = scaler_X.fit_transform(sps_tri.squeeze().numpy())
        v_pls = pls.transform(sps_scaled)
        v = v_pls

    elif method == 'AE':
        print("="*80)
        print("AutoEncoder")
        sp_training = SP[Y_train == args.y_target]
        print(f"AE Data Shape: {sp_training.shape}")
        autoencoder = SimSpecAutoEncoder(h).to(DEVICE)
        autoencoder.load_state_dict(torch.load(os.path.join(ckpt_dir, f'baseline-sp-ae-{h}.pt'), map_location=DEVICE))
        encoder = autoencoder.encoder
        with torch.no_grad():
            sps_tri_tensor = sps_tri.to(DEVICE)
            encoded_features = encoder(sps_tri_tensor)[0].cpu().numpy()
            v = encoded_features.squeeze()
            print(v.shape)

    elif method == 'SIMCLR':
        print("="*80)
        print("SIMCLR")
        sp_training = SP[Y_train == args.y_target]
        print(f"SIMCLR Data Shape: {sp_training.shape}")
        encoder = SimSpecEncoder(h).to(DEVICE)
        encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, f'baseline-sp-simclr-{h}.pt'), map_location=DEVICE))
        with torch.no_grad():
            sps_tri_tensor = sps_tri.to(DEVICE)
            encoded_features = encoder(sps_tri_tensor)[0].cpu().numpy()
            v = encoded_features.squeeze()
            print(v.shape)

    elif method == 'SIMCLR-CLIP':
        print("="*80)
        print("SIMCLR-CLIP")
        sp_training = SP[Y_train == args.y_target]
        print(f"SIMCLR-CLIP Data Shape: {sp_training.shape}")
        encoder = SimSpecEncoder(h).to(DEVICE)
        encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, f'baseline-simclr-clip-{h}.pt'), map_location=DEVICE))
        with torch.no_grad():
            sps_tri_tensor = sps_tri.to(DEVICE)
            encoded_features = encoder(sps_tri_tensor)[0].cpu().numpy()
            v = encoded_features.squeeze()
            print(v.shape)

    # Phase I data
    phase_1 = []
    for _ in range(m):
        idx = np.random.choice(len(v), n, False)
        phase_1.append(v[idx, :])
    phase_1 = np.array(phase_1)
    print(f"Phase I data shape: {phase_1.shape}")

    mu = np.mean(phase_1, axis=1)
    grand_mu = np.mean(phase_1, axis=(0,1))
    S_matrices = []  
    # print(f"Grand mu: {grand_mu}")

    for k in range(m):
        X_k = phase_1[k]  # Shape (n, p)
        S_k = np.cov(X_k, rowvar=False, ddof=1)  # Covariance matrix for sample k
        S_matrices.append(S_k)
    S = np.mean(S_matrices, axis=0)
    S_inv = np.linalg.inv(S)
    alpha = alpha
    critical_F = stats.f.ppf(1 - alpha, p, m*n - m-p+1)
    UCL = ((p * (m+1)*(n-1))/ (m*n - m - p + 1)) * critical_F
    print(f"UCL: {UCL}.")

    pred_normal = []
    for (sps, vs, _) in normal_data:
        if method == 'pca':
            sps_standardized = scaler.transform(sps.squeeze().numpy())
            # sps_standardized = sps.squeeze().numpy()
            vs = pca.transform(sps_standardized)
        elif method == 'pls':
            sps_standardized = scaler_X.transform(sps.squeeze().numpy())
            # sps_standardized = sps.squeeze().numpy()
            vs = pls.transform(sps_standardized)
            # print(vs.shape)

        elif method in ['AE', 'SIMCLR', 'SIMCLR-CLIP']:
            with torch.no_grad():
                encoded_features = encoder(sps.to(DEVICE))[0].cpu().numpy()
                vs = encoded_features.squeeze()
                # print(vs.shape)
        
        # Check normality (sanity check only; disabled in testing stage)
        # for i in range(vs.shape[1]):
        #     stat, p_value = stats.shapiro(vs[:, i])  # Shapiro-Wilk test
        #     print(f"Variable {i+1}: p-value = {p_value:.4f} (Shapiro-Wilk test)")
        # break

        diff = (np.mean(vs, axis=0) - grand_mu).reshape(-1, 1)
        T2 = n * (diff.T @ S_inv @ diff).squeeze()
        pred_normal.append(T2 < UCL)
        # print(T2)
    pred_normal = np.array(pred_normal)

    pred_abnormal = []
    for (sps, vs, _) in abnormal_data:
        if method == 'pca':
            sps_standardized = scaler.transform(sps.squeeze().numpy())
            # sps_standardized = sps.squeeze().numpy()
            vs = pca.transform(sps_standardized)
        elif method == 'pls':
            scaler = StandardScaler()
            sps_standardized = scaler_X.transform(sps.squeeze().numpy())
            # sps_standardized = sps.squeeze().numpy()
            vs = pls.transform(sps_standardized)   

        elif method in ['AE', 'SIMCLR', 'SIMCLR-CLIP']: 
            with torch.no_grad():
                encoded_features = encoder(sps.to(DEVICE))[0].cpu().numpy()
                vs = encoded_features.squeeze()

        diff = (np.mean(vs, axis=0) - grand_mu).reshape(-1, 1)
        T2 = n * (diff.T @ S_inv @ diff).squeeze()
        # print(T2)
        pred_abnormal.append(T2 >= UCL)
    pred_abnormal = np.array(pred_abnormal)

    TP = sum(pred_abnormal)
    TN = sum(pred_normal)
    FP = len(pred_normal) - TN
    FN = len(pred_abnormal) - TP

    false_alarms =  1 - sum(pred_normal) / len(pred_normal)
    mis_detections =  1 - sum(pred_abnormal) / len(pred_abnormal)
    F1 = 2 * TP / (2*TP + FP + FN)

    # F1 = (2 * len(abnormal_acc)) / (2 * len(abnormal_acc) + (1 - normal_acc) + (1 - abnormal_acc))

    print("False Alarms: ")
    print(100*np.round(false_alarms, prec))
    print("Mis Detections: ")
    print(100*np.round(mis_detections, prec))
    print("F1: ")
    print(100*np.round(F1, prec))

if h != 1:
    print("="*80)
    print("Hotelling T2")
    alpha = ALPHA
    # v_tri: extracted features on validation set (good part)
    # v_val0: extracted features on test set (good part)
    # v_val1: extracted features on test set (bad part)
    T2_control_chart('dcca', n=n_samples, p=args.h, alpha=alpha)
    T2_control_chart('pca', n=n_samples, p=args.h, alpha=alpha)
    T2_control_chart('pls', n=n_samples, p=args.h, alpha=alpha)
    T2_control_chart('AE', n=n_samples, p=args.h, alpha=alpha)
    T2_control_chart('SIMCLR', n=n_samples, p=args.h, alpha=alpha)
    T2_control_chart('SIMCLR-CLIP', n=n_samples, p=args.h, alpha=alpha)

