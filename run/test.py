import re
import sys
import argparse
from sklearn.model_selection import train_test_split
# This needs to be customized later (also needed to be customized when using GL)
sys.path.append('c:\\Users\\xysong\\Desktop\\Research\\DCCA-Image-Spectrum-Matching')
sys.path.append('../')
from util.utils import *
from dataset import *
from models.dcca import *
from models.decnet import *
from models.dl_baseline import *
from objective.loss import *
from const import *
from models.simulation import *
from scipy import stats 
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=lambda s: re.split(' |, ', s),
                    default="1Dr", help='comma or space delimited list of SP dataset names')
# parser.add_argument('--dset', type=str, required=True, help='finetuning dataset')
parser.add_argument('--GL',action='store_true')
parser.add_argument('--h', help='hidden dimensions', default=16, type=int)
parser.add_argument('--h_ft', help='hidden dimensions used in decision net', default=16, type=int)
parser.add_argument('--op', help='overparameterized network', action='store_true')
parser.add_argument('--freeze', help='freeze parameters of feature extraction network', action='store_true')
parser.add_argument('--lr', help='learning rate', default = 1e-4, type=float)
parser.add_argument('--n_epochs', help='max_epochs', default = 1000, type=int)
parser.add_argument('--bsz_tri', help='training batch size', default = 16, type=int)
parser.add_argument('--bsz_val', help='validation batch size', default = 16, type=int)
parser.add_argument('--tv_ratio', help='train-validation ratio', default = 0.1, type=float)
parser.add_argument('--alpha', help='alpha values for validation', default = 0.1, type=float)
# Encoders specification
parser.add_argument('--ckpt_name', help="ckpt name (tag of ckpt)", type=str)
parser.add_argument('--n_log', help='logging frequency', default = 25, type=int)
parser.add_argument('--tag', help="checkpoint tag", default='0', type=str)
parser.add_argument('--feature_level', help='feature level analysis', action='store_true')
args = parser.parse_args()
cmd = " ".join(sys.argv)

ALPHA=args.alpha


# FOR point-wise detection
# dset = torch.load(os.path.join(FT_DATA_DIR, args.dset, "data_abnormal.pt"))
# Reg 2 (for best so far)
# n_reg = 32
# dset = [dset[i] for i in range(n_reg, len(dset), 1)]
# Reg 1
# dset = torch.load(os.path.join(CKPT_SAVE_DIR, "DCCA", f"dcca_{args.ckpt_name}.pt"))['abn_set']

# dset = torch.load(os.path.join(CKPT_SAVE_DIR, "DCCA", f"dcca_{args.ckpt_name}.pt"))['abn_set']
torch.manual_seed(2024)
np.random.seed(2024)

# FOr process shift detection
print(args.dset)
SPECTRA_DATA = SP(parts=args.dset, normalize=True)
SPECTRA_DATA.gather(False)
CT_DATA = CT(parts=args.dset)
CT_DATA.read()
CT_DATA.gather()

aligner = Aligner(parts=args.dset, ct=CT_DATA, sp=SPECTRA_DATA)
aligner.align()
dataset = aligner.gather()
dset = Aligner.clip_window(dataset)

# dset = data_bad
model = DCCA(h=args.h).to(DEVICE)
bsz = 8
bsz_bootstrap = 10

with torch.no_grad():
    model.sp_encoder.load_state_dict(torch.load(os.path.join(CKPT_SAVE_DIR, "DCCA", f"dcca_{args.ckpt_name}.pt"))['sp_encoder'])
    model.ct_encoder.load_state_dict(torch.load(os.path.join(CKPT_SAVE_DIR, "DCCA", f"dcca_{args.ckpt_name}.pt"))['ct_encoder'])
    tri_set = torch.load(os.path.join(CKPT_SAVE_DIR, "DCCA", f"dcca_{args.ckpt_name}.pt"))['tri_set']
    val_set = torch.load(os.path.join(CKPT_SAVE_DIR, "DCCA", f"dcca_{args.ckpt_name}.pt"))['val_set']
    test_set = torch.load(os.path.join(CKPT_SAVE_DIR, "DCCA", f"dcca_{args.ckpt_name}.pt"))['test_set']

    abn_ldr = torch.utils.data.DataLoader(dset, batch_size=bsz, shuffle=True, drop_last=True)
    tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=bsz, shuffle=True, drop_last=True)
    val_ldr = torch.utils.data.DataLoader(val_set, batch_size=bsz, shuffle=True, drop_last=True)
    tst_ldr = torch.utils.data.DataLoader(test_set, batch_size=bsz, shuffle=True, drop_last=True)

    # dldr=tst_ldr
    # dldr=tri_ldr
    # dldr=val_ldr
    dldr=tst_ldr

    corrs = []
    # for (ct, sp, _, _) in dldr:
    for (ct, sp) in dldr:

        # cts = torch.cat([x for x, _, _, _ in dset], dim=0).unsqueeze(1)
        # sps = torch.cat([y.unsqueeze(0) for x, y, _, _ in dset], dim=0)
        # y = torch.tensor([y for _, _, y, _ in dset])
        # print(cts.shape, sps.shape)
        u, v = model(ct.to(DEVICE), sp.to(DEVICE))
        # u, v = np.array(u.squeeze().cpu()), np.array(v.squeeze().cpu())
        # print(u.shape, v.shape)s

        criterion = CCA(6, False, DEVICE)
        loss, corr_vec = criterion(u, v)
        # print(-loss)
        # print(corr_vec)
        corrs.append(-loss.cpu())

    print("Normal pairs correlations: ", np.mean(corrs))

    dldr = abn_ldr


    corrs = []
    # for (ct, sp, _, _) in dldr:
    for (ct, sp) in dldr:

        # cts = torch.cat([x for x, _, _, _ in dset], dim=0).unsqueeze(1)
        # sps = torch.cat([y.unsqueeze(0) for x, y, _, _ in dset], dim=0)
        # y = torch.tensor([y for _, _, y, _ in dset])
        # print(cts.shape, sps.shape)
        u, v = model(ct.to(DEVICE), sp.to(DEVICE))
        # u, v = np.array(u.squeeze().cpu()), np.array(v.squeeze().cpu())
        # print(u.shape, v.shape)

        criterion = CCA(6, False, DEVICE)
        loss, corr_vec = criterion(u, v)
        # print(-loss)
        # print(corr_vec)
        corrs.append(-loss.cpu())

    print("Abnormal pairs correlations: ", np.mean(corrs))

    print("\nStart actual testing...")

    # Bootstrap sample size
    cts_tri = torch.cat([ct for (ct, _) in tri_set], dim=0).unsqueeze(1)
    sps_tri = torch.cat([sp.unsqueeze(0) for (_, sp) in tri_set], dim=0)
    cts_val = torch.cat([ct for (ct, _) in val_set], dim=0).unsqueeze(1)
    sps_val = torch.cat([sp.unsqueeze(0) for (_, sp) in val_set], dim=0)
    print("DCCA Train set shape: ", cts_tri.shape, sps_tri.shape)
    print("DCCA Val set shape: ", cts_val.shape, sps_val.shape)
    bootstrap_sample_size = bsz_bootstrap
    n_bootstrap_samples = 2000
    bootstrap_sp, bootstrap_ct = [], []

    # get features
    if args.feature_level:
        v_tri = model.sp_encoder(sps_tri.to(DEVICE))[0].detach().cpu().squeeze()
        # v_val = model.sp_encoder(sps_val.to(DEVICE))[0].detach().cpu().squeeze()

    for _ in range(n_bootstrap_samples):
        idx = np.random.choice(len(sps_val), bootstrap_sample_size, False)
        bootstrap_sp.append(sps_val[idx])
        bootstrap_ct.append(cts_val[idx])

    # corrs = []
    # for ct, sp in zip(bootstrap_ct, bootstrap_sp):
    #     # print(ct.shape, sp.shape)
    #     u, v = model(ct.to(DEVICE), sp.to(DEVICE))
    #     # print(u.shape, v.shape)

    #     criterion = CCA(6, False, DEVICE)
    #     loss, corr_vec = criterion(u, v)
    #     # print(-loss)
    #     # print(corr_vec)
    #     corrs.append(-loss.cpu())
    # print("Mean of bootstrap sample correlations: ", np.mean(corrs))
    # plt.hist(corrs, bins=25)
    # plt.savefig('bootstrap.jpg')

    corrs = []
    for ct, sp in zip(bootstrap_ct, bootstrap_sp):
        # print(ct.shape, sp.shape)
        if not args.feature_level:
            example_matching_sample_idx = get_min_dist_samples_idx(sp, sps_tri, dims=(1,2,3))
        else:
            v_cand = model.sp_encoder(sp.to(DEVICE))[0].detach().cpu().squeeze()
            example_matching_sample_idx = get_min_dist_samples_idx(v_cand, v_tri, dims=1)
        # print(example_matching_sample_idx)
        ct_matching = cts_tri[np.array(example_matching_sample_idx)]
        u, v = model(ct_matching.to(DEVICE), sp.to(DEVICE))
        # print(u.shape, v.shape)

        criterion = CCA(6, False, DEVICE)
        loss, corr_vec = criterion(u, v)
        # print(-loss)
        # print(corr_vec)
        corrs.append(-loss.cpu())
    print("Mean of bootstrap sample correlations: ", np.mean(corrs))

    alpha = ALPHA
    T = np.quantile(corrs, alpha)
    print("Threshold: ", T)

    # Test normal
    cts_test_normal = torch.cat([ct for (ct, _) in test_set], dim=0).unsqueeze(1)
    sps_test_normal = torch.cat([sp.unsqueeze(0) for (_, sp) in test_set], dim=0)
    n_normal_samples = 200
    normal_sample_size = bsz
    print("Normal sample pool size: ", len(sps_test_normal))

    normal_sp, normal_ct = [], []
    for _ in range(n_normal_samples):
        idx = np.random.choice(len(sps_test_normal), normal_sample_size, False)
        normal_sp.append(sps_test_normal[idx])
        normal_ct.append(cts_test_normal[idx])

    corrs0 = []
    for ct, sp in zip(normal_ct, normal_sp):
        # print(ct.shape, sp.shape)
        if not args.feature_level:
            example_matching_sample_idx = get_min_dist_samples_idx(sp, sps_tri, dims=(1,2,3))
        else:
            v_cand = model.sp_encoder(sp.to(DEVICE))[0].detach().cpu().squeeze()
            example_matching_sample_idx = get_min_dist_samples_idx(v_cand, v_tri, dims=1)

        # print(example_matching_sample_idx)
        ct_matching = cts_tri[np.array(example_matching_sample_idx)]
        u, v = model(ct_matching.to(DEVICE), sp.to(DEVICE))
        # print(u.shape, v.shape)

        criterion = CCA(6, False, DEVICE)
        loss, corr_vec = criterion(u, v)
        # print(-loss)
        # print(corr_vec)
        corrs0.append(-loss.cpu())
    print("Mean of normal sample correlations: ", np.mean(corrs0))
    normal_acc = sum(corrs0 >= T) / n_normal_samples

    # Test abnormal
    cts_test_abnormal = torch.cat([ct for (ct, _) in dset], dim=0).unsqueeze(1)
    sps_test_abnormal = torch.cat([sp.unsqueeze(0) for (_, sp) in dset], dim=0)
    print("Abnormal sample pool size: ", len(sps_test_abnormal))
    n_abnormal_samples = 200
    abnormal_sample_size = bsz

    abnormal_sp, abnormal_ct = [], []
    for _ in range(n_abnormal_samples):
        idx = np.random.choice(len(sps_test_abnormal), abnormal_sample_size, False)
        abnormal_sp.append(sps_test_abnormal[idx])
        abnormal_ct.append(cts_test_abnormal[idx])

    corrs1 = []
    for ct, sp in zip(abnormal_ct, abnormal_sp):
        # print(ct.shape, sp.shape)
        if not args.feature_level:
            example_matching_sample_idx = get_min_dist_samples_idx(sp, sps_tri, dims=(1,2,3))
        else:
            v_cand = model.sp_encoder(sp.to(DEVICE))[0].detach().cpu().squeeze()
            example_matching_sample_idx = get_min_dist_samples_idx(v_cand, v_tri, dims=1)
            
        # print(example_matching_sample_idx)
        ct_matching = cts_tri[np.array(example_matching_sample_idx)]
        u, v = model(ct_matching.to(DEVICE), sp.to(DEVICE))
        # print(u.shape, v.shape)

        criterion = CCA(6, False, DEVICE)
        loss, corr_vec = criterion(u, v)
        # print(-loss)
        # print(corr_vec)
        corrs1.append(-loss.cpu())
    print("Mean of abnormal sample correlations: ", np.mean(corrs1))
    abnormal_acc = sum(corrs1 < T) / n_abnormal_samples

    f1 = (2 * abnormal_acc) / (2 * abnormal_acc + (1 - normal_acc) + (1 - abnormal_acc))

    false_alarms = 1 - normal_acc
    mis_detections = 1 - abnormal_acc

    prec=5
    print("False Alarms: ", 100*np.round(false_alarms, prec))
    print("Mis Detections: ", 100*np.round(mis_detections, prec))
    print("F1: ", 100*np.round(f1, prec))

    # Controlling False Alarm Rate
    print("Start experiments on controlling Type-I errors: ")
    alphas=[0.05, 0.1, 0.2, 0.3]

    for target_alpha in alphas:
        T_adjusted = np.quantile(corrs0, target_alpha)
        print(f"\nAdjusted threshold for Type I error {target_alpha}: ", T_adjusted)
        # Verification of type I errors
        normal_acc = sum(corrs0 >= T_adjusted) / n_normal_samples
        # Test mis-detection rate
        abnormal_acc = sum(corrs1 < T_adjusted) / n_abnormal_samples

        f1 = (2 * abnormal_acc) / (2 * abnormal_acc + (1 - normal_acc) + (1 - abnormal_acc))
        false_alarms = 1 - normal_acc
        mis_detections = 1 - abnormal_acc

        print("False Alarms: ", 100*np.round(false_alarms, prec))
        print("Mis Detections: ", 100*np.round(mis_detections, prec))
        print("F1: ", 100*np.round(f1, prec))
    


# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# colors = ['#9467bd', '#17becf', '#bcbd22'] 
colors = ['#d62728', '#1f77b4', '#7f7f7f']
# plt.hist(corrs, bins=25, alpha=0.6, label='Bootstrap', color=colors[2])
plt.hist(corrs0, bins=25, alpha=0.6, label='Normal', color=colors[1])
plt.hist(corrs1, bins=25, alpha=0.6, label='Abnormal', color=colors[0])
plt.vlines(x=T, ymin=0, ymax=150, color='black', label=f'Threshold: {round(T, 2)}')
plt.legend()
y_max = 30
plt.ylim((0,y_max))
plt.xticks(np.arange(0,6.5,0.5))
plt.yticks(np.arange(0, y_max + 10, 10))
plt.xlabel("Canonical Correlation Score", fontsize=12)
plt.ylabel("Count", fontsize=12)
# EXP_NAME = r"$T_{cir}, T_{crk}$"
# EXP_NAME = "Case Study 1"
EXP_NAME = "Case Study"
plt.title(f"Canonical Correlation Distribution ({EXP_NAME})", fontsize=14)
# plt.hist(corrs2, bins=25, alpha=0.5)
plt.savefig(os.path.join('Document', 'real data plot', f"{EXP_NAME}.png"), dpi=200)
# plt.show()

# Comparing T2 hotelling    
def hotelling_T2(method='dcca', p=6, m=200, n=8, alpha=0.05):
    # Phase I data
    print("Hotelling T2")
    # Bootstrap sample size
    cts_tri = torch.cat([ct for (ct, _) in tri_set], dim=0).unsqueeze(1)
    sps_tri = torch.cat([sp.unsqueeze(0) for (_, sp) in tri_set], dim=0)
    cts_val = torch.cat([ct for (ct, _) in val_set], dim=0).unsqueeze(1)
    sps_val = torch.cat([sp.unsqueeze(0) for (_, sp) in val_set], dim=0)
    print(sps_tri.shape, cts_tri.shape)

    if method == 'dcca':
        print("="*80)
        print("DCCA")
        sp_val_ldr = torch.utils.data.DataLoader(sps_val, batch_size=500, drop_last=False)
        for sp in sp_val_ldr:
            # print(sp.shape)
            v = model.sp_encoder(sp.to(DEVICE))[0].detach().cpu().squeeze()
    elif method == 'pca':
        print("="*80)
        print("PCA")
        sp_training = sps_tri.reshape((-1, 160)).to(DEVICE)
        print(f"PCA Training Shape: {sp_training.shape}")
        scaler = StandardScaler()
        sp_training = scaler.fit_transform(sp_training.squeeze().detach().cpu().numpy())
        pca = PCA(n_components=p).fit(sp_training)
        explained_variance_ratio = pca.explained_variance_ratio_
        # Total variance explained by all components
        total_variance_explained = np.sum(explained_variance_ratio)
        print(f"Explained Variance Ratio: {explained_variance_ratio}")
        print(f"Total Variance Explained: {total_variance_explained}")
        # Standardize: Zero mean, unit variance
        sps_tri_standardized = scaler.fit_transform(sps_val.detach().cpu().reshape((-1, 160)).numpy())
        v_pca = pca.transform(sps_tri_standardized)
        v = v_pca
    elif method == 'pls':
        print("="*80)
        print("PLS")
        sp_training = sps_tri.reshape((-1, 160)).to(DEVICE)
        ct_training = cts_tri.reshape((-1, 250*730)).to(DEVICE)
        from sklearn.cross_decomposition import PLSRegression
        print(f"PLS CT Training Shape: {ct_training.shape}")
        print(f"PLS SP Training Shape: {sp_training.shape}")
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        sp_training_scaled = scaler_X.fit_transform(sp_training.squeeze().detach().cpu().numpy())
        print(f"Reshaped CTs: {ct_training.reshape(len(ct_training), -1).shape}")
        ct_training_scaled = scaler_Y.fit_transform(ct_training.reshape(len(ct_training), -1).detach().cpu().numpy())
        pls = PLSRegression(n_components=p).fit(sp_training_scaled, ct_training_scaled)
        # Evalute features on validation set
        sps_scaled = scaler_X.fit_transform(sps_val.detach().reshape((-1, 160)).numpy())
        v_pls = pls.transform(sps_scaled)
        v = v_pls

    elif method == 'simclr-clip':
        print("="*80)
        print("SIMCLR-CLIP")
        simclr_clip_model = CLIPSimCLRREAL(h=args.h).to(DEVICE)
        simclr_clip_model.sp_encoder.load_state_dict(torch.load(os.path.join(CKPT_SAVE_DIR, "SIMCLR-CLIP", f'SIMCLR-CLIP_4B_h=6.pt'))['sp_encoder'])
        sp_val_ldr = torch.utils.data.DataLoader(sps_val, batch_size=500, drop_last=False)
        for sp in sp_val_ldr:
            # print(sp.shape)
            v = simclr_clip_model.sp_encoder(sp.to(DEVICE))[0].detach().cpu().squeeze()

    elif method == 'simclr':
        print("="*80)
        print("SIMCLR")
        simclr_model = SimCLRREAL(h=args.h).to(DEVICE)
        simclr_model.sp_encoder.load_state_dict(torch.load(os.path.join(CKPT_SAVE_DIR, "SIMCLR", f'SIMCLR_4B_h=6.pt'))['sp_encoder'])
        sp_val_ldr = torch.utils.data.DataLoader(sps_val, batch_size=500, drop_last=False)
        for sp in sp_val_ldr:
            # print(sp.shape)
            v = simclr_model.sp_encoder(sp.to(DEVICE))[0].detach().cpu().squeeze()

    elif method == 'AE':
        print("="*80)
        print("AE")
        ae_model = SpecAutoEncoder(h=args.h).to(DEVICE)
        ae_model.encoder.load_state_dict(torch.load(os.path.join(CKPT_SAVE_DIR, "AE", f'AE_4B_h=6.pt'))['model'])
        sp_val_ldr = torch.utils.data.DataLoader(sps_val, batch_size=500, drop_last=False)
        for sp in sp_val_ldr:
            # print(sp.shape)
            v = ae_model.encoder(sp.to(DEVICE))[0].detach().cpu().squeeze()

    # v = np.array(v)
    print(v.shape)
    phase_1 = []
    for _ in range(m):
        idx = np.random.choice(len(v), n, False)
        # phase_1.append(np.log(np.abs(v[idx, :]) + 1e-6))
        phase_1.append(v[idx, :])
    phase_1 = np.array(phase_1)
    print(f"Phase I data shape: {phase_1.shape}")

    mu = np.mean(phase_1, axis=1)
    grand_mu = np.mean(phase_1, axis=(0,1))
    S_matrices = []  
    print(f"Grand mu: {grand_mu}")

    for k in range(m):
        X_k = phase_1[k]  # Shape (n, p)
        S_k = np.cov(X_k, rowvar=False, ddof=1)  # Covariance matrix for sample k
        S_matrices.append(S_k)
    S = np.mean(S_matrices, axis=0)
    S_inv = np.linalg.inv(S)
    # alpha = 0.05
    critical_F = stats.f.ppf(1 - alpha, p, m*n - m-p+1)
    UCL = ((p * (m+1)*(n-1))/ (m*n - m - p + 1)) * critical_F
    print(f"UCL: {UCL}.")

    pred_normal = []
    t_scores_normal = []

    for (_, sps) in zip(normal_ct, normal_sp):
        if method == 'dcca':
            vs = model.sp_encoder(sps.to(DEVICE))[0].detach().cpu().squeeze().numpy()
        elif method == 'pca':
            sps_standardized = scaler.transform(sps.detach().cpu().reshape((-1, 160)).numpy())
            vs = pca.transform(sps_standardized)
        elif method == 'pls':
            sps_standardized = scaler_X.transform(sps.detach().cpu().reshape((-1, 160)).numpy())
            vs = pls.transform(sps_standardized)
        elif method == 'simclr-clip':
            vs = simclr_clip_model.sp_encoder(sps.to(DEVICE))[0].detach().cpu().squeeze().numpy()
        elif method == 'simclr':
            vs = simclr_model.sp_encoder(sps.to(DEVICE))[0].detach().cpu().squeeze().numpy()
        elif method == 'AE':
            vs = ae_model.encoder(sps.to(DEVICE))[0].detach().cpu().squeeze().numpy()

        # # Make normal
        # vs = np.log(np.abs(vs) + 1e-6)
        # for i in range(vs.shape[1]):
        #     stat, p_value = stats.shapiro(vs[:, i])  # Shapiro-Wilk test
        #     print(f"Variable {i+1}: p-value = {p_value:.4f} (Shapiro-Wilk test)")
        # break
        
        diff = (np.mean(vs, axis=0) - grand_mu).reshape(-1, 1)
        T2 = n * (diff.T @ S_inv @ diff).squeeze()
        t_scores_normal.append(T2)
        pred_normal.append(T2 < UCL)
        # print(T2)
    pred_normal = np.array(pred_normal)
    t_scores_normal = np.array(t_scores_normal)

    pred_abnormal = []
    t_scores_abnormal = []
    for (_, sps) in zip(abnormal_ct, abnormal_sp):
        if method == 'dcca':
            vs = model.sp_encoder(sps.to(DEVICE))[0].detach().cpu().squeeze().numpy()
        elif method == 'pca':
            sps_standardized = scaler.transform(sps.detach().cpu().reshape((-1, 160)).numpy())
            vs = pca.transform(sps_standardized)
        elif method == 'pls':
            sps_standardized = scaler_X.transform(sps.detach().cpu().reshape((-1, 160)).numpy())
            vs = pls.transform(sps_standardized)
        elif method == 'simclr-clip':
            vs = simclr_clip_model.sp_encoder(sps.to(DEVICE))[0].detach().cpu().squeeze().numpy()
        elif method == 'simclr':
            vs = simclr_model.sp_encoder(sps.to(DEVICE))[0].detach().cpu().squeeze().numpy()
        elif method == 'AE':
            vs = ae_model.encoder(sps.to(DEVICE))[0].detach().cpu().squeeze().numpy()

        # # Make normal
        # vs = np.log(np.abs(vs) + 1e-6)

        diff = (np.mean(vs, axis=0) - grand_mu).reshape(-1, 1)
        T2 = n * (diff.T @ S_inv @ diff).squeeze()
        t_scores_abnormal.append(T2)
        # print(T2)
        pred_abnormal.append(T2 >= UCL)
    pred_abnormal = np.array(pred_abnormal)
    t_scores_abnormal = np.array(t_scores_abnormal)

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

    # Adjusted threshold
    threshold = np.quantile(t_scores_normal, 1 - alpha)
    print(f"Adjusted threshold for Type I error {alpha}: ", threshold)
    false_alarms =  1 - sum(t_scores_normal < threshold) / len(t_scores_normal)
    mis_detections =  1 - sum(t_scores_abnormal >= threshold) / len(t_scores_abnormal)
    F1 = 2 * sum(t_scores_abnormal >= threshold) / (2*sum(t_scores_abnormal >= threshold) + sum(t_scores_normal >= threshold) + sum(t_scores_abnormal < threshold))
    print("False Alarms: ")
    print(100*np.round(false_alarms, prec))
    print("Mis Detections: ")
    print(100*np.round(mis_detections, prec))
    print("F1: ")
    print(100*np.round(F1, prec))


alpha=ALPHA
hotelling_T2(method='dcca', p=6, m=1000, alpha=alpha)
hotelling_T2(method='pca', p=10, m=1000, alpha=alpha)
hotelling_T2(method='pls', p=6, m=1000, alpha=alpha)
hotelling_T2(method='simclr-clip', p=6, m=1000, alpha=alpha)
hotelling_T2(method='simclr', p=6, m=1000, alpha=alpha)
hotelling_T2(method='AE', p=6, m=1000, alpha=alpha)
