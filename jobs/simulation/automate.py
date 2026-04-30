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
import argparse
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(2024)
torch.manual_seed(2024)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='data directory')
parser.add_argument('--ckpt_dir', help='data directory')
parser.add_argument('--flag', help="experiment to automate")
parser.add_argument('--h', help="h", type=int)
parser.add_argument('--window_size', help="n_window", type=int, default=25)
parser.add_argument('--y_target', help="Target Y value as normal", type=int, default=0)
parser.add_argument('--n_abn', help="n_abn", type=int, default=0)
args = parser.parse_args()

data_dir = os.path.join('Data', args.data_dir)
ckpt_dir = os.path.join('checkpoint', args.ckpt_dir)

if args.flag == "DGP-AE":
# This script accelerate training process of DGP AE using GPU
    ctscans = torch.load(os.path.join(data_dir, 'ct-raw.pt'))
    print(ctscans.shape)
    model = run_sim_imgae(ctscans, h=args.h, bsz_tri=128, bsz_val=200, lr=5e-4)
    torch.save(model.state_dict(), os.path.join(data_dir, 'ct_ae-dgp.pt'))

elif args.flag == "AE":
    sp_name = '2fc'
    SP = torch.load(os.path.join(data_dir,  f'sp_{sp_name}.pt'))
    SP = torch.tensor(SP.unsqueeze(1), dtype=torch.float32).to(DEVICE)
    print(SP.shape)
    model, triset, valset = run_sim_specae(SP, h=args.h, bsz_tri=64, bsz_val=32, lr=1e-3)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'baseline-sp-ae-{args.h}.pt'))

elif args.flag == "DCCA":
# This script accelerate training process of DCCA using GPU
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.data_dir != 'simple':
        # training CT
        CT = torch.load(os.path.join(data_dir,  'ct.pt'))
        # training SP
        sp_name = '2fc'
        SP = torch.load(os.path.join(data_dir,  f'sp_{sp_name}.pt'))
        SP = torch.tensor(SP.unsqueeze(1), dtype=torch.float32)
        print(CT.shape, SP.shape)
        Y = torch.load(os.path.join(data_dir,  'label.pt'))
        print(Counter(np.array(Y)))
        # mask0 = Y == 1
    else:
        # Deprecated! not useful
        print("Testing simple simulation from one root...")
        # load simple simulation data
        CT = torch.load(os.path.join(data_dir, 'ct_train.pt'))
        SP = torch.load(os.path.join(data_dir,  f'sp_train.pt'))
        SP = torch.tensor(SP.unsqueeze(1), dtype=torch.float32)
        print(CT.shape, SP.shape)
        Y = torch.load(os.path.join(data_dir,  'y_train.pt'))
        print(Counter(np.array(Y)))
        # Y = Y.to(DEVICE)

    mask0 = Y == args.y_target

    # Number of abnormal
    cts0, cts1 = CT[mask0], CT[~mask0]
    sps0, sps1 = SP[mask0], SP[~mask0]
    # cts = torch.cat([cts0, cts1[0:n_abn]], dim=0)
    # sps = torch.cat([sps0, sps1[0:n_abn]], dim=0)
    # print(sps.shape)
    # ys = torch.cat([torch.zeros(len(sps0), dtype=torch.uint8), torch.ones(n_abn, dtype=torch.uint8)])
    # print(Counter(np.array(ys)))
    # data = list(zip(cts, sps, ys))
    # print(len(data))

    data = list(zip(cts0, sps0, Y[mask0]))

    h=args.h
    window_size = args.window_size
    # without reg
    if args.n_abn == 0:
        print("Training without regularization on abnormal samples...")
        dcca_model, tri_set, val_set = run_dcca(data, h=h, bsz_tri=500, bsz_val=args.window_size, lr=1e-2, n_epoch=500, n_log=50)
        ckpt_name = f'hypo-good-only-{h}D-window-{window_size}'
    else:
    # With reg
        print("Training with regularization on abnormal samples...")
        n_abn = args.n_abn
        adv_data = (CT[mask0][0:n_abn], SP[~mask0][0:n_abn])
        dcca_model, tri_set, val_set = run_dcca(data, h=h, bsz_tri=2000, bsz_val=500, lr=1e-3, n_epoch=2000, n_log=100, adv=True, adv_data=adv_data, beta=0.1)
        ckpt_name = f'hypo-good-only-{h}D-{n_abn}'

    os.makedirs(os.path.join(ckpt_dir, ckpt_name), exist_ok=True)
    torch.save(dcca_model, os.path.join(ckpt_dir, ckpt_name, 'model.pt'))
    torch.save(tri_set, os.path.join(ckpt_dir, ckpt_name, 'tri_set.pt'))
    torch.save(val_set, os.path.join(ckpt_dir, ckpt_name, 'val_set.pt'))

elif args.flag == "CLS":

    os.makedirs(ckpt_dir, exist_ok=True)
    # training CT
    CT = torch.load(os.path.join(data_dir,  'ct.pt'))
    # training SP
    sp_name = '2fc'
    SP = torch.load(os.path.join(data_dir,  f'sp_{sp_name}.pt'))
    SP = torch.tensor(SP.unsqueeze(1), dtype=torch.float32)
    print(CT.shape, SP.shape)

    # Load labels
    Y = torch.load(os.path.join(data_dir,  'label.pt'))
    print(Counter(np.array(Y)))

    mask0 = Y == args.y_target
    print(mask0.shape)
    sps0, sps1 = SP[mask0], SP[~mask0]
    print(sps0.shape, sps1.shape)
    # Number of abnormal
    # n_abn = len(sps1)
    n_abn = args.n_abn
    sps = torch.cat([sps0, sps1[0:n_abn]], dim=0)
    print(sps.shape)
    ys = torch.cat([torch.zeros(len(sps0)), torch.ones(n_abn)])
    print(ys.shape)

    data = list(zip(sps, torch.tensor(ys, dtype=torch.uint8)))
    model = SimSpecAutoEncoder(args.h).to(DEVICE)
    rose = True
    naive_model, tri_set, val_set = run_ft(data, args.h, model.encoder, bsz_tri=256, bsz_val=256, freeze=False, lr=1e-3, n_epoch=500, n_log=100, ROSE=rose)

    # Run on test set
    SP_test = torch.load(os.path.join(data_dir,  'sp_2fc-test.pt'))
    SP_test = torch.tensor(SP_test.unsqueeze(1), dtype=torch.float32)
    if args.y_target == 1:
        Y_test = 1 - torch.load(os.path.join(data_dir, 'label-test.pt'))
    else:
        Y_test = torch.load(os.path.join(data_dir, 'label-test.pt'))
    print("Test set: ", Counter(np.array(Y_test)))
    val_set = list(zip(SP_test, Y_test))
    tst_ldr = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=True)

    test_cls_on_dset(tst_ldr, naive_model)

    # Save models
    ckpt_name = f'naive-cls-{n_abn}'
    os.makedirs(os.path.join(ckpt_dir, ckpt_name), exist_ok=True)
    torch.save(naive_model, os.path.join(ckpt_dir, ckpt_name, 'model.pt'))
    torch.save(data, os.path.join(ckpt_dir, ckpt_name, 'data.pt'))

elif args.flag == 'SIMCLR':
    sp_name = '2fc'
    SP = torch.load(os.path.join(data_dir,  f'sp_{sp_name}.pt'))
    SP = torch.tensor(SP.unsqueeze(1), dtype=torch.float32).to(DEVICE)
    print(SP.shape)
    model = train_simclr(SP, h=args.h ,epochs=100, batch_size=128, lr=1e-3)
    torch.save(model.encoder.state_dict(), os.path.join(ckpt_dir, f'baseline-sp-simclr-{args.h}.pt'))

elif args.flag == 'SIMCLR-CLIP':
    CT = torch.load(os.path.join(data_dir,  'ct.pt'))
    # training SP
    sp_name = '2fc'
    SP = torch.load(os.path.join(data_dir,  f'sp_{sp_name}.pt'))
    SP = torch.tensor(SP.unsqueeze(1), dtype=torch.float32)
    Y = torch.load(os.path.join(data_dir,  'label.pt'))
    print(CT.shape, SP.shape)
    mask0 = Y == args.y_target
    # Number of abnormal
    cts0, cts1 = CT[mask0], CT[~mask0]
    sps0, sps1 = SP[mask0], SP[~mask0]
    data = list(zip(cts0, sps0))

    model = train_clip_simclr(data, h=args.h, epochs=100, batch_size=128, lr=1e-3)
    torch.save(model.sp_encoder.state_dict(), os.path.join(ckpt_dir, f'baseline-simclr-clip-{args.h}.pt'))

