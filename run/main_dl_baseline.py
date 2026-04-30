import numpy as np
import json
import argparse
import os
import re
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# This needs to be customized later (also needed to be customized when using GL)
sys.path.append('c:\\Users\\xysong\\Desktop\\Research\\DCCA-Image-Spectrum-Matching')
sys.path.append('../')
from util.utils import *
from dataset import *
from models.dcca import *
from models.dl_baseline import *
from objective.loss import *
from const import *

parser = argparse.ArgumentParser()
# parser.add_argument('--dset', help="name of dataset", default='1Dr', type=str)
parser.add_argument('--dset', type=lambda s: re.split(' |, ', s),
                    default="1Dr", help='comma or space delimited list of SP dataset names')
parser.add_argument('--normalize',action='store_true')
parser.add_argument('--GL',action='store_true')
parser.add_argument('--flag', help="experiment to automate")
parser.add_argument('--h', help='hidden dimensions', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default = 1e-4, type=float)
parser.add_argument('--lr_step', help='learning rate scheduler stepsize', default = 1000, type=int)
parser.add_argument('--n_epochs', help='max_epochs', default = 1000, type=int)
parser.add_argument('--bsz_tri', help='training batch size', default = 16, type=int)
parser.add_argument('--bsz_val', help='validation batch size', default = 16, type=int)
parser.add_argument('--tv_ratio', help='train-validation ratio', default = 0.2, type=float)
parser.add_argument('--n_log', help='logging frequency', default = 25, type=int)
parser.add_argument('--tag', help="checkpoint tag", default='0', type=str)
args = parser.parse_args()
dset_name = "-".join(args.dset)
cmd = " ".join(sys.argv)

exp = args.flag

# Logging directories
ckpt_dir = os.path.join(CKPT_SAVE_DIR, exp)
os.makedirs(ckpt_dir, exist_ok=True)
log_dir = os.path.join(CKPT_LOG_DIR, exp)
os.makedirs(log_dir, exist_ok=True)
file_name = os.path.join(log_dir, f'{exp}_{dset_name}_{args.tag}.txt')
f = open(file_name, 'w')
f.write(f'{exp} training logs\n')

# Configuration
f.write("="*80 + "\nCommand: " + cmd + "\n")
f.write("="*80 + "\n" + str(args) + "\n" + "="*80 + "\n")
device_information(f)
# get_configuration(args, f)

# Dataset Loading
f.write("="*80 + "\n")
f.write("Datast Loading\n")
# TODO: Rewrite data processing part
SPECTRA_DATA = SP(parts=args.dset, normalize=args.normalize, f=f)
SPECTRA_DATA.gather(False, f)
CT_DATA = CT(parts=args.dset)
CT_DATA.read(f=f)
CT_DATA.gather(f=f)

aligner = Aligner(parts=args.dset, ct=CT_DATA, sp=SPECTRA_DATA)
aligner.align()
dataset = aligner.gather()
dataset = Aligner.clip_window(dataset)
print(len(dataset))

SAVE_TEST=True
test_set = None
if SAVE_TEST:
    n_test_size = 40
    idx = np.random.choice(len(dataset), n_test_size, False)
    test_set, dset = [], []
    for i in range(len(dataset)):
        if i in idx:
            test_set.append(dataset[i])
        else:
            dset.append(dataset[i])
    dataset = dset
print(len(dataset))
# Dataset split and loader declaration
tri_set, val_set = train_test_split(dataset, test_size=args.tv_ratio, random_state=2023)
tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=args.bsz_tri, shuffle=True, drop_last=True)
val_ldr = torch.utils.data.DataLoader(val_set, batch_size=args.bsz_val, shuffle=True, drop_last=True)

if args.flag == "SIMCLR-CLIP":
    f.write(f"Training SIMCLR-CLIP...\n")
    # Model declaration
    model = train_clip_simclr_real(tri_ldr, h=args.h, epochs=args.n_epochs, batch_size=args.bsz_tri, lr=args.lr)
    # Save models
    ckpt_name = os.path.join(ckpt_dir, f'{exp}_{dset_name}_{args.tag}.pt')
    torch.save({
        'sp_encoder': model.sp_encoder.state_dict(),
        'ct_encoder': model.ct_encoder.state_dict(),
        'model': model.state_dict(),
        'tri_set': tri_set,
        'val_set': val_set,
        'test_set': test_set
    }, ckpt_name)
    f.write(f"Training End...\n")
    f.write("="*80 + "\n")
    f.close()

elif args.flag == "SIMCLR":
    f.write(f"Training SIMCL...\n")
    # Model declaration
    model = train_simclr_real(tri_ldr, h=args.h, epochs=args.n_epochs, batch_size=args.bsz_tri, lr=args.lr)
    # Save models
    ckpt_name = os.path.join(ckpt_dir, f'{exp}_{dset_name}_{args.tag}.pt')
    torch.save({
        'sp_encoder': model.sp_encoder.state_dict(),
        'model': model.state_dict(),
        'tri_set': tri_set,
        'val_set': val_set,
        'test_set': test_set
    }, ckpt_name)
    f.write(f"Training End...\n")
    f.write("="*80 + "\n")
    f.close()

elif args.flag == "AE":
    f.write(f"Training Autoencoder...\n")
    # Model declaration
    model = train_spec_autoencoder(tri_ldr, val_ldr, h=args.h, num_epochs=args.n_epochs, lr=args.lr, device=DEVICE)
    # Save models
    ckpt_name = os.path.join(ckpt_dir, f'{exp}_{dset_name}_{args.tag}.pt')
    torch.save({
        'model': model.encoder.state_dict(),
        'tri_set': tri_set,
        'val_set': val_set,
        'test_set': test_set
    }, ckpt_name)
    f.write(f"Training End...\n")
    f.write("="*80 + "\n")
    f.close()