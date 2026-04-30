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
from models.img_autoencoder import *
from models.spec_autoencoder import *
from const import *

parser = argparse.ArgumentParser()
# parser.add_argument('--dset', help="name of dataset", default='1Dr', type=str)
parser.add_argument('--dset', type=lambda s: re.split(' |, ', s),
                    default="1Dr", help='comma or space delimited list of dataset names')
parser.add_argument('--normalize',action='store_true')
parser.add_argument('--GL',action='store_true')
parser.add_argument('--h', help='hidden dimensions', default=16, type=int)
parser.add_argument('--lr', help='learning rate', default = 1e-1, type=float)
parser.add_argument('--lr_step', help='learning rate scheduler stepsize', default = 1000, type=int)
parser.add_argument('--n_epochs', help='max_epochs', default = 1000, type=int)
parser.add_argument('--bsz_tri', help='training batch size', default = 16, type=int)
parser.add_argument('--bsz_val', help='validation batch size', default = 16, type=int)
parser.add_argument('--tv_ratio', help='train-validation ratio', default = 0.1, type=float)
parser.add_argument('--ckpt_name', help="checkpoint name (tag of checkpoints)", default=None, type=str)
parser.add_argument('--tag', help="checkpoint tag", default='0', type=str)
args = parser.parse_args()
dset_name = "-".join(args.dset)
cmd = " ".join(sys.argv)

# Path declaration: 
# TODO: revise this later
if args.GL:
    path = os.path.join(DATA_SPEC_DIR, "SPAE") 
    # will change later based on spectra data location
else:
    path = os.path.join(DATA_SPEC_DIR, "SPAE")

# Logging directories
ckpt_dir = os.path.join(CKPT_SAVE_DIR, "SPAE")
os.makedirs(ckpt_dir, exist_ok=True)
log_dir = os.path.join(CKPT_LOG_DIR, "SPAE")
os.makedirs(log_dir, exist_ok=True)
file_name = os.path.join(log_dir, f'spec_ae_{dset_name}_{args.tag}.txt')
f = open(file_name, 'w')
f.write('Spectra Autoencoder training logs\n')

# Configuration
f.write("="*80 + "\nCommand: " + cmd + "\n")
f.write("="*80 + "\n" + str(args) + "\n" + "="*80 + "\n")
device_information(f)
get_configuration(args, f)

# Dataset loading
f.write("="*80 + "\n")
f.write("Datast Loading\n")
SPECTRA_DATA = SP(parts=args.dset, normalize=args.normalize, f=f)
dataset = SPECTRA_DATA.gather_ae(True, f).unsqueeze(-2)
# if args.normalize:
#     dataset = SP.normalize(dataset)

tri_set, val_set = train_test_split(dataset, test_size=args.tv_ratio, random_state=2023)
f.write(f"Training set shape: {tri_set.shape}\n")
f.write(f"Validation set shape: {val_set.shape}\n")
f.write("="*80 + "\n")


f.write("="*80 + "\n")
# Model declaration
model = SpecAutoEncoder(args.h).to(DEVICE)

ckpt_name = args.ckpt_name
if ckpt_name is not None:
    ckpt_path = os.path.join(ckpt_dir, f'spec_ae_{ckpt_name}.pt')
    pre_trained = torch.load(ckpt_path)
    model.load_state_dict(pre_trained['model'])
    f.write(f'Checkpoint {ckpt_name} loaded successfully!\n')
    print(f'Checkpoint {ckpt_name} loaded successfully!')
else:
    f.write('No checkpoints loaded.\n')
    print('No checkpoints loaded.')


bsz_tri, bsz_val = args.bsz_tri, args.bsz_val
tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=bsz_tri, shuffle=True)
val_ldr = torch.utils.data.DataLoader(val_set, batch_size=bsz_val, shuffle=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
model, train_loss, val_loss = train_spectra_autoencoder(model, args.n_epochs, optimizer, scheduler, tri_ldr, val_ldr, f)

ckpt_name = os.path.join(ckpt_dir, f'spec_ae_{dset_name}_{args.tag}.pt')
torch.save({
    'train_loss': train_loss,
    'encoder': model.encoder.state_dict(),
    'decoder': model.decoder.state_dict(),
    'model': model.state_dict()
}, ckpt_name)
f.write(f"Training End...\n")
f.write("="*80 + "\n")
f.close()