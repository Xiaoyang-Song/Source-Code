import numpy as np
from PIL import Image
import argparse
import os
import yaml
import torch
import torch.utils.data
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import sys
# This needs to be customized later (also needed to be customized when using GL)
sys.path.append('c:\\Users\\xysong\\Desktop\\Research\\DCCA-Image-Spectrum-Matching')
from util.utils import *
from dataset import *
from models.img_autoencoder import *
from const import *
import re

parser = argparse.ArgumentParser()
# parser.add_argument('--dset', help="name of dataset", default='1Dr', type=str)
parser.add_argument('--dset', type=lambda s: re.split(' |, ', s),
                    default="1Dr", help='comma or space delimited list of dataset names')
# parser.add_argument('--dset', type=str, required=True, help='finetuning dataset')
parser.add_argument('--n_non_defects', help='number of non-defective images', default = 100, type=int)
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

# Path declaration
# if args.GL:
#     path = os.path.join(DATA_PROCESSED_DIR, args.dset, 'final')
# else:
#     path = os.path.join(DATA_PROCESSED_DIR, args.dset, 'final')

# Logging directories
ckpt_dir = os.path.join(CKPT_SAVE_DIR, "IMAE")
os.makedirs(ckpt_dir, exist_ok=True)
log_dir = os.path.join(CKPT_LOG_DIR, "IMAE")
os.makedirs(log_dir, exist_ok=True)
file_name = os.path.join(log_dir, f'img_ae_{dset_name}_{args.tag}.txt')
f = open(file_name, 'w')
f.write('Image Autoencoder training logs\n')

# Configuration
f.write("="*80 + "\nCommand: " + cmd + "\n")
f.write("="*80 + "\n" + str(args) + "\n" + "="*80 + "\n")
device_information(f)
get_configuration(args, f)


# Dataset loading
f.write("="*80 + "\n")
f.write("Datast Loading\n")
CT_DATA = CT(parts=args.dset)
CT_DATA.read(f)
data = CT_DATA.gather(f)
dataset = CT_DATA.filter_non_defects(args.n_non_defects, f)
dataset = dataset.unsqueeze(1)

# Dataset Loading (optional)
# dataset = torch.load(os.path.join(FT_DATA_DIR, args.dset, "data_tri.pt"))

tri_set, val_set = train_test_split(dataset, test_size=args.tv_ratio, random_state=2023)
# f.write(f"Training set shape: {tri_set.shape}\n")
# f.write(f"Validation set shape: {val_set.shape}\n")
f.write("="*80 + "\n")


# Model declaration
model = AutoEncoder(args.h).to(DEVICE)

ckpt_name = args.ckpt_name
if ckpt_name is not None:
    ckpt_path = os.path.join(ckpt_dir, f'img_ae_{ckpt_name}.pt')
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
model, train_loss, val_loss = train_autoencoder(model, args.n_epochs, optimizer, scheduler, tri_ldr, val_ldr, f)


ckpt_name = os.path.join(ckpt_dir, f'img_ae_{dset_name}_{args.tag}.pt')
torch.save({
    'train_loss': train_loss,
    'encoder': model.encoder.state_dict(),
    'decoder': model.decoder.state_dict(),
    'model': model.state_dict()
}, ckpt_name)
f.write(f"Training End...\n")
f.close()