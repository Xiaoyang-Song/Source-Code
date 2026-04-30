import torch
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from torchinfo import summary
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
import numpy as np
from tqdm import tqdm
import sys
# For local lab computer
sys.path.append('c:\\Users\\xysong\\Desktop\\Research\\DCCA-Image-Spectrum-Matching')
from models.spec_autoencoder import *
from sklearn.metrics import confusion_matrix
from const import *


class DecNet(nn.Module):
    def __init__(self, h, encoder, op=False, freeze=False):
        super().__init__()
        self.op = op
        self.h = h
        self.freeze = freeze
        self.encoder = encoder
        self.fc1 = nn.Linear(h, 16)
        if self.op:
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, 16)
            self.fc4 = nn.Linear(16, 1)
        else:
            self.fc4 = nn.Linear(16, 1)
        # Freeze parameters of the encoder
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        In: (B, window_size, channel, n_wl) or (window_size, channel, n_wl)
        Out: (B, 1) representing the probability
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        feat = self.encoder(x)[0].squeeze()
        feat = feat[:, 0:self.h]
        out = self.fc1(feat)
        out = F.relu(out)
        if self.op:
            out = self.fc2(out)
            out = F.relu(out)
            out = self.fc3(out)
            out = F.relu(out)
        out = self.fc4(out)
        out = torch.sigmoid(out)
        # print(out)
        return out
    
def get_pred(probs):
    return [1 if p >= 0.5 else 0 for p in probs.detach().cpu()]

def train_decnet(model, optimizer, tri_ldr, val_ldr, n_epochs=200, n_log=1, f=None):

    for i in range(n_epochs):
        y_true, preds, n_correct, n_total = [], [], 0, 0
        for (x, y, s) in tri_ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            decisions = model(x)
            loss = F.binary_cross_entropy(decisions, y.unsqueeze(-1).to(torch.float32))
            loss.backward()
            optimizer.step()

            pred = get_pred(decisions)
            preds += pred
            y_true += list(y.detach().cpu())
            n_correct += sum(np.array(pred) == np.array(y.detach().cpu()))
            n_total += len(y)
        # Compute confusion matrix
        conf_mat = confusion_matrix(y_true, preds)
        # Logging
        if (i + 1) % n_log == 0:
            print(f"Epoch {i+1} - Training ACC: {n_correct/n_total:.4f}")
            print(f"Epoch {i+1} - Truth: {dict(Counter(np.array(y_true)))}")
            print(f"Epoch {i+1} - Training Predictions: {dict(Counter(preds))}")
            print(f"Epoch {i+1} - Confusion Matrix: \n{conf_mat}\n")
            if f is not None:
                f.write(f"Epoch {i+1} - Training ACC: {n_correct/n_total:.4f}\n")
                f.write(f"Epoch {i+1} - Truth: {dict(Counter(np.array(y_true)))}\n")
                f.write(f"Epoch {i+1} - Preds: {dict(Counter(preds))}\n")
                f.write(f"Epoch {i+1} - Confusion Matrix: \n{conf_mat}\n")
        # Evaluation
        with torch.no_grad():
            y_true, preds, n_correct, n_total = [], [], 0, 0
            for (x, y, s)in val_ldr:
                x, y = x.to(DEVICE), y.to(DEVICE)
                decisions = model(x)
                pred = get_pred(decisions)
                preds += pred
                y_true += list(y.detach().cpu())
                n_correct += sum(np.array(pred) == np.array(y.detach().cpu()))
                n_total += len(y)
        # Compute confusion matrix
        conf_mat = confusion_matrix(y_true, preds)
        # Logging
        if (i + 1) % n_log == 0:
            print(f"Epoch {i+1} - Eval ACC: {n_correct/n_total:.4f}")
            print(f"Epoch {i+1} - Truth: {dict(Counter(np.array(y_true)))}")
            print(f"Epoch {i+1} - Eval Predictions: {dict(Counter(preds))}")
            print(f"Epoch {i+1} - Confusion Matrix: \n{conf_mat}\n")
            if f is not None:
                f.write(f"Epoch {i+1} - Training ACC: {n_correct/n_total:.4f}\n")
                f.write(f"Epoch {i+1} - Truth: {dict(Counter(np.array(y_true)))}\n")
                f.write(f"Epoch {i+1} - Preds: {dict(Counter(preds))}\n")
                f.write(f"Epoch {i+1} - Confusion Matrix: \n{conf_mat}\n")

    return model
    

if __name__ == '__main__':
    print("DecNet Test Suites.")
    from models.spec_autoencoder import *
    ckpt_name = '1Dr-4B-4D_ne'
    sp_model = SpecAutoEncoder()
    ckpt_path = os.path.join(CKPT_SAVE_DIR, 'SPAE', f'spec_ae_{ckpt_name}.pt')
    pre_trained = torch.load(ckpt_path, map_location=DEVICE)
    sp_model.load_state_dict(pre_trained['model'])
    print(f'Checkpoint {ckpt_name} loaded successfully!')

    encoder = sp_model.encoder
    net = DecNet(encoder).to(DEVICE)
    eg = torch.normal(0, 1, (4, 5, 1, 32)).to(DEVICE)
    out = net(eg)
    