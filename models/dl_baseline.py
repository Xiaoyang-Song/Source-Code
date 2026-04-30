import torch
import sys
sys.path.append('c:\\Users\\xysong\\Desktop\\Research\\DCCA-Image-Spectrum-Matching')
sys.path.append('../')
import numpy as np
from torchinfo import summary
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from const import *
from models.spec_autoencoder import *
from models.img_autoencoder import *
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from dataset import *
from models.dcca import *
from models.dl_baseline import *
from objective.loss import *
from const import *



class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLRREAL(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.sp_encoder = SpecEncoder(h)
        self.projector = ProjectionHead(in_dim=h, hidden_dim=64, out_dim=h)

    def forward(self, x):
        # print(x.shape)
        h = self.sp_encoder(x)[0].squeeze()
        # print(h.shape)
        z = self.projector(h)
        # print(z.shape)
        z = F.normalize(z, dim=1)
        return h, z

def nt_xent_loss(z1, z2, temperature=0.5):

    batch_size = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)

    similarity = F.cosine_similarity(
        z.unsqueeze(1),
        z.unsqueeze(0),
        dim=2
    )

    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)

    similarity = similarity / temperature
    similarity = similarity.masked_fill(mask, -1e9)

    positives = torch.cat([
        torch.diag(similarity, batch_size),
        torch.diag(similarity, -batch_size)
    ])

    denominator = torch.logsumexp(similarity, dim=1)

    loss = -positives + denominator
    return loss.mean()

def augment(x, noise_std=0.01, scale_std=0.01, drop_prob=0.05):
    # 1. Additive Gaussian noise
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise

    return x_aug

def train_simclr_real(loader, h, epochs=100, batch_size=256, lr=1e-3):

    model = SimCLRREAL(h).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        total_loss = 0

        for _, batch in loader:

            x = batch.to(DEVICE)

            x1 = augment(x)
            x2 = augment(x)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss/len(loader):.4f}")

    return model


class CLIPSimCLRREAL(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.sp_encoder = SpecEncoder(h)
        self.ct_encoder = Encoder(h)
        self.h = h

    def forward(self, img, spec):
        # CT Forward
        # (B, ws_ct, 1, 250, 730)
        B, ws_ct, c_ct, H, W = img.shape
        img = img.reshape((B * ws_ct, c_ct, H, W))
        out1 = self.ct_encoder(img)[0]
        out1 = out1.reshape((B, ws_ct, self.h))
        # Take the mean of CT scan features in the window
        out1 = torch.mean(out1, dim=1) 
        # (B, h)
        out2 = self.sp_encoder(spec)[0].squeeze()
        # print(out1.shape, out2.shape)
        return out1, out2

def cross_modal_nt_xent_loss(z1, z2, temperature=1):
    """
    Cross-modal contrastive loss.

    Args:
        z1: Tensor of shape [B, D], embeddings from modality 1
        z2: Tensor of shape [B, D], embeddings from modality 2
        temperature: scaling factor

    Returns:
        Scalar loss
    """
    # Normalize so dot product = cosine similarity
    # z1 = F.normalize(z1, dim=1)
    # z2 = F.normalize(z2, dim=1)

    # Cross-modal similarity matrix: [B, B]
    logits = torch.matmul(z1, z2.T) / temperature

    # Positive pairs are on the diagonal
    labels = torch.arange(z1.shape[0], device=z1.device)

    # Direction 1: modality 1 queries modality 2
    loss_12 = F.cross_entropy(logits, labels)

    # Direction 2: modality 2 queries modality 1
    loss_21 = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_12 + loss_21)
    
def train_clip_simclr_real(tri_ldr, h, epochs=100, batch_size=256, lr=1e-3):
    model = CLIPSimCLRREAL(h).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for x, y in tri_ldr:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # print(x.shape, y.shape)

            hx, hy = model(x, y)

            loss = cross_modal_nt_xent_loss(hx, hy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss/len(tri_ldr):.4f}")

    return model


# AE
def train_spec_autoencoder(t_loader, v_loader, h=6, num_epochs=256, lr=1e-3, device=DEVICE):

    train_loss_avg, val_loss_avg = [], []
    print('Training Started...')

    model = SpecAutoEncoder(h).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for epoch in range(num_epochs):
        model.train()
        train_loss_avg.append(0)
        num_batches = 0
        for _, specs in t_loader:
            specs = specs.to(device)

            optimizer.zero_grad()

            out = model(specs)
            loss = 0
            for i in range(specs.shape[0]):
                loss += F.mse_loss(out[i], specs[i], reduction='sum')
            loss /= specs.shape[0]
    
            # print(loss.item())
            # backpropagation
            loss.backward()
            optimizer.step()
            # logging
            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
        # print('Epoch [%d / %d] average reconstruction error per index: %f' % (epoch+1, num_epochs, train_loss_avg[-1]**0.5))

        with torch.no_grad():
            val_loss_avg.append(0)
            num_batches_val = 0
            for _, specs in v_loader:
                specs = specs.to(device)
                out = model(specs)
                loss = 0
                for i in range(specs.shape[0]):
                    loss += F.mse_loss(out[i], specs[i], reduction='sum')
                loss /= specs.shape[0]

                # logging
                val_loss_avg[-1] += loss.item()
                num_batches_val += 1

            val_loss_avg[-1] /= num_batches_val
            print('Epoch [%d / %d] average reconstruction error [Validation]: %f' % (epoch+1, num_epochs, val_loss_avg[-1]))

    return model