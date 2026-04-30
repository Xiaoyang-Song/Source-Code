import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from torch.utils.tensorboard import SummaryWriter
from models.spec_autoencoder import *
from models.img_autoencoder import *

# Define device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_layers(layer_dims):
    layers = []
    for l_id in range(len(layer_dims) - 1):
        if l_id == len(layer_dims) - 2:
            layers.append(nn.Sequential(
                # nn.BatchNorm1d(
                #     num_features=layer_dims[l_id], affine=False),
                nn.Linear(layer_dims[l_id], layer_dims[l_id + 1]),
            ))
        else:
            layers.append(nn.Sequential(
                nn.Linear(layer_dims[l_id], layer_dims[l_id + 1]),
                nn.Sigmoid(),
                # nn.BatchNorm1d(
                #     num_features=layer_dims[l_id + 1], affine=False),
            ))
    return layers


class MLP(nn.Module):
    def __init__(self, input_dim, layer_dims: list, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dims = [self.input_dim] + layer_dims + [self.output_dim]
        # Model Declaration
        self.layers = make_layers(self.layer_dims)
        self.model = nn.ModuleList(self.layers)
        # self.initialize()

        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, output_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        out = self.sig(self.fc1(x))
        return self.fc2(out)

        for layer in self.model:
            # print(x.shape)
            x = layer(x)
            x = (x - x.mean()/x.std())
        return x


class DCCA(nn.Module):
    def __init__(self, h=16):
        super().__init__()
        self.h = h
        self.ct_encoder = Encoder(self.h)
        self.sp_encoder = SpecEncoder(self.h)

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


def train_DCCA(model, optimizer, criterion, tri_ldr, val_ldr, writer=None, n_epochs=100, n_log=10, f=None, adv_ldr=None, beta=0.1):
    tri_iter_count, val_iter_count = 0, 0
    tri_log, val_log = [], []
    for epoch in tqdm(range(n_epochs)):
        model.train()
        tri_loss, val_loss = 0, 0
        n_tri, n_val = 0, 0
        # for step, (x, y, _, _) in enumerate(tri_ldr):
        for step, (x, y) in enumerate(tri_ldr):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Optimization
            optimizer.zero_grad()

            out1, out2 = model(x, y)
            if adv_ldr is None:
                loss, corr_vec = criterion(out1, out2)
                loss.backward()
            else:
                loss, corr_vec = criterion(out1, out2)
                loss_adv, n_adv = 0, 0
                for x_adv, y_adv in adv_ldr:
                    x_adv, y_adv = x_adv.to(DEVICE), y_adv.to(DEVICE)
                    # out3, out4 = model(x_adv, y_adv)
                    out3, out4 = model(x, y_adv)
                    loss_adv_tmp, _ = criterion(out3, out4)
                    loss_adv += loss_adv_tmp
                    n_adv += 1

                total_loss = loss - beta * (loss_adv / n_adv)
                total_loss.backward()

            optimizer.step()
 
            # Log correlation statistics
            feature_name = [f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
            corr_vec = [corr.detach() for corr in corr_vec]
            if writer is not None:
                writer.add_scalar("Loss/Train", loss.detach(), tri_iter_count)
                writer.add_scalar("Corr/Train", -loss.detach(), tri_iter_count)
                writer.add_scalars(f"Correlation/Train", dict(zip(feature_name, corr_vec)), tri_iter_count)

            tri_iter_count += 1
            tri_loss += (loss.detach() * len(y))
            n_tri += len(y)

        tri_log.append(tri_loss / n_tri)
        if (epoch + 1) % n_log == 0:
            print(f"Epoch {epoch:<3} | Train Loss: {tri_loss / n_tri:.5f} | Corr: {-tri_loss / n_tri:.5f}")
            if f is not None:
                f.write(f"Epoch {epoch:<3} | Train Loss: {tri_loss / n_tri:.5f} | Corr: {-tri_loss / n_tri:.5f}\n")
            
        # Validation step
        with torch.no_grad():
            # for step, (x, y, _, _) in enumerate(val_ldr):
            for step, (x, y) in enumerate(val_ldr):
                x, y = x.to(DEVICE), y.to(DEVICE)
                out1, out2 = model(x, y)
                loss, corr_vec = criterion(out1, out2)
                # Log correlation statistics
                feature_name = [f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
                corr_vec = [corr.detach() for corr in corr_vec]
                if writer is not None:
                    writer.add_scalar("Loss/Eval", loss.detach(), val_iter_count)
                    writer.add_scalar("Corr/Eval", -loss.detach(), val_iter_count)
                    writer.add_scalars(f"Correlation/Eval", dict(zip(feature_name, corr_vec)), val_iter_count)

                val_iter_count += 1
                val_loss += (loss.detach() * len(y))
                n_val += len(y)

            val_log.append(val_loss / n_val)
            if (epoch + 1) % n_log == 0:
                print(f"Epoch {epoch:<3} | Eval Loss: {val_loss / n_val:.5f} | Corr: {-val_loss / n_val:.5f}")
            if f is not None:
                f.write(f"Epoch {epoch:<3} | Eval Loss: {val_loss / n_val:.5f} | Corr: {-val_loss / n_val:.5f}\n")

    return model, tri_log, val_log, writer

if __name__ == 'main':
    print('DCCA Test Suites.')