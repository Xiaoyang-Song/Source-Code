import torch
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
# from const import *
# from dataset import *

# Define device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNNEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        # 1D Conv
        self.conv1 = nn.Conv1d(1, 256, 6)
        self.conv2 = nn.Conv1d(256, 32, 6)
        # Maxpooling
        self.maxpool1 = nn.MaxPool1d(3, return_indices=True)
        self.maxpool2 = nn.MaxPool1d(3, return_indices=True)
        # Flatten
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        In: (B, window_size, channel, n_wl)
        Out: (B, window_size, n_features)
        """
        B, ws, c, wl = x.shape
        # Reshape to (B x window_size, channel, n_wl)
        x = x.reshape((-1, c, wl))
        # Forward
        out = self.conv1(x)
        out = F.relu(out)
        out, idx1 = self.maxpool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out, idx2 = self.maxpool2(out)
        out = self.flatten(out)
        # Reshape to (B, window_size, n_features)
        out = out.reshape((B, ws, -1))
        return out, (idx1, idx2)
    
class LSTMEncoder(nn.Module):
    def __init__(self, h=6):
        super().__init__()
        self.lstm1 = nn.LSTM(32, 128, 1, batch_first=True)
        self.lstm2 = nn.LSTM(128, h, 1, batch_first=True)
    
    def forward(self, x):
        """
        In: (B, window_size, n_features)
        Out: (B, n_layer, n_h)
        """
        out, (hn, cn) = self.lstm1(x)
        out, (hn, cn) = self.lstm2(out)
        hn = hn.permute(1, 0, 2)
        return hn

class SpecEncoder(nn.Module):
    def __init__(self, h=6):
        super().__init__()
        self.cnn = CNNEncoder()
        self.lstm = LSTMEncoder(h=h)

    def forward(self, x):
        out, (idx1, idx2) = self.cnn(x)
        hn = self.lstm(out)
        return hn, (idx1, idx2)
    
class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(32, 256, 6)
        self.conv2 = nn.ConvTranspose1d(256, 1, 6)
        self.maxunpool1 = nn.MaxUnpool1d(3)
        self.maxunpool2 = nn.MaxUnpool1d(3)
    
    def forward(self, x, indices):
        """
        In: (B, window_size, n_features)
        Out: (B, window_size, channel, n_wl)
        """
        idx1, idx2 = indices
        # Reshape to (B x window_size, channel, n_features)
        B, ws, n_f = x.shape
        out = x.reshape((B * ws, n_f, 1))
        # 1D Conv
        out = self.maxunpool1(out, idx2, torch.Size([5, 16, 4]))
        out = self.conv1(out)
        out = F.relu(out)
        out = self.maxunpool2(out, idx1)
        out = self.conv2(out)
        # Reshape to (B, window_size, channel, n_wl)
        out = out.reshape((B, ws, 1, -1))
        return out

class LSTMDecoder(nn.Module):
    def __init__(self, h=6, ws=5):
        super().__init__()
        self.ws = ws
        self.lstm1 = nn.LSTM(h, 128, 1, batch_first=True)
        self.lstm2 = nn.LSTM(128, 32, 1, batch_first=True)

    def forward(self, x):
        """
        In: (B, n_layer, n_h)
        Out: 
        """
        B, n_layer, n_h = x.shape
        out = x.repeat((1, self.ws, 1))
        out, (hn, cn) = self.lstm1(out)
        out, (hn, cn) = self.lstm2(out)
        return out

class SpecDecoder(nn.Module):
    def __init__(self, h=6, ws=5):
        super().__init__()
        self.ws = ws
        self.cnn = CNNDecoder()
        self.lstm = LSTMDecoder(h=h, ws=self.ws)


    def forward(self, x, indices):
        out = self.lstm(x)
        out = self.cnn(out, indices)
        return out

class SpecAutoEncoder(nn.Module):
    def __init__(self, h=6):
        super().__init__()
        self.encoder = SpecEncoder(h).to(DEVICE)
        self.decoder = SpecDecoder(h).to(DEVICE)

    def forward(self, x):
        out, indices = self.encoder(x)
        out = self.decoder(out, indices)
        return out


def train_spectra_autoencoder(model, num_epochs, optimizer, scheduler, t_loader, v_loader, f=None, device=DEVICE):
    train_loss_avg, val_loss_avg = [], []
    print('Training Started...')
    if f is not None:
        f.write("Training Started...\n")
    for epoch in range(num_epochs):
        model.train()
        train_loss_avg.append(0)
        num_batches = 0
        for specs in t_loader:
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
        if f is not None:
            f.write('Epoch [%d / %d] average reconstruction error: %f\n' % (epoch+1, num_epochs, train_loss_avg[-1]))
            # f.write('Epoch [%d / %d] average reconstruction error per index: %f' % (epoch+1, num_epochs, train_loss_avg[-1]**0.5))
        scheduler.step()

        with torch.no_grad():
            val_loss_avg.append(0)
            num_batches_val = 0
            for specs in v_loader:
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
            # print('Epoch [%d / %d] average reconstruction error per index: %f' % (epoch+1, num_epochs, val_loss_avg[-1]**0.5))
            if f is not None:
                f.write('Epoch [%d / %d] average reconstruction error [Validation]: %f\n' % (epoch+1, num_epochs, val_loss_avg[-1]))
                # f.write('Epoch [%d / %d] average reconstruction error per index: %f' % (epoch+1, num_epochs, val_loss_avg[-1]**0.5))
            
    return model, train_loss_avg, val_loss_avg


if __name__ == '__main__':
    print("Spectra Autoencoder Architecture Test Suite.")
    # Test architecture
    encoder = SpecEncoder().to(DEVICE)
    decoder = SpecDecoder().to(DEVICE)
    eg = torch.zeros((4, 5, 1, 32)).to(DEVICE)
    # print(eg.shape)
    out, indices = encoder(eg)
    # print(out.shape)
    out = decoder(out, indices)
    # print(out.shape)

    # Step by step decomposition test
    cnn_encoder = CNNEncoder().to(DEVICE)
    out, indices = cnn_encoder(eg)
    lstm_encoder = LSTMEncoder().to(DEVICE)
    hn = lstm_encoder(out)
    lstm_decoder = LSTMDecoder().to(DEVICE)
    out = lstm_decoder(hn)
    cnn_decoder = CNNDecoder().to(DEVICE)
    out = cnn_decoder(out, indices)


    # Test trainer
    print(DEVICE)
    dataset = torch.ones((100, 5, 1, 32))
    tri_ldr = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    val_ldr = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    model = SpecAutoEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    model, train_loss, val_loss = train_spectra_autoencoder(model, 100, optimizer, scheduler, tri_ldr, val_ldr, None)

