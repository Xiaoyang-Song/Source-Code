import torch
import numpy as np
from torchinfo import summary
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from const import *
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from objective.loss import *
from util.utils import *
import cv2
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA as SKCCA
import scipy

def form_sim_sp_label_dset(SP, Y):
    return list(zip(SP, Y))
class SimEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 3 x 32 x 32 ->  16 x 28 x 28
        self.pool1 = nn.MaxPool2d((2, 2), return_indices=True)  # 16 x 28 x 28 -> 16 x 14 x 14
        self.conv2 = nn.Conv2d(32, 64, 5)   # 16 x 14 x 14 -> 32 x 10 x 10
        self.pool2 = nn.MaxPool2d((2, 2), return_indices=True)  # 32 x 10 x 10 -> 32 x 5 x 5
        # 32 x 5 x 5 -> 256 units dense layer
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, hidden)  # 256 -> h

    def forward(self, x):
        x, indices1 = self.pool1(self.conv1(x))
        x, indices2 = self.pool2(self.conv2(x))
        x = x.view(-1, 64 * 5 * 5)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, indices1, indices2

class SimEncoder0(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)  # 3 x 32 x 32 ->  16 x 28 x 28
        self.pool1 = nn.MaxPool2d((2, 2), return_indices=True)  # 16 x 28 x 28 -> 16 x 14 x 14
        self.conv2 = nn.Conv2d(4, 4, 5)   # 16 x 14 x 14 -> 32 x 10 x 10
        self.pool2 = nn.MaxPool2d((2, 2), return_indices=True)  # 32 x 10 x 10 -> 32 x 5 x 5
        # 32 x 5 x 5 -> 256 units dense layer
        self.fc1 = nn.Linear(4 * 5 * 5, 16)
        self.fc2 = nn.Linear(16, hidden)  # 256 -> h

    def forward(self, x):
        x, indices1 = self.pool1(self.conv1(x))
        x, indices2 = self.pool2(self.conv2(x))
        x = x.view(-1, 4 * 5 * 5)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, indices1, indices2
    
class SimDecoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 512)
        self.fc2 = nn.Linear(512, 64 * 5 * 5)
        self.unpool1 = nn.MaxUnpool2d((2, 2))
        self.conv1 = nn.ConvTranspose2d(64, 32, 5)
        self.unpool2 = nn.MaxUnpool2d((2, 2))
        self.conv2 = nn.ConvTranspose2d(32, 1, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, indices1, indices2):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = out.reshape((-1, 64, 5, 5))
        out = self.unpool1(out, indices2)
        out = F.relu(self.conv1(out))
        out = self.unpool2(out, indices1)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out


class SimDecoder0(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 128)
        self.fc2 = nn.Linear(128, 32 * 5 * 5)
        self.unpool1 = nn.MaxUnpool2d((2, 2))
        self.conv1 = nn.ConvTranspose2d(32, 16, 5)
        self.unpool2 = nn.MaxUnpool2d((2, 2))
        self.conv2 = nn.ConvTranspose2d(16, 1, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, indices1, indices2):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = out.reshape((-1, 32, 5, 5))
        out = self.unpool1(out, indices2)
        out = F.relu(self.conv1(out))
        out = self.unpool2(out, indices1)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out

class SimAutoEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = SimEncoder(hidden_dim)
        self.decoder = SimDecoder(hidden_dim)

    def forward(self, x):
        output, indc1, indc2 = self.encoder(x)
        output = self.decoder(output, indc1, indc2)
        return output
    
class SimAutoEncoder0(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = SimEncoder0(hidden_dim)
        self.decoder = SimDecoder0(hidden_dim)

    def forward(self, x):
        output, indc1, indc2 = self.encoder(x)
        output = self.decoder(output, indc1, indc2)
        return output



def train_sim_autoencoder(model, num_epochs, optimizer, scheduler, t_loader, v_loader, device=DEVICE):

    train_loss_avg, val_loss_avg = [], []
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss_avg.append(0)
        num_batches = 0
        num_wrong, n = [], 0
        for image_batch in t_loader:
            image_batch = image_batch.to(device)
            # autoencoder reconstruction
            image_batch_recon = model(image_batch)
            # reconstruction error
            # loss = F.binary_cross_entropy(image_batch_recon, image_batch)
            loss = F.mse_loss(image_batch_recon, image_batch)
            # ic(loss)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            train_loss_avg[-1] += loss.item()
            num_wrong.append(torch.sum((image_batch != image_batch_recon)))
            n += len(image_batch)
            num_batches += 1
        percent_wrong = torch.sum(torch.tensor(num_wrong)) / (32 * 32 * n)
        print(f'Epoch [%d / %d] average wrong pixels: %f' % (epoch+1, num_epochs, percent_wrong))
        train_loss_avg[-1] /= num_batches
        print(f'Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
        scheduler.step()

        # Validation Step
        with torch.no_grad():
            val_loss_avg.append(0)
            num_wrong_val, n_val, n_batches_val = [], 0, 0
            for image_batch in v_loader:
                image_batch = image_batch.to(device)
                # autoencoder reconstruction
                image_batch_recon = model(image_batch)
                # reconstruction error
                # loss = F.binary_cross_entropy(image_batch_recon, image_batch)
                loss = F.mse_loss(image_batch_recon, image_batch)

                val_loss_avg[-1] += loss.item()
                num_wrong_val.append(torch.sum((image_batch != image_batch_recon)))
                n_val += len(image_batch)
                n_batches_val += 1

            percent_wrong_val = torch.sum(torch.tensor(num_wrong_val)) / (32 * 32 * n_val)
            print(f'Epoch [%d / %d] average wrong pixels [validation]: %f' % (epoch+1, num_epochs, percent_wrong_val))
            val_loss_avg[-1] /= n_batches_val
            print(f'Epoch [%d / %d] average reconstruction error [validation]: %f' % (epoch+1, num_epochs, val_loss_avg[-1]))

    return model, train_loss_avg, val_loss_avg

# TODO: Image autoencoder driver for simulation study
def run_sim_imgae(data, h=6, bsz_tri=32, bsz_val=32, lr=1e-3):
    tri_set, val_set = train_test_split(data, test_size=0.2, random_state=2023)
    tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=bsz_tri, shuffle=True)
    val_ldr = torch.utils.data.DataLoader(val_set, batch_size=bsz_val, shuffle=True)

    model = SimAutoEncoder(h).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)
    model, train_loss, val_loss = train_sim_autoencoder(model, 500, optimizer, scheduler, tri_ldr, val_ldr)
    return model


class Sine(nn.Module):
    def __init__(self, omega=30):  # Omega is a scaling factor (default from SIREN)
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)
# TODO: Spectra autoencoder for simulation study
class SimSpecEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, 5)
        self.conv2 = nn.Conv1d(128, 64, 5)
        self.pool1 = nn.MaxPool1d(2, return_indices=True)
        self.pool2 = nn.MaxPool1d(2, return_indices=True)
        self.fc0 = nn.Linear(64, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, hidden)
        # self.act = nn.ReLU()
        self.relu_act = nn.ReLU()
        self.sin_act = Sine(omega=30) # SIREN activation
    
    def forward(self, x):
        out = self.relu_act(self.conv1(x))
        out, indice1 = self.pool1(out)
        out = self.relu_act(self.conv2(out))
        out, indice2 = self.pool2(out)
        out = self.relu_act(self.fc0(out.squeeze()))
        out = self.relu_act(self.fc1(out))
        # out = torch.sin(self.fc1(out))
        out = self.fc2(out)
        return out, indice1, indice2
    
class SimSpecEncoder0(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, hidden)
    
    def forward(self, x):
        # print(x)
        x = x.squeeze()
        out = F.relu(self.fc(x))
        out = self.fc2(out)
        # print(out)
        return out, 1

class SimSpecDecoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(64, 128, 5)
        self.conv2 = nn.ConvTranspose1d(128, 1, 5)
        # self.conv1 = nn.ConvTranspose1d(16, 16, 5)
        # self.conv2 = nn.ConvTranspose1d(16, 1, 5)
        self.pool1 = nn.MaxUnpool1d(2)
        self.pool2 = nn.MaxUnpool1d(2)
        self.fc0 = nn.Linear(hidden, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
    
    def forward(self, x, indice1, indice2):
        out = F.relu(self.fc0(x))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.unsqueeze(-1)
        # print(out.shape)
        out = self.pool1(out, indice2)
        out = F.relu(self.conv1(out))
        out = self.pool2(out, indice1)
        out = self.conv2(out)
        return out

class SimSpecFNNEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, hidden)

    def forward(self, x):
        out = F.relu(self.fc1(x.squeeze()))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.fc6(out)
        return out
    
class SimSpecFNNDecoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.fc6(out).unsqueeze(1)
        return out
    
    
class SimSpecAutoEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.encoder = SimSpecEncoder(hidden)
        self.decoder = SimSpecDecoder(hidden)

    def forward(self, x):
        out, indice1, indice2 = self.encoder(x)
        out = self.decoder(out, indice1, indice2)
        return out
    
class SimSpecFNNAutoEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.encoder = SimSpecFNNEncoder(hidden)
        self.decoder = SimSpecFNNDecoder(hidden)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

def train_sim_spec_autoencoder(model, num_epochs, optimizer, scheduler, t_loader, v_loader, device=DEVICE):
    train_loss_avg, val_loss_avg = [], []
    print('Training Started...')
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
    
            # backpropagation
            loss.backward()
            optimizer.step()
            # logging
            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
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

    return model, train_loss_avg, val_loss_avg

def run_sim_specae(data, h=6, bsz_tri=64, bsz_val=32, lr=1e-3):
    tri_set, val_set = train_test_split(data, test_size=0.2, random_state=2023)
    tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=bsz_tri, shuffle=True)
    val_ldr = torch.utils.data.DataLoader(val_set, batch_size=bsz_val, shuffle=True)

    model = SimSpecAutoEncoder(h).to(DEVICE)
    # model = SimSpecFNNAutoEncoder(h)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)
    model, train_loss, val_loss = train_sim_spec_autoencoder(model, 500, optimizer, scheduler, tri_ldr, val_ldr)
    return model, tri_set, val_set
# TODO: DCCA model for simulation study
class SIM_DCCA(nn.Module):
    def __init__(self, h=6):
        super().__init__()
        self.h = h
        self.ct_encoder = SimEncoder(self.h).to(DEVICE)
        self.sp_encoder = SimSpecEncoder(self.h).to(DEVICE)

        # self.ct_encoder = SimEncoder0(self.h).to(DEVICE)
        # self.sp_encoder = SimSpecEncoder0(self.h).to(DEVICE)
        # self.sp_encoder = SimSpecFNNEncoder(self.h).to(DEVICE)

    def forward(self, img, spec):
        out1 = self.ct_encoder(img)[0]
        out2 = self.sp_encoder(spec)[0]
        # out2 = self.sp_encoder(spec)
        # print(out1.shape, out2.shape)
        return out1, out2

# New encoders
class ImageEncoder(nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 8 * 8, output_dim)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class SpectralEncoder(nn.Module):
    def __init__(self, input_dim=100, output_dim=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class SIM_DCCA_ROOT(nn.Module):
    def __init__(self, h=6):
        super().__init__()
        self.h = h
        self.ct_encoder = ImageEncoder(self.h).to(DEVICE)
        self.sp_encoder = SpectralEncoder(100, self.h).to(DEVICE)

    def forward(self, img, spec):
        out1 = self.ct_encoder(img)
        out2 = self.sp_encoder(spec).squeeze()
        # print(out1.shape, out2.shape)
        return out1, out2
    
    
def train_sim_DCCA(model, optimizer, criterion, tri_ldr, val_ldr, n_epochs=100, n_log=10, adv_data=None, beta=1):
    tri_iter_count, val_iter_count = 0, 0
    tri_log, val_log = [], []
    for epoch in tqdm(range(n_epochs)):
        model.train()
        tri_loss, val_loss = 0, 0
        n_tri, n_val = 0, 0
        for step, (x, y, _) in enumerate(tri_ldr):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # print(y.shape)

            # Typical Optimization
            optimizer.zero_grad()
            out1, out2 = model(x, y)
            if adv_data is None:
                loss, corr_vec = criterion(out1, out2)
            else:
                loss1, corr_vec = criterion(out1, out2)
                x_adv, y_adv = adv_data
                x_adv, y_adv = x_adv.to(DEVICE), y_adv.to(DEVICE)
                out3, out4 = model(x_adv, y_adv)
                loss_adv, _ = criterion(out3, out4)

                loss = loss1 - beta * loss_adv


            # def compute_l1_loss(w):
            #         return torch.sqrt((torch.abs(w)**2).sum())

            # l1_weight = 0.01
            # l1_parameters = []
            # for parameter in model.parameters():
            #     l1_parameters.append(parameter.view(-1))
            # l1 = l1_weight * compute_l1_loss(torch.cat(l1_parameters))

            # loss += l1



            loss.backward()
            optimizer.step()



            # Optimization for L-BGFS
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                out1, out2 = model(x, y)
                loss, corr_vec = criterion(out1, out2)


                def compute_l1_loss(w):
                    return torch.sqrt((torch.abs(w)**2).sum())

                l1_weight = 0.01
                l1_parameters = []
                for parameter in model.parameters():
                    l1_parameters.append(parameter.view(-1))
                l1 = l1_weight * compute_l1_loss(torch.cat(l1_parameters))

                loss += l1
                
                if loss.requires_grad:
                    loss.backward()
                return loss

            # optimizer.step(closure)
            # # Reevaluate
            # out1, out2 = model(x, y)
            # loss = -torch.abs(F.cosine_similarity(out1, out2).mean())
            # loss, corr_vec = criterion(out1, out2)
            # optimizer.step()

            # Log correlation statistics
            feature_name = [f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
            corr_vec = [corr.detach() for corr in corr_vec]

            tri_iter_count += 1
            if adv_data is None:
                tri_loss += (loss.detach() * len(y))
            else:
                tri_loss += (loss1.detach() * len(y))
            n_tri += len(y)

        tri_log.append(tri_loss / n_tri)
        if (epoch + 1) % n_log == 0:
            if adv_data is None:
                print(f"Epoch {epoch:<3} | Train Loss: {tri_loss / n_tri:.5f} | Corr: {-tri_loss / n_tri:.5f}")
            else:
                print(f"Epoch {epoch:<3} | Train Loss: {tri_loss / n_tri:.5f} | Corr: {-tri_loss / n_tri:.5f}")
            
        # Validation step
        # print('EVAL')
        with torch.no_grad():
            for step, (x, y, _) in enumerate(val_ldr):
                x, y = x.to(DEVICE), y.to(DEVICE)
                out1, out2 = model(x, y)
                loss, corr_vec = criterion(out1, out2)
                # loss = -torch.abs(F.cosine_similarity(out1, out2).mean())
                # Log correlation statistics
                feature_name = [f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
                corr_vec = [corr.detach() for corr in corr_vec]

                val_iter_count += 1
                val_loss += (loss.detach() * len(y))
                n_val += len(y)

            val_log.append(val_loss / n_val)
            if (epoch + 1) % n_log == 0:
                print(f"Epoch {epoch:<3} | Eval Loss: {val_loss / n_val:.5f} | Corr: {-val_loss / n_val:.5f}")

    return model, tri_log, val_log


# TODO: DCCA driver for simulation study
def run_dcca(data, h=6, bsz_tri=32, bsz_val=32, lr=1e-3, sp_ckpt_name=None, ct_ckpt_name=None, fix_ct=False, n_epoch=200, n_log=100, adv=False, adv_data=None, beta=1):
    tri_set, val_set = train_test_split(data, test_size=0.2, random_state=2023)
    print(len(tri_set), len(val_set))
    # print(tri_set[0])
    tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=bsz_tri, shuffle=True, drop_last=True)
    val_ldr = torch.utils.data.DataLoader(val_set, batch_size=bsz_val, shuffle=False, drop_last=True)

    model = SIM_DCCA(h).to(DEVICE)
    # model = SIM_DCCA_ROOT(h).to(DEVICE)

    if sp_ckpt_name is not None:
        sp_ae = SimSpecAutoEncoder(h).to(DEVICE)
        ckpt_path = os.path.join('checkpoint', 'simulation', f'{sp_ckpt_name}.pt')
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        sp_ae.load_state_dict(ckpt)
        model.sp_encoder = sp_ae.encoder
        print(f'Spectra encoder checkpoint {sp_ckpt_name} loaded successfully!')
    else:
        print('No Spectra encoder checkpoints loaded.')

    if ct_ckpt_name is not None:
        ct_ae = SimAutoEncoder(h).to(DEVICE)
        ckpt = torch.load(os.path.join('checkpoint', 'simulation', f'{ct_ckpt_name}.pt'), map_location=DEVICE)
        ct_ae.load_state_dict(ckpt)
        model.ct_encoder = ct_ae.encoder
        if fix_ct:
            for param in model.ct_encoder.parameters():
                param.requires_grad = False
        print(f'CT scan encoder checkpoint {ct_ckpt_name} loaded successfully!')
    else:
        print('No CT scan encoder checkpoints loaded.')

    criterion = CCA(h, False, DEVICE)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    if not adv:
        model, train_loss, val_loss = train_sim_DCCA(model, optimizer, criterion, tri_ldr, val_ldr, n_epoch, n_log)
    else:
        assert adv_data is not None
        model, train_loss, val_loss = train_sim_DCCA(model, optimizer, criterion, tri_ldr, val_ldr, n_epoch, n_log, adv_data=adv_data, beta=beta)
    return model, tri_set, val_set



# TODO: Finetune driver for simulation study
def get_pred(probs):
    return [1 if p >= 0.5 else 0 for p in probs]


class SimDecNet(nn.Module):
    def __init__(self, h, encoder=None, op=False, freeze=False):
        super().__init__()
        self.op = op
        self.h = h
        self.freeze = freeze
        self.encoder = encoder
        self.fc1 = nn.Linear(h, 16)
        self.fc2 = nn.Linear(16, 16)
        if self.op:
            self.fc3 = nn.Linear(16, 16)
            self.fc4 = nn.Linear(16, 1)
        else:
            self.fc4 = nn.Linear(16, 1)
        # Freeze parameters of the encoder
        if self.freeze and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.encoder(x)[0].squeeze()
        # feat = x
        out = self.fc1(feat)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        if self.op:
            out = self.fc3(out)
            out = F.relu(out)
        out = self.fc4(out)
        out = torch.sigmoid(out)
        # print(out)
        return out

class IdleDecNet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

def train_sim_decnet(model, optimizer, tri_ldr, val_ldr, n_epochs=200, n_log=1):

    for i in tqdm(range(n_epochs)):
        y_true, preds, n_correct, n_total = [], [], 0, 0
        for (x, y) in tri_ldr:
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
            print(f"Epoch {i+1} - Confusion Matrix: \n{conf_mat}")
            print(f"Epoch {i+1} - False Alarm: {conf_mat[0][1]/(conf_mat[0][0] + conf_mat[0][1]):.4f}")
            print(f"Epoch {i+1} - Mis-Detection: {conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1]):.4f}\n")
        # Evaluation
        with torch.no_grad():
            y_true, preds, n_correct, n_total = [], [], 0, 0
            for (x, y)in val_ldr:
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
            print(f"Epoch {i+1} - Confusion Matrix: \n{conf_mat}")
            print(f"Epoch {i+1} - False Alarm: {conf_mat[0][1]/(conf_mat[0][0] + conf_mat[0][1]):.4f}")
            print(f"Epoch {i+1} - Mis-Detection: {conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1]):.4f}\n")

    return model


def run_ft(data, h_ft, encoder, freeze=True, op=False, bsz_tri=32, bsz_val=32, lr=1e-3, n_epoch=500, n_log=10, ROSE=False):
    tri_set, val_set = train_test_split(data, test_size=0.2, random_state=2023)
    # print(tri_set[0])
    if ROSE:
        tri_cls = [label for _, label in tri_set]
        # Need to get weight for every image in the dataset
        class_count = Counter(np.array(tri_cls))
        print(class_count)
        class_weights = [len(tri_cls)/c for c in pd.Series(class_count).sort_index().values]
        # print(class_weights)
        # Can't iterate over class_count because dictionary is unordered
        print("Handling imbalanced data using weighted sampler: class weight = ", class_weights)

        sample_weights = [0] * len(tri_set)
        for idx, (image, label) in enumerate(tri_set):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight

        print("Sample weights counter: ", Counter(sample_weights))

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(tri_set), replacement=True)  
        tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=bsz_tri, sampler=sampler)
    else:
        tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=bsz_tri, shuffle=True)
    val_ldr = torch.utils.data.DataLoader(val_set, batch_size=bsz_val, shuffle=True)
    

    # model = IdleDecNet().to(DEVICE)
    model = SimDecNet(h=h_ft, encoder=encoder, op=op, freeze=freeze).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train_sim_decnet(model, optimizer, tri_ldr, val_ldr, n_epoch, n_log)
    return model, tri_set, val_set

def test_cls_on_dset(ldr, model):
    with torch.no_grad():
        y_true, preds, n_correct, n_total = [], [], 0, 0
        for (x, y) in ldr:
            # print(x.shape, y.shape)
            x, y = x.to(DEVICE), y.to(DEVICE)
            decisions = model(x)
            pred = get_pred(decisions)
            preds += pred
            y_true += list(y.cpu())
            n_correct += sum(np.array(pred) == np.array(y.cpu()))
            n_total += len(y)
        # Compute confusion matrix
        conf_mat = confusion_matrix(y_true, preds)
        print(f"Test ACC: {n_correct/n_total:.4f}")
        print(f"Test - Truth: {dict(Counter(np.array(y_true)))}")
        print(f"Test - Predictions: {dict(Counter(preds))}")
        print(f"Test - Confusion Matrix: \n{conf_mat}")
        print(f"Test - False Alarm: {conf_mat[0][1]/(conf_mat[0][0] + conf_mat[0][1]):.4f}")
        print(f"Test - Mis-Detection: {conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1]):.4f}")


# TODO: Oneshot driver for simulation study
class SimOneShot(nn.Module):
    def __init__(self, h=16, h_mlp=16, target='acc'):
        super().__init__()
        self.h = h
        self.h_mlp = h_mlp
        self.ct_encoder = SimEncoder(self.h)
        self.sp_encoder = SimSpecEncoder(self.h)
        self.target = target
        if target == 'stat':
            self.mlp = MLP(h, h_mlp) # three-layers MLP
        elif target == 'acc' or target == 'gdv':
            self.mlp = BinaryMLP(h, h_mlp)

    def forward(self, img, spec):
        out1 = self.ct_encoder(img)[0]
        out2 = self.sp_encoder(spec)[0]
        # Predictions
        # ON CCA features
        # cca = SKCCA(n_components=self.h)
        # cca.fit(out1.detach().numpy(), out2.detach().numpy())
        # cca_ct, cca_sp = cca.transform(out1.detach().numpy(), out2.detach().numpy())
        # out3 = self.mlp(torch.tensor(cca_ct, dtype=torch.float32))

        out3 = self.mlp(out1)
        return out1, out2, out3

def train_sim_oneshot(model, optimizer, criterion, tri_ldr, val_ldr, n_epochs=100, n_log=10, beta=1, target='acc'):
    tri_iter_count, val_iter_count = 0, 0
    tri_log, val_log = [], []
    for epoch in tqdm(range(n_epochs)):
        model.train()
        tri_loss, val_loss = 0, 0
        mse_log, dcca_log = 0, 0
        n_tri, n_val = 0, 0
        y_true, preds, n_correct, n_total = [], [], 0, 0 # for prediction
        
        # GDV specific
        if target == 'gdv':
            feat, label = [], []
            gdv_dcca_loss = []

        for step, (ct, sp, y, s) in enumerate(tri_ldr):
            ct, sp, s, y = ct.to(DEVICE), sp.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
            # print(ct.shape, sp.shape, s.shape, y.shape)

            out1, out2, out3 = model(ct, sp)
            dcca_loss, corr_vec = criterion(out1, out2)

            if target == 'stat':
                mse = F.mse_loss(s, out3.squeeze())

            elif target == 'acc':
                mse = F.binary_cross_entropy(out3, y.unsqueeze(-1).to(torch.float32))
                pred = get_pred(out3.cpu())
                preds += pred
                y_true += list(y.cpu())
                n_correct += sum(np.array(pred) == np.array(y.cpu()))
                n_total += len(y)

            elif target == 'gdv':
                feat.append(out1)
                label.append(y)
                gdv_dcca_loss.append(dcca_loss)

            # Perform update for non-GDV cases
            if target != 'gdv':
                loss = dcca_loss + beta * mse
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Log correlation statistics
                feature_name = [f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
                corr_vec = [corr.detach() for corr in corr_vec]

                tri_iter_count += 1
                tri_loss += (loss.detach() * len(y))
                mse_log += (mse.detach() * len(y))
                dcca_log += (dcca_loss.detach() * len(y))
                n_tri += len(y)

        if target == 'acc':
            # Compute confusion matrix
            conf_mat = confusion_matrix(y_true, preds)

        # For GDV case, perform update at the end of each epoch
        elif target == 'gdv':
            feat = torch.cat(feat)
            label = torch.cat(label).squeeze()
            
            gdv = cal_gdv(feat, label)
            dcca_loss = torch.mean(dcca_loss)
            loss = dcca_loss + beta * gdv
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        # For non-GDV cases, compute relevant statistics
        if target != 'gdv':
            tri_log.append(tri_loss / n_tri)

        # Logging
        if (epoch + 1) % n_log == 0:
            if target != 'gdv':
                print(f"Epoch {epoch:<3} | Train Loss: {tri_loss / n_tri:.5f} | Corr: {-dcca_log / n_tri:.5f} | {NAMES[target]}: {mse_log / n_tri:.5f}")
            else:
                print(f"Epoch {epoch:<3} | Train Loss: {loss.detach():.5f} | Corr: {-dcca_loss.detach():.5f} | {NAMES[target]}: {gdv.detach():.5f}")

            if target == 'acc':
                print(f"Epoch {epoch} - Training ACC: {n_correct/n_total:.4f}")
                print(f"Epoch {epoch} - Truth: {dict(Counter(np.array(y_true)))}")
                print(f"Epoch {epoch} - Training Predictions: {dict(Counter(preds))}")
                print(f"Epoch {epoch} - Confusion Matrix: \n{conf_mat}\n")

        # Validation step
        with torch.no_grad():
            mse_log, dcca_log = 0, 0
            y_true, preds, n_correct, n_total = [], [], 0, 0 # for prediction

            if target == 'gdv':
                feat, label = [], []
                gdv_dcca_loss = []

            for step, (ct, sp, y, s) in enumerate(val_ldr):
                ct, sp, s, y = ct.to(DEVICE), sp.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
                out1, out2, out3 = model(ct, sp)
                dcca_loss, corr_vec = criterion(out1, out2)

                if target == 'stat':
                    mse = F.mse_loss(s, out3.squeeze())

                elif target == 'acc':
                    mse = F.binary_cross_entropy(out3, y.unsqueeze(-1).to(torch.float32))
                    pred = get_pred(out3)
                    preds += pred
                    y_true += list(y.cpu())
                    n_correct += sum(np.array(pred) == np.array(y.cpu()))
                    n_total += len(y)

                elif target == 'gdv':
                    feat.append(out1)
                    label.append(y)
                    gdv_dcca_loss.append(dcca_loss)
                    
                if target != 'gdv':
                    loss = dcca_loss + beta * mse

                    # Log correlation statistics
                    feature_name = [f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
                    corr_vec = [corr.detach() for corr in corr_vec]

                    val_iter_count += 1
                    val_loss += (loss.detach() * len(y))
                    mse_log += (mse.detach() * len(y))
                    dcca_log += (dcca_loss.detach() * len(y))
                    n_val += len(y)
        
            if target == 'acc':
                # Compute confusion matrix
                conf_mat = confusion_matrix(y_true, preds)

            elif target == 'gdv':
                feat = torch.cat(feat)
                label = torch.cat(label).squeeze()
                gdv = cal_gdv(feat, label)
                dcca_loss = torch.mean(dcca_loss)
                loss = dcca_loss + beta * gdv

            if target != 'gdv':
                val_log.append(val_loss / n_val)

        if (epoch + 1) % n_log == 0:
            if target != 'gdv':
                print(f"Epoch {epoch:<3} | Eval Loss: {val_loss / n_tri:.5f} | Corr: {-dcca_log / n_val:.5f} | {NAMES[target]}: {mse_log / n_val:.5f}")
            else:
                print(f"Epoch {epoch:<3} | Eval Loss: {loss.detach():.5f} | Corr: {-dcca_loss.detach():.5f} | {NAMES[target]}: {gdv.detach():.5f}")

            if target == 'acc':
                print(f"Epoch {epoch} - Eval ACC: {n_correct/n_total:.4f}")
                print(f"Epoch {epoch} - Truth: {dict(Counter(np.array(y_true)))}")
                print(f"Epoch {epoch} - Training Predictions: {dict(Counter(preds))}")
                print(f"Epoch {epoch} - Confusion Matrix: \n{conf_mat}\n")
            

    return model, tri_log, val_log


def run_sim_oneshot(data, h=6, h_mlp=6, target='acc', beta=1, bsz_tri=32, bsz_val=32, lr=1e-3, sp_ckpt_name=None, ct_ckpt_name=None, n_epoch=200, n_log=10):
    tri_set, val_set = train_test_split(data, test_size=0.2, random_state=2023)
    tri_ldr = torch.utils.data.DataLoader(tri_set, batch_size=bsz_tri, shuffle=True)
    val_ldr = torch.utils.data.DataLoader(val_set, batch_size=bsz_val, shuffle=True)

    model = SimOneShot(h=h, h_mlp=h_mlp, target=target).to(DEVICE) 
    # Criterion declaration
    criterion = CCA(h, False, DEVICE)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    model, train_loss, val_loss, = train_sim_oneshot(model, optimizer, criterion, tri_ldr, val_ldr, n_epoch, n_log, beta, target)
    return model, tri_set, val_set

def check_correlation(U, V):
    U = torch.tensor(U, dtype=torch.float32)
    V = torch.tensor(V, dtype=torch.float32)
    criterion = CCA(6, False, DEVICE)
    loss, _ = criterion(U, V)
    print(-loss)


##################################

# Naive function to generate uniform random image
def generate_img(n, theta):
    data = []
    for _ in range(n):
        P = np.random.rand(32 * 32).reshape((32, 32))
        mask = P >= theta
        I = np.zeros((32, 32))
        I[mask] = 1
        data.append(I)
    data = torch.tensor(data, dtype=torch.float32)
    return data

def generate_circle(img_size, num_circles):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    for _ in range(num_circles):
        radius = np.random.randint(img_size // 15, img_size // 5)
        # radius = np.random.randint(img_size // 10, img_size // 5)
        # radius = np.random.randint(1, 3)
        center = (np.random.randint(radius, img_size - radius), np.random.randint(radius, img_size - radius))
        cv2.circle(img, center, radius, 1, -1)
    return img

def generate_cracks(img_size, num_cracks):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    for _ in range(num_cracks):
        start = (np.random.randint(img_size), np.random.randint(img_size))
        end = (np.random.randint(img_size), np.random.randint(img_size))
        # thickness = np.random.randint(1, 3)
        thickness = np.random.randint(1, 2)
        cv2.line(img, start, end, 1, thickness)
    return img

def generate_imperfect_circle(img_size, num_circles):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    for _ in range(num_circles):
        radius = np.random.randint(img_size // 10, img_size // 5)
        center = (np.random.randint(radius, img_size - radius), np.random.randint(radius, img_size - radius))
        
        # Create an array to hold the points of the imperfect circle
        points = []
        
        for angle in range(0, 360):
            angle_rad = np.deg2rad(angle)
            perturbation = np.random.uniform(-radius * 0.3, radius * 0.3)  # Random perturbation
            r = radius + perturbation
            x = int(center[0] + r * np.cos(angle_rad))
            y = int(center[1] + r * np.sin(angle_rad))
            points.append((x, y))
        
        # Draw the imperfect circle using the points
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [points], 1)
    
    return img

def generate_sphere_cross_sections(radius, num_slices, image_size):
    # Create a grid of points
    x = np.linspace(-radius, radius, image_size)
    y = np.linspace(-radius, radius, image_size)
    xv, yv = np.meshgrid(x, y)
    
    # Calculate the z coordinates for each slice
    z_values = np.linspace(-radius, radius, num_slices)
    imgs = []
    for i, z in enumerate(z_values):
        # Equation of a sphere (x^2 + y^2 + z^2 = r^2)
        slice_image = np.sqrt(radius**2 - xv**2 - yv**2 - z**2)
        
        # Create a binary image: 1 inside the sphere, 0 outside
        binary_image = np.where(slice_image >= 0, 1, 0)
        current_radius = np.sqrt(radius**2 - z**2) if radius**2 - z**2 >= 0 else 0
        
        # Create a binary image
        binary_image = np.zeros((image_size, image_size), dtype=np.uint8)
        
        # Compute the coordinates of the disk representing the cross-section
        rr, cc = disk((image_size // 2, image_size // 2), current_radius, shape=(image_size, image_size))
        
        # Set the pixels inside the disk to 1
        binary_image[rr, cc] = 1
        imgs.append(binary_image)
        # Save the image
        # plt.imshow(binary_image, cmap='gray')
        # plt.axis('off')
        # plt.savefig(f'sphere_slice_{i:03d}.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
    return np.array(imgs)

def generate_binary_image(img_size, num_circles, num_cracks):
    circle_img = generate_circle(img_size, num_circles)
    cracks_img = generate_cracks(img_size, num_cracks)
    binary_img = np.maximum(circle_img, cracks_img)
    return binary_img

def generate_circle_img(img_size, num_circles):
    return generate_circle(img_size, num_circles)

def generate_cracks_img(img_size, num_cracks):
    return generate_cracks(img_size, num_cracks)

def generate_imperfect_circle_img(img_size, num_circles):
    return generate_imperfect_circle(img_size, num_circles)

def check_AE(datum, model):
    fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns
    axs[0].imshow(datum.squeeze().numpy())
    axs[0].set_title('OG')
    axs[1].imshow(model(datum.unsqueeze(0)).squeeze().detach().numpy())
    axs[1].set_title('RE')
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()
    plt.close()

def save(ctscans):
    # Labeling
    stats = torch.sum(ctscans, dim=(1, 2, 3))
    print(stats.shape)
    # plt.plot(stats)
    print(torch.mean(stats))
    T = 50
    Y = stats >= T
    print(Y.sum())
    # Save
    data_dir = os.path.join('Data', 'simulation')
    os.makedirs(data_dir, exist_ok=True)
    torch.save(ctscans, os.path.join(data_dir, 'ct-raw.pt'))
    torch.save(ctscans[0:1000], os.path.join(data_dir, 'ct.pt'))
    torch.save(stats[0:1000], os.path.join(data_dir, 'stats.pt'))
    torch.save(Y[0:1000], os.path.join(data_dir, 'label.pt'))
    torch.save(ctscans[1000:2000], os.path.join(data_dir, 'ct-test.pt'))
    torch.save(stats[1000:2000], os.path.join(data_dir, 'stats-test.pt'))
    torch.save(Y[1000:2000], os.path.join(data_dir, 'label-test.pt'))

def generate_sinusoidal_waves(V):
    spectra = []
    for x in np.linspace(0, 2, 16):
        gx = torch.sin(V[:,0] * V[:,1] * x) + torch.sin(V[:,2] * V[:,3] * x)**2 + torch.cos(V[:,4] * V[:,5] * x)
        spectra.append(gx)
    spectra = torch.column_stack(spectra)
    return spectra

# def generate_simple_sinusoidal_waves(V):
#     spectra = []
#     M = torch.normal(0, 0.01, size=(6, 16), dtype=torch.float32)
#     gx = torch.sin(torch.tensor(np.linspace(0, 2, 16)))
#     spectra = torch.matmul(V, M) + gx
#     return spectra

def generate_simple_sinusoidal_waves(V):
    spectra = []
    MW = torch.normal(1, 1, size=(6, 1), dtype=torch.float32)
    MA = torch.normal(1, 1, size=(6, 1), dtype=torch.float32)
    MB = torch.normal(1, 1, size=(6, 1), dtype=torch.float32)
    W, A, B = torch.matmul(V, MW), torch.matmul(V, MA), torch.matmul(V, MB)
    # 2 * pi approx 6
    for x in np.linspace(0, 2 * np.pi, 16):
        spectra.append(A*torch.sin(W * x + B))
    spectra = torch.column_stack(spectra)
    return spectra, MW, MA, MB

def generate_moderate_sinusoidal_waves(V):
    spectra = []
    M = torch.normal(0, 1, size=(6, 16), dtype=torch.float32)
    gx = torch.sin(torch.tensor(np.linspace(0, 4, 16))) 
    spectra = torch.matmul(V, M) + gx
    return spectra

def generate_waves_using_AE(V):
    model = SimSpecAutoEncoder(6)
    ckpt = torch.load(os.path.join('checkpoint', 'simulation', 'sp_ae.pt'))
    model.load_state_dict(ckpt)
    with torch.no_grad():
        spectra = model.decoder(V) # this needs to be fixed
    return spectra

def generate_simple_waves(V):
    M = torch.normal(1, 1, size=(6, 16), dtype=torch.float32)
    spectra = torch.matmul(V, M)
    return spectra

def generate_moderate_waves(V):
    M = torch.normal(0, 0.01, size=(6, 16), dtype=torch.float32)
    spectra = torch.matmul(V, M)
    print(spectra.shape)
    print(torch.mean(spectra, dim=1).shape)
    spectra = (spectra - torch.mean(spectra, dim=1, keepdim=True)) / torch.std(spectra, dim=1, keepdim=True)
    spectra = torch.sin(spectra)
    return spectra

def generate_2fc_waves(V):
    M = torch.normal(1, 1, size=(6, 32), dtype=torch.float32)
    spectra = torch.matmul(V, M)
    spectra = F.relu(spectra)
    M2 = torch.normal(0, 1, size=(32, 16), dtype=torch.float32)
    spectra = torch.log(torch.abs(1 / (1 - torch.matmul(spectra, M2))))
    # spectra = torch.log(torch.abs(torch.matmul(spectra, M2) / (1 - torch.matmul(spectra, M2))))
    return spectra, M, M2

def generate_3fc_waves(V):
    M = torch.normal(0, 0.01, size=(6, 32), dtype=torch.float32)
    spectra = torch.matmul(V, M)
    spectra = F.relu(spectra)
    M2 = torch.normal(0, 0.01, size=(32, 32), dtype=torch.float32)
    spectra = torch.log(torch.abs(torch.matmul(spectra, M2)) + 1)

    # spectra = F.relu(spectra)
    M3 = torch.normal(0, 0.01, size=(32, 16), dtype=torch.float32)
    spectra = F.relu(torch.matmul(spectra, M3))
    return spectra

def generate_nfc_waves(V, n):
    M = torch.normal(0, 1, size=(6, 32), dtype=torch.float32)
    spectra = torch.matmul(V, M)
    spectra = F.relu(spectra)**0.5
    M = torch.normal(0, 1, size=(32, 16), dtype=torch.float32)
    spectra = torch.matmul(spectra, M)
    return spectra

def tsne_separability(data, Y):
    # Check separability
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(data)
    # print(X_embedded.shape)
    colors = ['green' if l==1 else 'red' for l in Y]
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5)

def visualize_features(CT, SP, Y, stats, model):
    fig, ax = plt.subplots(2, 2)  # 1 row, 2 columns
    dataset = list(zip(SP, Y, stats))
    feat, label, stats = get_features_sim(dataset, model.sp_encoder)
    # Visualize spectra feature distribution using 2D PCA
    pca = PCA(n_components=2)
    pc = pca.fit_transform(feat)
    df = pd.DataFrame(data=pc, columns = ['PC1', 'PC2'])
    df['label'] = label

    ax[0][0].set_xlabel('PC1', fontsize = 12)
    ax[0][0].set_ylabel('PC2', fontsize = 12)
    ax[0][0].set_title('PCA - SP', fontsize = 13)

    targets = [0, 1]
    colors = ['r', 'g' ]
    for target, color in zip(targets,colors):
        indicesToKeep = df['label'] == target
        ax[0][0].scatter(df.loc[indicesToKeep, 'PC1'], df.loc[indicesToKeep, 'PC2'],c=color, s=50, alpha=0.5)
    ax[0][0].legend(targets)
    ax[0][0].grid()

    # TSNE
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(feat)
    # print(X_embedded.shape)
    colors = ['green' if l==1 else 'red' for l in label]
    ax[0][1].scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5)
    ax[0][1].set_title('TSNE - SP', fontsize = 13)   
    feat = torch.tensor(feat)
    print('SP feature GDV:', cal_gdv(feat, Y, D=6))

    # CT
    dataset = list(zip(CT, SP, Y, stats))
    feat, label, stats = get_features_ft_ct(dataset, model.ct_encoder)
    # Visualize CT scan feature distribution using 2D PCA
    pca = PCA(n_components=2)
    pc = pca.fit_transform(feat)
    df = pd.DataFrame(data=pc, columns = ['PC1', 'PC2'])
    df['label'] = label
    ax[1][0].set_xlabel('PC1', fontsize = 12)
    ax[1][0].set_ylabel('PC2', fontsize = 12)
    ax[1][0].set_title('PCA - CT', fontsize = 13)

    targets = [0, 1]
    colors = ['r', 'g' ]
    for target, color in zip(targets,colors):
        indicesToKeep = df['label'] == target
        ax[1][0].scatter(df.loc[indicesToKeep, 'PC1'], df.loc[indicesToKeep, 'PC2'],c=color, s=50, alpha=0.5)
    ax[1][0].legend(targets)
    ax[1][0].grid()

    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(feat)
    # print(X_embedded.shape)
    colors = ['green' if l==1 else 'red' for l in label]
    ax[1][1].scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5)
    ax[1][1].set_title('TSNE - CT', fontsize = 13)
    feat = torch.tensor(feat)
    print("CT feature GDV:", cal_gdv(feat, Y, D=6))

    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()


def evaluate_ft_acc_test(model, test_name, data_dir=os.path.join('Data', 'simulation')):
    SP_TEST = torch.load(os.path.join(data_dir, f'{test_name}-test.pt'))
    SP_TEST = SP_TEST.unsqueeze(1)
    SP_TEST = torch.tensor(SP_TEST, dtype=torch.float32)
    Y_TEST = torch.load(os.path.join(data_dir,  'label-test.pt'))
    # Evaluate
    out = model(SP_TEST)
    pred = get_pred(out)
    acc = sum(np.array(pred) == np.array(torch.tensor(Y_TEST, dtype=torch.uint8)))
    print('Test accuracy: ', acc / len(out))



# Used for hypo test experiments


def compute_cca(u, v, h):
    criterion = CCA(h, False, 'cpu')
    if h == 1:
       corr_loss, corr_vec = criterion(torch.tensor(u).unsqueeze(1), torch.tensor(v).unsqueeze(1))
       return -corr_loss, corr_vec[0]
    else:
       corr_loss, corr_vec = criterion(torch.tensor(u), torch.tensor(v))
       return -corr_loss, corr_vec

def randomized_corr(u, v, n=10, m=10, h=1, merge=False):
    assert len(u) == len(v)
    corrs = []
    for _ in range(m):
        idx1, idx2 = np.random.choice(len(u), n, False), np.random.choice(len(u), n, False)
        u_cca, v_cca = find_cca_feature(u[idx1], v[idx1], h, max_iter=5000, verbose=False)
        if h == 1:
            corrs.append(scipy.stats.pearsonr(u_cca[:,0], v_cca[:,0])[0])
        else:
            corrs.append(np.array([scipy.stats.pearsonr(u_cca[:,i], v_cca[:,i])[0] for i in range(h)]))
            
    if h != 1 and merge:
        return np.sum(corrs, axis=1)
    else:
        return np.array(corrs)

# def randomized_corr_customized(u, v, n=10, m=10, h=1, merge=False):
#     assert len(u) == len(v)
#     corrs = []
#     for _ in range(m):
#         idx1, idx2 = np.random.choice(len(u), n, False), np.random.choice(len(u), n, False)
#         corr, corr_vec = compute_cca(u[idx1], v[idx1], h)
#         if h == 1 or merge:
#             corrs.append(corr)
#         else:
#             corrs.append(np.array(corr_vec))
#     return np.array(corrs)

def randomized_corr_customized(u, v, n=10, m=10, h=1, merge=False):
    assert len(u) == len(v)
    corrs = []
    for _ in range(m):
        idx1, idx2 = np.random.choice(len(u), n, False), np.random.choice(len(u), n, False)
        corr, corr_vec = compute_cca(u[idx1], v[idx1], h)
        if h == 1 or merge:
            corrs.append(corr)
        else:
            corrs.append(np.array(corr_vec))
    return np.array(corrs)

def get_threshold(bootstrap_samples, alpha, h, merge=False):
    if h == 1 or merge:
        return np.quantile(bootstrap_samples, alpha)
    else:
        return [np.quantile(bootstrap_samples[:,i], alpha) for i in range(6)]
    
def distance(incoming_sp, bank, dims=(1, 2)):
    dist = torch.sum(torch.abs(bank - incoming_sp), dim=dims)
    return torch.argmin(dist), torch.min(dist)

def get_min_dist_samples_idx(window_sp, bank, dims=(1,2)):
    sample_idx = []
    for sp in window_sp:
        idx, dist = distance(sp, bank, dims=dims)
        sample_idx.append(idx)
    return sample_idx

def generate_simulation_test_samples(sp, v, y, n_samples, n_test):
    sps, vs, ys = [], [], []
    for _ in range(n_test):
        idx = np.random.choice(len(sp), n_samples, False)
        # Save
        sps.append(sp[idx])
        vs.append(v[idx])
        ys.append(y[idx])
    # print(len(sps), len(vs), len(ys))
    return sps, vs, ys


def test_single_sample(window_sp, window_v, sp_bank, u_bank, v_bank=None, h=1, merge=False):
    example_matching_sample_idx = get_min_dist_samples_idx(window_sp, sp_bank)
    if v_bank is not None: # Feature level matching
        example_matching_sample_idx = get_min_dist_samples_idx(torch.tensor(window_v), torch.tensor(v_bank), dims=1)
    u_matching = u_bank[example_matching_sample_idx]
    # Return correlations
    if h == 1 or merge:
        # u_cca, v_cca = find_cca_feature(u_matching, window_v, h, max_iter=5000, verbose=False)
        # return scipy.stats.pearsonr(u_cca[:,0], v_cca[:,0])[0]
        # Customized
        return compute_cca(u_matching, window_v, h)[0]
    else:
        # TODO: change this later
        # return sum([scipy.stats.pearsonr(u_cca[:,i], v_cca[:,i])[0] for i in range(h)])
        # return np.array([scipy.stats.pearsonr(u_cca[:,i], v_cca[:,i])[0] for i in range(h)])
        return np.array(compute_cca(u_matching, window_v, h)[1])

def test(data, sp_bank, u_bank, n_samples=25, n_test=15, h=1, verbose=False, merge=False, v_bank=None):
    corrs = []
    print("Test set size: ", len(data))
    for window_sp, window_v, y_sp in data:
        if verbose:
            print(Counter(np.array(y_sp)))
        corrs.append(test_single_sample(window_sp, window_v, sp_bank, u_bank, v_bank, h, merge))
    return np.array(corrs)

def get_acc(corrs, T, label, h=1, merge=False):
    # label: 0 means normal, 1 means abnormal
    preds = []
    if h == 1 or merge:
        for rho, y in zip(corrs, label):
            preds.append((rho >= T) if not y else (rho < T))
        return sum(preds) / len(corrs)
    else:
        for i in range(h):
            pred = []
            for rho, y in zip(corrs[:,i], label):
                pred.append((rho >= T[i]) if not y else (rho < T[i]))
            preds.append(pred)
        preds = np.all(preds, axis=0)
        return sum(preds) / len(corrs)
    
def find_cca_feature(u, v, n=6, max_iter=2500, verbose=False):
    cca = SKCCA(n_components=n, max_iter=max_iter)
    if n == 1:
        u, v = u.reshape(-1, 1), v.reshape(-1,1)
    cca.fit(u, v)
    X_c, Y_c = cca.transform(u, v)
    score = cca.score(u, v)
    # print(score)
    # print(X_c.shape, Y_c.shape)
    for i in range(n):
        if verbose:
            print(scipy.stats.pearsonr(X_c[:,i], Y_c[:,i]))
    return X_c, Y_c

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def extract_uv(dset, ct_encoder, sp_encoder):
    with torch.no_grad():
        cts = torch.cat([x for x, y, l in dset], dim=0).unsqueeze(1)
        sps = torch.cat([y for x, y, l in dset], dim=0).unsqueeze(1)
        y = torch.tensor([y for _, _, y in dset])
        print(cts.shape, sps.shape)
        u = ct_encoder(cts)
        v = sp_encoder(sps)
        if type(u) == tuple:
            u = u[0]
            v = v[0]
        # u = ct_encoder(cts)
        # v = sp_encoder(sps)
        print(u.shape, v.shape)
        u, v = np.array(u.squeeze()), np.array(v.squeeze())
        print(u.shape, v.shape)
        return u, v, cts, sps, y
    
def get_test_results(data, sp_bank, u_bank, n_test, h, threshold, label, merge=False, v_bank=None):
    corrs = test(data, sp_bank, u_bank, n_test=n_test, h=h, merge=merge, v_bank=v_bank)
    print("Mean correlations: ", np.round(np.mean(corrs, axis=0), 3))
    acc = get_acc(corrs, threshold, label, h, merge=merge)
    print("acc: ", acc)
    return corrs, acc

def get_acc_cls_baseline(data, model):
    preds = []
    for (sp, _, y) in tqdm(data):
        sp = sp.to(DEVICE)
        decisions = model(sp)
        pred = np.array(get_pred(decisions.detach().cpu()))
        preds.append(pred)
    return preds


# SIMCLR

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

class SimCLR(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.encoder = SimSpecEncoder(h)
        self.projector = ProjectionHead(in_dim=h, hidden_dim=64, out_dim=h)

    def forward(self, x):
        # print(x.shape)
        h, _, _ = self.encoder(x)
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
    """
    x: Tensor of shape [B, 1, 16]
    """

    B, C, D = x.shape

    # 1. Additive Gaussian noise
    noise = torch.randn_like(x) * noise_std

    # 2. Feature scaling (per sample)
    scale = 1 + scale_std * torch.randn(B, 1, 1, device=x.device)

    # 3. Feature masking
    mask = (torch.rand(B, C, D, device=x.device) > drop_prob).float()

    x_aug = x * scale
    x_aug = x_aug + noise
    x_aug = x_aug * mask

    return x_aug

def train_simclr(data, h, epochs=100, batch_size=256, lr=1e-3):
    tri_set, val_set = train_test_split(data, test_size=0.2, random_state=2023)
    print(len(tri_set), len(val_set))
    loader = torch.utils.data.DataLoader(tri_set, batch_size=batch_size, shuffle=True)
    model = SimCLR(h).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        total_loss = 0

        for batch in loader:

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


# CLIP-SIMCLR
class CLIPSimCLR(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.sp_encoder = SimSpecEncoder(h)
        self.ct_encoder = SimEncoder(h)

    def forward(self, ct, sp):
        # print(x.shape)
        h_ct, _, _ = self.ct_encoder(ct)
        h_sp, _, _ = self.sp_encoder(sp)

        h_ct = F.normalize(h_ct, dim=1)
        h_sp = F.normalize(h_sp, dim=1)

        return h_ct, h_sp

def cross_modal_nt_xent_loss(z1, z2, temperature=0.5):
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
    
def train_clip_simclr(data, h, epochs=100, batch_size=256, lr=1e-3):
    tri_set, val_set = train_test_split(data, test_size=0.2, random_state=2023)
    print(len(tri_set), len(val_set))
    loader = torch.utils.data.DataLoader(tri_set, batch_size=batch_size, shuffle=True)
    model = CLIPSimCLR(h).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            hx, hy = model(x, y)

            loss = cross_modal_nt_xent_loss(hx, hy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss/len(loader):.4f}")

    return model

if __name__ == '__main__':
    print("Simulation experiment test")
    # model = SimAutoEncoder(6)

    # spec_encoder = SimSpecEncoder(6)
    # sp = torch.ones((8, 1, 16))
    # print(sp.dtype)
    # out, idx1, idx2 = spec_encoder(sp)
    # print(out.shape)
    # spec_decoder = SimSpecDecoder(6)
    # sp = spec_decoder(out, idx1, idx2)
    # print(sp.shape)

    # Run
    data_dir = os.path.join('Data', 'simulation')
    CT = torch.load(os.path.join(data_dir,  'ct.pt'))
    # SP = torch.load(os.path.join(data_dir,  'sp.pt'))
    # SP = torch.load(os.path.join(data_dir,  'sp_mix.pt'))
    # SP = torch.load(os.path.join(data_dir,  'sp_2fc.pt'))
    SP = torch.load(os.path.join(data_dir,  'sp_linear.pt'))
    SP = SP.unsqueeze(1)
    SP = torch.tensor(SP, dtype=torch.float32)
    print(SP.shape)
    stats = torch.load(os.path.join(data_dir, 'stats.pt'))
    Y = torch.load(os.path.join(data_dir,  'label.pt'))


    # GPU training from cmd line
    # DCCA
    data = list(zip(CT, SP))
    # model = run_dcca(data, h=6, bsz_tri=800, bsz_val=64, lr=1e-3, ct_ckpt_name='ct_ae', sp_ckpt_name='sp_ae', fix_ct=True)  
    model = run_dcca(data, h=6, bsz_tri=800, bsz_val=200, lr=1e-2, n_epoch=1000)

    # Oneshot
    # data = list(zip(CT, SP, torch.tensor(Y, dtype=torch.uint8), stats))
    # model = run_sim_oneshot(data, bsz_tri=256, bsz_val=200, n_epoch=100, target='gdv')
    # dataset = list(zip(SP, torch.tensor(Y, dtype=torch.uint8), stats))
    # feat, label, stats = get_features_sim(dataset, model.sp_encoder)
    # vis_pca_feat_ft(feat, label, stats)
    # feat = torch.tensor(feat)
    # print(cal_gdv(feat, Y, D=6))