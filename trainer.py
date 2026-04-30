import torch.nn as nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from objective.loss import CCA
from models.dcca import *
from models.window_lstm import *
from models.model import *
from dataset import *
from icecream import ic
from const import *

class TRAINER():
    def __init__(self, tag, save_every_n = 10, n_epochs=100):
        self.n_epochs = n_epochs
        self.save_every_n = save_every_n
        self.tag = tag
        # Logging and Saving
        self.save_path = os.path.join(CKPT_SAVE_DIR, 'DCCA')
        self.writer = SummaryWriter(os.path.join(CKPT_LOG_DIR, 'DCCA', f'{tag}'))
    
    def train(self, model, optimizer, criterion, tri_ldr, val_ldr):

        tri_iter_count, val_iter_count = 0, 0
        for epoch in tqdm(range(self.n_epochs)):
            model.train()
            for step, (x, y) in enumerate(tri_ldr):
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out1, out2 = model(x, y)
                # print(out1)
                # print(out2)
                loss, corr_vec = criterion(out1, out2)
                loss.backward()
                optimizer.step()

                self.writer.add_scalar("Loss/Train", loss.detach(), tri_iter_count)
                self.writer.add_scalar("Corr/Train", -loss.detach(), tri_iter_count)
                # Log correlation statistics
                feature_name = [
                    f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
                corr_vec = [corr.detach() for corr in corr_vec]
                self.writer.add_scalars(
                    f"Correlation/Train", dict(zip(feature_name, corr_vec)), tri_iter_count)

                tri_iter_count += 1
                # print(
                #     f"Epoch {epoch:<3} | Step {step:<4} | Train Loss: {loss:.5f} | Corr: {-loss:.5f}")
            print(f"Epoch {epoch:<3} | Train Loss: {loss:.5f} | Corr: {-loss:.5f}")
                
            # Validation step
            with torch.no_grad():
                for step, (x, y) in enumerate(val_ldr):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out1, out2 = model(x, y)
                    loss, corr_vec = criterion(out1, out2)
                    # Log validation statistics
                    self.writer.add_scalar(
                        "Loss/Eval", loss.detach(), val_iter_count)
                    self.writer.add_scalar(
                        "Corr/Eval", -loss.detach(), val_iter_count)
                    # Log correlation statistics
                    feature_name = [
                        f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
                    corr_vec = [corr.detach() for corr in corr_vec]
                    self.writer.add_scalars(
                        f"Correlation/Eval", dict(zip(feature_name, corr_vec)), val_iter_count)
                    val_iter_count += 1
                    print(
                        f"Epoch {epoch:<3} | Step {step:<4} | Eval Loss: {loss:.5f} | | Corr: {-loss:.5f}")
                    
            if (epoch + 1) % self.save_every_n == 0:
                torch.save(model.state_dict(), os.path.join(self.save_path, f'DCCA-{self.tag}-[{epoch + 1}].pt'))



def train(model, train, val, tri_bsz, val_bsz, optimizer,
          max_epoch, writer):
    ic("Training begins")
    # TODO: wrap CCA loss in DCCA model
    criterion = CCA(4, False, DEVICE)
    # criterion = CCA(8, True, DEVICE)
    model.train()

    tri_iter_count, val_iter_count = 0, 0
    for epoch in range(max_epoch):
        tri_dset = sample_batch(train, tri_bsz)
        val_dset = sample_batch(val, val_bsz)
        for step, batch in enumerate(tri_dset):
            # Zero Existing Gradients
            optimizer.zero_grad()
            b1, b2 = [], []
            for (img, spec) in batch:
                img = img.unsqueeze(0).float().to(DEVICE)
                spec = spec.unsqueeze(0).float().to(DEVICE)
                # WindowDCCA Forward Pass
                out1, out2 = model(img, spec)
                # ic(out1)
                # ic(out2)
                b1.append(out1)
                b2.append(out2)
                # print(out1)
                # print(out2)
                # ic(out1.shape)
            b1 = torch.cat(b1, dim=0)
            b2 = torch.cat(b2, dim=0)
            # ic(b1)
            # ic(b2)
            # ic(b1.shape)
            loss, corr_vec = criterion(b1, b2)
            # print(corr_vec)
            # ic(loss)
            # Backward Pass
            loss.backward()
            # ic(loss)
            # Gradient Update
            optimizer.step()

            # Log statistics
            writer.add_scalar("Loss/Train", loss.detach(), tri_iter_count)
            writer.add_scalar("Corr/Train", -loss.detach(), tri_iter_count)
            # Log correlation statistics
            feature_name = [
                f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
            corr_vec = [corr.detach() for corr in corr_vec]
            writer.add_scalars(
                f"Correlation/Train", dict(zip(feature_name, corr_vec)), tri_iter_count)

            tri_iter_count += 1
            print(
                f"Epoch {epoch:<3} | Step {step:<4} | Train Loss: {loss:.5f} | Corr: {-loss:.5f}")

        # Evaluation
        with torch.no_grad():
            if val_dset is not None:
                for step, batch in enumerate(val_dset):
                    b1, b2 = [], []
                    for (img, spec) in batch:
                        img = img.unsqueeze(0).float().to(DEVICE)
                        spec = spec.unsqueeze(0).float().to(DEVICE)
                        # WindowDCCA Forward Pass
                        out1, out2 = model(img, spec)
                        b1.append(out1)
                        b2.append(out2)
                    b1 = torch.cat(b1, dim=0)
                    b2 = torch.cat(b2, dim=0)
                    loss, corr_vec = criterion(b1, b2)
                    # Log validation statistics
                    writer.add_scalar(
                        "Loss/Eval", loss.detach(), val_iter_count)
                    writer.add_scalar(
                        "Corr/Eval", -loss.detach(), val_iter_count)
                    # Log correlation statistics
                    feature_name = [
                        f"Feature {idx + 1}" for idx in np.arange(len(corr_vec))]
                    corr_vec = [corr.detach() for corr in corr_vec]
                    writer.add_scalars(
                        f"Correlation/Eval", dict(zip(feature_name, corr_vec)), val_iter_count)
                    val_iter_count += 1
                    print(
                        f"Epoch {epoch:<3} | Step {step:<4} | Eval Loss: {loss:.5f} | | Corr: {-loss:.5f}")


def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    ic("Training...")
