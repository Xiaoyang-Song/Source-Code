import torch
import numpy as np
from icecream import ic
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from const import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

###### Utility function for parameters ######
def get_configuration(cfg, f=None):
    print("\n" + "="*80)
    print("Experiment Configuration")
    print(f">> Dataset: {cfg.dset}")
    print(f">> Learning rate: {cfg.lr}")
    print(f">> Learning rate scheduler step: {cfg.lr_step}")
    print(f">> Max epochs: {cfg.n_epochs}")
    print(f">> Batch size (tri): {cfg.bsz_tri}")
    print(f">> Batch size (val): {cfg.bsz_val}")
    print(f">> Train-Val Ratio: {cfg.tv_ratio}")
    print(f">> Pretrained checkpoint: {cfg.ckpt_name}")
    print(f">> File identifier: {cfg.tag}")
    print("="*80 + "\n")

    if f is not None:
        f.write("\n" + "="*80 + "\n")
        f.write("Experiment Configuration\n")
        f.write(f">> Dataset: {cfg.dset}\n")
        f.write(f">> Learning rate: {cfg.lr}\n")
        f.write(f">> Learning rate scheduler step: {cfg.lr_step}\n")
        f.write(f">> Max epochs: {cfg.n_epochs}\n")
        f.write(f">> Batch size (tri): {cfg.bsz_tri}\n")
        f.write(f">> Batch size (val): {cfg.bsz_val}\n")
        f.write(f">> Train-Val Ratio: {cfg.tv_ratio}\n")
        f.write(f">> Pretrained checkpoint: {cfg.ckpt_name}\n")
        f.write(f">> File identifier: {cfg.tag}\n")
        f.write("="*80 + "\n\n")


######  Utility Functions for Feature Extraction ######
def get_features_ft(dataset_ft, encoder):
    with torch.no_grad():
        feat, label, stats = [], [], []
        for (x, y, stat) in dataset_ft:
            f = encoder(x.unsqueeze(0))[0].squeeze().detach().numpy()
            feat.append(f)
            label.append(y)
            stats.append(stat)
        feat = np.array(feat)
        return feat, label, stats
    
def get_features_sim(dataset_ft, encoder):
    with torch.no_grad():
        feat, label, stats = [], [], []
        for (x, y, stat) in dataset_ft:
            f = encoder(x)[0].squeeze().detach().numpy()
            feat.append(f)
            label.append(y)
            stats.append(stat)
        feat = np.array(feat)
        return feat, label, stats
    
def get_features_ft_ct(dataset_ft, encoder):
    with torch.no_grad():
        feat, label, stats = [], [], []
        for (ct, sp, y, stat) in dataset_ft:
            f = encoder(ct)[0].squeeze().detach().numpy()
            feat.append(f)
            label.append(y)
            stats.append(stat)
        feat = np.array(feat)
        return feat, label, stats


def vis_pca_feat_ft(feat, label, stats):
    # Visualize spectra feature distribution using 2D PCA
    pca = PCA(n_components=2)
    pc = pca.fit_transform(feat)
    df = pd.DataFrame(data=pc, columns = ['PC1', 'PC2'])
    df['label'] = label

    fig = plt.figure(figsize = (4, 4))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1', fontsize = 12)
    ax.set_ylabel('PC2', fontsize = 12)
    ax.set_title('2D PCA visualization', fontsize = 15)

    targets = [0, 1]
    colors = ['r', 'g' ]
    for target, color in zip(targets,colors):
        indicesToKeep = df['label'] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1'], df.loc[indicesToKeep, 'PC2'],c=color, s=50, alpha=0.5)
    ax.legend(targets)
    # plt.scatter(df.loc[:,'PC1'], df.loc[:, 'PC2'], c=stats, cmap='viridis', edgecolors='k', alpha=0.7)
    # cbar = plt.colorbar()
    # cbar.set_label('Values')
    ax.grid()
    plt.savefig('out.png')

def vis_umap_feat_ft(feat, label, stats):
    # Visualization using umap
    mapper = umap.UMAP().fit(feat)
    # p = umap.plot.points(mapper, labels=y_vis, color_key=colors)
    p = umap.plot.points(mapper, labels=label)
    umap.plot.show(p)
    # torch.save(umap.plot.show(p), "umap.png")



def flip_pixels(data: list):
    data = np.array(data)
    data = torch.tensor(1 - (data / 255), dtype=torch.float32).unsqueeze(1)
    return data

def get_defect_masks(data: torch.tensor):
    white_mask = (torch.sum(data - WHITE_IMAGES_PROCESSED,
                  dim=(-1, -2)) == 0).squeeze()
    # print(f"There are {white_mask.sum()} non-defects images.")
    defect_mask = ~white_mask
    # print(f"There are {defect_mask.sum()} defects images.")
    return white_mask, defect_mask

######  Utility Functions for Dataset Construction  ######
def get_num_white(img_window, white_img):
    # ic(img_window.shape)
    return sum(torch.sum(img_window - white_img, dim=1) == 0).item()


def get_basic_white_stats(dset, white):
    num_white = []
    for (img, spec) in dset:
        # ic(img.shape)
        # ic(white.shape)
        num_white.append(get_num_white(img, white))
    # counter = Counter(num_white)
    return num_white

######  Utility Functions for Gradient Visualization  ######
def autoencoder_gradient(encoder, decoder, img):
    img = img.unsqueeze(0)
    img.requires_grad = True
    feat, idx1, idx2 = encoder(img)
    feat.retain_grad()
    reconstructed_img = decoder(feat, idx1, idx2)
    loss = F.mse_loss(img, reconstructed_img)
    loss.backward()
    # grad = torch.abs(img.grad.data)
    grad = -img.grad.data
    # Plot
    plt.figure(figsize=(16, 8))
    # Raw Image
    plt.subplot(1, 3, 1)
    plt.imshow(img.detach().squeeze().cpu(), cmap='binary')
    plt.title("Raw Image")
    # Reconstructed Image
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_img.detach().squeeze().cpu(), cmap='binary')
    plt.title("Reconstructed Image")
    # Gradient Heatmap
    plt.subplot(1, 3, 3)
    # plt.imshow(grad.squeeze().detach(), cmap='seismic')
    plt.imshow(torch.abs(grad).squeeze().detach(), cmap='viridis')
    plt.title("Gradient Heatmap")
    # plt.subplot(1, 3, 3), plt.imshow(grad.squeeze().detach())
    plt.show()
    plt.savefig("gradmap.png", dpi=400)
    return grad
