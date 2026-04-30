import torch
import numpy as np
from models.dcca import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering


def decouple_dset(dset):
    img_lst, spec_lst = [], []
    for (img, spec) in dset:
        img_lst.append(img.detach().cpu())
        spec_lst.append(spec.detach().cpu())
    return img_lst, spec_lst


def decouple_batched_dset(dset):
    img_batch = []
    spec_batch = []
    for batch in dset:
        img = [i for i, _ in batch]
        spec = [s for _, s in batch]
        img_batch.append(img)
        spec_batch.append(spec)
    assert len(img_batch) == len(dset)
    assert len(spec_batch) == len(dset)
    return img_batch, spec_batch


def img_clustering(feats, n, seed=2023):
    kmean = KMeans(n_clusters=n, random_state=seed).fit(feats)
    kmean.fit(feats)
    label = kmean.predict(feats)
    return kmean, label
