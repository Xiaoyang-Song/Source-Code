import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
from icecream import ic
from util.utils import *
from const import *
from tqdm import tqdm
import argparse
import math
import re

def extract_feature(data, encoder, bsz, f = None, device=DEVICE):
    print('Extracting features...')
    if f is not None:
        f.write('Extracting features...\n')
    feats = []
    loader = torch.utils.data.DataLoader(data, batch_size=bsz, shuffle=False)
    
    with torch.no_grad():
        for data_batch in tqdm(loader):
            data_batch = data_batch.to(device)
            feats.append(encoder(data_batch)[0].to('cpu')) # transfer to CPU to save space
            data_batch = data_batch.to('cpu')
        return torch.cat(feats, dim=0)

class CT():
    """
    This class is used to train Autoencoder of CT scans; not for DCCA network.
    """
    def __init__(self, parts:list):
        self.parts = parts

        # Data
        self.raw = {}
        self.data = None

        # Features
        self.features = None

        # Physical statistics
        self.statistics = None

    def read(self, f=None):
        for part in self.parts:
            path = os.path.join(DATA_PROCESSED_DIR, part, 'final')
            # path = os.path.join(DATA_RAW_DIR, part) # Only for illustration plots in intro section
            image_addr_list = sorted(os.listdir(path))
            if f is not None:
                f.write(f'Reading part {part}...\nNumber of samples: {len(image_addr_list)}\n')
            print(f'Reading part {part}...\nNumber of samples: {len(image_addr_list)}')
            part_data = []
            for i in tqdm(image_addr_list):
                full_path = os.path.join(path, i)
                imgobj = Image.open(full_path).convert('L')
                img = np.asarray(imgobj)
                part_data.append(img)
            # Flip black and white pixels
            part_data = flip_pixels(part_data)
            self.raw[part] = part_data

            # For plotting only
            # self.raw[part] = torch.tensor(np.array(part_data), dtype=torch.float32).unsqueeze(1)

    def prepare_statistics(self):
        self.statistics = {}
        for part in self.parts:
            self.statistics[part] = torch.sum(self.raw[part], dim=(1, 2, 3))
            #TODO: Add more statistics later

    def prepare_features(self, encoder, bsz, f=None):
        self.features = {}
        for part in self.parts:
            self.features[part] = extract_feature(self.raw[part], encoder, bsz, f)
            print(self.features[part].shape)
            
    
    def gather(self, f=None):
        self.data = []
        for part in self.parts:
            self.data += self.raw[part]
            # print(np.array(part_data).shape)
        print(f"Finished gathering data from parts: {self.parts}.")
        print(f"{len(self.data)} CT scan images in total.")
        # print(np.array(self.data).shape)
        if f is not None:
            f.write(f"Finished gathering data from parts {self.parts}.\n{len(self.data)} CT scan images in total.\n")
        self.data = torch.cat(self.data, dim=0)
        return self.data
        
    def filter_non_defects(self, n = 10, f=None):
        assert self.data is not None
        non_defect_mask, defect_mask = get_defect_masks(self.data)
        print(f"Among {len(self.data)} CT scan images, {defect_mask.sum()} ({100 * defect_mask.sum() / len(self.data):.4f}%) images contain defects.")
        if f is not None:
            f.write(f"Among {len(self.data)} CT scan images, {defect_mask.sum()} ({100 * defect_mask.sum() / len(self.data):.4f}%) images contain defects.\n")
        # Filter non-defects
        non_defects, defects = self.data[non_defect_mask], self.data[defect_mask]
        self.dataset = torch.cat([defects, non_defects[0:n]])
        if f is not None:
            f.write(f"There are {self.dataset.shape[0]} CT scan images in the dataset in total.\n")
        print(f"There are {self.dataset.shape[0]} CT scan images in the dataset in total.")
        return self.dataset


class SP():
    def __init__(self, parts:list, normalize=False, window_size=5, f=None):
        self.parts = parts
        self.wsize = window_size
        # Normalization
        self.normalize = normalize
        if self.normalize:
            self.sp_min, self.sp_max = self.get_min_max(self.parts)
            # print(self.sp_max, self.sp_min)
        # Data: raw & processed
        self.raw = None
        self.stacked_data = None
        self.processed_data = None
        self.data = None

        # Feature
        self.features = None
        # Order of processing spectra: read -> clip_location -> clip_wavelength -> gather
        self.read(f)
        self.clip_location(f)
        self.clip_wavelength(f)


    def get_min_max(self, parts):
        sp_min = min([SPECTRA_MIN[part] for part in parts])
        sp_max = max([SPECTRA_MAX[part] for part in parts])
        return sp_min, sp_max

    def locate_active(self, avg_spectra, part):
        if self.normalize:
            threshold = (SPECTRA_THRESHOLD[part] - self.sp_min) / (self.sp_max - self.sp_min)
        else:
            threshold = SPECTRA_THRESHOLD[part]

        print("Active Threshold: ", threshold)
        up_j, down_j = 0, 0
        up_pos, down_pos = np.zeros(3), np.zeros(3)
        for i in range(len(avg_spectra) - 1):
            if (avg_spectra[i] > threshold) & (avg_spectra[i - 1] < threshold):
                up_pos[up_j] = i
                up_j += 1
            if (avg_spectra[i] > threshold) & (avg_spectra[i + 1] < threshold):
                down_pos[down_j] = i
                down_j += 1
        return up_pos, down_pos
    
    @staticmethod
    def sliding_window(spectra, size=5):
        window_lst = []
        # sliding windows (overlapped)
        for i in tqdm(np.arange(len(spectra))):
            win = spectra[i:i+size]
            if win.shape[0] == size:
                window_lst.append(list(np.array(win)))
        return window_lst
    
    @staticmethod
    def normalize(spectra):
        # UNUSED
        if type(spectra) == np.ndarray:
            return (spectra - np.min(spectra)) / (np.max(spectra) - np.min(spectra))
        else:
            return (spectra - torch.min(spectra)) / (torch.max(spectra) - torch.min(spectra))
    
    def read(self, f=None):
        self.raw = {}
        if f is not None:
            f.write("-"*60 + "\n")
            f.write("Reading data...\n")
        for part in self.parts:
            path = os.path.join(DATA_SPEC_DIR, SPECTRA_NAME[part])
            raw = np.loadtxt(open(path, "rb"), delimiter="\t")
            full_spectra = raw[1:, 1:]
            if self.normalize:
                full_spectra = (full_spectra - self.sp_min) / (self.sp_max - self.sp_min)
            if f is not None:
                f.write(f'>> Reading part {part}...\n---- Spectra Shape: {full_spectra.shape}\n')
            print(f'>> Reading part {part}...\n---- Spectra Shape: {full_spectra.shape}\n')
            self.raw[part] = full_spectra
        if f is not None:
            f.write("-"*60 + "\n")

    def prepare_features(self, encoder, bsz, f=None):
        self.features = {}
        for part in self.parts:
            print(self.processed_data[part].shape)
            self.features[part] = extract_feature(self.processed_data[part], encoder, bsz, f).squeeze()
            print(self.features[part].shape)

    def clip_location(self, f=None):
        assert self.raw is not None
        self.stacked_data = {}
        if f is not None:
            f.write("-"*60 + "\n")
            f.write("Processing data\n")
        for part in self.parts:
            assert part in self.raw
            full_spectra = self.raw[part]
            avg_intensity = np.mean(full_spectra, axis=0)

            # Locate active signals
            up_pos, down_pos = self.locate_active(avg_intensity, part)
            len_spectra = int(min(down_pos - up_pos))

            # Clip ranges
            shift_pos = np.floor((down_pos - up_pos - np.ones(3) * len_spectra) / 2).astype(int)
            start_pos = up_pos.astype(int) + shift_pos
            end_pos = up_pos.astype(int) + shift_pos + np.ones(3).astype(int) * len_spectra

            # Stack three laser scans
            stacked_spectra = np.zeros((full_spectra.shape[0], len_spectra, 3))
            stacked_spectra[:, :, 0] = full_spectra[:, start_pos[0]:end_pos[0]]
            stacked_spectra[:, :, 1] = np.flip(full_spectra[:, start_pos[1]:end_pos[1]], axis=1)
            stacked_spectra[:, :, 2] = full_spectra[:, start_pos[2]:end_pos[2]]

            self.stacked_data[part] = stacked_spectra

            print(f">> Locating active signals for part {part}.")
            print(f"---- The length of active signals is: {len_spectra}\n---- S: {up_pos}\n---- E: {down_pos}")
            print(f"---- Processed signal shape: {stacked_spectra.shape}\n")
            if f is not None:
                f.write(f">> Locating active signals for part {part}.\n")
                f.write(f"---- The length of active signals is: {len_spectra}\n---- S: {up_pos}\n---- E: {down_pos}\n")
                f.write(f"---- Processed signal shape: {stacked_spectra.shape}\n")
        if f is not None:
            f.write("-"*60 + "\n")
    
    def clip_wavelength(self, lb=394, ub=398, f=None):
        assert self.stacked_data is not None
        self.processed_data = {}
        print(f"\n Locating active wavelength for AL7075.")
        if f is not None:
            f.write(f"\n>> Locating active wavelength for AL7075.\n")
        for part in self.parts:
            spectra = self.stacked_data[part][W_MASK,:, :][RESOLUTION_MASK]
            # spectra = self.stacked_data[part]
            n_frames, n_wl = spectra.shape[1], spectra.shape[0]
            # Take the mean of three laser scans
            # self.processed_data[part] = torch.tensor(np.mean(spectra, axis=2).transpose(), dtype=torch.float32).unsqueeze(-2)
            self.processed_data[part] = torch.tensor(spectra, dtype=torch.float32).reshape((n_frames, n_wl, 3))

            print(f"---- Processed signal shape for part {part}: {spectra.shape}")
            if f is not None:
                f.write(f"---- Processed signal shape for part {part}: {spectra.shape}\n")
        if f is not None:
            f.write("-"*60 + "\n")

    def gather_ae(self, sliding_window, f=None):
        self.data_ae = []
        if f is not None:
            f.write("-"*60 + "\n")
            f.write("Gathering data from different parts\n")
        for part in self.parts:
            part_data = self.raw[part][W_MASK,:][RESOLUTION_MASK].reshape((-1, 32))
            if f is not None:
                f.write(f'>> Gathering part {part}...\n---- Spectra Shape: {part_data.shape}\n')
            print(f'>> Gathering part {part}...\n---- Spectra Shape: {part_data.shape}')

            if sliding_window:
                # Temporarily, only use the first scan
                # TODO: Change this later...
                windowed_data = SP.sliding_window(part_data, self.wsize)
                # windowed_data = windowed_data.unsqueeze(-2) # channel dimension
                self.data_ae += windowed_data
            else:
                self.data_ae += list(np.array(part_data))
        
        if f is not None:
            f.write(f">> Finished gathering data from parts {self.parts}.\n---- {len(self.data_ae)} spectra windows of size {self.wsize} in total.\n")
            f.write(f"---- Data shape: {np.array(self.data_ae).shape}\n")
        print(f">> Finished gathering data from parts {self.parts}.\n---- {len(self.data_ae)} spectra windows of size {self.wsize} in total.")
        print(f"---- Data shape: {np.array(self.data_ae).shape}")
        # Casting
        self.data_ae = torch.tensor(np.array(self.data_ae), dtype=torch.float32)
        return self.data_ae
    
    def gather(self, sliding_window, f=None):
        self.data = []
        if f is not None:
            f.write("-"*60 + "\n")
            f.write("Gathering data from different parts\n")
        for part in self.parts:
            # dtype of part_data: torch.tensor
            part_data = self.processed_data[part]
            if f is not None:
                f.write(f'>> Gathering part {part}...\n---- Spectra Shape: {part_data.shape}\n')
            print(f'>> Gathering part {part}...\n---- Spectra Shape: {part_data.shape}')

            if sliding_window:
                # Temporarily, only use the first scan
                # TODO: Change this later...
                windowed_data = SP.sliding_window(part_data[:,:,0], self.wsize)
                # windowed_data = windowed_data.unsqueeze(-2) # channel dimension
                self.data += windowed_data
            else:
                self.data += list(np.array(part_data))
        
        if f is not None:
            f.write(f">> Finished gathering data from parts {self.parts}.\n---- {len(self.data)} spectra windows of size {self.wsize} in total.\n")
            f.write(f"---- Data shape: {np.array(self.data).shape}\n")
        print(f">> Finished gathering data from parts {self.parts}.\n---- {len(self.data)} spectra windows of size {self.wsize} in total.")
        print(f"---- Data shape: {np.array(self.data).shape}")
        # Casting
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        return self.data


class Aligner():
    def __init__(self, parts: list, ct: CT, sp: SP):
        self.parts = parts
        self.ct = ct
        self.sp = sp

        # Aligned (CT, SP) & (CT feats, SP feats) Tuples
        self.aligned_raw = None
        self.aligned_feats_pre = None
        self.aligned_feats_post = None

        # Aligned (CT stats, SP) Tuples
        self.aligned_ct_stats_sp = None

        # Gathered data
        self.gathered_tuple_raw = None
        self.gathered_tuple_feats_pre = None
        self.gathered_tuple_feats_post = None

    @staticmethod
    def clip_window(data, size=5):
        # Heuristic function
        for idx in range(len(data)):
            data[idx] = list(data[idx]) # Tuple to list (required as tuples are immutable types)

            # Clip Spectra data
            l_sp_window = data[idx][1].shape[0]
            assert l_sp_window >= 5 and l_sp_window <= 6, f'Spectra windows size is {l_sp_window}, expected 5 <= l < = 6.'
            if l_sp_window > 5:
                data[idx][1] = data[idx][1][0:5]
            # Process the shape of spectra windows
            # Now only using the first scan (TODO: Change this later)
            data[idx][1] = data[idx][1][:,:,0].reshape((5, 1, 32))

            # Clip CT images (TODO: Change this later)
            # OPTION-1: use window with clipping
            # l_ct_window = data[idx][0].shape[0]
            # assert l_ct_window >= 23 and l_ct_window <= 24, f'CT scan windows size is {l_ct_window}, expected 23 <= l < = 24.'
            # if l_ct_window > 23:
            #     data[idx][0] = data[idx][0][0:23, :, :, :]

            # OPTION-2: use the mean of images in the window
            data[idx][0] = torch.mean(data[idx][0], keepdim=True, dim=0)

            # Transform back to tuples
            data[idx] = tuple(data[idx])
        return data

    def align(self, l=35):
        self.aligned_raw, self.aligned_feats_post = {}, {}

        ct_raw, sp_raw = self.ct.raw, self.sp.processed_data
        for part in self.parts:
            print("Aligner step size: ", STEP_SIZE[part])
            ct_r, sp_r = ct_raw[part], sp_raw[part]
            # ct, sp = ct_feats[part], sp_feats[part]
            n_ct, n_sp = len(ct_r), len(sp_r)
            r_ct, r_sp = n_ct / l, n_sp / l

            # Align CT scans and spectra
            step_size = STEP_SIZE[part]
            part_raw, part_feats = [], []
            cur_s = 0
            cur_e = cur_s + step_size
            while cur_e <= l:
                ct_s_idx, ct_e_idx = int(cur_s * r_ct), int(cur_e * r_ct)
                sp_s_idx, sp_e_idx = int(cur_s * r_sp), int(cur_e * r_sp)
                ct = ct_r[ct_s_idx: ct_e_idx, :, :, :]
                sp = sp_r[sp_s_idx: sp_e_idx, :]

                part_raw.append((ct, sp))

                cur_s += step_size
                cur_e += step_size

            self.aligned_raw[part] = part_raw

    def append_stats(self, functions):
        assert self.gathered_tuple_raw is not None
        for idx in range(len(self.gathered_tuple_raw)):
            stats = []
            ct_window = self.gathered_tuple_raw[idx][0]
            for f in functions:
                score = f(ct_window)
                stats.append(score)
            self.gathered_tuple_raw[idx] += (torch.tensor(stats),)
        return self.gathered_tuple_raw

    
    def align_sp_ct_stats(self, step_size, l=35):
        self.aligned_ct_stats_sp = {}

        ct_stats, sp_raw = self.ct.statistics, self.sp.processed_data
        for part in self.parts:
            ct_s, sp_r = ct_stats[part], sp_raw[part]
            n_ct, n_sp = len(ct_s), len(sp_r)
            r_ct, r_sp = n_ct / l, n_sp / l

            # Align CT scans statistics and spectra raw data
            part_raw = []
            
            cur_s = 0
            cur_e = cur_s + step_size
            while cur_e <= l:
                ct_s_idx, ct_e_idx = int(cur_s * r_ct), int(cur_e * r_ct)
                sp_s_idx, sp_e_idx = int(cur_s * r_sp), int(cur_e * r_sp)
                
                st = ct_s[ct_s_idx: ct_e_idx].mean(axis=0)
                sp = sp_r[sp_s_idx: sp_e_idx, :].mean(axis=0)

                part_raw.append((sp, st))

                cur_s += step_size
                cur_e += step_size

            self.aligned_ct_stats_sp[part] = part_raw

    def gather(self):
        assert self.aligned_raw is not None

        self.gathered_tuple_raw = []
        for part in self.parts:
            self.gathered_tuple_raw += self.aligned_raw[part]

        return self.gathered_tuple_raw
    
    def get_aligned_feats_pre(self, ct_encoder, sp_encoder):
        self.aligned_feats_pre = {}
        for part in self.parts:
            feats_pre = []
            for (ct, sp) in tqdm(self.aligned_raw[part]):
                ct, sp = ct.to(DEVICE), sp.to(DEVICE)
                ct_feats = ct_encoder(ct.unsqueeze(0))[0].squeeze()
                sp_feats = sp_encoder(sp.unsqueeze(0))[0].squeeze()
                feats_pre.append((ct_feats, sp_feats))
                # print((ct_feats.shape, sp_feats.shape))
            self.aligned_feats_pre[part] = feats_pre

    def gather_aliged_feats_pre(self):
        assert self.aligned_feats_pre is not None
        self.gathered_tuple_feats_pre = []

        for part in self.parts:
            self.gathered_tuple_feats_pre += self.aligned_feats_pre[part]


#######################################################################
def cal_avg_area(data):
    window_sum = torch.sum(data, dim=(1, 2, 3))
    return torch.mean(window_sum)

def cal_avg_num_holes(data):
    stats = []
    for img in data:
        img = np.array(1 - img, dtype=np.uint8).squeeze() * 255
        contours, hierarchy = cv.findContours(img.squeeze(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        stats.append(len(contours) - 1)
    return torch.mean(torch.tensor(stats, dtype=torch.float32))

def cal_avg_dist_holes(data):
    stats = []
    for img in data:
        img = np.array(1 - img, dtype=np.uint8).squeeze() * 255
        contours, hierarchy = cv.findContours(img.squeeze(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        num_holes = len(contours) - 1
        # If there are no defects, set zero to be the default value
        if num_holes <= 1:
            stats.append(0)
            continue
        # Compute defect cluster centers
        centers = []
        for i in range(num_holes):
            print(contours[i+1].shape)
            defect_contour = contours[i+1].reshape((-1, 2))
            print(defect_contour.shape)
            # plt.scatter(defect_contour[:, 0], defect_contour[:, 1], color='blue')
            xc, yc = defect_contour[:, 0].mean(), defect_contour[:, 1].mean()
            centers.append([xc, yc])
        # Compute pairwise distances
        dist = pair_wise_dist(centers)
        stats.append(dist)
    
    # return average pairwise distances in a window
    return torch.mean(torch.tensor(stats, dtype=torch.float32))

def pair_wise_dist(centers):
    dist = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers), 1):
            dist.append(math.dist(centers[i], centers[j]))
    return np.mean(dist)




#######################################################################

def filter_dataset(data, stats, gmask, bmask, include_ct=False):
    dataset = []
    normals, abnormals = [], []
    for idx, item in enumerate(data):
        if include_ct:
            if gmask[idx]:
                dataset.append((item[0], item[1], 1, stats[idx]))
                normals.append((item[0], item[1]))
            elif bmask[idx]:
                dataset.append((item[0], item[1], 0, stats[idx]))
                abnormals.append((item[0], item[1]))
        else:
            if gmask[idx]:
                dataset.append((item[1], 1, stats[idx]))
                normals.append((item[1], 1, stats[idx]))
            elif bmask[idx]:
                dataset.append((item[1], 0, stats[idx]))
                abnormals.append((item[1], 0, stats[idx]))
    return dataset, normals, abnormals

def form_ft_dataset(dsets, metric='avg_area', lb=None, ub=None, include_ct=False, train_on_good=False):
    with torch.no_grad():
        sp = SP(parts=dsets, normalize=True)
        sp.gather(sliding_window=False)
        ct = CT(dsets)
        ct.read()
        ct.gather()
        aligner = Aligner(dsets, ct, sp)
        aligner.align()
        data = aligner.gather()
        data = Aligner.clip_window(data)
    # Form finetuning dataset
    dset_name = "-".join(dsets)
    if metric == 'avg_area':
        assert lb is not None and ub is not None
        data_dir = os.path.join(FT_DATA_DIR, f"{dset_name}_aa_{lb}_{ub}")
        os.makedirs(data_dir, exist_ok=True)

        stats_avg = []
        for item in data:
            # window_sum = torch.sum(item[0], dim=(1, 2, 3))
            # stats_avg.append(torch.mean(window_sum))

            # New way of labeling
            window_avg = torch.mean(item[0], dim=(0,1))
            # print(window_avg.shape)
            stats_avg.append(torch.sum(window_avg > 0.1))
        # Log relevant information
        stats_avg = np.array(stats_avg)
        gmask, bmask = stats_avg <= lb, stats_avg > ub
        print(f"Number of good windows: {sum(gmask)}")
        print(f"Number of bad windows: {sum(bmask)}")
        print(f"Number of undefined windows: {len(stats_avg) - sum(bmask) - sum(gmask)}")
        f = open(os.path.join(data_dir, 'info.txt'), 'w')
        f.write(f"Number of good windows: {sum(gmask)}\n")
        f.write(f"Number of bad windows: {sum(bmask)}\n")
        f.write(f"Number of undefined windows: {len(stats_avg) - sum(bmask) - sum(gmask)}\n")
        f.close()
        # Save files
        dataset, normals, abnormals = filter_dataset(data, stats_avg, gmask, bmask, include_ct)
        if include_ct:
            torch.save(dataset, os.path.join(data_dir, "data-ct.pt"))
        else:
            torch.save(dataset, os.path.join(data_dir, "data.pt"))
        torch.save(gmask, os.path.join(data_dir, "gmask.pt"))
        torch.save(bmask, os.path.join(data_dir, "bmask.pt"))

        if train_on_good:
            torch.save(normals, os.path.join(data_dir, "data_normal.pt"))
            torch.save(abnormals, os.path.join(data_dir, "data_abnormal.pt"))
        # Make visualization plots
        plt.plot(stats_avg)
        plt.xlabel("Windows")
        plt.ylabel("Avg Area of Defects")
        plt.axhline(y = ub, color = 'r', linestyle = '-') 
        plt.axhline(y = lb, color = 'g', linestyle = '-') 
        plt.savefig(os.path.join(data_dir, 'info.jpg'))
    else:
        raise NotImplementedError



if __name__ == '__main__':
    # print('Dataset Test Suites.')
    # with torch.no_grad():
    #     part = '4D'
    #     # Test spectra class
    #     sp = SP(parts=[part])
    #     sp.gather(True)

    #     # Test "gather()" for AE training
    #     # sp.gather_ae(True)

    #     ct = CT([part])
    #     ct.read()
    #     ct.gather()

    #     aligner = Aligner([part], ct, sp)
    #     aligner.align()
    #     data = aligner.gather()
    #     data = aligner.append_stats(['a'])
    #     for item in data:
    #         print((item[0].shape, item[1].shape, item[2].shape))
    #     print(f"Number of tuples: {len(data)}")
    # data = Aligner.clip_window(data)
    # for item in data:
    #     print((item[0].shape, item[1].shape))

    
    
    # Driver code for preparing finetuning dataset
    # TODO: incorporate the following code in to another file later
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type=lambda s: re.split(' |, ', s),
                        default="1Dr", help='comma or space delimited list of SP dataset names')
    parser.add_argument('--include_ct',action='store_true')
    parser.add_argument('--good_only',action='store_true')
    parser.add_argument('--lb', help='lowerbound', default=50, type=int)
    parser.add_argument('--ub', help='upperbound', default=50, type=int)
    args = parser.parse_args()
    
    form_ft_dataset(args.dset, lb=args.lb, ub=args.ub, include_ct=args.include_ct, train_on_good=args.good_only)