import torch
import os
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import healpy as hp
import random

class HealpixDataset(Dataset):
    def __init__(self, file_address, transform=None, tmp_dir:str=''):
        with open(file_address, 'r') as f:
            # lines = f.readlines()
            lines = f.read().splitlines()
            #This line is a comment. The secong line after this comment is the healpix resolution used for sampling. The third line is the root directory where the files are located. Then, all file names are listed
            self.resolution = int(lines[1])
            self.root_dir = lines[2]
            if tmp_dir:
                self.root_dir = os.path.join(tmp_dir, os.path.basename(self.root_dir))
            self.list_filenames = lines[3:]
            nside = hp.order2nside(self.resolution)  # == 2 ** sampling_resolution
            self.nPix = hp.nside2npix(nside)

        self.transform = transform

    def __len__(self):
        return len(self.list_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.list_filenames[idx]
        sample_location = os.path.join(self.root_dir, filename)
        sample = {"features": np.load(sample_location), "filename":filename}
        assert sample["features"].shape[0] == self.nPix, "Number of pixels in loaded data doesn't match with database resolution"
        if self.transform:
            sample = self.transform(sample)
        return sample

class HEALPixNCrop(object):
    """Crop the given healpix sampling into n crop.

    .. Note::
        This function assumes that the HEALPix sampling is in NEST mode.

    Args:
        sampling_res (int): sampling resolution of input data in HEALPix NEST format
        patch_res (int): Resolution of the patches
        n (int): number of patches crop from the HEALPix sampling. negative n means all patches
        mode (str): "fixed_shuffle": each time you get same patches while patches are shuffled all around the sphere
                    "fixed_north_to_south_sweep" patches are taken from north pole to south pole evenly spaced w.r.t HEALPix ring scheme
                    "random": patches are changing at every call


    """
    def __init__(self, sampling_res, patch_res, n, mode="random"):
        patch_width = hp.order2nside(patch_res)
        nside_pixels = hp.order2nside(sampling_res)
        self.nPix = hp.nside2npix(nside_pixels)
        nside_patch = nside_pixels // patch_width
        self.n_divisions = hp.nside2npix(nside_patch)
        assert self.n_divisions % 12 == 0, "the minimum number of pathes is equal to 12 which consists of 12 big rhombus of HEALPix sampling"
        assert n <= self.n_divisions, "Number of requested crop should be less or equal than number of divisions that respecting the patch size"
        self.nPix_per_patch = patch_width * patch_width
        if n < 0:
            self.n = self.n_divisions
        else:
            self.n = n
        self.mode = mode

        if self.mode == "fixed_shuffle":
            self.patch_ids = random.sample(range(self.n_divisions), self.n) # Return a n length list of unique elements chosen from range(self.n_divisions)
        elif self.mode == "fixed_north_to_south_sweep":
            id_in_ring_scheme = np.linspace(0, self.n_divisions-1, num=self.n, dtype=int)
            self.patch_ids = hp.ring2nest(nside_patch, id_in_ring_scheme)

    def __call__(self, sample):
        assert len(sample["features"]) == self.nPix, "size doesn't match with init size"

        if self.mode == "random":
            self.patch_ids = random.sample(range(self.n_divisions), self.n) # Return a n length list of unique elements chosen from range(self.n_divisions)

        crops = [[]] * self.n
        for i in range(self.n):
            crops[i] = sample["features"][self.nPix_per_patch * self.patch_ids[i]: self.nPix_per_patch * (self.patch_ids[i]+1), :]

        return {"features": np.stack(crops), "patch_id": self.patch_ids}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, to_range_minusOne_to_plusOne=True):
        self.toRange_minusOne_to_plusOne = to_range_minusOne_to_plusOne

    def __call__(self, sample):
        sample["features"] = torch.FloatTensor(sample["features"]).div(255.)
        if self.toRange_minusOne_to_plusOne:
            sample["features"] = sample["features"] * 2.0 - 1.0
        if "patch_id" in sample:
            sample["patch_id"] = torch.LongTensor(sample["patch_id"])

        return sample

