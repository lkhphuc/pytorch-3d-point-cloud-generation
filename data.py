# %%
import os

import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

#%%
class PointCloud2dDataset(Dataset):
    """2D dataset rendered from ShapeNet for Point Cloud Generation
    Get all views of 1 model
    Return: dict()
        image_in (np.ndarray): [views, heights, width, channels]
        depth (np.ndarray): [angles, height, width]
        mask (np.ndarray): [angles, height, width]
    """

    def __init__(self, cfg, loadNovel=False, loadFixedOut=True, loadTest=False):
        """
        Args:
            cfg (dict): ArgParse with all configs
        """
        self.cfg = cfg
        self.loadNovel = loadNovel
        self.loadFixedOut = loadFixedOut
        self.load = "test" if loadTest else "train"
        list_file = f"{cfg.path}/{cfg.category}_{self.load}.list"
        self.CADs = []
        with open(list_file) as file:
            for line in file:
                id = line.strip().split("/")[1]
                self.CADs.append(id)
            self.CADs.sort()

    def __len__(self):
        return len(self.CADs)

    def __getitem__(self, idx):
        CAD = self.CADs[idx]
        image_in = np.load(
            f"{self.cfg.path}/{self.cfg.category}_inputRGB/{CAD}.npy")
        image_in = image_in / 255.0

        if self.loadNovel:
            raw_data = scipy.io.loadmat(
                f"{self.cfg.path}/{self.cfg.category}_depth/{CAD}.mat")
            depth = raw_data["Z"]
            trans = raw_data["trans"]
            mask = depth != 0
            depth[~mask] = self.cfg.renderDepth

            return {"image_in": image_in, "depth": depth, "mask": mask, "trans": trans}

        if self.loadFixedOut:
            raw_data = scipy.io.loadmat(
                f"{self.cfg.path}/{self.cfg.category}_depth_fixed{self.cfg.outViewN}/{CAD}.mat")
            depth = raw_data["Z"]
            mask = depth != 0
            depth[~mask] = self.cfg.renderDepth

            mask = np.transpose(mask, [1, 2, 0])
            depth = np.transpose(depth, [1, 2, 0])

            return {"image_in": image_in, "depth_fixedOut": depth, "mask_fixedOut": mask}

    def collate_fn(self, batch):
        """Convert a list of models to a batch of view
        Args:
            batch: (list) [chunkSize, ]
                each element of list batch has shape
                [viewN, height, width, channels]
        """
        # Shape: [chunkSize, viewN, height, width, channels] 
        batch_n = {key: np.array([d[key] for d in batch]) for key in batch[0]}
        modelIdx = np.random.permutation(cfg.chunkSize)[:cfg.batchSize]
        angleIdx = np.random.randint(24, size=[cfg.batchSize])
        return {
            inputImage: batch_n["image_in"][modelIdx, angleIdx],
            depthGT: batch_n["depth_fixedOut"][modelIdx],
            maskGT: batch_n["mask_fixedOut"][modelIdx]
        }

    def collate_fn_fixed(self, batch):
        """Convert a list of models to a batch of view
        Args:
            batch: (list) [chunkSize, ]
                each element of list batch has shape
                [viewN, height, width, channels]
        Return: {}
            inputImage: [batchSize, height, width, channels]
            depth_fixedOut: [batchSize, height, width, 8]
            mask_fixedOut: [batchSize, height, width, 8]
        """
        # Shape: [chunkSize, viewN, height, width, channels] 
        batch_n = {key: np.array([d[key] for d in batch]) for key in batch[0]}
        modelIdx = np.random.permutation(cfg.chunkSize)[:cfg.batchSize]
        # 24 is the number of rendered images for a single CAD models
        angleIdx = np.random.randint(24, size=[cfg.batchSize])
        return {
            "inputImage": batch_n["image_in"][modelIdx, angleIdx],
            "depthGT": batch_n["depth_fixedOut"][modelIdx],
            "maskGT": batch_n["mask_fixedOut"][modelIdx]
        }

# %%
### TEST
if __name__ == "__main__":
    import options
    cfg = options.get_arguments(training=True)
    ds = PointCloud2dDataset(cfg)
    dl = DataLoader(ds, batch_size=cfg.chunkSize, shuffle=False, collate_fn=ds.collate_fn_fixed)

# %%
