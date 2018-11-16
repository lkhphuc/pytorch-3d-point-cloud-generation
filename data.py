"""Pytorch Dataset and Dataloader for 3D PCG"""
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class PointCloud2dDataset(Dataset):
    """2D dataset rendered from ShapeNet for Point Cloud Generation
    Get all views of 1 model
    Return: dict()
        image_in (np.ndarray): [views, heights, width, channels]
        depth (np.ndarray): [angles, height, width]
        mask (np.ndarray): [angles, height, width]
    """

    def __init__(self, cfg, loadNovel=False, loadFixedOut=True,
                 loadTest=False, transforms=None):
        self.cfg = cfg
        self.loadNovel = loadNovel
        self.loadFixedOut = loadFixedOut
        self.load = "test" if loadTest else "train"
        self.transforms = transforms
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
            return {"image_in": image_in, "depth": depth, "mask": mask}

    def collate_fn(self, batch):
        """Convert a list of models with many views to
        a batch of random views of different models
        Args:
            batch: (list) [chunkSize, ]
                each element of list batch has shape
                [viewN, height, width, channels]
        Return: {}
            inputImage: [batchSize, height, width, channels]
            targetTrans: (numpy.ndarray)[batchSize, novelN, 4]
            depthGT: [batchSize, novelN, height, width, 1]
            maskGT: [batchSize, novelN, height, width, 1]
        """
        # Shape: [chunkSize, viewN, height, width, channels] 
        batch_n = {key: np.array([d[key] for d in batch]) for key in batch[0]}
        # Shape: [batchSize,]
        modelIdx = np.random.permutation(self.cfg.chunkSize)[:self.cfg.batchSize]
        # Shape: [batchSize, novelN]
        modelIdxTile = np.tile(modelIdx, [self.cfg.novelN, 1]).T
        # 24 is the number of rendered images for a single CAD models
        # Shape: [batchSize,]
        angleIdx = np.random.randint(24, size=[self.cfg.batchSize])
        # Shape: [batchSize, novelN]
        sampleIdx = np.random.randint(
            self.cfg.sampleN, size=[self.cfg.batchSize, self.cfg.novelN])
        images = batch_n["image_in"][modelIdx, angleIdx]
        targetTrans = batch_n["trans"][modelIdxTile, sampleIdx]
        depthGT = np.expand_dims(
            batch_n["depth"][modelIdxTile, sampleIdx], axis=-1)
        maskGT = np.expand_dims(
            batch_n["mask"][modelIdxTile, sampleIdx], axis=-1).astype(np.int)
        if self.transforms:
            images = torch.stack(
                [self.transforms(images[i])
                 for i in range(images.shape[0])], dim=0)
            depthGT = torch.stack(
                [self.transforms(depthGT[i])
                 for i in range(self.cfg.batchSize)], dim=0)
            maskGT = torch.stack(
                [self.transforms(maskGT[i])
                 for i in range(self.cfg.batchSize)], dim=0)

        return {"inputImage": images,
                "targetTrans": targetTrans,
                "depthGT": depthGT,
                "maskGT": maskGT }

    def collate_fn_fixed(self, batch):
        """Convert a list of models with many views to
        a batch of some fixed views of different models
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
        modelIdx = np.random.permutation(self.cfg.chunkSize)[:self.cfg.batchSize]
        # 24 is the number of rendered images for a single CAD models
        angleIdx = np.random.randint(24, size=[self.cfg.batchSize])

        images = batch_n["image_in"][modelIdx, angleIdx]
        depthGT = np.transpose(batch_n["depth"][modelIdx], axes=[0, 2, 3, 1])
        maskGT = np.transpose(batch_n["mask"][modelIdx], axes=[0, 2, 3, 1]).astype(np.int)
        if self.transforms:
            images = torch.stack(
                [self.transforms(images[i])
                 for i in range(images.shape[0])], dim=0)
            depthGT = torch.stack(
                [self.transforms(depthGT[i])
                 for i in range(depthGT.shape[0])], dim=0)
            maskGT = torch.stack(
                [self.transforms(maskGT[i])
                 for i in range(maskGT.shape[0])], dim=0)

        return {
            "inputImage": images,
            "depthGT": depthGT,
            "maskGT": maskGT,
        }

if __name__ == "__main__":
    import options
    from torchvision import transforms

    CFG = options.get_arguments()
    tfms = transforms.ToTensor()

    ds_fixed = PointCloud2dDataset(CFG, transforms=tfms)
    dl_fixed = DataLoader(ds_fixed, batch_size=CFG.chunkSize, shuffle=False, collate_fn=ds_fixed.collate_fn_fixed)

    ds_novel = PointCloud2dDataset(CFG, loadNovel=True, transforms=tfms)
    dl_novel = DataLoader(ds_novel, batch_size=CFG.chunkSize, shuffle=False, collate_fn=ds_novel.collate_fn)

