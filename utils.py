"""Utilities for 2D-3D conversion and building models"""
import os

import numpy as np
import scipy.misc
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import data
from PCGModel import Structure_Generator


# compute projection from source to target
def projection(Vs, Vt):
    VsN = Vs.size(0)
    VtN = Vt.size(0)
    Vt_rep = torch.repeat(Vt[None, :, :], [VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = torch.repeat(Vs[:, None, :], [1, VtN, 1])  # [VsN,VtN,3]
    diff = Vt_rep - Vs_rep
    dist = torch.sqrt(torch.reduce_sum(diff**2, axis=[2]))  # [VsN,VtN]
    idx = torch.to_int32(torch.argmin(dist, axis=1))
    proj = torch.gather_nd(Vt_rep, torch.stack([torch.range(VsN), idx], axis=1))
    minDist = torch.gather_nd(dist, torch.stack([torch.range(VsN), idx], axis=1))
    return proj, minDist


def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)


def imread(fname):
    return scipy.misc.imread(fname) / 255.0


def imsave(fname, array):
    scipy.misc.toimage(array, cmin=0.0, cmax=1.0).save(fname)


# make image summary from image batch
def imageSummary(opt, tag, image, H, W):
    blockSize = opt.visBlockSize
    imageOne = torch.batch_to_space(
        image[:blockSize**2], crops=[[0, 0], [0, 0]], block_size=blockSize)
    imagePermute = torch.reshape(imageOne, [H, blockSize, W, blockSize, -1])
    imageTransp = torch.transpose(imagePermute, [1, 0, 3, 2, 4])
    imageBlocks = torch.reshape(imageTransp,
                             [1, H * blockSize, W * blockSize, -1])
    summary = torch.summary.image(tag, imageBlocks)
    return summary



# save model
def saveModel(opt, sess, saver, it):
    saver.save(sess, "models_{0}/{1}_it{2}.ckpt".format(
        opt.group, opt.model, it))

def build_structure_generator(cfg):
    return Structure_Generator()

def make_optimizer(cfg, model):
    # params = []
    # for key, value in model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = cfg.lr
    #     weight_decay = cfg.lrDecay
    #     params += [{"params": [value], "lr": cfg.lr, "weight_decay": weight_decay}]
    params = model.parameters()
    optimizer = torch.optim.SGD(params, cfg.lr)
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    return CosineAnnealingLR(optimizer, 10)

def make_data_fixed(cfg, tfms):
    ds_tr = data.PointCloud2dDataset(cfg, loadNovel=False, loadFixedOut=True, loadTest=False, transforms=tfms)
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.chunkSize, shuffle=True,
        collate_fn=ds_tr.collate_fn_fixed)

    ds_test = data.PointCloud2dDataset(cfg, loadNovel=False, loadFixedOut=True, loadTest=True, transforms=tfms)
    dl_test = DataLoader(
        ds_test, batch_size=cfg.chunkSize, shuffle=False,
        collate_fn=ds_test.collate_fn_fixed)
    return [dl_tr, dl_test]

