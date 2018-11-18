"""Utilities for 2D-3D conversion and building models"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
import torchvision

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

def build_structure_generator(cfg):
    model = Structure_Generator()
    if cfg.load is not None:
        LOAD_PATH = f"models/{cfg.loadPath}_{cfg.experiment}"
        print(cfg.load)
        if cfg.load == 0:
            model.load_state_dict(torch.load(f"{LOAD_PATH}/best.pth"))
        else: model.load_state_dict(
            torch.load(f"{LOAD_PATH}/{cfg.load}.pth"))
    return model

def make_optimizer(cfg, model):
    params = model.parameters()
    if cfg.optim.lower() in 'adam':
        if cfg.trueWD: return optim.Adam(params, cfg.lr, weight_decay=0)
        else : return optim.Adam(params, cfg.lr, weight_decay=cfg.wd)
    elif cfg.optim.lower() in 'sgd':
        return optim.SGD(params, cfg.lr)

def make_lr_scheduler(cfg, optimizer):
    if not cfg.lrSched: return None
    elif cfg.lrSched.lower() in 'steplr':
        return lr_scheduler.StepLR(optimizer, cfg.lrStep, cfg.lrDecay)
    elif cfg.lrSched.lower() in 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, cfg.lrDecay)

def make_data_fixed(cfg, tfms):
    ds_tr = data.PointCloud2dDataset(cfg, loadNovel=False, loadFixedOut=True,
                                     loadTest=False, transforms=tfms)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.chunkSize, shuffle=True,
                       drop_last=True, collate_fn=ds_tr.collate_fn_fixed)

    ds_test = data.PointCloud2dDataset(cfg, loadNovel=False, loadFixedOut=True,
                                       loadTest=True, transforms=tfms)
    dl_test = DataLoader(ds_test, batch_size=cfg.chunkSize, shuffle=False,
                         drop_last=True, collate_fn=ds_test.collate_fn_fixed)
    return [dl_tr, dl_test]

def make_data_novel(cfg, tfms):
    ds_tr = data.PointCloud2dDataset(
        cfg, loadNovel=True, loadFixedOut=False,
        loadTest=False, transforms=tfms)
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.chunkSize, shuffle=True,
        drop_last=True, collate_fn=ds_tr.collate_fn)

    ds_test = data.PointCloud2dDataset(
        cfg, loadNovel=True, loadFixedOut=False,
        loadTest=True, transforms=tfms)
    dl_test = DataLoader(
        ds_test, batch_size=cfg.chunkSize, shuffle=False,
        drop_last=True, collate_fn=ds_test.collate_fn)
    return [dl_tr, dl_test]


# logging
def save_best_model(model_path, model, df_hist):
    if df_hist['val_loss'].tail(1).iloc[0] <= df_hist['val_loss'].min():
        torch.save(model.state_dict(), f"{model_path}/best.pth")

def log_hist(logger, df_hist):
    last = df_hist.tail(1)
    best = df_hist.sort_values('val_loss').head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary['name'] = ['Last', 'Best']
    logger.debug(summary[['name', 'epoch', 'train_loss', 'val_loss']])
    logger.debug('')

def write_on_board_losses_stg1(writer, df_hist):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars('loss', {
        'train': row.train_loss,
        'val': row.val_loss,
    }, row.epoch)
    writer.add_scalars('loss_XYZ', {
        'train': row.train_loss_XYZ,
        'val': row.val_loss_XYZ,
    }, row.epoch)
    writer.add_scalars('loss_mask', {
        'train': row.train_loss_mask,
        'val': row.val_loss_mask,
    }, row.epoch)

def write_on_board_losses_stg2(writer, df_hist):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars('loss', {
        'train': row.train_loss,
        'val': row.val_loss,
    }, row.epoch)
    writer.add_scalars('loss_depth', {
        'train': row.train_loss_depth,
        'val': row.val_loss_depth,
    }, row.epoch)
    writer.add_scalars('loss_mask', {
        'train': row.train_loss_mask,
        'val': row.val_loss_mask,
    }, row.epoch)

def write_on_board_images_stg1(writer, images, epoch):
    writer.add_image('RGB', images['RGB'], epoch)
    writer.add_image('depth/GT', images['depthGT'], epoch)
    writer.add_image('depth/pred', images['depth'], epoch)
    writer.add_image('depth/pred_mask', images['depth_mask'], epoch)
    writer.add_image('mask/GT', images['maskGT'], epoch)
    writer.add_image('mask/pred', images['mask'], epoch)

def write_on_board_images_stg2(writer, images, epoch):
    writer.add_image('RGB', images['RGB'], epoch)
    writer.add_image('depth/GT', images['depthGT'], epoch)
    writer.add_image('depth/pred', images['depth'], epoch)
    writer.add_image('mask/GT', images['maskGT'], epoch)
    writer.add_image('mask/pred', images['mask'], epoch)
    writer.add_image('mask/rendered', images['mask_rendered'], epoch)

def make_grid(t):
    return torchvision.utils.make_grid(t, normalize=True)

