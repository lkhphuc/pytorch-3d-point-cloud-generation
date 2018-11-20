import logging
import os

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import data
import options
import utils
from trainer import Trainer_stg1

if __name__ == "__main__":

    print("=======================================================")
    print("Find optimal LR for structure generator with fixed viewpoints")
    print("=======================================================")

    print("Setting configurations...")
    cfg = options.get_arguments()

    EXPERIMENT = f"{cfg.model}_{cfg.experiment}_findLR"

    print("Create Dataloader")
    dataloaders = utils.make_data_fixed(cfg)

    print("Define losses")
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    criterions = [l1_loss, bce_loss]

    print("Build Structure Generator")
    model = utils.build_structure_generator(cfg).to(cfg.device)

    print("Create optimizer and scheduler")
    optimizer = utils.make_optimizer(cfg, model)

    print("Create tensorboard logger")
    writer = SummaryWriter(comment="_"+EXPERIMENT)

    trainer = Trainer_stg1(cfg, dataloaders, criterions)
    trainer.findLR(model, optimizer, writer, start_lr=cfg.startLR, end_lr=cfg.endLR, num_iters=cfg.itersLR)

    writer.close()
