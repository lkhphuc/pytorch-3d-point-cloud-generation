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
    print("Pretrain structure generator with fixed viewpoints")
    print("=======================================================")

    print("Setting configurations...")
    cfg = options.get_arguments()

    EXPERIMENT = f"{cfg.model}_{cfg.experiment}"
    MODEL_PATH = f"models/{EXPERIMENT}"

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

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
    scheduler = utils.make_lr_scheduler(cfg, optimizer)

    print("Create logger")
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        logger.addHandler(logging.FileHandler(filename=f"logs/{EXPERIMENT}.log"))

    print("Create tensorboard logger")
    writer = SummaryWriter(comment="_"+EXPERIMENT)

    def on_after_epoch(model, df_hist, images, epoch):
        utils.save_best_model(MODEL_PATH, model, df_hist)
        utils.log_hist(logger, df_hist)
        utils.write_on_board_losses_stg1(writer, df_hist)
        utils.write_on_board_images_stg1(writer, images, epoch)

    trainer = Trainer_stg1(cfg, dataloaders, criterions, on_after_epoch)

    hist = trainer.train(model, optimizer, scheduler)
    hist.to_csv(f"logs/{EXPERIMENT}.csv", index=False)

    writer.close()
