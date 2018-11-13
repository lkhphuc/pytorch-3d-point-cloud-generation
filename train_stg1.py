import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

import data
import options
import utils
from trainer import Trainer_stg1


if __name__ == "__main__":

    print("=======================================================")
    print("Pretrain structure generator with fixed viewpoints")
    print("=======================================================")

    print("setting configurations...")
    cfg = options.get_arguments()

    EXPERIMENT = f"{cfg.model}_{cfg.experiment}"
    MODEL_PATH = f"models/{EXPERIMENT}"

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(
            logging.FileHandler(filename=f"logs/{EXPERIMENT}.log"))

    writer = SummaryWriter(comment="_"+EXPERIMENT)

    print("create Dataloader")
    tfms = transforms.ToTensor()
    dataloader = utils.make_data_fixed(cfg, tfms)

    print("define losses")
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    criterions = [l1_loss, bce_loss]

    print("build Structure Generator")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = utils.build_structure_generator(cfg).to(device)

    print("create optimizer and scheduler")
    optimizer = utils.make_optimizer(cfg, model)
    scheduler = utils.make_lr_scheduler(cfg, optimizer)

    def on_after_epoch(model, df_hist, images, epoch):
        utils.save_best_model(MODEL_PATH, model, df_hist)
        utils.write_on_board_losses(writer, df_hist)
        utils.write_on_board_images(writer, images, epoch)
        utils.log_hist(logger, df_hist)

    trainer = Trainer_stg1(cfg, dataloader, criterions, device, on_after_epoch)

    hist = trainer.train(model, optimizer, scheduler)
    hist.to_csv(f"{EXPERIMENT}/hist.csv", index=False)

    writer.close()
