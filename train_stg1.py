import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

import data
import options
import utils
from trainer import Trainer_stg1


def save_best_model(model_path, model, df_hist):
    if df_hist['val_loss'].tail(1).iloc[0] <= df_hist['val_loss'].min():
        torch.save(model.state_dict(), f"{model_path}/best.pth")


def write_on_board(writer, df_hist):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars('loss', {
        'train': row.train_loss,
        'val': row.val_loss,
    }, row.epoch)


def log_hist(logger, df_hist):
    last = df_hist.tail(1)
    best = df_hist.sort_values('val_loss').head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary['name'] = ['Last', 'Best']
    logger.debug(summary[['name', 'epoch', 'train_loss', 'val_loss']])
    logger.debug('')

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
    ce_loss = nn.CrossEntropyLoss()
    criterions = [l1_loss, ce_loss]

    print("build Structure Generator")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = utils.build_structure_generator(cfg).to(device)

    print("create optimizer and scheduler")
    optimizer = utils.make_optimizer(cfg, model)
    scheduler = utils.make_lr_scheduler(cfg, optimizer)

    def on_after_epoch(model, df_hist):
        save_best_model(MODEL_PATH, model, df_hist)
        write_on_board(writer, df_hist)
        log_hist(logger, df_hist)

    trainer = Trainer_stg1(cfg, dataloader, criterions, device, on_after_epoch)

    hist = trainer.train(model, optimizer, scheduler)
    hist.to_csv(f"{EXPERIMENT}/hist.csv", index=False)

    writer.close()
