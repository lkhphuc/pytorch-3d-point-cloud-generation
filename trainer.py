import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import transform
from utils import make_grid


class Trainer_stg1:
    def __init__(self, cfg, data_loaders, criterions, on_after_epoch=None):
        self.cfg = cfg
        self.data_loaders = data_loaders
        self.l1 = criterions[0]
        self.sigmoid_bce = criterions[1]
        self.history = []
        self.on_after_epoch = on_after_epoch

    def train(self, model, optimizer, scheduler):
        print("======= TRAINING START =======")

        for epoch in range(self.cfg.startEpoch, self.cfg.endEpoch):
            print(f"Epoch {epoch}:")

            lr = None
            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_lr()[0]

            train_epoch_loss = self._train_on_epoch(model, optimizer, scheduler)
            val_epoch_loss = self._val_on_epoch(model)

            hist = {
                'epoch': epoch,
                'train_loss_XYZ': train_epoch_loss["epoch_loss_XYZ"],
                'train_loss_mask': train_epoch_loss["epoch_loss_mask"],
                'train_loss': train_epoch_loss["epoch_loss"],
                'val_loss_XYZ': val_epoch_loss["epoch_loss_XYZ"],
                'val_loss_mask': val_epoch_loss["epoch_loss_mask"],
                'val_loss': val_epoch_loss["epoch_loss"],
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                images = self._make_images_board(model, self.data_loaders[1])
                self.on_after_epoch(model, pd.DataFrame(self.history),
                                    images, lr, epoch)

        print("======= TRAINING DONE =======")

        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer, scheduler):
        model.train()

        data_loader = self.data_loaders[0]
        running_loss_XYZ = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0

        for batch in data_loader:

            input_images = batch['inputImage'].float().to(self.cfg.device)
            depthGT = batch['depthGT'].float().to(self.cfg.device)
            maskGT = batch['maskGT'].float().to(self.cfg.device)

            # ------ define ground truth------
            # Shape: [H,W]
            XGT, YGT = torch.meshgrid([
                torch.arange(self.cfg.outH),
                torch.arange(self.cfg.outW)])
            XGT, YGT = XGT.float(), YGT.float()
            # Shape [2V,H,W]
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]),
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0)
            # Shape: [1, 2V, H, W] (Expand to new dim)
            XYGT = torch.cat([XYGT[None, :]
                              for i in range(depthGT.size(0))], dim=0)\
                        .to(self.cfg.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN * 3, :,  :]
                mask = (maskLogit > 0).byte()

                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT)
                loss_XYZ += self.l1(depth.masked_select(mask),
                                    depthGT.masked_select(mask))
                loss_mask = self.sigmoid_bce(maskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_XYZ

                # Update weights
                loss.backward()
                # True Weight decay
                if self.cfg.trueWD is not None:
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.data.add_(
                                -self.cfg.trueWD * group['lr'], param.data)
                optimizer.step()

            if scheduler: scheduler.step()

            running_loss_XYZ += loss_XYZ.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_XYZ = running_loss_XYZ / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        print(f"\tTrain loss: {epoch_loss}")

        return {"epoch_loss_XYZ": epoch_loss_XYZ,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }

    def _val_on_epoch(self, model):
        model.eval()

        data_loader = self.data_loaders[1]
        running_loss_XYZ = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0

        for batch in data_loader:

            input_images = batch['inputImage'].float().to(self.cfg.device)
            depthGT = batch['depthGT'].float().to(self.cfg.device)
            maskGT = batch['maskGT'].float().to(self.cfg.device)

            # ------ define ground truth------
            # Shape: [H,W]
            XGT, YGT = torch.meshgrid([
                torch.arange(self.cfg.outH),
                torch.arange(self.cfg.outW)])
            XGT, YGT = XGT.float(), YGT.float()
            # Shape [V,H,W]
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]),
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0)
            # Shape: [1, 2V, H, W] (Expand to new dim)
            XYGT = torch.cat([XYGT[None, :]
                              for i in range(depthGT.size(0))], dim=0)\
                        .to(self.cfg.device)

            with torch.set_grad_enabled(False):
                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN*3,:,:]
                mask = (maskLogit > 0).byte()

                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT)
                loss_XYZ += self.l1(
                    depth.masked_select(mask), depthGT.masked_select(mask))
                loss_mask = self.sigmoid_bce(maskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_XYZ

            running_loss_XYZ += loss_XYZ.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_XYZ = running_loss_XYZ / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        print(f"\tVal loss: {epoch_loss}")

        return {"epoch_loss_XYZ": epoch_loss_XYZ,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }

    def _make_images_board(self, model, dataloader):
        batch = next(iter(dataloader))
        input_images = batch['inputImage'].float().to(self.cfg.device)
        depthGT = batch['depthGT'].float().to(self.cfg.device)
        maskGT = batch['maskGT'].float().to(self.cfg.device)

        with torch.set_grad_enabled(False):
            XYZ, maskLogit = model(input_images)
            XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
            depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN * 3, :,  :]
            mask = (maskLogit > 0).float()

        return {'RGB': make_grid(input_images[:64]),
                'depth': make_grid(1-depth[:64, 0:1, :, :]),
                'depth_mask': make_grid(((1-depth)*mask)[:64, 0:1, :, :]),
                'depthGT': make_grid(1-depthGT[:64, 0:1, :, :]),
                'mask': make_grid(torch.sigmoid(maskLogit[:64, 0:1,:, :])),
                'maskGT': make_grid(maskGT[:64, 0:1, :, :]),
                }

    def findLR(self, model, optimizer, writer,
               start_lr=1e-7, end_lr=10, num_iters=50):

        model.train()
        data_loader = self.data_loaders[0]

        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iters)
        losses = []

        for lr in lrs:
            # Update LR
            for group in optimizer.param_groups:
                group['lr'] = lr

            batch = next(iter(data_loader))
            input_images = batch['inputImage'].float().to(self.cfg.device)
            depthGT = batch['depthGT'].float().to(self.cfg.device)
            maskGT = batch['maskGT'].float().to(self.cfg.device)

            # ------ define ground truth------
            # Shape: [H,W]
            XGT, YGT = torch.meshgrid([
                torch.arange(self.cfg.outH),
                torch.arange(self.cfg.outW)])
            XGT, YGT = XGT.float(), YGT.float()
            # Shape [V,H,W]
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]),
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0)
            # Shape: [1, 2V, H, W] (Expand to new dim)
            XYGT = torch.cat([XYGT[None, :]
                              for i in range(depthGT.size(0))], dim=0)\
                        .to(self.cfg.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN * 3, :,  :]
                mask = (maskLogit > 0).byte()

                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT)
                loss_XYZ += self.l1(depth.masked_select(mask),
                                    depthGT.masked_select(mask))
                loss_mask = self.sigmoid_bce(maskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_XYZ

                # Update weights
                loss.backward()
                # True Weight decay

                if self.cfg.trueWD is not None:
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.data = param.data.add(
                                -self.cfg.trueWD * group['lr'], param.data)
                optimizer.step()

            losses.append(loss.item())

        fig, ax = plt.subplots()
        ax.plot(lrs, losses)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('loss')
        ax.set_xscale('log')
        writer.add_figure('findLR', fig)

class Trainer_stg2:
    def __init__(self, cfg, data_loaders, criterions, on_after_epoch=None):
        self.cfg = cfg
        self.data_loaders = data_loaders
        self.l1 = criterions[0]
        self.sigmoid_bce = criterions[1]
        self.history = []
        self.on_after_epoch = on_after_epoch

    def train(self, model, optimizer, scheduler):
        print("======= TRAINING START =======")

        for epoch in range(self.cfg.startEpoch, self.cfg.endEpoch):
            print(f"Epoch {epoch}:")
            if scheduler: scheduler.step()
            train_epoch_loss = self._train_on_epoch(model, optimizer)
            val_epoch_loss = self._val_on_epoch(model)

            hist = {
                'epoch': epoch,
                'train_loss_depth': train_epoch_loss["epoch_loss_depth"],
                'train_loss_mask': train_epoch_loss["epoch_loss_mask"],
                'train_loss': train_epoch_loss["epoch_loss"],
                'val_loss_depth': val_epoch_loss["epoch_loss_depth"],
                'val_loss_mask': val_epoch_loss["epoch_loss_mask"],
                'val_loss': val_epoch_loss["epoch_loss"],
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                images = self._make_images_board(model, self.data_loaders[1])
                self.on_after_epoch(model, pd.DataFrame(self.history),
                                    images, epoch)

        print("======= TRAINING DONE =======")

        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer, scheduler):
        model.train()

        data_loader = self.data_loaders[0]
        running_loss_depth = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0

        for batch in data_loader:

            input_images = batch['inputImage'].float().to(self.cfg.device)
            renderTrans = batch['targetTrans'].float().to(self.cfg.device)
            depthGT = batch['depthGT'].float().to(self.cfg.device)
            maskGT = batch['maskGT'].float().to(self.cfg.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                XYZ, maskLogit = model(input_images)
                mask = (maskLogit > 0).byte()

                # ------ build transformer ------
                fuseTrans = F.normalize(self.cfg.fuseTrans, p=2, dim=1)
                XYZid, ML = transform.fuse3D(
                    self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
                newDepth, newMaskLogit, collision = transform.render2D(
                    self.cfg, XYZid, ML, renderTrans)  # [B,N,H,W,1]

                # ------ Compute loss ------
                loss_depth = self.l1(newDepth.masked_select(collision==1),
                                     depthGT.masked_select(collision==1))
                loss_mask = self.sigmoid_bce(newMaskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_depth

                # Update weights
                loss.backward()
                # True Weight decay

                if self.cfg.trueWD is not None:
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.data = param.data.add(
                                -self.cfg.trueWD * group['lr'], param.data)
                optimizer.step()

            running_loss_depth += loss_depth.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_depth = running_loss_depth / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        print(f"\tTrain loss: {epoch_loss}")

        return {"epoch_loss_depth": epoch_loss_depth,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }

    def _val_on_epoch(self, model):
        model.eval()

        data_loader = self.data_loaders[1]
        running_loss_depth = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0

        for batch in data_loader:

            input_images = batch['inputImage'].float().to(self.cfg.device)
            renderTrans = batch['targetTrans'].float().to(self.cfg.device)
            depthGT = batch['depthGT'].float().to(self.cfg.device)
            maskGT = batch['maskGT'].float().to(self.cfg.device)

            with torch.set_grad_enabled(False):
                XYZ, maskLogit = model(input_images)
                mask = (maskLogit > 0).byte()

                # ------ build transformer ------
                fuseTrans = F.normalize(self.cfg.fuseTrans, p=2, dim=1)
                XYZid, ML = transform.fuse3D(
                    self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
                newDepth, newMaskLogit, collision = transform.render2D(
                    self.cfg, XYZid, ML, renderTrans)  # [B,N,H,W,1]

                # ------ Compute loss ------
                loss_depth = self.l1(newDepth.masked_select(collision==1),
                                     depthGT.masked_select(collision==1))
                loss_mask = self.sigmoid_bce(newMaskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_depth

            running_loss_depth += loss_depth.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_depth = running_loss_depth / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        print(f"\tVal loss: {epoch_loss}")

        return {"epoch_loss_depth": epoch_loss_depth,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }

    def _make_images_board(self, model, dataloader):
        batch = next(iter(dataloader))
        input_images = batch['inputImage'].float().to(self.cfg.device)
        renderTrans = batch['targetTrans'].float().to(self.cfg.device)
        depthGT = batch['depthGT'].float().to(self.cfg.device)
        maskGT = batch['maskGT'].float().to(self.cfg.device)

        with torch.set_grad_enabled(False):
            XYZ, maskLogit = model(input_images)
            mask = (maskLogit > 0).byte()

            # ------ build transformer ------
            fuseTrans = F.normalize(self.cfg.fuseTrans, p=2, dim=1)
            XYZid, ML = transform.fuse3D(
                self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
            newDepth, newMaskLogit, collision = transform.render2D(
                self.cfg, XYZid, ML, renderTrans)  # [B,N,1,H,W]


        return {'RGB': make_grid(input_images[:16]),
                'depth': make_grid(
                    ((1 - newDepth) * (collision==1).float())[:16, 0, 0:1, :, :]),
                'depthGT': make_grid(1-depthGT[:16, 0, 0:1, :, :]),
                'mask': make_grid(
                    torch.sigmoid(maskLogit[:16, 0:1,:, :])),
                'mask_rendered': make_grid(
                    torch.sigmoid(newMaskLogit[:16, 0, 0:1,:, :])),
                'maskGT': make_grid(maskGT[:16, 0, 0:1, :, :]),
                }
