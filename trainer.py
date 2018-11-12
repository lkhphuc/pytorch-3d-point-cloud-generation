import pandas as pd
import torch


class Trainer_stg1:
    def __init__(self, cfg, data_loaders, criterions, device,on_after_epoch=None):
        self.cfg = cfg
        self.data_loaders = data_loaders
        self.l1 = criterions[0]
        self.ce = criterions[1]
        self.device = device
        self.history = []
        self.on_after_epoch = on_after_epoch

    def train(self, model, optimizer, scheduler):
        print("======= TRAINING START =======")

        for epoch in range(self.cfg.startEpoch, self.cfg.endEpoch):
            train_epoch_loss = self._train_on_epoch(model, optimizer, scheduler)
            val_epoch_loss = self._val_on_epoch(model, optimizer)

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
                self.on_after_epoch(model, pd.DataFrame(self.history))

        print("======= TRAINING DONE =======")
        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer, scheduler):
        model.train()

        data_loader = self.data_loaders[0]
        running_loss_XYZ = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0

        for batch in data_loader:

            input_images = batch['inputImage'].float().to(self.device)
            depthGT = batch['depthGT'].float().to(self.device)
            maskGT = batch['maskGT'].float().to(self.device)

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
            XYGT = torch.cat(
                [XYGT[None, :] for i in range(depthGT.size(0))], dim=0)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN * 3, :,  :]
                mask = (maskLogit > 0).long()

                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT) + self.l1(depth[mask], depthGT[mask])
                loss_mask = self.ce(maskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_XYZ

                # Update weights
                loss.backward()
                optimizer.step()

            running_loss_XYZ += loss_XYZ.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_XYZ = running_loss_XYZ / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)


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

            input_images = batch['inputImage'].float().to(self.device)
            depthGT = batch['depthGT'].float().to(self.device)
            maskGT = batch['maskGT'].float().to(self.device)

            # ------ define ground truth------
            # Shape: [H,W]
            XGT, YGT = torch.meshgrid([
                torch.arange(self.cfg.outH),
                torch.arange(self.cfg.outW)])
            XGT, YGT = XGT.float(), YGT.float()
            # Shape [V,H,W]
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]),
                YGT.repeat([self.cfg.outViewN, 1,1])], dim=0)
            # Shape: [1, 2V, H, W] (Expand to new dimension)
            XYGT = XYGT[None, :]

            with torch.set_grad_enabled(False):
                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.outViewN * 2:self.outViewN * 3, :,  :]
                mask = (maskLogit > 0).float()

                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT) + self.l1(depth[mask], depthGT[mask])
                loss_mask = self.ce(maskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_XYZ

            running_loss_XYZ += loss_XYZ.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_XYZ = running_loss_XYZ / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        return {"epoch_loss_XYZ": epoch_loss_XYZ,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }
