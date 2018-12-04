import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch

import transform
import utils


class TrainerStage1:
    '''Train loop and evaluation for stage 1 Structure generator'''
    def __init__(self, cfg, data_loaders, criterions,
                 on_after_epoch=None, on_after_batch=None):
        self.cfg = cfg
        self.data_loaders = data_loaders
        self.l1 = criterions[0]
        self.sigmoid_bce = criterions[1]
        self.iteration = 0
        self.epoch = 0
        self.history = []
        self.on_after_epoch = on_after_epoch
        self.on_after_batch = on_after_batch

    def train(self, model, optimizer, scheduler):
        print("======= TRAINING START =======")

        for self.epoch in range(self.cfg.startEpoch, self.cfg.endEpoch):
            print(f"Epoch {self.epoch}:")

            train_epoch_loss = self._train_on_epoch(model, optimizer)
            val_epoch_loss = self._val_on_epoch(model)

            hist = {
                'epoch': self.epoch,
                'train_loss_XYZ': train_epoch_loss["epoch_loss_XYZ"],
                'train_loss_mask': train_epoch_loss["epoch_loss_mask"],
                'train_loss': train_epoch_loss["epoch_loss"],
                'val_loss_XYZ': val_epoch_loss["epoch_loss_XYZ"],
                'val_loss_mask': val_epoch_loss["epoch_loss_mask"],
                'val_loss': val_epoch_loss["epoch_loss"],
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                images = self._make_images_board(model)
                self.on_after_epoch(model, pd.DataFrame(self.history),
                                    images, self.epoch)

        print("======= TRAINING DONE =======")
        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer):
        model.train()

        data_loader = self.data_loaders[0]
        running_loss_XYZ = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0

        for self.iteration, batch in enumerate(data_loader, self.iteration):
            input_images, depthGT, maskGT = utils.unpack_batch_fixed(batch, self.cfg.device)
            # ------ define ground truth------
            XGT, YGT = torch.meshgrid([
                torch.arange(self.cfg.outH), # [H,W]
                torch.arange(self.cfg.outW)]) # [H,W]
            XGT, YGT = XGT.float(), YGT.float()
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]), 
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0) #[2V,H,W]
            XYGT = XYGT.unsqueeze(dim=0).to(self.cfg.device) # [1,2V,H,W] 

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

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

                # ------ Update weights ------
                loss.backward()
                # True Weight decay
                if self.cfg.trueWD is not None:
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.data.add_(
                                -self.cfg.trueWD * group['lr'], param.data)
                optimizer.step()

            if self.on_after_batch is not None:
                if self.cfg.lrSched.lower() in "cyclical":
                    self.on_after_batch(self.iteration)
                else: self.on_after_batch(self.epoch)

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
            input_images, depthGT, maskGT = utils.unpack_batch_fixed(batch, self.cfg.device)
            # ------ define ground truth------
            XGT, YGT = torch.meshgrid([
                torch.arange(self.cfg.outH), # [H,W]
                torch.arange(self.cfg.outW)]) # [H,W]
            XGT, YGT = XGT.float(), YGT.float()
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]), 
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0) #[2V,H,W]
            XYGT = XYGT.unsqueeze(dim=0).to(self.cfg.device) # [1,2V,H,W] 

            with torch.set_grad_enabled(False):
                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN*3,:,:]
                mask = (maskLogit > 0).byte()
                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT)
                loss_XYZ += self.l1(depth.masked_select(mask),
                                    depthGT.masked_select(mask))
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

    def _make_images_board(self, model):
        model.eval()
        num_imgs = 64

        batch = next(iter(self.data_loaders[1]))
        input_images, depthGT, maskGT = utils.unpack_batch_fixed(batch, self.cfg.device)

        with torch.set_grad_enabled(False):
            XYZ, maskLogit = model(input_images)
            XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
            depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN * 3, :,  :]
            mask = (maskLogit > 0).float()

        return {'RGB': utils.make_grid(input_images[:num_imgs]),
                'depth': utils.make_grid(1-depth[:num_imgs, 0:1, :, :]),
                'depth_mask': utils.make_grid(
                    ((1-depth)*mask)[:num_imgs, 0:1, :, :]),
                'depthGT': utils.make_grid(
                    1-depthGT[:num_imgs, 0:1, :, :]),
                'mask': utils.make_grid(
                    torch.sigmoid(maskLogit[:num_imgs, 0:1,:, :])),
                'maskGT': utils.make_grid(maskGT[:num_imgs, 0:1, :, :]),
                }

    def findLR(self, model, optimizer, writer,
               start_lr=1e-7, end_lr=10, num_iters=50):
        model.train()

        losses = []
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iters)

        for lr in lrs:
            # Update LR
            for group in optimizer.param_groups: group['lr'] = lr

            batch = next(iter(self.data_loaders[0]))
            input_images, depthGT, maskGT = utils.unpack_batch_fixed(batch, self.cfg.device)
            # ------ define ground truth------
            XGT, YGT = torch.meshgrid([torch.arange(self.cfg.outH), # [H,W]
                                       torch.arange(self.cfg.outW)]) # [H,W]
            XGT, YGT = XGT.float(), YGT.float()
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]), 
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0) #[2V,H,W]
            XYGT = XYGT.unsqueeze(dim=0).to(self.cfg.device) #[1,2V,H,W]

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

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


class TrainerStage2:
    '''Train loop and evaluation for stage 2 with pseudo-renderer'''
    def __init__(self, cfg, data_loaders, criterions,
                 on_after_epoch=None, on_after_batch=None):
        self.cfg = cfg
        self.data_loaders = data_loaders
        self.l1 = criterions[0]
        self.sigmoid_bce = criterions[1]
        self.iteration = 0
        self.epoch = 0
        self.history = []
        self.on_after_epoch = on_after_epoch
        self.on_after_batch = on_after_batch

    def train(self, model, optimizer, scheduler):
        print("======= TRAINING START =======")

        for self.epoch in range(self.cfg.startEpoch, self.cfg.endEpoch):
            print(f"Epoch {self.epoch}:")

            train_epoch_loss = self._train_on_epoch(model, optimizer)
            val_epoch_loss = self._val_on_epoch(model)

            hist = {
                'epoch': self.epoch,
                'train_loss_depth': train_epoch_loss["epoch_loss_depth"],
                'train_loss_mask': train_epoch_loss["epoch_loss_mask"],
                'train_loss': train_epoch_loss["epoch_loss"],
                'val_loss_depth': val_epoch_loss["epoch_loss_depth"],
                'val_loss_mask': val_epoch_loss["epoch_loss_mask"],
                'val_loss': val_epoch_loss["epoch_loss"],
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                images = self._make_images_board(model)
                self.on_after_epoch(
                    model, pd.DataFrame(self.history),
                    images, self.epoch, self.cfg.saveEpoch)

        print("======= TRAINING DONE =======")
        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer):
        model.train()

        data_loader = self.data_loaders[0]
        running_loss_depth = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0
        fuseTrans = self.cfg.fuseTrans

        for self.iteration, batch in enumerate(data_loader, self.iteration):
            input_images, renderTrans, depthGT, maskGT = utils.unpack_batch_novel(batch, self.cfg.device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                XYZ, maskLogit = model(input_images)
                # ------ build transformer ------
                XYZid, ML = transform.fuse3D(
                    self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
                newDepth, newMaskLogit, collision = transform.render2D(
                    self.cfg, XYZid, ML, renderTrans)  # [B,N,H,W,1]
                # ------ Compute loss ------
                loss_depth = self.l1(
                    newDepth.masked_select(collision==1),
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

            if self.on_after_batch is not None:
                if self.cfg.lrSched.lower() in "cyclical":
                    self.on_after_batch(self.iteration)
                else: self.on_after_batch(self.epoch)

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
        fuseTrans = self.cfg.fuseTrans

        for batch in data_loader:
            input_images, renderTrans, depthGT, maskGT = utils.unpack_batch_novel(batch, self.cfg.device)

            with torch.set_grad_enabled(False):
                XYZ, maskLogit = model(input_images)
                # ------ build transformer ------
                XYZid, ML = transform.fuse3D(
                    self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
                newDepth, newMaskLogit, collision = transform.render2D(
                    self.cfg, XYZid, ML, renderTrans)  # [B,N,H,W,1]
                # ------ Compute loss ------
                loss_depth = self.l1(
                    newDepth.masked_select(collision==1),
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

    def _make_images_board(self, model):
        model.eval()
        num_imgs = 64
        fuseTrans = self.cfg.fuseTrans

        batch = next(iter(self.data_loaders[1]))
        input_images, renderTrans, depthGT, maskGT = utils.unpack_batch_novel(batch, self.cfg.device)

        with torch.set_grad_enabled(False):
            XYZ, maskLogit = model(input_images)
            # ------ build transformer ------
            XYZid, ML = transform.fuse3D(
                self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
            newDepth, newMaskLogit, collision = transform.render2D(
                self.cfg, XYZid, ML, renderTrans)  # [B,N,1,H,W]

        return {'RGB': utils.make_grid( input_images[:num_imgs]),
                'depth': utils.make_grid(
                    ((1-newDepth)*(collision==1).float())[:num_imgs, 0, 0:1, :, :]),
                'depthGT': utils.make_grid(
                    1-depthGT[:num_imgs, 0, 0:1, :, :]),
                'mask': utils.make_grid(
                    torch.sigmoid(maskLogit[:num_imgs, 0:1,:, :])),
                'mask_rendered': utils.make_grid(
                    torch.sigmoid(newMaskLogit[:num_imgs, 0, 0:1, :, :])),
                'maskGT': utils.make_grid(
                    maskGT[:num_imgs, 0, 0:1, :, :]),
                }

    def findLR(self, model, optimizer, writer,
               start_lr=1e-7, end_lr=10, num_iters=50):

        model.train()

        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iters)
        losses = []
        fuseTrans = self.cfg.fuseTrans

        for lr in lrs:
            # Update LR
            for group in optimizer.param_groups:
                group['lr'] = lr

            batch = next(iter(self.data_loaders[0]))
            input_images, renderTrans, depthGT, maskGT = utils.unpack_batch_novel(batch, self.cfg.device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                XYZ, maskLogit = model(input_images)
                # ------ build transformer ------
                XYZid, ML = transform.fuse3D(
                    self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
                newDepth, newMaskLogit, collision = transform.render2D(
                    self.cfg, XYZid, ML, renderTrans)  # [B,N,H,W,1]
                # ------ Compute loss ------
                loss_depth = self.l1(
                    newDepth.masked_select(collision==1),
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
                                -self.cfg.trueWD * group['lr'],
                                param.data)
                optimizer.step()

            losses.append(loss.item())

        fig, ax = plt.subplots()
        ax.plot(lrs, losses)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('loss')
        ax.set_xscale('log')
        writer.add_figure('findLR', fig)


class Validator:
    '''Perform Validation on the trained Structure generator'''
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.device = cfg.device
        self.dataset = dataset
        self.history = []
        self.CADs = dataset.CADs
        self.result_path = f"results/{cfg.model}_{cfg.experiment}"

    def eval(self, model):
        print("======= EVALUATION START =======")

        fuseTrans = self.cfg.fuseTrans
        for i in range(len(self.dataset)):
            cad = self.dataset[i]
            input_images = torch.from_numpy(cad['image_in'])\
                                .permute((0,3,1,2))\
                                .float().to(self.cfg.device)
            points24 = np.zeros([self.cfg.inputViewN, 1], dtype=np.object)

            XYZ, maskLogit = model(input_images)
            mask = (maskLogit > 0).float()
            # ------ build transformer ------
            XYZid, ML = transform.fuse3D(
                self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
            
            XYZid, ML = XYZid.permute([0, 2, 1]), ML.squeeze()
            for a in range(self.cfg.inputViewN):
                xyz = XYZid[a] #[VHW, 3]
                ml = ML[a] #[VHW]
                points24[a, 0] = (xyz[ml > 0]).detach().cpu().numpy()

            pointMeanN = np.array([len(p) for p in points24[:, 0]]).mean()
            scipy.io.savemat(
                f"{self.result_path}/{self.CADs[i]}.mat", 
                {"image": cad["image_in"], "pointcloud": points24})

            print(f"{pointMeanN:.2f} points save to {self.result_path}/{self.CADs[i]}.mat")
            self.history.append(
                {"cad": self.CADs[i], "average points": pointMeanN})

        print("======= EVALUATION DONE =======")
        return pd.DataFrame(self.history)

    def eval_dist(self):
        print("======= EVALUATION START =======")
        CADN = len(self.CADs)

        pred2GT_all = np.ones([CADN, self.cfg.inputViewN]) * np.inf
        GT2pred_all = np.ones([CADN, self.cfg.inputViewN]) * np.inf
        with torch.set_grad_enabled(False):
            for m, cad in enumerate(self.CADs):
                # load GT
                obj = scipy.io.loadmat(f"{self.cfg.path}/{self.cfg.category}_testGT/{cad}.mat")
                Vgt = torch.from_numpy(np.concatenate([obj["V"], obj["Vd"]], axis=0)).to(self.device).float()
                VgtN = len(Vgt)
                # load prediction
                Vpred24 = scipy.io.loadmat(f"{self.result_path}/{cad}.mat")["pointcloud"][:, 0]
                assert (len(Vpred24) == self.cfg.inputViewN)

                for a in range(self.cfg.inputViewN):
                    Vpred = torch.from_numpy(Vpred24[a]).to(self.device).float()
                    VpredN = len(Vpred)
                    # rotate CAD model to be in consistent coordinates
                    Vpred[:, 1], Vpred[:, 2] = Vpred[:, 2], -Vpred[:, 1]
                    # compute test error in both directions
                    pred2GT_all[m, a] = self._computeTestError(Vpred, Vgt, type="pred->GT")
                    GT2pred_all[m, a] = self._computeTestError(Vgt, Vpred, type="GT->pred")

                info = {"cad": cad,
                        "pred->GT": pred2GT_all[m].mean()*100,
                        "GT->pred": GT2pred_all[m].mean()*100,}
                print(info)
                self.history.append(info)

        print("======= EVALUATION DONE =======")
        return pd.DataFrame(self.history)

    def _computeTestError(self, Vs, Vt, type):
        """compute test error for one prediction"""
        VsN, VtN = len(Vs), len(Vt)
        if type == "pred->GT":
            evalN, VsBatchSize, VtBatchSize = min(VsN, 200), 200, 100000
        if type == "GT->pred":
            evalN, VsBatchSize, VtBatchSize = min(VsN, 200), 200, 40000
        # randomly sample 3D points to evaluate (for speed)
        randIdx = np.random.permutation(VsN)[:evalN]
        Vs_eval = Vs[randIdx]
        minDist_eval = np.ones([evalN]) * np.inf
        # for batches of source vertices
        VsBatchN = int(np.ceil(evalN / VsBatchSize))
        VtBatchN = int(np.ceil(VtN / VtBatchSize))
        for b in range(VsBatchN):
            VsBatch = Vs_eval[b * VsBatchSize:(b + 1) * VsBatchSize]
            minDist_batch = np.ones([len(VsBatch)]) * np.inf
            for b2 in range(VtBatchN):
                VtBatch = Vt[b2 * VtBatchSize:(b2 + 1) * VtBatchSize]
                _, minDist = self._projection(VsBatch, VtBatch)
                minDist = minDist.detach().cpu().numpy()
                minDist_batch = np.minimum(minDist_batch, minDist)
            minDist_eval[b * VsBatchSize:(b + 1) * VsBatchSize] = minDist_batch
        return np.mean(minDist_eval)

    def _projection(self, Vs, Vt):
        '''compute projection from source to target'''
        VsN = Vs.size(0)
        VtN = Vt.size(0)
        diff = Vt[None, :, :] - Vs[:, None, :]
        dist = (diff**2).sum(dim=2).sqrt()
        idx = torch.argmin(dist, dim=1)
        # proj = Vt_rep[np.arange(VsN), idx, :]
        proj = None
        minDist = dist[np.arange(VsN), idx]

        return proj, minDist
