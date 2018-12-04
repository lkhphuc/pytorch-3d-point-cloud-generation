import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse3D(cfg, XYZ, maskLogit, fuseTrans):
    """Fuse multiple depth views into a 3D point cloud representation
    Args:
    output of structure generator
        XYZ (tensor:[B,3V,H,W]): x,z,y of V different fixed views
        maskLogit (tensor:[B,V,H,W]): mask of V different fixed views
    output of render module
        fuseTrans (Tensor:[V, 4])
    Return:
        XYZid (Tensor [B,3,VHW]): point clouds
        ML (Tensor [B,1,VHW]): depth stack
     """
    # 2D to 3D coordinate transformation
    invKhom = cfg.Khom2Dto3D.inverse() # [4x4]
    invKhomTile = invKhom.repeat([cfg.batchSize, cfg.outViewN, 1, 1]) #[B,V,4x4]

    # viewpoint rigid transformation
    q_view = fuseTrans # [V, 4]
    t_view = torch.Tensor([0, 0, -cfg.renderDepth]) \
                  .repeat([cfg.outViewN, 1]).to(cfg.device) # [V,3]
    RtHom_view = transParamsToHomMatrix(q_view, t_view) # [V,4,4]

    RtHomTile_view = RtHom_view.unsqueeze(0).repeat([cfg.batchSize, 1, 1, 1])
    invRtHomTile_view = RtHomTile_view.inverse() # [B,V,4,4]

    # effective transformation
    RtHomTile = torch.matmul(invRtHomTile_view, invKhomTile) # [B,V,4,4]
    RtTile = RtHomTile[:, :, :3, :] # [B,V,3,4]

    # transform depth stack
    ML = maskLogit.clone().reshape([cfg.batchSize, 1, -1]) # [B,1,VHW]
    XYZhom = get3DhomCoord(XYZ, cfg)  # [B,V,4,HW]
    XYZid = torch.matmul(RtTile, XYZhom)  # [B,V,3,HW]

    # fuse point clouds
    XYZid = XYZid.permute([0, 2, 1, 3]).reshape([cfg.batchSize, 3, -1]) #[B,3,VHW]

    return XYZid, ML

def render2D(cfg, XYZid, ML, renderTrans):  # [B,1,VHW]
    """Render 2D depth views from fused 3D point clouds
    Args:
        XYZid (Tensor [B,3,VHW]): point clouds
        ML (Tensor [B,1,BHW]): depth stack
        renderTrans (Tensor [B, novelN, 4])
    Return: (Tensor [B,N,1,H,W])
        newDepth: depth map for novel views
        newMaskLogit: mask logit for depth views
        collision
    """
    offsetDepth, offsetMaskLogit = 10.0, 1.0

    # target rigid transformation
    q_target = renderTrans.reshape([cfg.batchSize * cfg.novelN, 4]) #[BN,4]
    t_target = torch.Tensor([0, 0, -cfg.renderDepth]) \
                    .repeat([cfg.batchSize * cfg.novelN, 1]) \
                    .float().to(cfg.device) # [BN,3]
    RtHom_target = transParamsToHomMatrix(q_target, t_target) \
                    .reshape([cfg.batchSize, cfg.novelN, 4, 4]) # [B,N,4,4]

    # 3D to 2D coordinate transformation
    mul = torch.Tensor([[cfg.upscale], [cfg.upscale], [1], [1]])
    KupHom = cfg.Khom3Dto2D * mul.to(cfg.device) #[4,4]
    KupHomTile = KupHom.repeat([cfg.batchSize, cfg.novelN, 1, 1]) #[B,N,4,4]

    # effective transformation
    RtHomTile = torch.matmul(KupHomTile, RtHom_target) # [B,N,4,4]
    RtTile = RtHomTile[:, :, :3, :] # [B,N,3,4]

    # transform depth stack
    XYZidHom = get3DhomCoord2(XYZid, cfg) # [B,4,VHW]
    XYZidHomTile = XYZidHom.unsqueeze(dim=1).repeat([1, cfg.novelN, 1, 1]) # [B,N,4,VHW]
    XYZnew = torch.matmul(RtTile, XYZidHomTile) # [B,N,3,VHW]
    Xnew, Ynew, Znew = torch.split(XYZnew, 1, dim=2) # [B,N,1,VHW]

    # concatenate all viewpoints
    MLcat = ML.repeat([1, cfg.novelN, 1]).reshape([-1]) # [BNVHW]
    XnewCat = Xnew.reshape([-1]) # [BNVHW]
    YnewCat = Ynew.reshape([-1]) # [BNVHW]
    ZnewCat = Znew.reshape([-1]) # [BNVHW]
    batchIdxCat, novelIdxCat, _ = torch.meshgrid([
        torch.arange(cfg.batchSize),
        torch.arange(cfg.novelN),
        torch.arange(cfg.outViewN * cfg.outH * cfg.outW)
    ]) # [B,N,VHW]
    batchIdxCat = batchIdxCat.reshape([-1]).to(cfg.device) # [BNVHW]
    novelIdxCat = novelIdxCat.reshape([-1]).to(cfg.device) # [BNVHW]

    # apply in-range masks
    XnewCatInt = XnewCat.round().long() # [BNVHW]
    YnewCatInt = YnewCat.round().long() # [BNVHW]
    maskInside = (XnewCatInt >= 0) & (XnewCatInt < cfg.upscale * cfg.W) \
               & (YnewCatInt >= 0) & (YnewCatInt < cfg.upscale * cfg.H) # [BNVHW]
    valueInt = torch.stack(
        [XnewCatInt, YnewCatInt, batchIdxCat, novelIdxCat], dim=1) # [BNVHW,4]
    valueFloat = torch.stack(
        [1 / (ZnewCat + offsetDepth + 1e-8), MLcat], dim=1) # [BNVHW,2]
    insideInt = valueInt[maskInside] # [U,4]
    insideFloat = valueFloat[maskInside] # [U,2]
    _, MLnewValid = torch.unbind(insideFloat, dim=1) # [U]
    # apply visible masks
    maskExist = MLnewValid > 0 # [U]
    visInt = insideInt[maskExist] # [U',4]
    visFloat = insideFloat[maskExist] # [U',2]
    invisInt = insideInt[~maskExist] # [U-U',4]
    invisFloat = insideFloat[~maskExist] # [U-U',2]
    XnewVis, YnewVis, batchIdxVis, novelIdxVis = torch.unbind(visInt, dim=1) #[U']
    iZnewVis, MLnewVis = torch.unbind(visFloat, dim=1)  # [U']
    XnewInvis, YnewInvis, batchIdxInvis, novelIdxInvis = torch.unbind(invisInt, dim=1) # [U-U']
    _, MLnewInvis = torch.unbind(invisFloat, dim=1) # [U-U']

    # map to upsampled inverse depth and mask (visible)
    # scatterIdx = torch.stack(
    #     [batchIdxVis, novelIdxVis, YnewVis, XnewVis], dim=1)  # [U,4]
    upNewiZMLCnt = torch.zeros([cfg.batchSize, cfg.novelN, 3,
                                 cfg.H*cfg.upscale, cfg.W*cfg.upscale]
                                ).to(cfg.device) #[B,N,3,uH,uW]
    countOnes = torch.ones_like(iZnewVis)
    scatteriZMLCnt = torch.stack([iZnewVis, MLnewVis, countOnes], dim=1) #[U,3]
    # upNewiZMLCnt[scatterIdx[:,0],
    #              scatterIdx[:,1],
    #              :,
    #              scatterIdx[:,2],
    #              scatterIdx[:,3]] = scatteriZMLCnt
    upNewiZMLCnt[batchIdxVis,
                 novelIdxVis,
                 :,
                 YnewVis,
                 XnewVis] = scatteriZMLCnt
    upNewiZMLCnt = upNewiZMLCnt.reshape([cfg.batchSize * cfg.novelN,
                                         3,
                                         cfg.H * cfg.upscale,
                                         cfg.W * cfg.upscale])  # [BN,3,uH,uW]
    # downsample back to original size
    newiZMLCnt = F.adaptive_max_pool2d(
        upNewiZMLCnt, output_size=(cfg.H, cfg.W)) # [BN,3,H,W]
    newiZMLCnt = newiZMLCnt.reshape(
        [cfg.batchSize, cfg.novelN, 3, cfg.H, cfg.W])  # [B,N,3,H,W]
    newInvDepth, newMaskLogitVis, collision = torch.split(newiZMLCnt, 1, dim=2)  # [B,N,1,H,W]

    # map to upsampled inverse depth and mask (invisible)
    scatterIdx = torch.stack(
        [batchIdxInvis, novelIdxInvis, YnewInvis, XnewInvis], dim=1)  # [U,4]
    upNewML = torch.zeros([cfg.batchSize, cfg.novelN, 1,
                           cfg.H*cfg.upscale, cfg.W*cfg.upscale]
                          ).to(cfg.device) # [B,N,1,uH,uW]
    scatterML = MLnewInvis.unsqueeze(-1)  # [U,1]
    upNewML[scatterIdx[:,0],
            scatterIdx[:,1],
            :,
            scatterIdx[:,2],
            scatterIdx[:,3]] = scatterML # [B,N,1,uH,uW]
    upNewML = upNewML.reshape([cfg.batchSize * cfg.novelN,
                               1,
                               cfg.H * cfg.upscale,
                               cfg.W * cfg.upscale])  # [BN,1,uH,uW]
    # downsample back to original size
    newML = F.adaptive_avg_pool2d(
        upNewML, output_size=(cfg.H, cfg.W)) # [BN,1,H,W]
    newMaskLogitInvis = newML.reshape(
        [cfg.batchSize, cfg.novelN, 1, cfg.H, cfg.W])  # [B,N,H,W,1]
    # combine visible/invisible
    newMaskLogitNotVis = torch.where(
        newMaskLogitInvis < 0,
        newMaskLogitInvis,
        torch.ones_like(newInvDepth) * (-offsetMaskLogit)) # [B,N,1,H,W]
    newMaskLogit = torch.where(newMaskLogitVis > 0,
                               newMaskLogitVis,
                               newMaskLogitNotVis) # [B,N,1,H,W]
    newDepth = 1 / (newInvDepth + 1e-8) - offsetDepth

    return newDepth, newMaskLogit, collision  # [B,N,1,H,W]

def quaternionToRotMatrix(q):
    # q = [V, 4]
    qa, qb, qc, qd = torch.unbind(q, dim=1) # [V,]
    R = torch.stack(
        [torch.stack([1 - 2 * (qc**2 + qd**2),
                      2 * (qb * qc - qa * qd),
                      2 * (qa * qc + qb * qd)]),
         torch.stack([2 * (qb * qc + qa * qd),
                      1 - 2 * (qb**2 + qd**2),
                      2 * (qc * qd - qa * qb)]),
         torch.stack([2 * (qb * qd - qa * qc),
                      2 * (qa * qb + qc * qd),
                      1 - 2 * (qb**2 + qc**2)])]
    ).permute(2, 0, 1)
    return R.to(q.device)

def transParamsToHomMatrix(q, t):
    """q = [V, 4], t = [V,3]"""
    N = q.size(0)
    R = quaternionToRotMatrix(q) # [V,3,3]
    Rt = torch.cat([R, t.unsqueeze(-1)], dim=2) # [V,3,4]
    hom_aug = torch.cat([torch.zeros([N, 1, 3]), torch.ones([N, 1, 1])],
                        dim=2).to(Rt.device)
    RtHom = torch.cat([Rt, hom_aug], dim=1) # [V,4,4]
    return RtHom

def get3DhomCoord(XYZ, cfg):
    ones = torch.ones([cfg.batchSize, cfg.outViewN, cfg.outH, cfg.outW]) \
                .to(XYZ.device)
    XYZhom = torch.cat([XYZ, ones], dim=1) \
                  .reshape([cfg.batchSize, 4, cfg.outViewN, -1])\
                  .permute([0, 2, 1, 3])
    return XYZhom  # [B,V,4,HW]

def get3DhomCoord2(XYZ, cfg):
    ones = torch.ones([cfg.batchSize, 1, cfg.outViewN * cfg.outH * cfg.outW]) \
                .to(XYZ.device)
    XYZhom = torch.cat([XYZ, ones], dim=1)
    return XYZhom  # [B,4,VHW]
