"""Parsing arguments from the commandline"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F


def parse_arguments():
    """Parse input arguments"""

    parser = argparse.ArgumentParser("Pytorch 3D Point Cloud Generation")

    # Training related
    parser.add_argument(
        "--experiment", default="0",
        help="name for experiment")
    parser.add_argument(
        "--model", default="PCG",
        help="name for model")

    parser.add_argument(
        "--path", default="data",
        help="path to data folder")
    parser.add_argument(
        "--category", default="03001627",
        help="category ID number")

    parser.add_argument(
        "--phase", type=str, default="stg1",
        help="stg1, stg2, eval")
    # For stage 2
    parser.add_argument(
        "--loadPath", type=str, default=None,
        help="path to load model")
    parser.add_argument(
        "--loadEpoch", type=int, default=None,
        help="model to load in loadPath, default load best")

    parser.add_argument(
        "--startEpoch", type=int, default=0,
        help="start training from epoch")
    parser.add_argument(
        "--endEpoch", type=int, default=1000,
        help="stop training at epoch")
    parser.add_argument(
        "--saveEpoch", type=int, default=None,
        help="checkpoint model every --saveEpoch, None: best model only"
    )

    parser.add_argument(
        "--chunkSize", type=int, default=100,
        help="number of unique CAD models in each batch")
    parser.add_argument(
        "--batchSize", type=int, default=100,
        help="number of unique images from chunkSize CADs models")

    # Optimizer
    parser.add_argument(
        "--optim", type=str, default='sgd',
        choices=['adam', 'sgd'],
        help="what optimizer to use (adam/sgd)")
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="max learning rate")
    parser.add_argument(
        "--wd", type=float, default=0.0,
        help="value for weight decay as implemented (L2 norm)")
    parser.add_argument(
        "--trueWD", type=float, default=0,
        help="value for TRUE weight decay")
    parser.add_argument(
        "--momentum", type=float, default=0,
        help="value formomentum, default=None")

    # LR scheduler
    parser.add_argument(
        "--lrSched", type=str, default=None,
        choices=['annealing', 'cyclical', 'restart'],
        help="What learning rate scheduler to use")
    parser.add_argument(
        "--lrBase", type=float, default=0.3,
        help="Base learning rate")
    parser.add_argument(
        "--lrStep", type=int, default=55,
        help="Step size (#epoch) of lrSched, 2 steps == 1 cycle // 1 step -> restart")
    parser.add_argument(
        "--lrGamma", type=float, default=0.9,
        help="Multiplicative factor of learning rate decay")
    parser.add_argument(
        "--lrRestart", type=str, default=None,
        help="How many step to warm restart SGD/Adam's lr")
    # For SGDR
    parser.add_argument(
        "--T_0", type=int, default=10,
        help="number of epoch per cycle")
    parser.add_argument(
        "--T_mult", type=int, default=10,
        help="multiplicative value for T0")


    parser.add_argument(
        "--gpu", type=int, default=0,
        help="which GPU to use")

    # For LR Finder only
    parser.add_argument(
        "--startLR", type=float, default=1e-7,
        help="start range of lr in LR Finder")
    parser.add_argument(
        "--endLR", type=float, default=10,
        help="end range of lr in LR Finder")
    parser.add_argument(
        "--itersLR", type=float, default=10,
        help="Number of iterations to explore LR")

    # Model related
    parser.add_argument(
        "--lambdaDepth", type=float, default=1.0,
        help="loss weight factor (depth)")
    parser.add_argument(
        "--std", type=float, default=0.1,
        help="initialization standard deviation")
    parser.add_argument(
        "--novelN", type=int, default=5,
        help="number of novel views simultaneously")
    parser.add_argument(
        "--outViewN", type=int, default=8,
        help="number of fixed views (output)")
    parser.add_argument(
        "--inSize", default="64x64",
        help="resolution of encoder input")
    parser.add_argument(
        "--outSize", default="128x128",
        help="resolution of decoder output")
    parser.add_argument(
        "--predSize", default="128x128",
        help="resolution of prediction")
    parser.add_argument(
        "--upscale", type=int, default=5,
        help="upscaling factor for rendering")

    return parser.parse_args()

def get_arguments():
    cfg = parse_arguments()
    # these stay fixed
    cfg.sampleN = 100
    cfg.renderDepth = 1.0
    cfg.BNepsilon = 1e-5
    cfg.BNdecay = 0.999
    cfg.inputViewN = 24
    # ------ below automatically set ------
    cfg.device = torch.device(
        f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    cfg.inH, cfg.inW = [int(x) for x in cfg.inSize.split("x")]
    cfg.outH, cfg.outW = [int(x) for x in cfg.outSize.split("x")]
    cfg.H, cfg.W = [int(x) for x in cfg.predSize.split("x")]
    cfg.Khom3Dto2D = torch.Tensor([[cfg.W, 0, 0, cfg.W / 2],
                                   [0, -cfg.H, 0, cfg.H / 2],
                                   [0, 0, -1, 0],
                                   [0, 0, 0, 1]]).float().to(cfg.device)
    cfg.Khom2Dto3D = torch.Tensor([[cfg.outW, 0, 0, cfg.outW / 2],
                                   [0, -cfg.outH, 0, cfg.outH / 2],
                                   [0, 0, -1, 0],
                                   [0, 0, 0, 1]]).float().to(cfg.device)
    cfg.fuseTrans = F.normalize(
        torch.from_numpy(
            np.load(f"{cfg.path}/trans_fuse{cfg.outViewN}.npy")),
        p=2, dim=1).to(cfg.device)

    print(f"EXPERIMENT: {cfg.model}_{cfg.experiment}")
    print("------------------------------------------")
    print(f"input:{cfg.inH}x{cfg.inW}, output:{cfg.outH}x{cfg.outW}, pred:{cfg.H}x{cfg.W}")
    print(f"viewN:{cfg.outViewN}(out), upscale:{cfg.upscale}, novelN:{cfg.novelN}")
    print(f"Device: {cfg.device}, depth_loss weight:{cfg.lambdaDepth}")
    print("------------------------------------------")

    return cfg
