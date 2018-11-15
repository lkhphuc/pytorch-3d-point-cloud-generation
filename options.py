"""Parsing arguments from the commandline"""
import argparse
import numpy as np

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
        "--training", type=bool, default=True,
        help="training or testing")
    parser.add_argument(
        "--load", default=None,
        help="load trained model to fine-tune/evaluate")

    parser.add_argument(
        "--startEpoch", type=int, default=0,
        help="resume training from iteration number")
    parser.add_argument(
        "--endEpoch", type=int, default=10000,
        help="endEpoch")

    parser.add_argument(
        "--chunkSize", type=int, default=100,
        help="Number of unique CAD models in each batch")
    parser.add_argument(
        "--batchSize", type=int, default=20,
        help="number of unique images from chunkSize CADs models")

    # Optimizer
    parser.add_argument(
        "--optim", type=str, default='adam',
        help="optimizer to use")
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="base learning rate (AE)")
    ## Adam 
    parser.add_argument(
        "--wd", type=float, default=0.0,
        help="weight decay as implemented in Adam optimizer (L2 norm)")
    parser.add_argument(
        "--trueWD", type=float, default=0.0,
        help="TRUE weight decay in Adam")
    ## SGD
    parser.add_argument(
        "--lambdaDepth", type=float, default=1.0,
        help="loss weight factor (depth)")

    # LR scheduler
    parser.add_argument(
        "--lrSched", type=str, default=None,
        help="What learning rate scheduler to use"
    )
    parser.add_argument(
        "--lrDecay", type=float, default=1.0,
        help="learning rate decay multiplier (gamma)")
    ## StepLR
    parser.add_argument(
        "--lrStep", type=int, default=1,
        help="how many epochs until update lr")
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="which GPU to use")

    # Model related
    parser.add_argument("--arch", default=None)
    parser.add_argument(
        "--novelN", type=int, default=5,
        help="number of novel views simultaneously")
    parser.add_argument(
        "--outViewN", type=int, default=8,
        help="number of fixed views (output)")
    parser.add_argument(
        "--std", type=float, default=0.1,
        help="initialization standard deviation")
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
    """Parse and and add constant arguments"""
    cfg = parse_arguments()
    # these stay fixed
    cfg.sampleN = 100
    cfg.renderDepth = 1.0
    cfg.BNepsilon = 1e-5
    cfg.BNdecay = 0.999
    cfg.inputViewN = 24
    # ------ below automatically set ------
    cfg.inH, cfg.inW = [int(x) for x in cfg.inSize.split("x")]
    cfg.outH, cfg.outW = [int(x) for x in cfg.outSize.split("x")]
    cfg.H, cfg.W = [int(x) for x in cfg.predSize.split("x")]
    cfg.visBlockSize = int(np.floor(np.sqrt(cfg.batchSize)))
    cfg.Khom3Dto2D = np.array(
        [[cfg.W, 0, 0, cfg.W / 2],
         [0, -cfg.H, 0, cfg.H / 2],
         [0, 0, -1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32)
    cfg.Khom2Dto3D = np.array(
        [[cfg.outW, 0, 0, cfg.outW / 2],
         [0, -cfg.outH, 0, cfg.outH / 2],
         [0, 0, -1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32)
    cfg.fuseTrans = np.load(f"{cfg.path}/trans_fuse{cfg.outViewN}.npy")

    print(f"{cfg.model}_{cfg.experiment}")
    print("------------------------------------------")
    print(f"batch size: {cfg.batchSize}, \ category: {cfg.category}")
    print(f"size: {cfg.inH}x{cfg.inW}(in), \
          {cfg.outH}x{cfg.outW}(out), \
          {cfg.H}x{cfg.W}(pred)")
    print(f"viewN: {cfg.outViewN}(out), upscale: {cfg.upscale}, novelN: {cfg.novelN}")
    print("------------------------------------------")
    if cfg.training:
        print(f"learning rate: {cfg.lr:.2e} \
              (decay: {cfg.lrDecay}, step size: {cfg.lrStep})")
        print(f"depth loss weight: {cfg.lambdaDepth}")

    return cfg
