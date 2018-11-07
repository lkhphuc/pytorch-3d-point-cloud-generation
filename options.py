"""Parsing arguments from the commandline"""
import argparse

import numpy as np

import util


def parse_arguments(training):
    """Parse input arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", default="data", help="path to data folder")
    parser.add_argument(
        "--category", default="03001627", help="category ID number")
    parser.add_argument("--group", default="0", help="name for group")
    parser.add_argument(
        "--model", default="test", help="name for model instance")
    parser.add_argument(
        "--load",
        default=None,
        help="load trained model to fine-tune/evaluate")
    parser.add_argument(
        "--std",
        type=float,
        default=0.1,
        help="initialization standard deviation")
    parser.add_argument(
        "--outViewN",
        type=int,
        default=8,
        help="number of fixed views (output)")
    parser.add_argument(
        "--inSize", default="64x64", help="resolution of encoder input")
    parser.add_argument(
        "--outSize", default="128x128", help="resolution of decoder output")
    parser.add_argument(
        "--predSize", default="128x128", help="resolution of prediction")
    parser.add_argument(
        "--upscale",
        type=int,
        default=5,
        help="upscaling factor for rendering")
    parser.add_argument(
        "--novelN",
        type=int,
        default=5,
        help="number of novel views simultaneously")
    parser.add_argument("--arch", default=None)
    if training:  # training
        parser.add_argument(
            "--batchSize",
            type=int,
            default=20,
            help="batch size for training")
        parser.add_argument(
            "--chunkSize",
            type=int,
            default=100,
            help="data chunk size to load")
        parser.add_argument(
            "--itPerChunk",
            type=int,
            default=50,
            help="training iterations per chunk")
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="base learning rate (AE)")
        parser.add_argument(
            "--lrDecay",
            type=float,
            default=1.0,
            help="learning rate decay multiplier")
        parser.add_argument(
            "--lrStep",
            type=int,
            default=20000,
            help="learning rate decay step size")
        parser.add_argument(
            "--lambdaDepth",
            type=float,
            default=1.0,
            help="loss weight factor (depth)")
        parser.add_argument(
            "--fromIt",
            type=int,
            default=0,
            help="resume training from iteration number")
        parser.add_argument(
            "--toIt",
            type=int,
            default=100000,
            help="run training to iteration number")
    else:  # evaluation
        parser.add_argument(
            "--batchSize",
            type=int,
            default=1,
            help="batch size for evaluation")

    return parser.parse_args()


def get_arguments(training):
    """Parse and and add constant arguments"""
    cfg = parse_arguments(training)
    # these stay fixed
    cfg.sampleN = 100
    cfg.renderDepth = 1.0
    cfg.BNepsilon = 1e-5
    cfg.BNdecay = 0.999
    cfg.inputViewN = 24
    # ------ below automatically set ------
    cfg.training = training
    cfg.inH, cfg.inW = [int(x) for x in cfg.inSize.split("x")]
    cfg.outH, cfg.outW = [int(x) for x in cfg.outSize.split("x")]
    cfg.H, cfg.W = [int(x) for x in cfg.predSize.split("x")]
    cfg.visBlockSize = int(np.floor(np.sqrt(cfg.batchSize)))
    cfg.Khom3Dto2D = np.array(
        [[cfg.W, 0, 0, cfg.W / 2], [0, -cfg.H, 0, cfg.H / 2], [0, 0, -1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32)
    cfg.Khom2Dto3D = np.array(
        [[cfg.outW, 0, 0, cfg.outW / 2], [0, -cfg.outH, 0, cfg.outH / 2],
         [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float32)
    cfg.fuseTrans = np.load("data/trans_fuse{0}.npy".format(cfg.outViewN))

    print("({0}) {1}".format(
        util.toGreen("{0}".format(cfg.group)),
        util.toGreen("{0}".format(cfg.model))))
    print("------------------------------------------")
    print("batch size: {0}, category: {1}".format(
        util.toYellow("{0}".format(cfg.batchSize)),
        util.toYellow("{0}".format(cfg.category))))
    print("size: {0}x{1}(in), {2}x{3}(out), {4}x{5}(pred)".format(
        util.toYellow("{0}".format(cfg.inH)),
        util.toYellow("{0}".format(cfg.inW)),
        util.toYellow("{0}".format(cfg.outH)),
        util.toYellow("{0}".format(cfg.outW)),
        util.toYellow("{0}".format(cfg.H)),
        util.toYellow("{0}".format(cfg.W))))
    if training:
        print("learning rate: {0} (decay: {1}, step size: {2})".format(
            util.toYellow("{0:.2e}".format(cfg.lr)),
            util.toYellow("{0}".format(cfg.lrDecay)),
            util.toYellow("{0}".format(cfg.lrStep))))
        print("depth loss weight: {0}".format(
            util.toYellow("{0}".format(cfg.lambdaDepth))))
    print("viewN: {0}(out), upscale: {1}, novelN: {2}".format(
        util.toYellow("{0}".format(cfg.outViewN)),
        util.toYellow("{0}".format(cfg.upscale)),
        util.toYellow("{0}".format(cfg.novelN))))
    print("------------------------------------------")
    if training:
        print(
            util.toMagenta("training model ({0}) {1}...".format(
                cfg.group, cfg.model)))

    return cfg
