import os

import numpy as np
import scipy.misc
import torch

import termcolor


# compute projection from source to target
def projection(Vs, Vt):
    VsN = Vs.size(0)
    VtN = Vt.size(0)
    Vt_rep = torch.repeat(Vt[None, :, :], [VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = torch.repeat(Vs[:, None, :], [1, VtN, 1])  # [VsN,VtN,3]
    diff = Vt_rep - Vs_rep
    dist = torch.sqrt(torch.reduce_sum(diff**2, axis=[2]))  # [VsN,VtN]
    idx = torch.to_int32(torch.argmin(dist, axis=1))
    proj = torch.gather_nd(Vt_rep, torch.stack([torch.range(VsN), idx], axis=1))
    minDist = torch.gather_nd(dist, torch.stack([torch.range(VsN), idx], axis=1))
    return proj, minDist


def mkdir(path):
    if not os.path.exists(path): os.makedirs(path)


def imread(fname):
    return scipy.misc.imread(fname) / 255.0


def imsave(fname, array):
    scipy.misc.toimage(array, cmin=0.0, cmax=1.0).save(fname)


# convert to colored strings
def toRed(content):
    return termcolor.colored(content, "red", attrs=["bold"])


def toGreen(content):
    return termcolor.colored(content, "green", attrs=["bold"])


def toBlue(content):
    return termcolor.colored(content, "blue", attrs=["bold"])


def toCyan(content):
    return termcolor.colored(content, "cyan", attrs=["bold"])


def toYellow(content):
    return termcolor.colored(content, "yellow", attrs=["bold"])


def toMagenta(content):
    return termcolor.colored(content, "magenta", attrs=["bold"])


# make image summary from image batch
def imageSummary(opt, tag, image, H, W):
    blockSize = opt.visBlockSize
    imageOne = torch.batch_to_space(
        image[:blockSize**2], crops=[[0, 0], [0, 0]], block_size=blockSize)
    imagePermute = torch.reshape(imageOne, [H, blockSize, W, blockSize, -1])
    imageTransp = torch.transpose(imagePermute, [1, 0, 3, 2, 4])
    imageBlocks = torch.reshape(imageTransp,
                             [1, H * blockSize, W * blockSize, -1])
    summary = torch.summary.image(tag, imageBlocks)
    return summary


# restore model
def restoreModelFromIt(opt, sess, saver, it):
    saver.restore(sess, "models_{0}/{1}_it{2}.ckpt".format(
        opt.group, opt.model, it))


# restore model
def restoreModel(opt, sess, saver):
    saver.restore(sess, "models_{0}/{1}.ckpt".format(opt.group, opt.load))


# save model
def saveModel(opt, sess, saver, it):
    saver.save(sess, "models_{0}/{1}_it{2}.ckpt".format(
        opt.group, opt.model, it))
