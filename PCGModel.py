import numpy as np
import scipy.ndimage.filters
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, 2, True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def deconv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 3, 2, padding=1, output_padding=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def linear_block(in_c, out_c):
    return nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )

class Encoder(nn.Module):
    """Build Encoder"""

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv2d_block(3, 96)
        self.conv2 = conv2d_block(96, 128)
        self.conv3 = conv2d_block(128, 192)
        self.conv4 = conv2d_block(192, 256)
        self.fc1 = linear_block(4096, 2048) # After flatten
        self.fc2 = linear_block(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x.view(-1, 4096))
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Decoder(nn.Module):
    """Build Decoder"""

    def __init__(self, outViewN):
        super(Decoder, self).__init__()
        self.outViewN = outViewN

        self.fc1 = linear_block(512, 1024)
        self.fc2 = linear_block(1024, 2048)
        self.fc3 = linear_block(2048, 4096)
        self.deconv1 = deconv2d_block(256, 192)
        self.deconv2 = deconv2d_block(192, 128)
        self.deconv3 = deconv2d_block(128, 96)
        self.deconv4 = deconv2d_block(96, 64)
        self.deconv5 = deconv2d_block(64, 48)
        self.pixel_conv = nn.Conv2d(48, outViewN * 4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.deconv1(x.view([-1, 256, 4, 4]))
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.pixel_conv(x)
        XYZ, maskLogit = torch.split(
            x, [self.outViewN * 3, self.outViewN], dim=1)

        return XYZ, maskLogit

class Structure_Generator(nn.Module):
    """Structure generator components in PCG"""

    def __init__(self, encoder=None, decoder=None):
        super(Structure_Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        if not self.encoder:
            self.encoder = Encoder()

        if not self.decoder:
            self.decoder = Decoder(outViewN=8)

    def forward(self, x):
        latent = self.encoder(x)
        XYZ, maskLogit = self.decoder(latent)

        return XYZ, maskLogit


# TESTING

if __name__ == '__main__':
    from options import get_arguments
    cfg = get_arguments()
    encoder = Encoder()
    decoder = Decoder(outViewN=cfg.outViewN)
    model = Structure_Generator(cfg)
