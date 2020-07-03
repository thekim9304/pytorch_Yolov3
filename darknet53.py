import time

import torch
import torch.nn as nn

class DarkConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, strides, padding=None):
        super(DarkConv, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=kernel_size,
                              stride=strides,
                              padding=padding,
                              bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=out_ch) # C from an expected input of size (N, C, H, W)
        self.lrelu = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.lrelu(x)
        return x

class DarkResidual(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2):
        super(DarkResidual, self).__init__()
        self.darkconv1 = DarkConv(in_ch=in_ch, out_ch=out_ch1, kernel_size=1, strides=1, padding=0)
        self.darkconv2 = DarkConv(in_ch=out_ch1, out_ch=out_ch2, kernel_size=3, strides=1, padding=1)

    def forward(self, inputs):
        x = self.darkconv1(inputs)
        x = self.darkconv2(x)
        x = torch.add(inputs, x)
        return x

class Darknet53(nn.Module):
    def __init__(self, in_ch):
        super(Darknet53, self).__init__()
        self.darkconv1 = DarkConv(in_ch=in_ch, out_ch=32, kernel_size=3, strides=1, padding=1)
        self.darkconv2 = DarkConv(in_ch=32, out_ch=64, kernel_size=3, strides=2, padding=1)

        # 1x residual blocks
        layers = [DarkResidual(in_ch=64, out_ch1=32, out_ch2=64) for _ in range(1)]
        self.residual_block1 = nn.Sequential(*layers)

        self.darkconv3 = DarkConv(in_ch=64, out_ch=128, kernel_size=3, strides=2, padding=1)
        # 2x residual blocks
        layers = [DarkResidual(in_ch=128, out_ch1=64, out_ch2=128) for _ in range(2)]
        self.residual_block2 = nn.Sequential(*layers)

        self.darkconv4 = DarkConv(in_ch=128, out_ch=256, kernel_size=3, strides=2, padding=1)
        # 8x residual blocks
        layers = [DarkResidual(in_ch=256, out_ch1=128, out_ch2=256) for _ in range(8)]
        self.residual_block3 = nn.Sequential(*layers)

        self.darkconv5 = DarkConv(in_ch=256, out_ch=512, kernel_size=3, strides=2, padding=1)
        # 8x residual blocks
        layers = [DarkResidual(in_ch=512, out_ch1=256, out_ch2=512) for _ in range(8)]
        self.residual_block4 = nn.Sequential(*layers)

        self.darkconv6 = DarkConv(in_ch=512, out_ch=1024, kernel_size=3, strides=2, padding=1)
        # 4x residual blocks
        layers = [DarkResidual(in_ch=1024, out_ch1=512, out_ch2=1024) for _ in range(4)]
        self.residual_block5 = nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.darkconv1(inputs)
        x = self.darkconv2(x)
        x = self.residual_block1(x)
        x = self.darkconv3(x)
        x = self.residual_block2(x)
        x = self.darkconv4(x)
        x = self.residual_block3(x)
        y0 = x
        x = self.darkconv5(x)
        x = self.residual_block4(x)
        y1 = x
        x = self.darkconv6(x)
        x = self.residual_block5(x)
        y2 = x
        return y0, y1, y2

def main():
    input = torch.randn(1, 3, 416, 416)
    model = Darknet53(3)
    model.eval()
    model.cuda()

    for _ in range(20):
        prevTime = time.time()
        model(input.cuda())
        curTime = time.time()
        sec = curTime-prevTime
        print(sec)

if __name__ == '__main__':
    main()