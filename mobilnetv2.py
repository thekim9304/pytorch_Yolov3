import time

import torch
import torch.nn as nn

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_ch),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(in_ch, hidden_dim, kernel_size=1, norm_layer=norm_layer))
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
                nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(out_ch)
            ])
        else:
            # pw
            layers.extend([
                # dw
                ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch)
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, inputs):
        if self.use_res_connect:
            return inputs + self.conv(inputs)
        else:
            return self.conv(inputs)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, s, t, norm_layer, n):
        super(InvertedResidualBlock, self).__init__()

        width_mult = 1.0
        round_nearest = 8

        output_channel = _make_divisible(out_ch * width_mult, round_nearest)
        block = []
        for i in range(n):
            stride = s if i == 0 else 1
            block.append(
                InvertedResidual(in_ch, output_channel, stride, expand_ratio=t, norm_layer=norm_layer)
            )
            in_ch = output_channel
        self.blocks = nn.Sequential(*block)

    def forward(self, inputs):
        return self.blocks(inputs)

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        width_mult = 1.0
        round_nearest = 8

        norm_layer = nn.BatchNorm2d
        inverted_residual_setting =[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult, round_nearest)
        self.block1 = ConvBNReLU(in_ch=3, out_ch=input_channel, kernel_size=3, stride=2, norm_layer=norm_layer)
        self.block2 = InvertedResidualBlock(in_ch=input_channel, out_ch=16, s=1, t=1, norm_layer=norm_layer, n=1)
        self.block3 = InvertedResidualBlock(in_ch=16, out_ch=24, s=2, t=6, norm_layer=norm_layer, n=2)
        self.block4 = InvertedResidualBlock(in_ch=24, out_ch=32, s=2, t=6, norm_layer=norm_layer, n=3)
        self.block5 = InvertedResidualBlock(in_ch=32, out_ch=64, s=2, t=6, norm_layer=norm_layer, n=4)
        self.block6 = InvertedResidualBlock(in_ch=64, out_ch=96, s=1, t=6, norm_layer=norm_layer, n=3)
        self.block7 = InvertedResidualBlock(in_ch=96, out_ch=160, s=2, t=6, norm_layer=norm_layer, n=3)
        self.block8 = InvertedResidualBlock(in_ch=160, out_ch=320, s=1, t=6, norm_layer=norm_layer, n=1)

        # self.block9 = nn.Conv2d(320, 1280, 1, 1, 0)

    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x1 = x
        x = self.block5(x)
        x = self.block6(x)
        x2 = x
        x = self.block7(x)
        x = self.block8(x)
        # x = self.block9(x)
        x3 = x
        return x1, x2, x3


def main():
    input = torch.randn(1, 3, 416, 416)
    model = MobileNetV2()

    model.eval()
    model.to('cuda')

    for _ in range(20):
        prevTime = time.time()
        model(input.to('cuda'))
        curTime = time.time()
        sec = curTime-prevTime
        print(sec)

    model2 = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)
    model2.eval()
    model2.to('cuda')

    for _ in range(20):
        prevTime = time.time()
        model2(input.to('cuda'))
        curTime = time.time()
        sec = curTime - prevTime
        print(sec)


if __name__ == '__main__':
    main()