import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import time


# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=(1, stride), padding=(0, padding), dilation=(1, dilation)))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()             # 激活函数
        self.dropout1 = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(n_outputs)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.bn)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, (1, 1)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    # 每一层的特征数（不包括输出，输出默认1）；输入和预测的时间步长
    def __init__(self, channel_layer, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(channel_layer)         # 从输入到输出的每一层特征数
        for i in range(num_levels - 1):
            dilation_size = 2 ** i
            in_channels = channel_layer[i]
            out_channels = channel_layer[i + 1]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):       # （批大小， 节点数， 通道数， 时间步长）
        x = torch.transpose(x, 1, 2)
        out = self.tcn(x)       # （批大小， 通道数， 节点数， 时间步长）
        return out


if __name__ == '__main__':
    batch_size = 64
    in_channel = 1
    node_num = 170
    channel_hide = [in_channel, 16, 16, 16, 16]  # 每一层的通道数
    len_input = 12
    len_output = 12
    time_len_inout = [len_input, len_output]
    data = torch.ones(batch_size, node_num, in_channel, len_input)
    net = TemporalConvNet(channel_hide)
    print(net)
    start = time.time()
    out = net(data)
    end = time.time()
    # print(out)
    print(out.shape)
    print(end - start)