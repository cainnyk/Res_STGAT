import torch
import torch.nn as nn
from a_net.gat_block import GAT
from a_net.tcn_block import TemporalConvNet
from a_utils.draw_utils import draw_adj
import torch.utils.tensorboard as tb
import time


class DueModel(nn.Module):
    def __init__(self, tcn_channel_hide, gat_channel_hide, nheads, adj, time_len_inout):
        super(DueModel, self).__init__()
        self.time_len_inout = time_len_inout
        self.tcn = TemporalConvNet(tcn_channel_hide)
        self.trans = nn.Conv2d(tcn_channel_hide[-1], gat_channel_hide[0], kernel_size=(1, time_len_inout[0]))
        self.gat = GAT(gat_channel_hide, nheads, adj)
        self.local_pre = nn.Linear(gat_channel_hide[0], gat_channel_hide[-1])
        self.activate = nn.LeakyReLU(0.1)

    def forward(self, x):                       # 输入[64, 170, 1, 60],输出[64, 170, 12]
        x = self.tcn(x)                         # （批大小， 通道数， 节点数， 时间步长）
        x = self.trans(x)
        x = self.tcn2out(x)                     # （批大小， 节点数， 预测时间步长）
        local = self.local_pre(x)
        neighbor = self.gat(x)
        return local + neighbor

    def tcn2out(self, x):
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        return x


if __name__ == '__main__':
    bitch_size = 1
    in_num = 24
    out_num = 12
    node_num = 3000
    fea = 3
    gat_channel_list = [64, 64, 64, 64, 12]
    tcn_channel_list = [fea, 64, 64, 64, 64]
    head_num = 3
    time_len_inout = [in_num, out_num]
    adj_mx = torch.rand(node_num, node_num) > 0.99
    adj_mx = adj_mx.float()
    adj_mx = adj_mx + adj_mx.T + torch.eye(node_num)
    data = torch.rand(bitch_size, node_num, fea, in_num)

    # wr = tb.SummaryWriter('test_log')
    net = DueModel(tcn_channel_list, gat_channel_list, head_num, adj_mx, time_len_inout)
    t0 = time.time()
    out = net(data)
    t1 = time.time()
    # wr.add_graph(net, PEMS08)
    # wr.close()
    # tensorboard --logdir=a_net\test_log

    print(out.shape)
    print(t1-t0)
    # print(net)
