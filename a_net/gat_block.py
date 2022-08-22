import torch
import torch.nn as nn
import torch.nn.functional as F
from a_net.gat_layer import GatLayer


class GAT(nn.Module):
    def __init__(self, channel_list, head_num, adj, dropout=0.1, alpha=0.1):
        super(GAT, self).__init__()
        self.name = 'gat'
        self.dropout = dropout
        self.head_num = head_num
        self.model_type = 0
        layers = []
        layer_num = len(channel_list)
        for i in range(layer_num - 2):
            in_channels = channel_list[i]
            out_channels = channel_list[i + 1]
            layers += [GatLayer(in_channels, out_channels, adj, dropout, alpha, concat=True)]
        self.attentions = [nn.Sequential(*layers) for _ in range(head_num)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GatLayer(channel_list[-2], channel_list[-1], adj, dropout=dropout,
                                           alpha=alpha, concat=False)
        self.out_line = nn.Linear(channel_list[-1], channel_list[-1])

    def forward(self, x):   # [批大小，节点数，特征数]
        x = F.dropout(x, self.dropout, training=self.training)
        # 计算并拼接由多头注意力所产生的特征矩阵
        cont = [att(x) for att in self.attentions]
        x = sum(cont)/self.head_num
        x = F.dropout(x, self.dropout, training=self.training)
        # 特征矩阵经由输出层得到最终的模型输出
        x = self.out_att(x)
        return self.out_line(x)


if __name__ == '__main__':
    bitch_size = 1
    in_num = 4
    out_num = 3
    node_num = 5
    channel_list = [in_num, 3, 3, out_num]
    head_num = 1
    adj_mx = torch.rand(node_num, node_num) > 0.6
    adj_mx = adj_mx.float()
    adj_mx = adj_mx * adj_mx.T
    data = torch.rand(bitch_size, node_num, in_num)
    net = GAT(channel_list, head_num, adj_mx, dropout=0.2, alpha=0.2)
    out = net(data)
    print(out.shape)
    print(net)