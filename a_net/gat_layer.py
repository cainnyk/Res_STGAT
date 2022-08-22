import torch
import torch.nn as nn
import torch.nn.functional as F
from a_utils.draw_utils import draw_adj


class GatLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, dropout, alpha, concat=True):
        super(GatLayer, self).__init__()
        self.adj = adj
        self.out_features = out_features
        self.dk = adj.shape[0] ** 0.5
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.drop = nn.Dropout(dropout)
        self.res_net = nn.Linear(in_features, out_features, bias=False) if in_features != out_features else None
        self.bn = nn.BatchNorm1d(adj.shape[0])

    def forward(self, h):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.drop(attention)
        h_prime = torch.matmul(attention, Wh)
        h_prime = self.bn(h_prime)
        res = h if self.res_net is None else self.res_net(h)
        if self.concat:
            return F.elu(h_prime + res)
        else:
            return h_prime + res

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)
        e = e/self.dk
        return self.leakyrelu(e)


if __name__ == '__main__':
    bitch_size = 64
    in_num = 16
    out_num = 16
    node_num = 120
    adj_mx = torch.rand(node_num, node_num) > 0.7
    adj_mx = adj_mx.float()
    print(adj_mx.dtype)
    adj_mx = adj_mx * adj_mx.T
    data = torch.rand(bitch_size, node_num, in_num)
    net = GatLayer(in_num, out_num, adj_mx, dropout=0.2, alpha=0.2)
    out = net(data)
    print(out.shape)