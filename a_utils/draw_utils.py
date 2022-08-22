import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_adj(adj):
    node_y = adj.shape[0]
    node_x = adj.shape[1]
    x = [range(node_x) for _ in range(node_y)]
    y = [[-i]*node_x for i in range(node_y)]
    co = (adj-adj.min())/(adj.max()-adj.min())*0.95+0.05
    plt.figure(dpi=100, figsize=(10, node_y*10/node_x))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.scatter(np.array(x), np.array(y),  marker='s',
                alpha=co,
                c=-adj, s=(600/node_x)**2)
    plt.show()

if __name__ == '__main__':
    node_num = 50
    # adj = torch.rand(node_num, node_num//2)
    # x = [[np.sqrt(i/10)+np.sin(j/10) for i in range(node_num)]for j in range(node_num)]
    x = [[i+j for i in range(node_num*3)] for j in range(node_num)]
    # adj = np.sin(np.array(x))
    adj = np.array(x)
    # adj = adj*(adj>0.7)
    draw_adj(adj)
