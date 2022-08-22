# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
from a_config.config import Config
import a_utils.utils as utils
from a_net.st_block import DueModel
from tensorboardX import SummaryWriter
import shutil
import matplotlib.pyplot as plt
from a_utils.draw_utils import draw_adj

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
# DEVICE = torch.device('cpu')
print("CUDA:", USE_CUDA, DEVICE)

config = Config()  # 获取配置参数
lr = config.lr
epoch = config.epoch
batch_size = config.batch_size

seed = config.seed
nheads = config.nheads
gat_channel_list = config.gat_channel_list
tcn_channel_list = config.tcn_channel_list

aim_fea = config.aim_fea
diffusion = config.diffusion
num_of_weeks = config.num_of_weeks
num_of_days = config.num_of_days
num_of_hours = config.num_of_hours
num_of_predict = config.num_of_predict
points_per_hour = config.points_per_hour
in_channel = config.in_channels
num_of_history = config.num_of_history
edge_path = config.edge_path

aim_feature = ['flow', 'ratio', 'speed']
# 特征数据的地址不要后缀
data_num = edge_path[:7]
path = data_num + aim_feature[aim_fea] + '_h' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)
data_prepared_path = '../a_data/' + path + '.npz'
adj_path = '../a_data/' + edge_path
params_path = '../a_params/' + path

# 数据集加载
train_loader, train_target_tensor, val_loader, val_target_tensor, \
test_loader, test_target_tensor, _mean, _std \
    = utils.load_data(data_prepared_path, num_of_hours, num_of_days, num_of_weeks,
              in_channel, DEVICE, batch_size, shuffle=True)

# 构造连接矩阵
num_of_vertices = test_target_tensor.shape[1]
adj_mx, distance_mx = utils.get_adj(adj_path, num_of_vertices, DEVICE, 0)
draw_adj(torch.sigmoid(100*adj_mx.to('cpu')))
for i in range(diffusion):
    adj_mx = torch.matmul(adj_mx, adj_mx)
draw_adj(torch.sigmoid(100*adj_mx.to('cpu')))

# 搭建神经网络
net = DueModel(tcn_channel_list, gat_channel_list, nheads,
               adj_mx.T, [num_of_history, len(num_of_predict)]).to(DEVICE)


def train_main():
    # 建立网络文件夹，若存在则更新
    if not os.path.exists(params_path):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif os.path.exists(params_path):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    # 设置损失函数，优化器以及网络记录*****************************/*/*/*/
    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(net.parameters())
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    # 打印网络结构
    print(net)
    # 打印每层的参数个数
    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_ep = 0
    best_val_loss = np.inf
    start_time = time()

    # train model ************************/////////*****************/////////***************
    train_line = []
    val_line = []
    for ep in range(epoch):
        temp = []
        params_filename = os.path.join(params_path, 'epoch_%s.params' % ep)
        # 验证部分
        val_loss = utils.compute_val_loss_mstgcn(net, val_loader, criterion, sw, ep, num_of_predict)
        val_line.append(val_loss)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        if val_loss < best_val_loss and ep > epoch-10:
            best_val_loss = val_loss
            best_ep = ep
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data     # 输入[64, 170, 3, 60],输出[64, 170, 12]
            optimizer.zero_grad()                   # 批大小，节点数，特征数，时间序列长度
            outputs = net(encoder_inputs)
            labels = labels[:, :, num_of_predict]                    # 只输出五分钟的预测值
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            temp.append(training_loss)
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)
        print('training_loss: %.2f, time: %.2f' % (sum(temp)/len(temp), time()-start_time))
        train_line.append(sum(temp)/len(temp))
        print('learning rate:', lr)
    print('\n============ finish training =============\nbest epoch:', best_ep)
    print('test begins:\n')
    plt.plot(train_line[1:], 'b', alpha=0.5)
    plt.plot(val_line[1:], 'r')
    plt.show()
    # apply the best model on the test set
    predict_main(best_ep, test_loader, test_target_tensor, _mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type):
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)
    net.load_state_dict(torch.load(params_filename))
    utils.predict_and_save_results_mstgcn(net, in_channel, data_loader, data_target_tensor, adj_mx,
                                          global_step, _mean, _std, params_path, type, num_of_predict)


if __name__ == "__main__":
    train_main()
    # predict_main(1, test_loader, test_target_tensor, _mean, _std, 'test')











