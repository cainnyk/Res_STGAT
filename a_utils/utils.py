import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import time
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    mx = d_mat_inv.dot(mx)
    return mx


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def get_adj(adj_path, num_of_vertices, device, eye=0):
    edges = pd.read_csv(adj_path).to_numpy()
    # 稀疏格式邻接表表示邻接矩阵 单向
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_of_vertices, num_of_vertices),
                        dtype=np.float32)
    distance_mx = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                        shape=(num_of_vertices, num_of_vertices),
                        dtype=np.float32)
    max_d = distance_mx.max()
    min_d = distance_mx.min()
    distance_mx = (max_min_normalization(distance_mx.toarray(), max_d, min_d)+1)/2
    # 构建对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 是否自环
    if eye > 0:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    adj = adj.to(device)
    # distance_mx = distance_mx.to(device)
    return adj, distance_mx


def load_data(data_prepared_path, num_of_hours, num_of_days, num_of_weeks,
              in_channel, DEVICE, batch_size, shuffle=True):
    filename = data_prepared_path
    print('load file:', filename)
    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, in_channel, :]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x = val_x[:, :, in_channel, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, in_channel, :]
    test_target = file_data['test_target']

    # # 减少测试集数量
    # test_x = test_x[:, :, :, :]
    # test_target = test_target[:, :, :]

    mean = file_data['mean'][:, :, in_channel, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, in_channel, :]  # (1, 1, 3, 1)
    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.float32).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.float32).to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.float32).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.float32).to(DEVICE)  # (B, N, T)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.float32).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.float32).to(DEVICE)  # (B, N, T)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std


def compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch, num_of_predict, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.PEMS08.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''
    # net.train(False)  # ensure dropout layers are in evaluation mode
    net.eval()
    with torch.no_grad():
        val_loader_length = len(val_loader)  # nb of batch
        tmp = []  # 记录了所有batch的loss
        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            labels = labels[:, :, num_of_predict]                    # 只输出五分钟的预测值
            loss = criterion(outputs, labels)  # 计算误差
            tmp.append(loss.item())
            if (limit is not None) and batch_index >= limit:
                break
        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
        print('\nepoch:', epoch, '\nvalidation_loss', validation_loss)
    return validation_loss


def predict_and_save_results_mstgcn(net, in_channel, data_loader, data_target_tensor, adj,
                                    global_step, _mean, _std, params_path, type, num_of_predict):
    '''
    :param net: nn.Module
    :param data_loader: torch.utils.PEMS08.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    ...
    '''
    net.train(False)  # ensure dropout layers are in test mode
    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)  # nb of batch
        prediction = []             # 存储所有batch的output
        input = []                  # 存储所有batch的input
        yk0 = time.time()
        for batch_index, batch_data in enumerate(data_loader):
            encoder_inputs, labels = batch_data
            input.append(encoder_inputs[:, :, in_channel].cpu().numpy())  # (batch, T', 1)
            outputs = net(encoder_inputs)
            prediction.append(outputs.detach().cpu().numpy())
        yk = time.time()
        input = np.concatenate(input, 0)
        input = re_normalization(input, _mean, _std)
        prediction = np.concatenate(prediction, 0)          # (batch, T', 1) 预测结果张量
        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差，先每个时刻，在全部
        prediction_length = prediction.shape[2]
        excel_list = np.zeros([prediction_length+1, 3])           # 预测误差矩阵
        print('\npredict num：\t   MAE       RMSE      MAPE')
        for i in num_of_predict:
            assert data_target_tensor.shape[0] == prediction.shape[0]   # 异常处理，表示满足这个条件才能继续
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i-num_of_predict[0]])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i-num_of_predict[0]]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i-num_of_predict[0]], 0)
            print('predict %s： \t %.4f   %.4f   %.4f' % (i+1, mae, rmse, mape))
            excel_list[i-num_of_predict[0], :] = np.array([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor[:, :, num_of_predict].reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor[:, :, num_of_predict].reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor[:, :, num_of_predict].reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('\ntotal average：\t %.4f   %.4f   %.4f' % (mae, rmse, mape))
        excel_list[-1, :] = np.array([mae, rmse, mape])

        print('单次推理用时：', (yk - yk0) / data_target_tensor.shape[0])
