import os
import numpy as np
from a_config.config import Config
import argparse
import configparser
import matplotlib.pyplot as plt

'''
1.搜索时间片的起止点，若没有则返回none
2.返回周，日，小时的当前起点序列
3.获得所有序列并拼接成训练集，验证集，测试集等，同时归一化，计算均值方差等
4.归一化函数
'''
def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history PEMS08
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on PEMS08
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    if label_start_idx + num_for_predict > sequence_length:
        return None
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    if len(x_idx) != num_of_depend:
        return None
    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None
    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict, 7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j] for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict, 24, points_per_hour)
        if not day_indices:
            return None, None, None, None
        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None
        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(node_feature_path, num_of_weeks, num_of_days, num_of_hours,
                              num_for_predict, points_per_hour=12, aim_fea=0, save=False):
    data_seq = np.load(node_feature_path)['data']  # shape=(sequence_length, num_of_vertices, num_of_features)
    data_name = node_feature_path[:7]
    all_samples = []
    for idx in range(data_seq.shape[0]):  # 按照时间片来取图
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue
        week_sample, day_sample, hour_sample, target = sample
        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]
        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)
        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)
        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, aim_fea, :]  # (1,N,T)
        sample.append(target)
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)
        all_samples.append(sample)
        # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)
    over_line = int(len(all_samples) * 1)
    # split_line1 = int(len(all_samples) * 0.12)
    # split_line2 = int(len(all_samples) * 0.16)
    # over_line = int(len(all_samples) * 0.2)

    # for i in zip()就是把多个列表合并共同遍历
    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:over_line])]

    # 把周日小时数据拼接在一起，all_sample原本里面有5组数据，[周,日,小时，target，时间片号]
    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]
    # 归一化，status=[三个特征的均值，标准差]大小为[1, 1, 3, 1], 后面是z-score归一化的东西
    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])

    aim_feature = ['flow', 'ratio', 'speed']
    if save:
        filename = data_name + aim_feature[aim_fea]\
                   + '_h' + str(num_of_hours)\
                   + '_d' + str(num_of_days) \
                   + '_w' + str(num_of_weeks)

        print('save file:', filename)
        # numpy的压缩算法
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                            )
    return all_data


def normalization(train, val, test):    # (B,N,F,T')
    '''

    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0, 1, 3), keepdims=True)        # (1, 1, 3, 1)
    std = train.std(axis=(0, 1, 3), keepdims=True)          # 就是对0,1,3维度的数据都算一遍
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


# prepare dataset
config = Config()  # 获取配置参数
edge_path = config.edge_path
node_feature_path = config.node_feature_path
aim_fea = config.aim_fea
num_of_weeks = config.num_of_weeks
num_of_days = config.num_of_days
num_of_hours = config.num_of_hours
num_for_predict = len(config.num_of_predict)
points_per_hour = config.points_per_hour

# 数据文件是(sequence_length, num_of_vertices, num_of_features)的
all_data = read_and_generate_dataset(node_feature_path, num_of_weeks, num_of_days, num_of_hours,
                                     num_for_predict, points_per_hour, aim_fea, save=True)
