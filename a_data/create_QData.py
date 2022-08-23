import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os


def get_aim_order(path, scope):
    link_gps = pd.read_table(path).to_numpy()
    left, right, down, up = scope
    small_map = np.array([0, 1, 2, 3])
    for i in range(link_gps.shape[0]):
        if left < link_gps[i, 1] < right:
            if down < link_gps[i, 2] < up:
                small_map = np.vstack([small_map, np.append(i, link_gps[i, :])])
    draw_map(link_gps, small_map[1:, 2:], scope)
    return small_map[1:, :2]


def draw_map(link_gps, small_map, scope):
    left, right, down, up = scope
    plt.scatter(link_gps[:, 1], link_gps[:, 2], s=0.1)
    plt.plot([left, right], [down, down], 'r')
    plt.plot([left, right], [up, up], 'r')
    plt.plot([right, right], [down, up], 'r')
    plt.plot([left, left], [down, up], 'r')
    plt.show()
    plt.scatter(small_map[:, 0], small_map[:, 1], s=1)
    plt.show()


def get_sel_edge(path, sel_road_id, data_path):
    edge = pd.read_csv(path).to_numpy()
    sel_edge = np.array([0, 0, 0])
    for rel in edge:
        from_id = np.where(rel[2] == sel_road_id)
        to_id = np.where(rel[3] == sel_road_id)
        if from_id[0].size and to_id[0].size:
            sel_edge = np.vstack((sel_edge, np.array([from_id[0][0], to_id[0][0], 1])))
    data_path = data_path + '/edge.csv'
    with open(data_path, 'w', newline='') as t:
        writer = csv.writer(t)
        writer.writerow(['from', 'to', 'cost'])
        writer.writerows(sel_edge[1:])


def get_seq(path, aim_idx, data_path):
    seq_len = 5856
    seq = np.zeros([seq_len, aim_idx.size, 1])
    with open(path, 'r', encoding='UTF-8') as file_content:
        node_num = 0
        time_num = 0
        time_id = 1e10
        for line in file_content:
            if node_num == aim_idx.size and time_id == seq_len:
                break
            sel_num = aim_idx[node_num] if node_num < aim_idx.size else 0
            if time_num == sel_num * seq_len:
                time_id = 0
                node_id = node_num
                node_num += 1
                print('read_node_num：', node_num, '\ttotal_node_num：', aim_idx.size)
            if time_id < seq_len:
                d = np.array(line[:-1].split(','), dtype=float)
                seq[time_id, node_id] = d[2]
                time_id += 1
            time_num += 1
    aim_path = data_path + '/node_feature'
    np.savez_compressed(aim_path, data=seq)


if __name__ == '__main__':
    aim_path = 'QDataL'
    if not os.path.exists(aim_path):
        os.makedirs(aim_path)
        print('create params directory %s' % (aim_path))
    # scope = [116.41, 116.46, 39.905, 39.95]
    scope = [116.425, 116.455, 39.92, 39.95]
    path_link_road = 'E:\\baidu_netdisk\Q_Traffic_Dataset\link_road.v2'
    path_link_gps = 'E:\\baidu_netdisk\Q_Traffic_Dataset\link_gps.v2'
    path_link_rel = 'E:\\baidu_netdisk\Q_Traffic_Dataset\link_rel.rel'
    path_link_speed = 'E:\\baidu_netdisk\Q_Traffic_Dataset\link_speed.v2'
    # 利用GPS选定路段
    sel_road = get_aim_order(path_link_gps, scope)
    print('选定路段数量：', sel_road.shape[0])
    # 获取路段连接
    get_sel_edge(path_link_rel, sel_road[:, 1], aim_path)
    # 获取路段历史平均车速
    get_seq(path_link_speed, sel_road[:, 0], aim_path)
    data_seq = np.load(aim_path + '/node_feature.npz')['data']
    print()
