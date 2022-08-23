class Config:
    def __init__(self):
        data_name = 1
        if data_name == 8:
            self.edge_path = 'PEMS08/edge.csv'
            self.node_feature_path = 'PEMS08/node_feature.npz'
        elif data_name == 4:
            self.edge_path = 'PEMS04/distance.csv'
            self.node_feature_path = 'PEMS04/pems04.npz'
        elif data_name == 0:
            self.edge_path = 'QDataX/edge.csv'
            self.node_feature_path = 'QDataX/node_feature.npz'
        elif data_name == 1:
            self.edge_path = 'QDataL/edge.csv'
            self.node_feature_path = 'QDataL/node_feature.npz'

        self.aim_fea = 0                    # 预测的目标特征（0：流量预测；2：速度预测）
        self.points_per_hour = 4           # 每小时时间片数
        self.num_of_predict = range(4)     # 预测的时间片数
        self.num_of_weeks = 0               # 周
        self.num_of_days = 0                # 日
        self.num_of_hours = 3               # 小时
        self.num_of_history = (self.num_of_weeks + self.num_of_days + self.num_of_hours) \
                         * self.points_per_hour
        self.diffusion = 2
        # 以下是网络参数配置
        self.in_channels = range(1)  # 3个输入通道，是一个范围
        self.lr = 0.01
        self.epoch = 100
        self.batch_size = 64
        # gat
        self.seed = 2022
        self.nheads = 2

        gat_filter = [64] * 8
        tcn_filter = [16] * 4
        self.gat_channel_list = gat_filter + [len(self.num_of_predict)]
        self.tcn_channel_list = [len(self.in_channels)] + tcn_filter

