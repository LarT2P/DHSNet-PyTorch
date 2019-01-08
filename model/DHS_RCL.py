import torch
import torch.nn as nn

class RCL_Module(nn.Module):
    def __init__(self, in_channels):
        super(RCL_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(65, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, smr):
        """
        RCL模块的正向传播

        :param x: 来自前面卷积网络的特征图, 因为通道数不唯一, 所以使用额外参数in_channels制定
        :param smr: 来自上一级得到的预测掩膜图, 通道数为1
        :return: RCL模块输出的预测掩膜图
        """
        # in_channelx1x1x64
        out1 = self.conv1(x)
        out1 = self.sigmoid(out1)
        out2 = self.sigmoid(smr)
        # 合并来自前一级的预测掩膜和对应前期卷积特征图, 并进行融合
        out = torch.cat((out1, out2), 1)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn(out)

        out_share = out
        for i in range(3):
            out = self.conv3(out)
            # 在RCL中, 使用求和的方式对共享特征和输出不同的时间步的特征结合
            out = torch.add(out, out_share)
            out = self.relu(out)
            out = self.bn(out)

        out = self.sigmoid(self.conv4(out))
        return out
