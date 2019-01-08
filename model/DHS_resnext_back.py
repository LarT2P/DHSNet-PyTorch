import torch
import torch.nn.functional as F
from torch import nn

from model import ResNeXt101


class R3Net(nn.Module):
    def __init__(self, new_block):
        super(R3Net, self).__init__()
        
        self.upsample = lambda x: F.interpolate(
            x, scale_factor=2, mode='nearest'
        )
        
        resnext = ResNeXt101()
        # 对应的五个阶段
        self.layer0 = resnext.layer0  # 1/4
        self.layer1 = resnext.layer1  # 1/4
        self.layer2 = resnext.layer2  # 1/8
        self.layer3 = resnext.layer3  # 1/16
        self.layer4 = resnext.layer4  # 1/32
        
        # 这里可以考虑去掉, 直接换成一个卷积与上采样的组合, 这样可以任意输入图片了.
        self.fc_line = nn.Linear(7 * 7 * 2048, 196)
        # 这里括号里的都是前期卷积的特征图的对应的通道数量 => in_channels
        self.rcl1 = new_block(1024)
        self.rcl2 = new_block(512)
        self.rcl3 = new_block(256)
        self.rcl4 = new_block(3)
        
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True
    
    def forward(self, x_1):
        # 获取五个阶段的特征输出
        x_4 = self.layer0(x_1)  # x/4
        x_4 = self.layer1(x_4)  # x/4
        x_8 = self.layer2(x_4)  # x/8
        x_16 = self.layer3(x_8)  # x/16 = 14
        x_32 = self.layer4(x_16)  # x/32
        
        bz = x_32.shape[0]
        x = x_32.view(bz, -1)
        # x = F.adaptive_avg_pool2d(x, (1, 1)).view(bz, -1)
        
        x = self.fc_line(x)  # generate the SMRglobal
        x = x.view(bz, 1, 14, -1)
        x1 = torch.sigmoid(x)
        
        x = self.rcl1.forward(x_16, x)
        x2 = torch.sigmoid(x)
        
        x = self.upsample(x)
        x = self.rcl2.forward(x_8, x)
        x3 = torch.sigmoid(x)
        
        x = self.upsample(x)
        x = self.rcl3.forward(x_4, x)
        x4 = torch.sigmoid(x)
        
        x = self.upsample(x)
        x = self.upsample(x)
        x = self.rcl4.forward(x_1, x)
        x5 = torch.sigmoid(x)
        
        # 训练的时候要对7个预测都进行监督(深监督)
        return x1, x2, x3, x4, x5
