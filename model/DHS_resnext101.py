import model.resnext_101_32x4d_ as resnext_101_32x4d_
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import resnext101_32_path


# class ResNeXt101(nn.Module):
#     def __init__(self):
#         super(ResNeXt101, self).__init__()
#         net = resnext_101_32x4d_.resnext_101_32x4d
#         net.load_state_dict(torch.load(resnext101_32_path))
#
#         net = list(net.children())
#         self.layer0 = nn.Sequential(*net[:4])
#         self.layer1 = net[4]
#         self.layer2 = net[5]
#         self.layer3 = net[6]
#         self.layer4 = net[7]
#
#     def forward(self, x):
#         layer0 = self.layer0(x)
#         layer1 = self.layer1(layer0)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)
#         return layer4

class ResNeXt101(nn.Module):
    def __init__(self, new_block):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        net.load_state_dict(torch.load(resnext101_32_path))
        
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:4])
        self.layer1 = net[4]
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]
        
        # self.upsample = lambda x: F.interpolate(
        #     x, scale_factor=2, mode='nearest'
        # )
        
        # 这里可以考虑去掉, 直接换成一个卷积与上采样的组合, 这样可以任意输入图片了.
        self.fc_line = nn.Linear(7 * 7 * 2048, 196)
        # 这里括号里的都是前期卷积的特征图的对应的通道数量 => in_channels
        self.rcl1 = new_block(1024)
        self.rcl2 = new_block(512)
        self.rcl3 = new_block(256)
        self.rcl4 = new_block(3)
        
        self.deconv_2_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1)
        self.deconv_3_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1) # x2
        self.deconv_4_5 = nn.ConvTranspose2d(1, 1, 8, 4, 2) # x4
        
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x_1):
        # 获取五个阶段的特征输出
        # layer0 = self.layer0(x)  # x/4 64
        # layer1 = self.layer1(layer0)  # x/4 256
        # layer2 = self.layer2(layer1)  # x/8 512
        # layer3 = self.layer3(layer2)  # x/16 1024
        # layer4 = self.layer4(layer3)  # x/32 2048
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
    
        x = self.deconv_2_3(x)
        x = self.rcl2.forward(x_8, x)
        x3 = torch.sigmoid(x)
    
        x = self.deconv_3_4(x)
        x = self.rcl3.forward(x_4, x)
        x4 = torch.sigmoid(x)
    
        x = self.deconv_4_5(x)
        x = self.rcl4.forward(x_1, x)
        x5 = torch.sigmoid(x)
    
        # 训练的时候要对7个预测都进行监督(深监督)
        return x1, x2, x3, x4, x5
