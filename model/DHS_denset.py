"""
测试DenseNet预训练模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable
from collections import OrderedDict
from model.DHS_RCL import RCL_Module
import re

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(
        self, num_input_features, growth_rate, bn_size, drop_rate, dilation
    ):
        """
        DenseNet Layer 定义 [BN-Relu-Conv1x1-BN-Relu-Conv3x3]
        
        :param num_input_features: 输入特征通道数
        :param growth_rate: 增长率, 也就是每个密集层的输出特征通道数
        :param bn_size: 两个卷积层中间过度的通道数量=bn_size * growth_rate
        :param drop_rate: dropout概率
        :param dilation: 卷积核扩张率
        """
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        # 1x1
        self.add_module(
            'conv1', nn.Conv2d(
                num_input_features, bn_size * growth_rate, kernel_size=1,
                stride=1, bias=False
            )
        ),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        # 3x3
        self.add_module(
            'conv2', nn.Conv2d(
                bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,
                padding=dilation, bias=False, dilation=dilation
            )
        ),
        self.drop_rate = drop_rate
    
    def forward(self, x):
        # https://user-images.githubusercontent.com/16298490/49634289-32f71900-fa37-11e8-8024-33a26237a1f5.png
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self, num_layers, num_input_features, bn_size, growth_rate, drop_rate,
        dilation=1
    ):
        """
        根据给定的层数构造DenseBlock
        
        :param num_layers: 构造num_layers个dnase layer
        :type num_layers:
        :param num_input_features: 各层的输入通道数
        :type num_input_features:
        :param bn_size: denselayer的bn_size参数
        :type bn_size:
        :param growth_rate: 增长率
        :type growth_rate:
        :param drop_rate: dropout概率
        :type drop_rate:
        :param dilation: 卷积扩张率
        :type dilation:
        """
        super(_DenseBlock, self).__init__()
        
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size,
                drop_rate, dilation
            )
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        """
        构造过渡层 [BN-ReLU-Conv1x1-Avepool2x2]
        
        :param num_input_features: 输入特征通道数量
        :type num_input_features:
        :param num_output_features: 输出特征通道数量
        :type num_output_features:
        """
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv', nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        )
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
  
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    
    def __init__(
        self, new_block, growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
        is_downsamples=[True, True, True, True], dilations=[1, 1, 1, 1, 1]
    ):
        """
        构造DenseNet
        
        :param growth_rate: 增长率, 也就是每一层输出多加k层
        :param block_config: 每个池化块有多少层
        :param num_init_features: 第一次卷积输出通道数
        :param bn_size:
        :param drop_rate: dropout概率
        :param num_classes: 输出类别数量
        :param is_downsamples: 是否下采样
        :param dilations: 卷积扩张率
        """
        
        super(DenseNet, self).__init__()
        
        self.is_downsamples = is_downsamples
        self.dilations = dilations
        self.upsample = lambda x: F.interpolate(
            x, scale_factor=2, mode='nearest'
        )
        
        # First convolution 初始的7x7卷积层
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Each denseblock 所有中间的danse blcok, 不会用到第四个_Transition
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers, num_input_features=num_features,
                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
                dilation=dilations[i]
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # ===================================================================
        # Linear layer
        self.fc_line = nn.Linear(num_features, 784)
        # 这里括号里的都是前期卷积的特征图的对应的通道数量 => in_channels
        self.layer1 = new_block(512)
        self.layer2 = new_block(256)
        self.layer3 = new_block(64)
        self.layer4 = new_block(3)
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x_1):
        x_2 = self.features.conv0(x_1)
        x_2 = self.features.norm0(x_2)
        x_2 = self.features.relu0(x_2)
        
        x_4 = self.features.pool0(x_2)
        x_4 = self.features.denseblock1(x_4)
        
        x_8 = self.features.transition1(x_4)
        x_8 = self.features.denseblock2(x_8)
        
        x_16 = self.features.transition2(x_8)
        x_16 = self.features.denseblock3(x_16)
        
        # x_32 = self.features.transition3(x_16)
        # x_32 = self.features.denseblock4(x_32)
        # x_32 = self.features.norm5(x_32)
        
        x = F.relu(x_16, inplace=True)
        
        bz = x_16.shape[0]
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(bz, -1)
        
        x = self.fc_line(x)  # generate the SMRglobal
        x = x.view(bz, 1, 28, -1)
        x1 = torch.sigmoid(x)
        
        x = self.layer1.forward(x_8, x)
        x2 = torch.sigmoid(x)
        
        x = self.upsample(x)
        x = self.layer2.forward(x_4, x)
        x3 = torch.sigmoid(x)
        
        x = self.upsample(x)
        x = self.layer3.forward(x_2, x)
        x4 = torch.sigmoid(x)
        
        x = self.upsample(x)
        x = self.layer4.forward(x_1, x)
        x5 = torch.sigmoid(x)
        
        return x1, x2, x3, x4, x5


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 32),
        **kwargs
    )
    
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        
        # # 1. filter out unnecessary keys
        # model_dict = model.state_dict()
        # pretrained_dict = {
        #     k: v for k, v in state_dict.items() if k in model_dict
        # }
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # # 3. load the new state dict
        # model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    net = densenet169(pretrained=True, new_block=RCL_Module).cuda()
    x = torch.Tensor(2, 3, 224, 224).cuda()
    x1, x2, x3, x4, x5 = net(Variable(x))
    print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
