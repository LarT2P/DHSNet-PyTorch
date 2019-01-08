#!/usr/bin/env python
import os
import numpy as np
import PIL.Image
import torch
from torch.utils import data


class MyData(data.Dataset):
    """
    对训练集数据进行获取/处理操作
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, transform=True):
        super(MyData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'Image_after')
        gt_root = os.path.join(self.root, 'Mask')
        file_imgnames = os.listdir(img_root)

        self.img_names = []
        self.gt_names = []
        self.names = []
        # 只选择jpg的图像, 以及对应的真实标注
        for i, name in enumerate(file_imgnames):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.gt_names.append(
                os.path.join(gt_root, name[:-4] + '.png')
            )
            # 汇总最后保存的名字
            self.names.append(name[:-4])

    def __len__(self):
        # 定义len(MyData)的效果
        return len(self.img_names)

    def __getitem__(self, index):
        # 定义索引效果, 依次载入图像, 调整大小, 对真值图像进行二值化
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.uint8)

        gt_file = self.gt_names[index]
        gt = PIL.Image.open(gt_file)
        gt = gt.resize((224, 224))
        gt = np.array(gt, dtype=np.int32)
        gt[gt != 0] = 1

        # 为了防止输入的数据不是三个通道, 或者真实标注是大于一个通道.
        if len(img.shape) < 3:
            img = np.stack((img, img, img), axis=2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]

        if self._transform:
            img, gt = self.transform(img, gt)
            return img, gt
        else:
            return img, gt

    def transform(self, img, gt):
        """
        对于图像数据进行处理.
        图片数据归一化, 调整维度, 转化为tensor.
        真实数据直接转化为tensor.

        :param img: 训练集图片
        :param gt: 真实标注
        :return: 调整后的图片与真实标注
        """
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std

        img = img.transpose(2, 0, 1)
        # Creates a Tensor from a numpy.ndarray.
        img = torch.from_numpy(img).float()

        gt = torch.from_numpy(gt).float()
        return img, gt


class MyTestData(data.Dataset):
    """
    对测试集数据进行读取

    root: director/to/images/
            structure:
            - root
                - images
                    - images (images here)
                - masks (ground truth)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, transform=True):
        super(MyTestData, self).__init__()
        self.root = root
        self._transform = transform

        # 仅获取验证集图像
        img_root = os.path.join(self.root, 'Image')
        file_names = os.listdir(img_root)

        self.img_names = []
        self.names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img_size = img.size
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.uint8)

        # 为了防止输入的数据不是三个通道, 或者真实标注是大于一个通道.
        if len(img.shape) < 3:
            img = np.stack((img, img, img), axis=2)
        if img.shape[2] > 3:
            img = img[:, :, :3]

        if self._transform:
            img = self.transform(img)
            return img, self.names[index], img_size
        else:
            return img, self.names[index], img_size

    def transform(self, img):
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img
