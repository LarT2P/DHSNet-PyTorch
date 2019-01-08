# todo:
#   [ ] CRF
#   [ ] RNN
#   [v] DenseNet
#   [v] ResNeXt
#   [ ] 这里的输出的数据的大小和原图是不一样的, 需要改成一样的

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import os
from matplotlib.pyplot import imsave
from dataset import MyData, MyTestData
from model.DHS_resnext101 import ResNeXt101
from model.DHS_RCL import RCL_Module
import utils.tools as tools
from tqdm import tqdm
from utils.evaluateFM import get_FM
import pandas as pd
import time

parser = argparse.ArgumentParser(description='DHS-Pytorch')
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--total_epochs', default=100, type=int)

parser.add_argument('--dataset', default='DUTS', type=str)
parser.add_argument('--lr', default=1e-5)
parser.add_argument('--data_root', default='./data/', type=str)
parser.add_argument('--cache', default='./cache/')
parser.add_argument('--pre', default='./prediction')
parser.add_argument('--val_rate', default=2)  # validate the model every n epoch


def train(model, progress_epoch, criterion, optimizer_feature, epoch, args):
    # 改为训练模式
    # model.train()
    # 测量一个epoch的时间
    time_begin_epoch = time.time()
    
    # 一个周期内部进行迭代计算
    for ib, (input, gt) in enumerate(progress_epoch):
        # 获取对应的5个掩膜预测结果
        inputs = Variable(input).cuda()
        # 224/16,/16,/8,/4,/1
        msk1, msk2, msk3, msk4, msk5 = model.forward(inputs)
        
        gt = Variable(gt.unsqueeze(1)).cuda()
        gt_14 = functional.interpolate(gt, size=14, mode='bilinear')
        gt_28 = functional.interpolate(gt, size=28, mode='bilinear')
        gt_56 = functional.interpolate(gt, size=56, mode='bilinear')
        
        loss = criterion(msk1, gt_14) + criterion(msk2, gt_14) \
               + criterion(msk3, gt_28) + criterion(msk4, gt_56) \
               + criterion(msk5, gt)
        
        model.zero_grad()
        loss.backward()
        optimizer_feature.step()
        
        # train_losses.append(round(float(loss.data.cpu()), 3))
        title = '{} Epoch {}/{}'.format('Training', epoch, args.total_epochs)
        progress_epoch.set_description(
            title + ' ' + 'loss:' + str(loss.data.cpu().numpy())
        )
    
    time_end_epoch = time.time()
    print("训练一个epoch的时间:", time_end_epoch - time_begin_epoch)


def test(
    model, val_loader, epoch,
    prediction_root, check_root_feature, check_root_opti, val_root
):
    time_begin_test = time.time()
    
    val_output_root = (prediction_root + '/epoch_current')
    if not os.path.exists(val_output_root):
        os.makedirs(val_output_root)
    
    print("\ngenerating output images")
    for ib, (input_, img_name, _) in enumerate(val_loader):
        inputs = Variable(input_).cuda()
        _, _, _, _, output = model.forward(inputs)
        # ques: 这里又一次sigmoid的用处?
        # output = functional.sigmoid(output)
        out = output.data.cpu().numpy()
        for i in range(len(img_name)):
            imsave(os.path.join(val_output_root, img_name[i] + '.png'),
                   out[i, 0], cmap='gray')
    print("\nevaluating mae....")
    
    # 计算F测度和平均绝对误差
    F_measure, mae = get_FM(
        salpath=val_output_root + '/', gtpath=val_root + '/masks/'
    )
    
    time_end_test = time.time()
    print("测试一次的时间:", time_end_test - time_begin_test)
    
    return epoch, F_measure, mae


def main(args):
    dataset = args.dataset
    bsize = args.batch_size
    root = args.data_root
    cache_root = args.cache
    prediction_root = args.pre
    
    train_root = root + dataset + '/train'
    val_root = root + dataset + '/val'  # validation dataset
    
    # mkdir( path [,mode] )：创建一个目录，可以是相对或者绝对路径，mode的默认模式是0777。
    # 如果目录有多级，则创建最后一级。如果最后一级目录的上级目录有不存在的，则会抛出一个OSError。
    # makedirs( path [,mode] )：创建递归的目录树，可以是相对或者绝对路径，mode的默认模式是
    # 0777。如果子目录创建失败或者已经存在，会抛出一个OSError的异常，Windows上Error 183即为
    # 目录已经存在的异常错误。如果path只有一级，与mkdir相同。
    check_root_opti = cache_root + '/opti'  # save checkpoint parameters
    if not os.path.exists(check_root_opti):
        os.makedirs(check_root_opti)
    
    check_root_feature = cache_root + '/feature'  # save checkpoint parameters
    if not os.path.exists(check_root_feature):
        os.makedirs(check_root_feature)
    
    # 获取调整后的数据集
    train_loader = torch.utils.data.DataLoader(
        MyData(train_root, transform=True),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        MyTestData(val_root, transform=True),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True
    )
    model = ResNeXt101(RCL_Module)
    model.cuda()
    criterion = nn.BCELoss()
    optimizer_feature = torch.optim.Adam(model.parameters(), lr=args.lr)
    # http://www.spytensor.com/index.php/archives/32/
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_feature, 'max', verbose=1, patience=10
    )
    
    progress = tqdm(
        range(args.start_epoch, args.total_epochs + 1), miniters=1,
        ncols=100, desc='Overall Progress', leave=True, position=0
    )
    
    offset = 1
    best = 0
    result = {'epoch': [], 'F_measure': [], 'MAE': []}
    for epoch in progress:
        # ===============================TRAIN=================================
        title = 'Training Epoch {}'.format(epoch)
        progress_epoch = tqdm(
            tools.IteratorTimer(train_loader), ncols=120,
            total=len(train_loader), smoothing=0.9, miniters=1,
            leave=True, position=offset, desc=title
        )
        
        train(model, progress_epoch, criterion, optimizer_feature, epoch, args)
        
        # ==============================TEST===================================
        if epoch % args.val_rate == 0:
            epoch, F_measure, mae = test(
                model, val_loader, epoch,
                prediction_root, check_root_feature, check_root_opti, val_root
            )
            
            result['epoch'].append(int(epoch))
            result['F_measure'].append(round(float(F_measure), 3))
            result['MAE'].append(round(float(mae), 3))
            df = pd.DataFrame(result).set_index('epoch')
            df.to_csv('./result.csv')
            
            if epoch == 0:
                best = F_measure - mae
            elif (F_measure - mae) > best:
                best = F_measure - mae
                # 存储最好的权重和偏置
                filename = ('%s/feature-best.pth' % check_root_feature)
                torch.save(model.state_dict(), filename)
                # 存储最好的优化器状态
                filename_opti = ('%s/opti-best.pth' % check_root_opti)
                torch.save(optimizer_feature.state_dict(), filename_opti)
            
            # 只在验证期间考虑更改学习率
            scheduler.step(best)


if __name__ == '__main__':
    main(parser.parse_args())
