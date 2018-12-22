import argparse
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import os
from matplotlib.pyplot import imsave
from dataset import MyData, MyTestData
from model import Feature
from model import RCL_Module
import utils.tools as tools
from tqdm import tqdm
from utils.evaluateFM import get_FM
import pandas as pd

parser = argparse.ArgumentParser(description='DHS-Pytorch')
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--total_epochs', default=100, type=int)

parser.add_argument('--dataset', default='DUTS', type=str)
parser.add_argument('--lr', default=1e-5)
parser.add_argument('--data_root', required=True)
parser.add_argument('--cache', default='./cache/')
parser.add_argument('--pre', default='./prediction')
parser.add_argument('--val_rate', default=2)  # validate the model every n epoch


def main(args):
    dataset = args.dataset
    bsize = args.batch_size
    root = args.data_root
    cache_root = args.cache
    prediction_root = args.pre

    train_root = root + dataset + '/train'
    val_root = root + dataset + '/val'  # validation dataset

    check_root_opti = cache_root + '/opti'  # save checkpoint parameters
    if not os.path.exists(check_root_opti):
        os.mkdir(check_root_opti)

    check_root_feature = cache_root + '/feature'  # save checkpoint parameters
    if not os.path.exists(check_root_feature):
        os.mkdir(check_root_feature)

    train_loader = torch.utils.data.DataLoader(
        MyData(train_root, transform=True),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        MyTestData(val_root, transform=True),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)

    model = Feature(RCL_Module)
    model.cuda()
    criterion = nn.BCELoss()
    optimizer_feature = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []

    progress = tqdm(range(args.start_epoch, args.total_epochs + 1), miniters=1,
                    ncols=100, desc='Overall Progress', leave=True, position=0)
    offset = 1

    best = 0
    evaluation = []
    result = {'epoch': [], 'F_measure': [], 'MAE': []}
    for epoch in progress:
        if (epoch != 0):
            print("\nloading parameters")
            model.load_state_dict(
                torch.load(check_root_feature + '/feature-current.pth'))
            optimizer_feature.load_state_dict(
                torch.load(check_root_opti + '/opti-current.pth'))
            #
        title = 'Training Epoch {}'.format(epoch)
        progress_epoch = tqdm(tools.IteratorTimer(train_loader), ncols=120,
                              total=len(train_loader), smoothing=.9,
                              miniters=1,
                              leave=True, position=offset, desc=title)

        for ib, (input, gt) in enumerate(progress_epoch):
            inputs = Variable(input).cuda()
            gt = Variable(gt.unsqueeze(1)).cuda()
            gt_28 = functional.interpolate(gt, size=28, mode='bilinear')
            gt_56 = functional.interpolate(gt, size=56, mode='bilinear')
            gt_112 = functional.interpolate(gt, size=112, mode='bilinear')

            msk1, msk2, msk3, msk4, msk5 = model.forward(inputs)

            loss = criterion(msk1, gt_28) + criterion(msk2, gt_28) + criterion(
                msk3,
                gt_56) + criterion(
                msk4, gt_112) + criterion(msk5, gt)
            model.zero_grad()
            loss.backward()
            optimizer_feature.step()

            train_losses.append(round(float(loss.data.cpu()), 3))
            title = '{} Epoch {}/{}'.format('Training', epoch,
                                            args.total_epochs)
            progress_epoch.set_description(
                title + ' ' + 'loss:' + str(loss.data.cpu().numpy()))

        filename = ('%s/feature-current.pth' % (check_root_feature))
        filename_opti = ('%s/opti-current.pth' % (check_root_opti))
        torch.save(model.state_dict(), filename)
        torch.save(optimizer_feature.state_dict(), filename_opti)

        # --------------------------validation on the test set every n epoch--------------
        if (epoch % args.val_rate == 0):
            fileroot = ('%s/feature-current.pth' % (check_root_feature))
            model.load_state_dict(torch.load(fileroot))
            val_output_root = (prediction_root + '/epoch_current')
            if not os.path.exists(val_output_root):
                os.mkdir(val_output_root)
            print("\ngenerating output images")
            for ib, (input, img_name, _) in enumerate(val_loader):
                inputs = Variable(input).cuda()
                _, _, _, _, output = model.forward(inputs)
                output = functional.sigmoid(output)
                out = output.data.cpu().numpy()
                for i in range(len(img_name)):
                    imsave(os.path.join(val_output_root, img_name[i] + '.png'),
                           out[i, 0],
                           cmap='gray')

            print("\nevaluating mae....")
            F_measure, mae = get_FM(salpath=val_output_root + '/',
                                    gtpath=val_root + '/gt/')
            evaluation.append([int(epoch), float(F_measure), float(mae)])
            result['epoch'].append(int(epoch))
            result['F_measure'].append(round(float(F_measure), 3))
            result['MAE'].append(round(float(mae), 3))
            df = pd.DataFrame(result).set_index('epoch')
            df.to_csv('./result.csv')

            if (epoch == 0):
                best = F_measure - mae
            elif ((F_measure - mae) > best):
                best = F_measure - mae
                filename = ('%s/feature-best.pth' % (check_root_feature))
                filename_opti = ('%s/opti-best.pth' % (check_root_opti))
                torch.save(model.state_dict(), filename)
                torch.save(optimizer_feature.state_dict(), filename_opti)


if __name__ == '__main__':
    main(parser.parse_args())
