import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image


def get_FM(salpath, gtpath):
    """
    对于预测出的结果和真实标注计算对应的F测度和平均绝对误差

    usage:
        F_measure, mae = get_FM(
            salpath=val_output_root + '/', gtpath=val_root + '/masks/'
        )

    :param salpath: 存放预测结果的路径
    :param gtpath: 存放真实标注的路径
    :return: F_measure, mae
    """
    gtdir = gtpath
    saldir = salpath

    files = os.listdir(gtdir)
    eps = np.finfo(float).eps

    m_pres = np.zeros(21)
    m_recs = np.zeros(21)
    m_fms = np.zeros(21)
    m_thfm = 0
    m_mea = 0
    it = 1
    for i, name in enumerate(files):
        if not os.path.exists(gtdir + name):
            print(gtdir + name, 'does not exist')
        gt = Image.open(gtdir + name)
        gt = np.array(gt, dtype=np.uint8)
        mask = Image.open(saldir + name).convert('L')
        mask = mask.resize((np.shape(gt)[1], np.shape(gt)[0]))
        mask = np.array(mask, dtype=np.float)
        # 防止预测出来的结果不是灰度图
        if len(mask.shape) != 2:
            mask = mask[:, :, 0]
        # 调整预测掩膜的值到0~1
        mask = (mask - mask.min()) / (mask.max() - mask.min() + eps)
        # 将真实标注二值化
        gt[gt != 0] = 1

        pres = []
        recs = []
        fms = []
        # 这里使用的是真值减去未二值化的预测结果来计算mea
        mea = np.abs(gt - mask).mean()
        # 这里利用固定的阈值(均值的二倍），来计算对应的F测度结果
        binary = np.zeros(mask.shape)
        th = 2 * mask.mean()
        if th > 1:
            th = 1
        binary[mask >= th] = 1
        sb = (binary * gt).sum()
        pre = sb / (binary.sum() + eps)
        rec = sb / (gt.sum() + eps)
        thfm = 1.3 * pre * rec / (0.3 * pre + rec + eps)

        # 使用0-1之间的阈值， 分别进行二值化处理，与真实标注对比，计算PR曲线和动态F测度
        for th in np.linspace(0, 1, 21):
            binary = np.zeros(mask.shape)
            binary[mask >= th] = 1
            pre = (binary * gt).sum() / (binary.sum() + eps)
            rec = (binary * gt).sum() / (gt.sum() + eps)
            fm = 1.3 * pre * rec / (0.3 * pre + rec + eps)
            pres.append(pre)
            recs.append(rec)
            fms.append(fm)
        fms = np.array(fms)
        pres = np.array(pres)
        recs = np.array(recs)
        # 这里的操作实际上就是计算了对应的均值, 只是利用了一个迭代的公式而避免了动态的存储扩展
        # 这里通过书写公式可以看出来, 是通过利用那个it-1将当前与之前求平均的结果进行了关联. it-1
        # 乘以之前的结果,可以得到对应的前面所有项加和的结果.
        # 另一种可以考虑的算法就是直接加和, 来计算总的结果后, 整体求均值, 但是这样会导致较大的数据
        m_mea = m_mea * (it - 1) / it + mea / it
        m_fms = m_fms * (it - 1) / it + fms / it
        m_recs = m_recs * (it - 1) / it + recs / it
        m_pres = m_pres * (it - 1) / it + pres / it
        m_thfm = m_thfm * (it - 1) / it + thfm / it
        it += 1
    return m_thfm, m_mea


if __name__ == '__main__':
    m_thfm, m_mea = get_FM()
    print(m_thfm)
    print(m_mea)
