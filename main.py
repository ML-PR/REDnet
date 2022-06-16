import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import REDNet10, REDNet20, REDNet30
import dataset as dg
from dataset import DenoisingDataset
from utils import AverageMeter
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet30', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--images_dir', type=str, default='data/train')
    parser.add_argument('--outputs_dir', type=str, default='weight')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--patch_size', type=int, default=161)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed) #生成随机种子

    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()

    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = dg.datagenerator(opt.images_dir, opt.patch_size, opt.batch_size)
    DDataset = DenoisingDataset(dataset, opt.sigma)
    dataloader = DataLoader(dataset=DDataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads,
                            pin_memory=True, drop_last=True)
    # 每个epoch训练
    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()
        PSNR = []
        SSIM = []
        LOSS = []
        # tqdm是训练可视化的模块
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            # 每个batch训练
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)   #模型输入

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

                ## 下面计算psnr，ssim，和loss
                p = preds
                label = labels
                preds_ps = p.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).byte().cpu().numpy()
                label_ps = label.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).byte().cpu().numpy()

                P = peak_signal_noise_ratio(label_ps, preds_ps)
                S = structural_similarity(label_ps, preds_ps, channel_axis=1)
                # 将当前batch的PSNR，SSIM，loss放入数组
                PSNR.append(P)
                SSIM.append(S)
                LOSS.append(loss)

        AVG_psnr = np.mean(PSNR)
        AVG_ssim = np.mean(SSIM)
        MSE_LOSS = np.sum(LOSS)
        print('epoch:', opt.num_epochs, '\tAVG_ssim:', AVG_ssim,'\tAVG_psnr:', AVG_psnr,'\tMSE_Loss:',MSE_LOSS,'\tepoch_loses:',epoch_losses)

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_sigma{}_epoch{}.pth'.format(opt.arch, opt.sigma, epoch)))