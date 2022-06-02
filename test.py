import argparse
import os
import glob
import torch
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import REDNet10, REDNet20, REDNet30
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet30', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--weights_path', type=str, default='weight/REDNet30_sigma25_epoch0.pth')
    parser.add_argument('--image_path', type=str, default='data/Set68')
    parser.add_argument('--outputs_dir', type=str, default='data1')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()

    file_list = glob.glob(opt.image_path + '/*.png')  # get name list of all .png files

    PSNR = []
    SSIM = []
    with torch.no_grad():
        for i in range(len(file_list)):
            img = cv2.imread(file_list[i], 0)
            img1 = np.array(img, dtype=np.float32) / 255.0
            np.random.seed(seed=0)
            img_noised = img1 + np.random.normal(0, opt.sigma/255.0, img.shape)
            img_noised = img_noised.astype(np.float32)

            filename = os.path.basename(file_list[i]).split('.')[0]
            cv2.imwrite(opt.outputs_dir + "/{}_noised_s{}.png".format(filename, opt.sigma), img_noised*255) # save noised image

            img_noised = transforms.ToTensor()(img_noised).unsqueeze(0).to(device)
            pre_img = model(img_noised)
            pre_img = pre_img.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).squeeze(2).byte().cpu().numpy()

            cv2.imwrite(opt.outputs_dir + "/{}_{}_s{}.png".format(filename, opt.arch, opt.sigma), pre_img)  # save have cleaned the-noised-image

            P = peak_signal_noise_ratio(img, pre_img)
            S = structural_similarity(img, pre_img, channel_axis=1)
            PSNR.append(P)
            SSIM.append(S)

    AVG_psnr = np.mean(PSNR)
    AVG_ssim = np.mean(SSIM)
    print('AVG_psnr:', AVG_psnr)
    print('AVG_ssim:', AVG_ssim)
