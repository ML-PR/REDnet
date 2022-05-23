import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

stride = 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name,patch_size):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(images_dir, patch_size, batch_size, verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(images_dir+'/*.png')  # get name list of all .png files
    # initrialize
    dataset = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i], patch_size)
        for patch in patches:
            dataset.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    dataset = np.array(dataset, dtype='uint8')
    dataset = np.expand_dims(dataset, axis=3)
    discard_n = len(dataset)-len(dataset)//batch_size*batch_size
    # because of batch namalization
    dataset = np.delete(dataset, range(discard_n), axis=0)
    dataset = dataset.astype('float32') / 255.0
    dataset = torch.from_numpy(dataset.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
    return dataset


if __name__ == '__main__':

    data = datagenerator('data/train', 40, 16)
    # print(data[0].shape)

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import io
import numpy as np
import PIL.Image as pil_image

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)


class Dataset(object):
    def __init__(self, images_dir, patch_size, sigma, use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.sigma = sigma   # noise level
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            label = tf.read_file(self.image_files[idx])
            label = tf.image.decode_jpeg(label, channels=1)  # 将JPEG编码图像解码为uint8张量.
            label = pil_image.fromarray(label.numpy())  # array转换成image
        else:
            label = pil_image.open(self.image_files[idx])  # mode:L

        # randomly crop patch from training set
        # crop_x = random.randint(0, label.width - self.patch_size)
        # crop_y = random.randint(0, label.height - self.patch_size)
        # label = label.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))  # patch_size * patch_size

        # additive jpeg noise
        # buffer = io.BytesIO()
        # label.save(buffer, format='jpeg', quality=self.jpeg_quality)
        # input = pil_image.open(buffer)

        input = np.array(input).astype(np.float32)
        label = np.array(label).astype(np.float32)

        input = np.transpose(input,axes=[0, 1])
        label = np.transpose(label,axes=[0, 1])

        # normalization
        input /= 255.0
        label /= 255.0

        return input, label

    def __len__(self):
        return len(self.image_files)
'''