'''
给一个文件夹里的图片加噪声
'''
import os
import skimage.io as io
from skimage import util
import cv2

proportion=0.0025#加入多少小噪声点，可以调大
def convert(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        path = input_dir + "/" + filename#获取文件路径
        img = io.imread(path)#读取图片
        img_noised = util.random_noise(img, mode="gaussian", var=0.0225)
        io.imsave('./train2/{}_noised.png'.format(filename), img_noised)

if __name__ == '__main__':
    input_dir = "./train"#输入数据文件夹
    output_dir = "./train2"#输出数据文件夹
    convert(input_dir, output_dir)

