import cv2
import numpy as np


def Gaussnoise_func(image, mean=0, var=0.005):
    '''
    添加高斯噪声
    mean : 均值
    var : 方差
    '''
    image = np.array(image / 255, dtype=float)  # 将像素值归一
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 产生高斯噪声
    out = image + noise  # 直接将归一化的图片与噪声相加

    '''
    将值限制在(-1/0,1)间，然后乘255恢复
    '''
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

    # 按间距中的绿色按钮以运行脚本。


# if __name__ == '__main__':
print('test')
mean = 2;
var = 2;
img = cv2.imread("monarch.bmp")
img_r = Gaussnoise_func(img, mean / 10, var / 100)

cv2.imwrite('./monarch_noised.png', img_r)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
