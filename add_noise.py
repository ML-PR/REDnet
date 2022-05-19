'''
给单个图片添加噪声
'''
import skimage.io as io
from skimage import util

img = io.imread('./data1/monarch.bmp')
img_noised = util.random_noise(img, mode="gaussian", var=0.01)
io.imsave('./data1/m5.png', img_noised)
#io.imshow(img)
#io.show()
#io.imshow(img_noised)
#io.show()
