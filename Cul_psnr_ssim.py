'''
计算PSNR SSIM
'''
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import skimage.io as io

img = io.imread('./data1/monarch.bmp')
img_Noise=io.imread('./data1/m4_REDNet30.png')
#var是标准化后的sigma**2
#峰值信噪比
PSNR = psnr(img, img_Noise);
print('PSNR:', PSNR)

#结构相似性
SSIM = ssim(img, img_Noise,channel_axis=2)
print('SSIM:', SSIM)