import cv2
import numpy as np
import math

im0 = cv2.imread('out_flow_image_1.png')
im1 = cv2.imread('out_flow_translated_pytorch.png')
#im1 = cv2.imread('out_flow_translated_dain_wo_sampling.png')

gt_rgb = np.array(im0, dtype=np.float32)
rec_rgb = np.array(im1, dtype=np.float32)

cv2.imwrite('diff.png', gt_rgb - rec_rgb)

diff_rgb = 128.0 + rec_rgb - gt_rgb
avg_interp_error_abs = np.mean(np.abs(diff_rgb - 128.0))

mse = np.mean((diff_rgb - 128.0) ** 2)

PIXEL_MAX = 255.0
psnr = 20 * math.log10(PIXEL_MAX / (math.sqrt(mse)+np.spacing(1)))

print('psnr', psnr)
print('l2', mse)


