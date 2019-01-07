import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from ADMM_super import ADMM_super

# ------------------------fspecial() in MATLAB to python3-----------------------
# https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python
# By rayryeng
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# ------------------------psnr() in MATLAB to python3---------------------------
# https://dsp.stackexchange.com/questions/38065/peak-signal-to-noise-ratio-psnr-in-python-for-an-image
# By Himanshu Tyagi and edited by Peter K.
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# original image
img = cv2.imread('./data/Couple512.png', cv2.IMREAD_GRAYSCALE) # Read image here
z = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
print(np.shape(z))

# blur kernel and downsampling factor
h = matlab_style_gauss2D((9, 9),1);

K = 2;
noise_level = 10/255;

# calculate the observed image
# y = imfilter(z,h,'circular');
y = scipy.ndimage.correlate(z, h, mode='wrap').transpose()
# y = downsample2(y,K);
y = np.transpose(np.transpose(y[0::K])[0::K])
y = y + noise_level*np.random.randn(len(y), len(y[0]))

# # %parameters
method = 'BM3D'
if method == 'RF':
    lam = 0.0002
elif method == 'NLM' or method == 'BM3D':
    lam = 0.001
else:
    lam = 0.01

# # %optional parameters
opts={}
opts['rho']     = 1
opts['gamma']   = 1
opts['max_itr'] = 20
opts['print']   = True
#
# # %main routine
out = ADMM_super(y,h,K,lam,method,opts)

# # %display
PSNR_output = psnr(out,z);
tt = 'PSNR = ' + str(PSNR_output) + ' dB'
print(tt)

plt.figure(1)
plt.subplot(121)
plt.title('Input')
plt.imshow(y.transpose(), cmap='gray')

plt.subplot(122)
plt.imshow(out, cmap='gray')
plt.title(tt)
plt.show()
