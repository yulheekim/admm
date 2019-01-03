import scipy.ndimage
import skimage.measure
import numpy as np
import cv2

def fdown(x,h,K):
    # tmp = imfilter(x,h,'circular');
    tmp = scipy.ndimage.correlate(x, h, mode='wrap').transpose()
    # y = downsample2(tmp,K);
    y = np.transpose(np.transpose(tmp[0::K])[0::K])
    return y

def upf(x,h,K):
    # tmp = upsample2(x,K);
    a=np.zeros((np.shape(x)[0]*K, np.shape(x)[1]*K))
    for i in range(len(a)):
        for j in range(len(a[0])):
            if i%K==0 and j%K==0:
                a[i][j]=x[int(i/K)][int(j/K)]
            else:
                a[i][j]=0
    # y = imfilter(tmp,h,'circular');
    y = scipy.ndimage.correlate(a, h, mode='wrap').transpose()
    return y
