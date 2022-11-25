import numpy as np 
from skimage.transform import resize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import logging

def gauss2D(shape,sigma):
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

def backprojection(img_hr, img_lr, maxIter):
    #function name-> update_superresolved_img
    #variable name:->backprojection_variable_p
    # #equation 10 yima 2008 cvpr 
    p = gauss2D((5, 5), 1)
    #p = gauss2D((3, 3), 0.01)
    p = np.multiply(p, p)
    p = np.divide(p, np.sum(p))

    #bp_factor = 0.005
    bp_factor = 0.01
    logging.info("backprojection factor=%s"%str(bp_factor))
    p = p*bp_factor


    for i in range(maxIter):
        img_lr_ds = resize(img_hr, img_lr.shape, anti_aliasing=1)
        img_diff = img_lr - img_lr_ds

        img_diff = resize(img_diff, img_hr.shape)
        img_hr += convolve2d(img_diff, p, 'same')
    return img_hr