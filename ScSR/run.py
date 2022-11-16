import numpy as np 
from os import listdir, mkdir
from os.path import isdir
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.transform import resize
#from scipy.misc import imresize
from tqdm import tqdm
import pickle
from ScSR import ScSR
from backprojection import backprojection
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import normalize

import logging
import math
from config_run import config
import os
from skimage.exposure import match_histograms

def normalize_signal(img, channel):
    if np.mean(img[:, :, channel]) * 255 > np.mean(img_lr_ori[:, :, channel]):
        ratio = np.mean(img_lr_ori[:, :, channel]) / (np.mean(img[:, :, channel]) * 255)
        img[:, :, channel] = np.multiply(img[:, :, channel], ratio)
    elif np.mean(img[:, :, channel]) * 255 < np.mean(img_lr_ori[:, :, channel]):
        ratio = np.mean(img_lr_ori[:, :, channel]) / (np.mean(img[:, :, channel]) * 255)
        img[:, :, channel] = np.multiply(img[:, :, channel], ratio)
    return img[:, :, channel]

def normalize_max(img):
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if img[m, n, 0] > 1:
                img[m, n, 0] = 1
            if img[m, n, 1] > 1:
                img[m, n, 1] = 1
            if img[m, n, 2] > 1:
                img[m, n, 2] = 1
    return img

#################################################################################

# Set which dictionary you want to use
D_size = 2048
US_mag = 3
lmbd = 0.1
patch_size= 3

dict_name = str(D_size) + '_US' + str(US_mag) + '_L' + str(lmbd) + '_PS' + str(patch_size)

with open('data/dicts/Dh_' + dict_name + '.pkl', 'rb') as f:
    Dh = pickle.load(f)
Dh = normalize(Dh)
with open('data/dicts/Dl_' + dict_name + '.pkl', 'rb') as f:
    Dl = pickle.load(f)
Dl = normalize(Dl)

### SET PARAMETERS
img_lr_dir = config.val_lr_path
img_hr_dir = config.val_hr_path
overlap = 1
lmbd = 0.1
upscale = 3
maxIter = 100

img_type = '.png'

#################################################################################

log_format = "%(asctime)s | %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(config.output_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("initialize logging")
logging.info("args = %s", str(config))

#################################################################

img_lr_file = listdir(img_lr_dir)
img_lr_file = [item for item in img_lr_file if img_type in item]
logging.info('img_lr_file: ',img_lr_file)

#for i in tqdm(range(len(img_lr_file))):
for i in range(len(img_lr_file)):
    logging.info("image number %d"%i)
    # Read test image
    img_name = img_lr_file[i]
    img_name_dir = list(img_name)
    img_name_dir = np.delete(np.delete(np.delete(np.delete(img_name_dir, -1), -1), -1), -1)
    img_name_dir = ''.join(img_name_dir)
    logging.info(img_name_dir)
    img_lr = imread( os.path.join(*[img_lr_dir, img_name]) )

    # Read and save ground truth image
    img_hr = imread( os.path.join(*[img_hr_dir, img_name]) )
    imsave(os.path.join(*[config.output_dir,"%04d_3HR.png"%i]), img_hr)
    img_hr_y = rgb2ycbcr(img_hr)[:, :, 0]

    # Change color space
    img_lr_ori = img_lr
    temp = img_lr
    img_lr = rgb2ycbcr(img_lr)
    img_lr_y = img_lr[:, :, 0]
    img_lr_cb = img_lr[:, :, 1]
    img_lr_cr = img_lr[:, :, 2]

    # Upscale chrominance to color SR images
    img_sr_cb = resize(img_lr_cb, (img_hr.shape[0], img_hr.shape[1]), order=0)
    img_sr_cr = resize(img_lr_cr, (img_hr.shape[0], img_hr.shape[1]), order=0)

    # Super Resolution via Sparse Representation
    logging.info('Before ScSR call')
    img_sr_y = ScSR(img_lr_y, img_hr_y.shape, upscale, Dh, Dl, lmbd, overlap)
    logging.info('After ScSR call')
    img_sr_y = backprojection(img_sr_y, img_lr_y, maxIter)
    logging.info('Backprojection done')

    img_sr_hmatched_y = match_histograms(image=img_sr_y,reference=img_lr_y,channel_axis=None)
    logging.info('img_sr_y: '+str(img_sr_y))
    logging.info('img_sr_hmatched_y: '+str(img_sr_hmatched_y))

    # Create colored SR images
    img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
    img_sr_hmatched = np.stack((img_sr_hmatched_y, img_sr_cb, img_sr_cr), axis=2)

    logging.info('img_sr: ',img_sr)
    with open(os.path.join(*[config.output_dir,'%04d_2SR_ycbcr.pkl'%i]), 'wb') as f:
        pickle.dump(img_sr, f, pickle.HIGHEST_PROTOCOL)

    img_sr = ycbcr2rgb(img_sr)
    img_sr_hmatched = ycbcr2rgb(img_sr_hmatched)
    
    imsave(os.path.join(*[config.output_dir,'%04d_2SR_y_b4postproc.png'%i]), img_sr_y)
    imsave(os.path.join(*[config.output_dir,'%04d_2SR_b4postproc.png'%i]), img_sr)

    # Signal normalization
    for channel in range(img_sr.shape[2]):
        img_sr[:, :, channel] = normalize_signal(img_sr, channel)

    # Maximum pixel intensity normalization
    img_sr = normalize_max(img_sr)

    # Bicubic interpolation for reference
    img_bc = resize(img_lr_ori, (img_hr.shape[0], img_hr.shape[1]))
    logging.info('img_bc: ',img_bc)
    imsave(os.path.join(*[config.output_dir,'%04d_1bicubic.png'%i]), img_bc)
    img_bc_y = rgb2ycbcr(img_bc)[:, :, 0]

    # Compute RMSE for the illuminance
    rmse_bc_hr = np.sqrt(mean_squared_error(img_hr_y, img_bc_y))
    rmse_bc_hr = np.zeros((1,)) + rmse_bc_hr
    rmse_sr_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_y))
    rmse_sr_hr = np.zeros((1,)) + rmse_sr_hr
    rmse_srhm_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_hmatched_y))
    rmse_srhm_hr = np.zeros((1,)) + rmse_srhm_hr
    #np.savetxt(os.path.join(*[config.output_dir, 'RMSE_bicubic.txt']), rmse_bc_hr)
    #np.savetxt(os.path.join(*[config.output_dir, 'RMSE_SR.txt']), rmse_sr_hr)
    logging.info('bicubic RMSE: '+str(rmse_bc_hr))
    logging.info('SR RMSE: '+str(rmse_sr_hr))
    logging.info('SR Hmatched RMSE: '+str(rmse_srhm_hr))

    y_psnr_bc_hr = 20*math.log10(255.0/rmse_bc_hr)
    y_psnr_sr_hr = 20*math.log10(255.0/rmse_sr_hr)
    y_psnr_srhm_hr = 20*math.log10(255.0/rmse_srhm_hr)
    logging.info('bicubic Y-Channel PSNR: '+str(y_psnr_bc_hr))
    logging.info('SR Y-Channel PSNR: '+str(y_psnr_sr_hr))
    logging.info('SRHM Y-Channel PSNR: '+str(y_psnr_srhm_hr))

    logging.info('img_sr: ',img_sr)
    imsave(os.path.join(*[config.output_dir,'%04d_2SR.png'%i]), img_sr)
    with open(os.path.join(*[config.output_dir,'%04d_2SR.pkl'%i]), 'wb') as f:
        pickle.dump(img_sr, f, pickle.HIGHEST_PROTOCOL)
    logging.info('img_srhm: ',img_sr_hmatched)
    imsave(os.path.join(*[config.output_dir,'%04d_2SRHM.png'%i]), img_sr_hmatched)
    with open(os.path.join(*[config.output_dir,'%04d_2SRHM.pkl'%i]), 'wb') as f:
        pickle.dump(img_sr_hmatched, f, pickle.HIGHEST_PROTOCOL)

    img_sr_testing = (np.clip(img_sr_hmatched,0,1)*255).astype(np.uint8)
    logging.info('img_sr_testing: ',img_sr_testing)
    imsave(os.path.join(*[config.output_dir,'%04d_2SR_testing.png'%i]), img_sr_testing)
    with open(os.path.join(*[config.output_dir,'%04d_2SR_testing.pkl'%i]), 'wb') as f:
        pickle.dump(img_sr_testing, f, pickle.HIGHEST_PROTOCOL)
