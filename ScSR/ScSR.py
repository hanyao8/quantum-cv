import numpy as np 
from os import listdir
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.transform import resize
import pickle
from featuresign import fss_yang
from scipy.signal import convolve2d
from tqdm import tqdm

from sklearn import linear_model

import my_algorithms

import logging
from config_run import config

def extract_lr_feat(img_lr):
    h, w = img_lr.shape
    img_lr_feat = np.zeros((h, w, 4))

    # First order gradient filters
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    img_lr_feat[:, :, 0] = convolve2d(img_lr, hf1, 'same')
    img_lr_feat[:, :, 1] = convolve2d(img_lr, vf1, 'same')

    # Second order gradient filters
    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    img_lr_feat[:, :, 2] = convolve2d(img_lr, hf2, 'same')
    img_lr_feat[:, :, 3] = convolve2d(img_lr, vf2, 'same')

    return img_lr_feat

def create_list_step(start, stop, step):
    list_step = []
    for i in range(start, stop, step):
        list_step = np.append(list_step, i)
    return list_step

def lin_scale(xh, us_norm):
    hr_norm = np.sqrt(np.sum(np.multiply(xh, xh)))

    if hr_norm > 0:
        lin_scale_factor = 1.2
        #logging.info("lin_scale_factor=%s"%str(lin_scale_factor))
        s = us_norm * lin_scale_factor / hr_norm
        #s = us_norm * 20.0 / hr_norm
        #s = us_norm * 1.2 / hr_norm
        #logging.info("s=%s"%str(s))
        xh = np.multiply(xh, s)
    return xh

def ScSR(img_lr_y, size, upscale, Dh, Dl, lmbd, overlap):

    #patch_size = 3
    patch_size = config.patch_size

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_hr = np.zeros(img_us.shape)
    img_hr_ctrl = np.zeros(img_us.shape)
    cnt_matrix = np.zeros(img_us.shape)

    #img_lr_y_feat = extract_lr_feat(img_hr)
    img_lr_y_feat = extract_lr_feat(img_us)

    gridx = np.append(create_list_step(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)

    count = 0

    logging.info(gridx)
    logging.info(gridy)

    cardinality = np.zeros(len(gridx)*len(gridy))

    for m in tqdm(range(0, len(gridx))):
        logging.info("Inside ScSR loop, iteration=%d"%m)
    #for m in range(0, len(gridx)):
        for n in range(0, len(gridy)):
            count += 1
            xx = int(gridx[m])
            yy = int(gridy[n])

            us_patch = img_us[yy : yy + patch_size, xx : xx + patch_size]
            us_mean = np.mean(np.ravel(us_patch, order='F'))
            us_patch = np.ravel(us_patch, order='F') - us_mean
            us_norm = np.sqrt(np.sum(np.multiply(us_patch, us_patch)))

            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order='F')
            feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))

            if feat_norm > 1:
                y = np.divide(feat_patch, feat_norm)
            else:
                y = feat_patch

            #b = np.dot(np.multiply(Dl.T, -1), y)
            #if len(b.shape)==1:
            #    b = b.reshape((b.shape[0],1))
            #print('fss_yang arg shapes: ')
            #print(Dl.shape)
            #A = np.matmul(Dl.T,Dl)
            #print(A.shape)
            #print(b.shape)
            #w = fss_yang(lmbd, A, b)

            if config.sc_algo=="sklearn_lasso":
                #reg = linear_model.Lasso(alpha=lmbd)
                #lasso_alpha = 1e-3
                #logging.info("lasso_alpha=%s"%str(lasso_alpha))
                reg = linear_model.Lasso(alpha=config.lasso_alpha,max_iter=1000)
                #print(Dl.shape)
                #print(y.shape)
                reg.fit(Dl,y)
                w = reg.coef_
                #logging.info("w="+str(w))
                
            elif config.sc_algo=="qubo_lasso":
                #w = qubo_lasso(Dl,y,alpha=0.1)
                w = my_algorithms.qubo_lasso(Dl,y,alpha=config.lasso_alpha)
            elif config.sc_algo=="qubo_bsc":
                #w = qubo_lasso(Dl,y,alpha=0.1)
                w = my_algorithms.qubo_bsc(Dl,y,alpha=config.bsc_alpha,h_bar=config.bsc_h_bar)
                if n==0:
                    logging.info("m=%d, n=0, w="%(m)+str(w))
                    logging.info("0norm="+str(np.matmul(np.where(np.abs(w)>0,1,0),np.ones(len(w)))))
            elif config.sc_algo=="fss":
                b = np.dot(np.multiply(Dl.T, -1), y)
                if len(b.shape)==1:
                    b = b.reshape((b.shape[0],1))
                #print('fss_yang arg shapes: ')
                #print(Dl.shape)
                A = np.matmul(Dl.T,Dl)
                #print(A.shape)
                #print(b.shape)
                w = fss_yang(lmbd, A, b)
                #logging.info("w="+str(w))

            cardinality[count-1] = np.matmul(np.where(np.abs(w)>0,1,0),np.ones(len(w)))

            hr_patch = np.dot(Dh, w)
            #logging.info(hr_patch)
            hr_patch = lin_scale(hr_patch, us_norm)
            #logging.info(us_norm)
            #logging.info(hr_patch)

            hr_patch = np.reshape(hr_patch, (patch_size, -1))
            hr_patch += us_mean
            #logging.info(us_mean)
            #logging.info(hr_patch)

            #img_hr[yy : yy + patch_size, xx : xx + patch_size] += hr_patch
            img_hr[yy : yy + patch_size, xx : xx + patch_size] = img_hr[yy : yy + patch_size, xx : xx + patch_size] + hr_patch
            img_hr_ctrl[yy : yy + patch_size, xx : xx + patch_size] = img_hr_ctrl[yy : yy + patch_size, xx : xx + patch_size] + us_mean
            #img_hr[yy : yy + patch_size, xx : xx + patch_size] = hr_patch
            #cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] += 1
            #logging.info("cnt_matrix")
            #logging.info(cnt_matrix[yy : yy + patch_size, xx : xx + patch_size])
            cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] = cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] + 1.0
            #logging.info(cnt_matrix[yy : yy + patch_size, xx : xx + patch_size])

    #logging.info(np.where(cnt_matrix < 1))
    #index = np.where(cnt_matrix < 1)[0]
    index_y,index_x = np.where(cnt_matrix < 1)
    #logging.info(index_y)
    #logging.info(index_x)
    #logging.info(index)
    assert len(index_y)==len(index_x)
    for i in range(len(index_y)):
        yy = index_y[i]
        xx = index_x[i]
        img_hr[yy][xx] = img_us[yy][xx]
        img_hr_ctrl[yy][xx] = img_us[yy][xx]
        cnt_matrix[yy][xx] = 1.0

    #img_hr[index] = img_us[index]
    #cnt_matrix[index] = 1
    img_hr = np.divide(img_hr, cnt_matrix)
    img_hr_ctrl = np.divide(img_hr_ctrl, cnt_matrix)
    #logging.info(cnt_matrix)
    #logging.info(cnt_matrix[40:60,40:60])

    logging.info("cardinality="+str(cardinality))
    logging.info("avg_cardinality="+str(np.mean(cardinality)))

    #return img_hr,img_us
    return img_hr,img_hr_ctrl
