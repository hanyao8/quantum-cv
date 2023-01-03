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

from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite

import os
import dimod

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

    if config.sc_algo=="qubo_bsc_dwave1": #submit to DWave hybrid solver
        logging.info("running qubo_bsc_dwave1 in ScSR")
        n_patches_per_qubo = 8
        qubo_size = n_patches_per_qubo*Dl.shape[1]
        client = Client.from_config(token=config.dwave_token)
        solver = client.get_solver(name='hybrid_binary_quadratic_model_version2')
        Q_dicts = my_algorithms.create_qubo1(img_lr_y, size, Dl, overlap, n_patches_per_qubo)
        flattened_m = np.zeros(qubo_size*len(Q_dicts))
        logging.info("Number of QUBO problems to solve: %d"%(len(Q_dicts)))
        #raise(Exception)
        total_qpu_access_time = 0
        for i in range(len(Q_dicts)):
            dwave_real_run = False
            if dwave_real_run:
                logging.info("i=%d"%i)
                computation = solver.sample_qubo(Q_dicts[i],time_limit=3.5)
                logging.info(str(computation.id))
                logging.info(str(computation.variables))
                logging.info(str(computation.result))
                logging.info(str(computation.energies))
                logging.info(str(computation.samples))
                #print(computation.sampleset.to_pandas_dataframe())
                logging.info(str(computation.sampleset.variables))
                logging.info(str(computation.sampleset.info))
                total_qpu_access_time += computation.sampleset.info['qpu_access_time']
                flattened_m[i*qubo_size:i*qubo_size+len(computation.samples[0])] = computation.samples[0]
                #raise(Exception)
            
        logging.info("total_qpu_access_time="+str(total_qpu_access_time))

        with open(os.path.join(*[config.output_dir,'qubo_bsc_dwave_flattened_mw.pkl']), 'wb') as f:
            pickle.dump(flattened_m*config.bsc_h_bar, f, pickle.HIGHEST_PROTOCOL)

    elif config.sc_algo=="qubo_bsc_dwave2": #submit to Dwave pure quantum solver
        logging.info("running qubo_bsc_dwave1 in ScSR")
        qubo_size = config.qubo_size

        sampler_advantage = DWaveSampler(solver={'topology__type':'pegasus'})
        ec_advantage = EmbeddingComposite(sampler_advantage)

        Q_dicts,sp_map_index,flattened_m = my_algorithms.create_qubo2(img_lr_y, size, Dl, overlap)
        #flattened_m = np.zeros(len(gridx)*len(gridy)*Dl.shape[1])

        dwave_samplesets = []
        logging.info("Number of QUBO problems to solve: %d"%(len(Q_dicts)))
        total_qpu_access_time = 0
        for i in range(len(Q_dicts)):
            dwave_real_run = False
            
            logging.info("i=%d"%i)
            if dwave_real_run:
                sampleset = ec_advantage.sample_qubo(Q_dicts[i],num_reads=config.num_reads)
            else:
                Q_tmp = np.random.randint(2,size=Dl.shape[1],dtype=np.int8)
                Q_tmp_dict = {i:Q_tmp[i] for i in range(Dl.shape[1])}
                sampleset = dimod.SampleSet.from_samples(dimod.as_samples(Q_tmp_dict), 'BINARY', 0)
            dwave_samplesets.append(sampleset)
            logging.info(str(sampleset.info))
            logging.info("sampleset size="+str(len(sampleset)))
            #to implement: clamping

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
            elif config.sc_algo=="qubo_bsc_dwave1":
                w = flattened_m[count*Dl.shape[1]:(count+1)*Dl.shape[1]]
                if n==0:
                    logging.info("m=%d, n=0, w="%(m)+str(w))
                    logging.info("0norm="+str(np.matmul(np.where(np.abs(w)>0,1,0),np.ones(len(w)))))
            elif config.sc_algo=="qubo_bsc_dwave2":
                w = flattened_m[count*Dl.shape[1]:(count+1)*Dl.shape[1]]
                if n==0:
                    logging.info("From Tabu search, yet to include quantum results")
                    logging.info("m=%d, n=0, w="%(m)+str(w))
                    logging.info("0norm="+str(np.matmul(np.where(np.abs(w)>0,1,0),np.ones(len(w)))))
                subproblems_per_qubo = int(qubo_size/config.subproblem_size)
                patch_samples = np.zeros((config.num_passes,config.num_reads,config.subproblem_size))
                patch_energies = np.zeros((config.num_passes,config.num_reads))
                patch_occurrences = np.zeros((config.num_passes,config.num_reads))
                for i in range(config.n_passes):
                    sampleset_no = int(count/subproblems_per_qubo)*config.n_passes+i
                    offset = (count%subproblems_per_qubo)*config.subproblem_size
                    sampleset = dwave_samplesets[sampleset_no]
                    for j in range(len(sampleset.record)):
                        patch_samples[i][j] = sampleset.record[0][0][offset:offset+config.subproblem_size]
                        patch_energies[i][j] = sampleset.record[0][1]
                        patch_occurrences[i][j] = sampleset.record[0][2]
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

            beta = 0.01
            if config.sc_algo=="qubo_bsc_dwave2":
                hr_patch = np.zeros(Dh.shape[0])
                p = np.zeros(config.num_reads*config.n_passes)
                Z = 0 #partition function
                for j in range(config.num_reads):
                    for k in range(config.n_passes):
                        w_tmp = w.copy()
                        w_tmp[sp_map_index[count][k]] = patch_samples[k][j]
                        pZ = np.exp(-beta*patch_energies[k][j])*patch_occurrences[k][j]
                        p[j*config.n_passes+k] = pZ
                        Z += pZ
                        hr_patch += pZ*np.dot(Dh, w_tmp)
                hr_patch = hr_patch/Z
                p = p/Z
                gibbs_entropy = -np.sum(p*np.log(p+1e-9))
                logging.info("Z="+str(Z))
                logging.info("p sum check: "+str(np.sum(p)))
                logging.info("gibbs_entropy="+str(gibbs_entropy))
            else:
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
