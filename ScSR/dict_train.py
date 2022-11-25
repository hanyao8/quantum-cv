import numpy as np 
from rnd_smp_patch import rnd_smp_patch
from patch_pruning import patch_pruning
from spams import trainDL
import pickle

from config_train import config
import logging
import os
from sklearn.decomposition import dict_learning_online
from sklearn.decomposition import DictionaryLearning

# ========================================================================
# Demo codes for dictionary training by joint sparse coding
# 
# Reference
#   J. Yang et al. Image super-resolution as sparse representation of raw
#   image patches. CVPR 2008.
#   J. Yang et al. Image super-resolution via sparse representation. IEEE 
#   Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010
# 
# Jianchao Yang
# ECE Department, University of Illinois at Urbana-Champaign
# For any questions, send email to jyang29@uiuc.edu
# =========================================================================

#dict_size   = 2048         # dictionary size
#lmbd        = 0.1          # sparsity regularization
#patch_size  = 3            # image patch size
#nSmp        = 100000       # number of patches to sample
#upscale     = 3            # upscaling factor

dict_size   = config.dict_size         # dictionary size
lmbd        = config.lmbd          # sparsity regularization
patch_size  = config.patch_size            # image patch size
nSmp        = config.nSmp       # number of patches to sample
upscale     = config.upscale            # upscaling factor

train_img_path = 'data/train_hr/'   # Set your training images dir

################################################################################

log_format = "%(asctime)s | %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(config.output_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("initialize logging")
logging.info("args = "+str(config))

################################################################################

# Randomly sample image patches
Xh, Xl = rnd_smp_patch(train_img_path, patch_size, nSmp, upscale)

# Prune patches with small variances
Xh, Xl = patch_pruning(Xh, Xl)
Xh = np.asfortranarray(Xh)
Xl = np.asfortranarray(Xl)

logging.info("Xh shape: "+str(Xh.shape))
logging.info("Xl shape: "+str(Xl.shape))

# Dictionary learning
logging.info("Learning Dh")
Dh = trainDL(Xh, K=dict_size, lambda1=lmbd, iter=100)

logging.info("Learning Dl")
Dl = trainDL(Xl, K=dict_size, lambda1=lmbd, iter=100)
#Dh = dict_learning_online(Xh,n_components=dict_size,alpha=lmbd,max_iter=100,verbose=True)
#Dl = dict_learning_online(Xl,n_components=dict_size,alpha=lmbd,max_iter=100,verbose=True)

#Dh_dict_learner = DictionaryLearning(n_components=dict_size,transform_algorithm='omp',transform_n_nonzero_coefs=dict_size,verbose=True)
#Dl_dict_learner = DictionaryLearning(n_components=dict_size,transform_algorithm='omp',transform_n_nonzero_coefs=dict_size,verbose=True)

logging.info("learner initialized")


#Dh = Dh_dict_learner.fit_transform(Xh)

#Dl = Dh_dict_learner.fit_transform(Xl)

logging.info("Done")

logging.info(Dh.shape)
logging.info(Dl.shape)

# Saving dictionaries to files
#with open('data/dicts/'+ 'Dh_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
#    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)
with open('data/dicts/'+ 'Dh_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '_test_' + config.exp_name + '.pkl', 'wb') as f:
    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)

#with open('data/dicts/'+ 'Dl_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
#    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
with open('data/dicts/'+ 'Dl_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '_test_' + config.exp_name + '.pkl', 'wb') as f:
    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
