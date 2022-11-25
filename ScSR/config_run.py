from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C
C.seed = 12345

############################################

#mode_val = "1img" #inference settings
mode_val = "1img_small"
#mode_val = "valset"

#algorithm for sparse coding
C.sc_algo = "sklearn_lasso"
C.lasso_alpha = 1e-4
#C.sc_algo = "qubo_lasso"
#C.sc_algo = "fss"

#C.Dl_path = "data/dicts/Dl_512_US3_L0.1_PS5.pkl"
#C.Dh_path = "data/dicts/Dh_512_US3_L0.1_PS5.pkl"
#C.Dl_path = "data/dicts/Dl_2048_US3_L0.1_PS3_test_exp_train_2022_11_22_16_43_24.pkl"
#C.Dh_path = "data/dicts/Dh_2048_US3_L0.1_PS3_test_exp_train_2022_11_22_16_43_24.pkl"
#C.Dl_path = "data/dicts/Dl_512_US3_L0.1_PS5_test_exp_train_2022_11_22_17_01_54.pkl"
#C.Dh_path = "data/dicts/Dh_512_US3_L0.1_PS5_test_exp_train_2022_11_22_17_01_54.pkl"
C.Dl_path = "data/dicts/Dl_128_US3_L0.1_PS5_test_exp_train_2022_11_22_17_08_59.pkl"
C.Dh_path = "data/dicts/Dh_128_US3_L0.1_PS5_test_exp_train_2022_11_22_17_08_59.pkl"
#C.Dl_path = "data/dicts/Dl_32_US3_L0.1_PS5_test_exp_train_2022_11_22_17_13_46.pkl"
#C.Dh_path = "data/dicts/Dh_32_US3_L0.1_PS5_test_exp_train_2022_11_22_17_13_46.pkl"
#C.Dl_path = "data/dicts/Dl_8_US3_L0.1_PS5_test_exp_train_2022_11_22_18_41_32.pkl"
#C.Dh_path = "data/dicts/Dh_8_US3_L0.1_PS5_test_exp_train_2022_11_22_18_41_32.pkl"
#C.Dl_path = "data/dicts/Dl_1024_US3_L0.1_PS5.pkl"
#C.Dh_path = "data/dicts/Dh_1024_US3_L0.1_PS5.pkl"
#C.Dl_path = "data/dicts/Dl_2048_US3_L0.1_PS3.pkl"
#C.Dh_path = "data/dicts/Dh_2048_US3_L0.1_PS3.pkl"

############################################

#C.D_size = 8
#C.D_size = 32
C.D_size = 128
#C.D_size = 512
#C.D_size = 1024
#C.D_size = 2048
C.US_mag = 3
C.lmbd = 0.1
#C.patch_size= 3
C.patch_size= 5

#C.overlap = 1
C.overlap = 3


############################################

C.root_dir = os.path.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.exp_name = 'exp_inference_'+exp_time
C.output_dir = os.path.join(*[C.root_dir,'output',C.exp_name])
os.makedirs(C.output_dir, exist_ok=True)


print("exp_name: %s"%C.exp_name)

############################################

#img_lr_dir = 'data/val_lr/'
#img_hr_dir = 'data/val_hr/'
#img_lr_dir = '/Users/hchoong/Desktop/eth/sa_a3nas/data/SR_testing_datasets/Set5/LR/'
#img_hr_dir = '/Users/hchoong/Desktop/eth/sa_a3nas/data/SR_testing_datasets/Set5/HR/'
#img_lr_dir = '/Users/hchoong/Desktop/github/quantum-cv/ScSR/data/val_lr/'
#img_hr_dir = '/Users/hchoong/Desktop/github/quantum-cv/ScSR/data/val_hr/'
#img_lr_dir = '/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_lr/'
#img_hr_dir = '/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_hr/'

val_hr_path = {
    "1img":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_single_hr",
    "1img_small":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_single_small_hr",
    "valset":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_hr"
}
C.val_hr_path = val_hr_path[mode_val]

val_lr_path = {
    "1img":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_single_lr",
    "1img_small":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_single_small_lr",
    "valset":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_lr"
}
C.val_lr_path = val_lr_path[mode_val]



