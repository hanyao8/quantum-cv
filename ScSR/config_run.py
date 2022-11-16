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
mode_val = "valset"

############################################

C.root_dir = os.path.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.exp_name = 'exp_'+exp_time
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
    "valset":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_hr"
}
C.val_hr_path = val_hr_path[mode_val]

val_lr_path = {
    "1img":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_single_lr",
    "valset":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_lr"
}
C.val_lr_path = val_lr_path[mode_val]



