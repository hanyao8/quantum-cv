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

mode_val = "1_img" #inference settings

############################################

C.root_dir = os.path.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.output_dir = os.path.join(*[C.root_dir,'output','output_'+exp_time])
os.makedirs(C.output_dir, exist_ok=True)
C.exp_name = 'exp_'+exp_time

print("exp_name: %s"%C.exp_name)

############################################

val_hr_path = {
    "1_img":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_single_hr"
}
C.val_hr_path = val_hr_path[mode_val]

val_lr_path = {
    "1_img":"/scratch_net/kringel/hchoong/github/quantum-cv/ScSR/data/val_single_lr"
}
C.val_lr_path = val_lr_path[mode_val]



