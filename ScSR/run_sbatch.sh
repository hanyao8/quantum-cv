#!/bin/bash
#SBATCH  --output=sbatch_log/run_%j.out

source activate quantum_cv2
python run.py


