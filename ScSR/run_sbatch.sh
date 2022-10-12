#!/bin/bash
#SBATCH  --output=log/run_%j.out

source activate quantum_cv2
python run.py


