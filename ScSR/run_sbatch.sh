#!/bin/bash
#SBATCH  --output=sbatch_log/run_%j.out

source activate quantum_cv3
python run.py


