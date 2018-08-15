#!/bin/bash

#SBATCH --time=01:00:00						## wallclock time hh:mm:ss
#SBATCH --gres=gpu:1 --constraint='pascal'		## use K80 or P100
#SBATCH --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL

module load goolfc/triton-2016a					## toolchain

srun make clean
srun make
srun --gres=gpu:1 nvprof --export-profile timeline.prof bin/cubble data.json save.json
