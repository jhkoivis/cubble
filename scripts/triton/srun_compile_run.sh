#!/bin/bash

#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=project_2001889
#SBATCH --reservation=openACC_course_thu

module load gcc/8.3.0 cuda/10.1.168

make -C final
srun make -C run
