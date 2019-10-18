#!/bin/bash
#SBATCH --job-name=myTest
#SBATCH --account=project_2001889
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=2

module load gcc/8.3.0 cuda/10.1.168 hpcx-mpi/2.5.0-cuda 

srun cubble input_parameters.json output.json
