#!/bin/sh
#SBATCH --time=06:00:00          # Run time in hh:mm:ss
#SBATCH --mem=32000              # Maximum memory required (in megabytes)
#SBATCH --job-name=VAE_CIFAR100
#SBATCH --partition=gpu
#SBATCH --reservation=cse496dl
#SBATCH --gres=gpu:6
#SBATCH --constraint=[gpu_k20|gpu_k40|gpu_p100]
#SBATCH --error=/work/cse496dl/cpack/Assignment_3/job.err
#SBATCH --output=/work/cse496dl/cpack/Assignment_3/job.out
#SBATCH --qos=short		 # 6 hour job run time max
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<chulwoo.pack@gmail.com>

module load singularity
singularity exec docker://unlhcc/sonnet-gpu python3 -u $@