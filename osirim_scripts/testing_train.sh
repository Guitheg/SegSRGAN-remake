#!/bin/sh

#SBATCH --job-name=TESTING_TRAIN
#SBATCH --output=/projets/srm4bmri/outputs/TESTING_TRAIN.out
#SBATCH --error=/projets/srm4bmri/outputs/TESTING_TRAIN.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=guigobin@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/pytorch-NGC-20-03-py3.sif
python=$HOME/SSG/env/bin/python
script=$HOME/SSG/src/SegSRGAN-remake/train.py

data=example.csv

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script}