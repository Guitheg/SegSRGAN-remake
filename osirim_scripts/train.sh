#!/bin/sh

#SBATCH --job-name=TRAIN_SEGSRGAN
#SBATCH --output=/projets/srm4bmri/outputs/TRAIN_SEGSRGAN.out
#SBATCH --error=/projets/srm4bmri/outputs/TRAIN_SEGSRGAN.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=guigobin@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA11/tf2-NGC-20-06-py3.sif
python=$HOME/SSG/env/bin/python
script=$HOME/SSG/src/SegSRGAN-remake/train.py

data=example.csv

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script} -n training_10_epochs -csv ${data} -ps 64 64 64 -lr 0.8 0.8 0.8 -e 10
