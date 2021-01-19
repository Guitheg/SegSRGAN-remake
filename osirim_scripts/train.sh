#!/bin/sh

#SBATCH --job-name=TRAIN_MRI_SRGAN
#SBATCH --output=/projets/srm4bmri/outputs/TRAIN_MRI_SRGAN.out
#SBATCH --error=/projets/srm4bmri/outputs/TRAIN_MRI_SRGAN.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=guigobin@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/tf2-NGC-20-03-py3.sif
python=$HOME/SSG/env/bin/python
script=$HOME/SSG/src/SegSRGAN-remake/train.py

training_name=train_mri_srgan
dataset=test_dataset

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script} -n ${training_name} -d ${dataset} -e 10