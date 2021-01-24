#!/bin/sh

#SBATCH --job-name=TRAIN_MAE_LOSS
#SBATCH --output=/projets/srm4bmri/outputs/TRAIN_MAE_LOSS.out
#SBATCH --error=/projets/srm4bmri/outputs/TRAIN_MAE_LOSS.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=guigobin@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/tf2-NGC-20-03-py3.sif
python=$HOME/SSG/env/bin/python
script=$HOME/SSG/src/SegSRGAN-remake/train.py

mri_test=/projets/srm4bmri/segsrgan/training_folder/batchs/complete_dataset/LR_hr1010.nii.gz
training_name=train_mae_loss
dataset=complete_dataset

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script} -n ${training_name} -d ${dataset} -e 60 -t ${mri_test}