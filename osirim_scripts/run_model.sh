#!/bin/sh

#SBATCH --job-name=RUN_MODEL
#SBATCH --output=/projets/srm4bmri/outputs/RUN_MODEL.out
#SBATCH --error=/projets/srm4bmri/outputs/RUN_MODEL.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=guigobin@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/tf2-NGC-20-03-py3.sif
python=$HOME/SSG/env/bin/python
script=$HOME/SSG/src/SegSRGAN-remake/run_model.py

mri=/projets/srm4bmri/segsrgan/training_folder/batchs/test_dataset/LR_hr1010.nii.gz
model=train_mri_srgan

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script} -n run_model -f ${mri} -m ${model}
