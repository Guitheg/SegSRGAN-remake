#!/bin/sh

#SBATCH --job-name=RUN_MODEL
#SBATCH --output=/projets/srm4bmri/outputs/RUN_MODEL.out
#SBATCH --error=/projets/srm4bmri/outputs/RUN_MODEL.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=guigobin@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA11/tf2-NGC-20-06-py3.sif
python=$HOME/envs/segsrgan/bin/python
script=$HOME/SSG/src/SegSRGAN-remake/run_model.py

mri=/projets/srm4bmri/segsrgan/dataset/lr/lr1010.nii.gz
output=/projets/srm4bmri/segsrgan/outputs/results/
model=/projets/srm4bmri/segsrgan/training_folder/checkpoints/training_10_epochs/

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script} -n run_model -f ${mri} -o ${output} -m ${model} -ps 64 64 64
