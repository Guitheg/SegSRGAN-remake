#!/bin/sh

#SBATCH --job-name=BUILD_DATASET
#SBATCH --output=/projets/srm4bmri/outputs/BUILD_DATASET.out
#SBATCH --error=/projets/srm4bmri/outputs/BUILD_DATASET.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=guigobin@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/tf2-NGC-20-03-py3.sif
python=$HOME/SSG/env/bin/python
script=$HOME/SSG/src/SegSRGAN-remake/build_dataset.py

dataset_name=complete_64_dataset_bs_32
csv=complete.csv

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script} -n ${dataset_name} -csv ${csv} --save_lr -bs 32
