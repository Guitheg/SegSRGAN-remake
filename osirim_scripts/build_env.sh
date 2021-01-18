#!/bin/sh

#SBATCH --job-name=BUILD_ENV
#SBATCH --output=/projets/srm4bmri/outputs/BUILD_ENV.out
#SBATCH --error=/projets/srm4bmri/outputs/BUILD_ENV.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=guigobin@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA11/tf2-NGC-20-06-py3.sif
python=$HOME/SSG/env/bin/python
script=$HOME/SSG/src/SegSRGAN-remake/build_env.py

home_path=/projets/srm4bmri/segsrgan/

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script} -f ${home_path}
