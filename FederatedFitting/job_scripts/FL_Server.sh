#!/bin/bash
#SBATCH --mem=64g                              # required number of memory
#SBATCH --nodes=1                               # nodes required for whole simulation

#SBATCH --cpus-per-task=40                       # CPUs for each task
## SBATCH --gpus-per-task=1                      # Uncomment if using gpu
##SBATCH --ntasks-per-node=1                     # Uncomment if using gpu 

#SBATCH --partition=cpu                    # <- or if Delta one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8

##SBATCH --gpus-per-node=1      # Uncomment if using gpu         # number of GPUs you want to use on 1 node
##SBATCH --gpu-bind=none        # Uncomment if using gpu


#SBATCH --job-name=FedFit_Distributed_server_day50_run3   # job name
#SBATCH --time=02:40:00                         # dd-hh:mm:ss for the job

#SBATCH -e FedFit_Distributed_server_day50_run3-err-%j.log
#SBATCH -o FedFit_Distributed_server_day50_run3-out-%j.log

#SBATCH --constraint="scratch"

#SBATCH --account=bbjo-delta-cpu
#SBATCH --mail-user=pp32@illinois.edu
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options


set -x

# Load necessary modules
source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda deactivate
conda deactivate  # just making sure
module purge
module reset  # load the default Delta modules

module load anaconda3_gpu
module list

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate /u/parthpatel7173/.conda/envs/fedfit

# Change directory to the cloned repo
cd /u/parthpatel7173/MMA_RadioWave/FederatedFitting

python ./examples/octopus/run_server.py --config ./examples/configs/FLserver.yaml --day "50"
# apptainer exec --nv \
#   MMA_GW_Inference_miniapp.sif \
#   python /app/examples/octopus/run_server.py --config <absolute path to FL server config file>/FLserver.yaml

