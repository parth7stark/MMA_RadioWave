#!/bin/bash
#SBATCH --mem=150g                              # required number of memory
#SBATCH --nodes=1                               # nodes required for whole simulation

#SBATCH --cpus-per-task=40                       # CPUs for each task
## SBATCH --gpus-per-task=1                      # Uncomment if using gpu
##SBATCH --ntasks-per-node=1                     # Uncomment if using gpu 

#SBATCH --partition=cpu                    # <- or if Delta one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8

##SBATCH --gpus-per-node=1      # Uncomment if using gpu         # number of GPUs you want to use on 1 node
##SBATCH --gpu-bind=none        # Uncomment if using gpu


#SBATCH --job-name=CurveFitting_withoutFederation_dayAll_tobs_Dingo68_epsilonthetaCorepfixed_run2   # job name
#SBATCH --time=01:00:00                         # dd-hh:mm:ss for the job

#SBATCH -e CurveFitting_withoutFederation_dayAll_tobs_Dingo68_epsilonthetaCorepfixed_run2-err-%j.log
#SBATCH -o CurveFitting_withoutFederation_dayAll_tobs_Dingo68_epsilonthetaCorepfixed_run2-out-%j.log

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
cd /u/parthpatel7173/MMA_RadioWave/FederatedFitting/Kara_files

# python ./with_ini_DL_multiproc.py --ini_file_path ./test_ini_DL.ini
# python ./with_ini_DL_epsilonFixed.py --ini_file_path test_ini_DL_epsilonFixed.ini
python with_ini_DL_epsilonthetaCorepFixed.py --ini_file_path test_ini_DL_epsilonthetaCorepFixed.ini

# python ./examples/octopus/run_server.py --config ./examples/configs/FLserver.yaml --day "50"
# apptainer exec --nv \
#   MMA_GW_Inference_miniapp.sif \
#   python /app/examples/octopus/run_server.py --config <absolute path to FL server config file>/FLserver.yaml

