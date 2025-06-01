#!/bin/bash
#SBATCH --mem=32g                              # required number of memory
#SBATCH --nodes=1                               # nodes required for whole simulation

#SBATCH --cpus-per-task=16                       # CPUs for each task/site
## SBATCH --gpus-per-task=1                      # Uncomment if using gpu
##SBATCH --ntasks-per-node=1                     # Uncomment if using gpu 

#SBATCH --partition=cpu                    # <- or if Delta one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8

##SBATCH --gpus-per-node=1      # Uncomment if using gpu         # number of GPUs you want to use on 1 node
##SBATCH --gpu-bind=none        # Uncomment if using gpu


#SBATCH --job-name=FedFit_Distributed_4sites_node2_day50   # job name
#SBATCH --time=03:15:00                         # dd-hh:mm:ss for the job

#SBATCH -e FedFit_Distributed_4sites_node2_day50-err-%j.log
#SBATCH -o FedFit_Distributed_4sites_node2_day50-out-%j.log

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

# python ./examples/octopus/run_site.py --config ./examples/configs/detector1.yaml
# apptainer exec --nv \
#   MMA_GW_Inference_miniapp.sif \
#   python /app/examples/octopus/run_detector.py --config <absolute path to FL detector0 config file>/detector1.yaml

# Launch 4 site processes in parallel
# srun -n 1 --cpus-per-task 4 python ./examples/octopus/run_site.py --config ./examples/configs/4_Kim.yaml --day "all"> detector4.log 2>&1 &
# srun -n 1 --cpus-per-task 4 python ./examples/octopus/run_site.py --config ./examples/configs/5_Remsi.yaml --day "all" > detector5.log 2>&1 &
# srun -n 1 --cpus-per-task 4 python ./examples/octopus/run_site.py --config ./examples/configs/6_Troja.yaml --day "all" > detector6.log 2>&1 &
# srun -n 1 --cpus-per-task 4 python ./examples/octopus/run_site.py --config ./examples/configs/7_Makhatini.yaml --day "all" > detector7.log 2>&1 &

python ./examples/octopus/run_site.py --config ./examples/configs/4_Kim.yaml --day "50"> 4_Kim_day50.log 2>&1 &
python ./examples/octopus/run_site.py --config ./examples/configs/5_Remsi.yaml --day "50" > 5_Remsi_day50.log 2>&1 &
python ./examples/octopus/run_site.py --config ./examples/configs/6_Troja.yaml --day "50" > 6_Troja_day50.log 2>&1 &
python ./examples/octopus/run_site.py --config ./examples/configs/7_Makhatini.yaml --day "50" > 7_Makhatini_day50.log 2>&1 &

wait
