#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=exclhost
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N FedFit_distributed_likelihood_same_node
#PBS -l walltime=00:10:00
#PBS -k doe
#PBS -j oe
#PBS -A MOFA
#PBS -M pp32@illinois.edu
#PBS -m abe

# Load necessary modules
ml use /soft/modulefiles
ml spack-pe-base/0.8.1
ml use /soft/spack/testing/0.8.1/modulefiles
ml apptainer/main
ml load e2fsprogs

export BASE_SCRATCH_DIR=/local/scratch/ # For Polaris
export APPTAINER_TMPDIR=$BASE_SCRATCH_DIR/apptainer-tmpdir
mkdir -p $APPTAINER_TMPDIR

export APPTAINER_CACHEDIR=$BASE_SCRATCH_DIR/apptainer-cachedir
mkdir -p $APPTAINER_CACHEDIR

# For internet access
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy=admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov

conda activate /home/parthpatel17/.conda/envs/fedfit

# Change directory to the cloned repo
cd /lus/eagle/projects/RAPINS/parth/MMA_RadioWave/FederatedFitting

# python ./examples/octopus/run_server.py --config ./examples/configs/FLserver.yaml
# apptainer exec --nv \
#   MMA_GW_Inference_miniapp.sif \
#   python /app/examples/octopus/run_server.py --config <absolute path to FL server config file>/FLserver.yaml

# -------------------------
# Launch Federated Components in Parallel
# -------------------------

# Run server
python ./examples/octopus/run_server.py --config ./examples/configs/FLserver.yaml \
    > server_stdout.log 2> server_stderr.log &

# Run Site 1
python ./examples/octopus/run_site.py --config ./examples/configs/detector0.yaml \
    > site1_stdout.log 2> site1_stderr.log &

# Run Site 2
python ./examples/octopus/run_site.py --config ./examples/configs/detector1.yaml \
    > site2_stdout.log 2> site2_stderr.log &

# Wait for all background jobs to complete
wait
