#!/bin/bash -l
#SBATCH --job-name=emri-1pa-mcmc
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 25:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie
#SBATCH --output=output_mcmc_%j.txt

# Example SLURM launcher for the GPU MCMC scripts in this repository (tested on
# the "sonic" cluster, UCD). Edit the SBATCH options above for your own cluster,
# and set SCRIPT below to the run you want to launch, e.g.:
#
#   SCRIPT=scripts/Case1_1PAScalarVS0PAScalar/1a_M1e6_mu50/MCMC_code_Case1.py
#
# Submit with: sbatch --chdir=<repo_root> launchers/gpumcmc.sh

SCRIPT="scripts/Case1_1PAScalarVS0PAScalar/1a_M1e6_mu50/MCMC_code_Case1.py"

cd "$SLURM_SUBMIT_DIR"

nvidia-smi -q | grep "Compute Capability"

conda activate few
module load cuda/12.6
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export TORCH_CUDA_ARCH_LIST='7.0,8.6'

python "$SCRIPT"

conda deactivate
