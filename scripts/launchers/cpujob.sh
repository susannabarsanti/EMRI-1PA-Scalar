#!/bin/bash -l
#SBATCH --job-name=emri-1pa-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 3
#SBATCH -t 00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie

# Example SLURM launcher for lightweight, GPU-free tasks. All MCMC launch
# scripts in this repository require a GPU (cupy) - use gpumcmc.sh for those -
# but trajectory integration alone does not. Edit SCRIPT below and the SBATCH
# options above for your own cluster.
#
# Submit with: sbatch --chdir=<repo_root> launchers/cpujob.sh

SCRIPT="scripts/dephasing_trajectories.py"

cd "$SLURM_SUBMIT_DIR"

conda activate few
python "$SCRIPT"
conda deactivate
