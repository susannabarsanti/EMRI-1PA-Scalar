#!/bin/bash -l
#SBATCH --job-name=hello
# speficity number of nodes 
#SBATCH -N 1
# specify the gpu queue

#SBATCH --partition=gpu
# Request 1 gpus
#SBATCH --gres=gpu:1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=1

# specify the walltime e.g 2 mins
#SBATCH -t 25:15:10


#SBATCH --exclude=sonicgpu1,sonicgpu2,sonicgpu3,sonicgpu4,sonicgpu5,sonicgpu6,sonicgpu8,sonicgpu9,sonicgpu10,sonicgpu11
#,sonicgpu12,sonicgpu13,sonicgpu14,sonicgpu15,sonicgpu16,sonicgpu17,sonicgpu18,sonicgpu19,sonicgpu20
#,sonicgpu12,sonicgpu13,sonicgpu14,sonicgpu15,sonicgpu16,sonicgpu17,sonicgpu18,sonicgpu19,sonicgpu20
#
# set to email at start,end and failed jobs
# 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie

#SBATCH --output=output_mcmc_1PAdinj1PAdrec_%j.txt



# run from current directory
cd $SLURM_SUBMIT_DIR

# command to use
nvidia-smi
nvidia-smi -q | grep "Compute Capability"

#conda activate few_kerr2 #in few_kerr2 c'è installato few di soton , lisa-on-gpu senza EqualArms e FoM con tue modifiche
conda activate few
module load cuda/12.6
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export TORCH_CUDA_ARCH_LIST='7.0,8.6'
nvidia-smi -q | grep "Compute Capability"
which nvcc
nvcc --version
echo $LD_LIBRARY_PATH
echo $CUDA_HOME
echo $LD_LIBRARY_PATH | grep cuda


#python circular2.py
#python fisher_bias.py
#python MCMC_code_kerr.py
#python MCMC_code_newPSD_1PAchargeVS1PAGR.py



#python MCMC_code_1pachargespinVS0pachargespin.py
#python MCMC_code_IMRI.py
#python MCMC_code_pn_Kerr.py


python MCMC_code_SPIN.py
#python mismatch_analysis.py



#python dephasing_code.py

#python code_plot_waveform.py
#python MCMC_code_SPIN.py


#python MCMC_code_1pachargeVS0pacharge.py 

#python MCMC_code_1pachargeVS0pachargei_toFifty.py 
#python MCMC_code_schwVSkerr.py
#python waveform_kerr.py
#python trajectory_test.py
#python MCMC_code_charge_spin_1PA.py 
#python fisher_logl.py
#python check_trajectory_1pa.py
#python fisher_kerr_gr.py
#python 1PASpinScalar.py 
conda deactivate
