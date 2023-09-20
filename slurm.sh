#!/bin/bash
#SBATCH --job-name=audio_lm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:A100-SXM4:8 # Adjust number of GPUs here
#SBATCH --time=01:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out


set -x -e

# CHANGE HERE THE CONDA EVN AND ANY STARTUP SCRIPTS
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
source /opt/hpc-sdk-22.5/env.sh
conda activate llm

# have the below in case of debugging nccl issues such as nccl timeout.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
#export NCCL_PROTO=simple
#export RDMAV_FORK_SAFE=1
#export FI_EFA_FORK_SAFE=1
#export FI_EFA_USE_DEVICE_RDMA=1
#export FI_PROVIDER=efa
#export FI_LOG_LEVEL=1
#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=ens

echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log.txt"

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="accelerate launch \
    --config_file audiolm.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="\
train.py \
    --batch_size 8 \
    --type fine
"


export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"