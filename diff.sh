#!/bin/bash
#SBATCH --job-name=audio_lm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100-SXM4:4 # Adjust number of GPUs here
#SBATCH --time=01:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out


set -x -e

# CHANGE HERE THE CONDA EVN AND ANY STARTUP SCRIPTS
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
#source /opt/hpc-sdk-22.5/env.sh
conda activate llm2

# print nvcc version
echo "NVCC VERSION: $(nvcc --version)"
# print torch cuda version
echo "TORCH CUDA VERSION: $(python -c 'import torch; print(torch.version.cuda)')"


# PRINT PROPERLY THE ABOVE
echo "$(ldconfig -v 2>/dev/null | grep "libnccl.so" | tail -n1 | sed -r 's/^.*\.so\.//')"


# have the below in case of debugging nccl issues such as nccl timeout.
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
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

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

echo "NUM_PROCESSES: $NUM_PROCESSES"

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"

MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "MASTER_PORT: $MASTER_PORT"

echo "SLURM PRoc ID: $SLURM_PROCID"

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="accelerate launch \
    --config_file single.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "
#export LAUNCHER="python -m torch.distributed.launch \
#    --nproc_per_node=$NUM_PROCESSES \
#    --nnodes=1 \
#    --node_rank=0 \
#    --master_addr=$MASTER_ADDR \
#    --master_port=$MASTER_PORT \
#    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="train.py \
    --batch_size 8 \
    --type fine
"

export CMD="$LAUNCHER $PROGRAM"

bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

# execure cmd without srun
#bash -c "$CMD" 2>&1 | tee -a $LOG_PATH


echo "END TIME: $(date)"