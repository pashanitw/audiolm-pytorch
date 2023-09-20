#!/bin/bash
#SBATCH --job-name=audio_lm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100-SXM4:8 # Adjust number of GPUs here
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

echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log.txt"


GPUS_PER_NODE=8
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
    --config_file audiolm.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_machines $NNODES \
    --num_processes $NUM_PROCESSES \
    --machine_rank \$SLURM_PROCID \
    "

# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="train.py"

GLOBAL_RANK=$((SLURM_NODEID * SLURM_NTASKS_PER_NODE + SLURM_PROCID))

export CMD="$LAUNCHER $PROGRAM"
#export CMD="$LAUNCHER --nproc_per_node $SLURM_NTASKS_PER_NODE --nnodes $SLURM_JOB_NUM_NODES --node_rank $SLURM_NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT $PROGRAM --rank $GLOBAL_RANK"

echo $CMD
echo "Job ID with appended string: ${SLURM_JOBID}"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "
srun $SRUN_ARGS --jobid $SLURM_JOB_ID -u bash -c "$CMD"

#bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

#for mrank in $(seq 0 $((SLURM_NNODES - 1)))
#do
#echo "$mrank address"=${All_ADDR[mrank]}
#bash -c "$LAUNCHER --machine_rank $mrank $PROGRAM" 2>&1 | tee -a $LOG_PATH
#done



echo "END TIME: $(date)"