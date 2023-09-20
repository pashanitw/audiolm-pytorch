#!/bin/bash

#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=01:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

nodes=( $( scontrol show hostnames $SLURM_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{ for(i=1; i<=NF; i++) if ($i != "127.0.1.1") print $i; }')
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
echo "Rendezvous Endpoint: $head_node_ip:29500"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo 'master_addr: '$master_addr
export MASTER_ADDR=$master_addr

source /opt/hpc-sdk-22.5/env.sh
conda activate OpenChatKit

#https://github.com/huggingface/accelerate/issues/1519

accelerate launch \
    train.py \
    --multi_gpu \
    --num_machines 2 \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port 1234