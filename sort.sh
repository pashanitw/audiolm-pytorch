#!/bin/bash

# Initialize variables
total_gpus_available=0
total_gpus_assigned=0
total_nodes=0
declare -A user_gpu_usage

# Get the list of all partitions
partitions=$(scontrol show partition | grep "PartitionName=" | sed 's/PartitionName=//g' | sed 's/ .*//')

# Loop through each partition
for partition in $partitions; do

  # Get the list of nodes for this partition (removing duplicates)
  nodes=$(sinfo -h -p $partition -N -o "%N" | sort | uniq)

  # Increment the total node count
  total_nodes=$((total_nodes + $(echo "$nodes" | wc -l)))

  # Loop through each node
  for node in $nodes; do
    # Get node details
    node_details=$(scontrol show node $node)

    # Get total GPUs available on the node from the Gres field
    gpus_available=$(echo "$node_details" | grep -Po 'Gres=gpu:[^:]*:\K\d+')

    # Get total GPUs assigned on the node from the AllocTRES field
    gpus_assigned=$(echo "$node_details" | grep -Po 'AllocTRES=.*gres/gpu=\K\d+')

    # If gpus_available is empty, set it to 0
    if [ -z "$gpus_available" ]; then
      gpus_available=0
    fi

    # If gpus_assigned is empty, set it to 0
    if [ -z "$gpus_assigned" ]; then
      gpus_assigned=0
    fi

    # Sum up the values
    total_gpus_available=$((total_gpus_available + gpus_available))
    total_gpus_assigned=$((total_gpus_assigned + gpus_assigned))

    # Print details for the node
    echo "Node: $node, Partition: $partition, GPUs Capacity: $gpus_available, GPUs Assigned: $gpus_assigned, GPUs Available: $((gpus_available - gpus_assigned))"

    # Get the list of jobs running on this node and accumulate GPU usage per user
    while read -r line; do
      user=$(echo $line | awk '{print $1}')
      gpu_info=$(echo $line | awk '{print $6}')
      if [[ $gpu_info == gres:gpu* ]]; then
        gpu_used=${gpu_info##*:}
      else
        gpu_used=0
      fi

    if [[ $gpu_used =~ ^[0-9]+$ ]]; then
      # Initialize user's GPU usage to 0 if not already set
      if [ -z ${user_gpu_usage[$user]} ]; then
        user_gpu_usage[$user]=0
      fi
      user_gpu_usage[$user]=$((user_gpu_usage[$user] + gpu_used))
    else
      # Initialize user's GPU usage to 0 if not already set
      if [ -z ${user_gpu_usage[$user]} ]; then
        user_gpu_usage[$user]=0
      fi
    fi
    done < <(squeue -w $node -o "%u %P %D %C %G %b" --noheader)
  done
done

# Print the total values and total number of nodes
echo "Total Nodes: $total_nodes"
echo "Total GPUs Available: $total_gpus_available"
echo "Total GPUs Assigned: $total_gpus_assigned"

# Print GPU usage per user in sorted order
echo "GPU Usage per User (sorted):"
for user in "${!user_gpu_usage[@]}"; do
  echo "$user: ${user_gpu_usage[$user]} GPUs"
done | sort -k2 -n -r
