#!/bin/bash

# Initialize variables
total_gpus_available=0
total_gpus_assigned=0
total_nodes=0

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
    echo "Node: $node, Partition: $partition, GPUs Capacity: $gpus_available, GPUs Assigned: $gpus_assigned, GPUs Available: $((gpus_available - gpus_assigned))",
      # Get the list of jobs running on this node
    jobs_on_node=$(squeue -w $node -o "%u %P %D %C %G %b" --noheader)

    if [ -n "$jobs_on_node" ]; then
      echo "Jobs running on Node: $node"
      echo "$jobs_on_node"
    else
      echo "No jobs running on Node: $node"
    fi
  done
done

# Print the total values and total number of nodes
echo "Total Nodes: $total_nodes"
echo "Total GPUs Available: $total_gpus_available"
echo "Total GPUs Assigned: $total_gpus_assigned"
