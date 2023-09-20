#!/bin/bash

# Get the list of all partitions
partitions=$(scontrol show partition | grep "PartitionName=" | sed 's/PartitionName=//g' | sed 's/ .*//')

for partition in $partitions; do
  # Get the list of nodes from sinfo command for the specific partition (removing duplicates)
  nodes=$(sinfo -h -p $partition -N -o "%N" | sort | uniq)

  # Count the number of unique nodes
  total_nodes=$(echo "$nodes" | wc -l)

  # Print the total number of nodes and list of nodes for the specific partition
  echo "Partition: $partition"
  echo "Total Nodes: $total_nodes"
  echo "List of Nodes:"
  echo "$nodes"
  echo ""
done
