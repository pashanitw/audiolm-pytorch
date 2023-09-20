#!/bin/bash
# Get the list of jobs running on this node
jobs_on_node=$(squeue -w $node -o "%u %P %D %C %G %b" --noheader)

if [ -n "$jobs_on_node" ]; then
  echo "Jobs running on Node: $node"
  echo "$jobs_on_node"
else
  echo "No jobs running on Node: $node"
fi
