#!/bin/bash

# Define the folder name
FOLDER_NAME="hubert"

# Define the URL of your checkpoint (replace with your actual URL)
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
KMEANS_URL="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin"

# Check if folder exists
if [ ! -d "$FOLDER_NAME" ]; then
    mkdir -p "$FOLDER_NAME"
fi

# Download the checkpoint into the folder with its original name

curl -L "$CHECKPOINT_URL" -o "${FOLDER_NAME}/$(basename $CHECKPOINT_URL)"
curl -L "$KMEANS_URL" -o "${FOLDER_NAME}/$(basename $KMEANS_URL)"

echo "Download completed!"
