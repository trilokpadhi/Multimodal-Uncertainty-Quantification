#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*"
}

log "Cloning the Ferret repository..."
git clone https://github.com/apple/ml-ferret

cd ml-ferret

log "Creating Conda environment..."
# Ensure conda is available in the script
if ! command -v conda &> /dev/null
then
    log "Conda could not be found. Please install Conda before running this script."
    exit 1
fi

conda create -n ferret python=3.10 -y

log "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ferret

log "Upgrading pip..."
pip install --upgrade pip  # enable PEP 660 support

log "Installing package in editable mode..."
pip install -e .

log "Installing additional required packages..."
pip install pycocotools
pip install protobuf==3.20.0

log "Installing additional packages for training cases..."
pip install ninja
pip install flash-attn --no-build-isolation

log "Ferret model installation script completed successfully."
