#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*"
}

log "Cloning the LLaVA repository..."
git clone https://github.com/haotian-liu/LLaVA.git

cd LLaVA

log "Creating Conda environment..."
# Ensure conda is available in the script
if ! command -v conda &> /dev/null
then
    log "Conda could not be found. Please install Conda before running this script."
    exit 1
fi

conda create -n llava python=3.10 -y

log "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava

log "Upgrading pip..."
pip install --upgrade pip  # enable PEP 660 support

log "Installing package in editable mode..."
pip install -e .

log "Installing additional packages for training cases..."
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

log "Upgrading to the latest code base..."
git pull
pip install -e .

log "LLaVA installation script completed successfully."

# If you encounter import errors, run the following command:
# pip install flash-attn --no-build-isolation --no-cache-dir
