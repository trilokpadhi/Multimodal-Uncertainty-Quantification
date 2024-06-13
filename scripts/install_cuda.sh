#!/bin/bash

# Exit on any error
# set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*"
}

log "Starting CUDA installation script."

# Clean up any existing CUDA and NVIDIA packages
log "Purging existing NVIDIA and CUDA packages..."
sudo apt-get purge -y 'nvidia*' 'cuda*'
sudo apt-get autoremove -y
sudo apt-get autoclean -y

# Add the NVIDIA GPG key
log "Adding NVIDIA GPG key..."
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# Add the CUDA repository
log "Adding CUDA repository..."
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list'

# Update the package lists
log "Updating package lists..."
sudo apt-get update

# Install CUDA
log "Installing CUDA..."
sudo apt-get install -y cuda-12-5

# Check for broken packages
log "Checking for broken packages..."
sudo apt-get check

# Fix broken packages if any
log "Fixing broken packages if any..."
sudo apt-get -f install -y

# Reinstall CUDA if needed
log "Reinstalling CUDA if needed..."
sudo apt-get install -y cuda

# Add CUDA to PATH
log "Adding CUDA to PATH..."
echo 'export PATH=/usr/local/cuda-12.5/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
log "Verifying CUDA installation..."
nvcc --version

log "CUDA installation script completed successfully."
