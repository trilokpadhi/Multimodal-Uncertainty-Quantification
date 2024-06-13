#!/bin/bash

# Exit on any error
# set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*"
}

log "Starting Miniconda installation script."

# Define Miniconda version and installation path
MINICONDA_VERSION="latest"
MINICONDA_INSTALLER="Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
INSTALL_PATH="$HOME/miniconda3"

# Download Miniconda installer
log "Downloading Miniconda installer..."
wget $MINICONDA_URL -O $MINICONDA_INSTALLER

# Run the Miniconda installer
log "Running the Miniconda installer..."
bash $MINICONDA_INSTALLER -b -p $INSTALL_PATH

# Clean up installer
log "Cleaning up installer..."
rm $MINICONDA_INSTALLER

# Initialize Miniconda
log "Initializing Miniconda..."
eval "$($INSTALL_PATH/bin/conda shell.bash hook)"
conda init

# Verify Miniconda installation
log "Verifying Miniconda installation..."
conda --version

log "Miniconda installation script completed successfully."
