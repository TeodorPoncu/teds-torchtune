#!/bin/bash

# Usage example:
# ./setup_env.sh myenv
#
# This will create a new Conda environment named "myenv" with Python 3.11,
# install PyTorch and related packages, and set up any additional requirements.

# Check if environment name is provided
if [ $# -eq 0 ]; then
    echo "Please provide a name for the Conda environment."
    echo "Usage: $0 <environment_name>"
    exit 1
fi

# Set environment name
ENV_NAME=$1

# Echo the environment name
echo "Setting up Conda environment: $ENV_NAME"

# Create Conda environment with Python 3.11
echo "Creating Conda environment '$ENV_NAME' with Python 3.11..."
conda create -n "$ENV_NAME" python=3.11 -y

# Activate the environment
echo "Activating environment '$ENV_NAME'..."
source activate "$ENV_NAME"

# Install PyTorch and related packages
echo "Installing PyTorch, torchvision, torchaudio, and CUDA in '$ENV_NAME'..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Check if requirements.txt exists and install packages
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt in '$ENV_NAME'..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping this step."
fi

# Install the current project in editable mode
pip install -e .
echo "Environment setup complete for '$ENV_NAME'!"
echo "To activate this environment, use: conda activate $ENV_NAME"