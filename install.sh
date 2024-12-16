#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check Python version
echo "Checking Python version..."
python3 --version

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set up environment variables (if needed)
# echo "Setting up environment variables..."
# export VARIABLE_NAME=value

# Additional setup steps (if needed)
# echo "Performing additional setup..."
# mkdir -p some_directory

echo "Installation complete." 