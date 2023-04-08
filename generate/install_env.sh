#!/bin/bash

# Update system packages
sudo apt update -y
sudo apt upgrade -y

# Install Python development headers and libraries
sudo apt install -y python3-dev python3-pip python3-venv

# Install OpenCV dependencies
sudo apt install -y libopencv-dev

# Add Blender repository
sudo add-apt-repository ppa:blender/ppa

# Update package list
sudo apt update

# Install Blender
sudo apt install blender

# Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install Python libraries
pip install --upgrade pip
pip install -r requirements.txt

# Deactivate the virtual environment (optional)
deactivate
