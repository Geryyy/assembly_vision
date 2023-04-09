#!/bin/bash
#Add the deadsnakes PPA repository to install Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
# Update system packages
sudo apt upgrade -y
# Install Python 3.10 and its development headers and libraries
sudo apt install -y python3.10 python3.10-dev python3.10-venv python3-pip
# Install OpenCV dependencies
sudo apt install -y libopencv-dev
# Add Blender repository
sudo add-apt-repository ppa:blender/ppa
# Update package list
sudo apt update
# Install Blender
sudo apt install blender
# Create a virtual environment (optional) using Python 3.10
python3.10 -m venv venv
source venv/bin/activate
# Install Python libraries
pip install --upgrade pip
pip install -r requirements.txt
# Deactivate the virtual environment (optional)
deactivate
