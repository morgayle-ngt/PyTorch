#!/bin/bash
# Update package list
sudo apt-get update

# Install OS-level packages via apt (or your distro’s package manager):
sudo apt-get install python3-apt python3-dbus python3-gi ufw -y
sudo apt-get install python3-gi-cairo gir1.2-gtk-3.0 -y

# Install Python3 and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Let pip auto-resolve the tensor compatibilities (no pinning)
pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio
   
# Verify installation
python3 --version
