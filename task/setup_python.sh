#!/bin/bash
# Update package list
apt-get update

# Install OS-level packages via apt (or your distro’s package manager):
apt-get install python3-apt python3-dbus python3-gi ufw -y
apt-get install python3-gi-cairo gir1.2-gtk-3.0 -y

# Install Python3 and pip
apt-get install -y python3 python3-pip python3-venv

# Verify installation
python3 --version
