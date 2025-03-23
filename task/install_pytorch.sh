#!/bin/bash
# Navigate to the task directory
cd ~/task

# Activate the virtual environment
source venv/bin/activate

# Install PyTorch and related packages
# pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# List installed packages and create requirements.txt
# pip freeze > requirements.txt

# Display the contents of requirements.txt
cat requirements.txt

# Skipped pip freeze and installing from stable point tested on Ubuntu 20.04
pip install --upgrade --no-cache-dir -r requirements.txt