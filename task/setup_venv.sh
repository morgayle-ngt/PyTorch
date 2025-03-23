#!/bin/bash
# Navigate to the task directory
cd ../task

# Create virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# List files inside the environment
ls -l venv/
ls -l venv/bin/