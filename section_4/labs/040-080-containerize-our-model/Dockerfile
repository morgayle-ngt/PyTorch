# Use official Python slim base image version 3.11
____ ____:____

# Set the working directory
____ /opt/app

# Install CPU versions of PyTorch
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Flask App
____ ____ .

# Create a user and group for running the app
RUN groupadd -r pytorch && useradd --no-log-init -r -g pytorch -m pytorch

# Change ownership of the app directory
RUN chown -R pytorch:pytorch /opt/app

# The pytorch user should run the container
____ ____

# Expose the port for Flask
____ 8000

# Command to run the app
____ ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "--preload", "app:app"]
