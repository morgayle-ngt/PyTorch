#!/bin/bash

# Installed required packages
pip install matplotlib notebook

# verify installed packages
python3 -c "import matplotlib; import notebook"

# run jupyter notebook server
cd ../tasks
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='admin1234'