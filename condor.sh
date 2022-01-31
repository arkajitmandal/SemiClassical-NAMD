#!/bin/bash

# Load Python
# (should be the same version used to create the virtual environment)
module load python/3.7.0

# Unpack your envvironment (with your packages), and activate it
tar -xzf py3.tar.gz 
python3 -m venv py3
source py3/bin/activate

# Run the Python script 
python3 $1  

# Deactivate environment 
deactivate
