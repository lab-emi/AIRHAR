#!/bin/bash

# Define the base directories for each dataset
CI4R_BASE_DIR="/path/to/ci4r/dataset"
DIAT_BASE_DIR="/path/to/diat/dataset"
GLASGOW_BASE_DIR="/path/to/glasgow/dataset"

# Prepare the Alabama dataset
echo "Preparing the CI4R dataset..."
python3 datasets/data_prepare_CI4R.py --path "$CI4R_BASE_DIR"

# Prepare the DIAT dataset
echo "Preparing the DIAT dataset..."
python3 datasets/data_prepare_DIAT.py --path "$DIAT_BASE_DIR"

# Prepare the Glasgow dataset
echo "Preparing the Glasgow dataset..."
python3 datasets/data_prepare_UoG20.py --path "$GLASGOW_BASE_DIR"

echo "All datasets have been prepared."
