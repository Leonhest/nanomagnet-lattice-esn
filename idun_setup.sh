#!/bin/bash
# Setup script for the NTNU Idun cluster.
# Run this once interactively on the login node before submitting jobs.
#
# Usage:
#   bash idun_setup.sh

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0

VENV_DIR="/cluster/home/$USER/esn-venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

echo "Creating virtual environment at $VENV_DIR ..."
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing dependencies ..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Submit jobs with: sbatch idun_optimize.slurm"
