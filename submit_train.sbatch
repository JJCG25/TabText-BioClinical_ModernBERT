#!/bin/bash
#SBATCH --job-name=tabtext_train
#SBATCH --account=molthesis
#SBATCH --partition=cpu-cvail01
#SBATCH --comment=notify           # Notificaciones por Discord DM
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

source .venv/bin/activate

python3 make_datasets.py
python3 train.py
