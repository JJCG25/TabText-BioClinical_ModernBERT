#!/bin/bash
#SBATCH --job-name=tabtext_embeddings
#SBATCH --account=molthesis
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --comment=notify           # Notificaciones por Discord DM
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

source .venv/bin/activate

python3 extract_embeddings.py
