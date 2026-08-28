#!/bin/bash
#SBATCH --job-name=tabtext_prepare
#SBATCH --account=mol_thesis
#SBATCH --partition=cpu-cvail01
#SBATCH --comment=notify           # Notificaciones por Discord DM
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4              # CPU cores per task
#SBATCH --mem=1G                       # Memory limit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

source .venv/bin/activate

python3 prepare_data.py
python3 build_patient_text.py
