#!/bin/bash
#SBATCH --job-name=tabtext_pipeline
#SBATCH --account=my_project       # TODO: tu proyecto/usuario
#SBATCH --partition=gpu-cvail01    # TODO: gpu-cvail01 si tienes acceso GPU, o cpu-cvail01 si no
#SBATCH --gres=gpu:1               # TODO: borra esta linea si usas partition cpu-cvail01
#SBATCH --comment=notify           # Notificaciones por Discord DM
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

# TODO: activa aqui el entorno una vez creado, por ejemplo:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate tabtext
# o con venv:
# source .venv/bin/activate

bash run_pipeline.sh
