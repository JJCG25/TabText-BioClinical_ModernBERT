#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Limite de cores de CPU a usar (n_jobs de sklearn/xgboost/lightgbm/catboost
# y tambien BLAS/OpenMP a bajo nivel). Ajusta con: TABTEXT_N_JOBS=8 ./run_pipeline.sh
# Usa TABTEXT_N_JOBS=-1 para no limitar nada (usa todos los cores disponibles).
export TABTEXT_N_JOBS="${TABTEXT_N_JOBS:-4}"

if [ "$TABTEXT_N_JOBS" != "-1" ]; then
  export OMP_NUM_THREADS="$TABTEXT_N_JOBS"
  export MKL_NUM_THREADS="$TABTEXT_N_JOBS"
  export OPENBLAS_NUM_THREADS="$TABTEXT_N_JOBS"
  export NUMEXPR_NUM_THREADS="$TABTEXT_N_JOBS"
fi

echo "============================================================"
echo "TabText pipeline"
echo "Directorio: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
if [ "$TABTEXT_N_JOBS" = "-1" ]; then
  echo "Cores de CPU: sin limite (usando todos los disponibles)"
else
  echo "Cores de CPU limitados a: $TABTEXT_N_JOBS"
fi
echo "============================================================"

echo
echo "============================================================"
echo "Verificando version de transformers (BioClinical-ModernBERT requiere >=4.48.0)"
echo "============================================================"

"$PYTHON_BIN" -c "
from packaging.version import Version
import transformers
v = transformers.__version__
print('transformers instalado:', v)
if Version(v) < Version('4.48.0'):
    raise SystemExit('transformers>=4.48.0 es requerido para ModernBERT (instalado: ' + v + ')')
"

run_step() {
  local name="$1"
  local script="$2"

  echo
  echo "============================================================"
  echo "Paso: $name"
  echo "Script: $script"
  echo "============================================================"

  "$PYTHON_BIN" "$script"
}

run_step "Preparar data" "prepare_data.py"
run_step "Construir textos" "build_patient_text.py"
run_step "Extraer embeddings" "extract_embeddings.py"
run_step "Crear dataset tabular + embeddings" "make_datasets.py"
run_step "Entrenar modelos con holdout test" "train.py"

echo
echo "============================================================"
echo "Pipeline terminado correctamente"
echo "============================================================"
