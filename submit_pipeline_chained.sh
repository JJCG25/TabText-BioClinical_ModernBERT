#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

jid1=$(sbatch --parsable submit_prepare.sbatch)
echo "Job 1 (prepare_data + build_patient_text, CPU): $jid1"

jid2=$(sbatch --parsable --dependency=afterok:"$jid1" submit_embeddings.sbatch)
echo "Job 2 (extract_embeddings, GPU): $jid2"

jid3=$(sbatch --parsable --dependency=afterok:"$jid2" submit_train.sbatch)
echo "Job 3 (make_datasets + train, CPU): $jid3"

echo
echo "Pipeline encadenado. Sigue el progreso con: squeue -u \$USER"
