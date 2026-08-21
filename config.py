# config.py

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = Path("/home/crisr_cvail/CCC-ML-Project/data/mice.xlsx")
SHEET_NAME = 0

TARGET_COL = "Desenlace combinado"

OUTPUT_DIR = PROJECT_DIR / "outputs_mice"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TEXT_DIR = OUTPUT_DIR / "texts"
TEXT_DIR.mkdir(exist_ok=True, parents=True)

EMB_DIR = OUTPUT_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True, parents=True)

MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42

SPLIT_STRATEGY = "valid"  # opciones: "oof" o "valid"

TRAIN_SIZE = 0.80
TEST_SIZE = 0.20

VALID_TRAIN_SIZE = 0.80
VALID_SIZE = 0.10
VALID_TEST_SIZE = 0.10

if SPLIT_STRATEGY not in {"oof", "valid"}:
    raise ValueError("SPLIT_STRATEGY debe ser 'oof' o 'valid'.")

SPLITS = ["train", "test"] if SPLIT_STRATEGY == "oof" else ["train", "valid", "test"]

TEXT_MODELS = {
    "bioclinical_modernbert_base": "thomas-sounack/BioClinical-ModernBERT-base",
}

MAX_LENGTH = {
    "bioclinical_modernbert_base": 2048,
}

BATCH_SIZE = {
    "bioclinical_modernbert_base": 2,
}
