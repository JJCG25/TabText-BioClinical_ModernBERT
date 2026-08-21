# prepare_data.py

import json
import re
import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    SHEET_NAME,
    TARGET_COL,
    OUTPUT_DIR,
    RANDOM_STATE,
    SPLIT_STRATEGY,
    SPLITS,
    TRAIN_SIZE,
    TEST_SIZE,
    VALID_TRAIN_SIZE,
    VALID_SIZE,
    VALID_TEST_SIZE,
)


LEAKAGE_COLS = [
    "Desenlace combinado",
    "desenlace_combinado",
    "desenlace cardiovascular",
    "descenlace cardiovascular",
    "Complicaciones cardiovasculares",
    "Mortalidad menor a 2 años",
    "mort 2años",
    "mortalidad 2 años",
    "Trasplante Cardíaco",
    "Traplante Cardíaco",
    "Enfermedad Cerebrovascular",
    "Asistencia Ventricular",
    "Estado egreso",
    "Fecha egreso",
    "Estancia hospitalaria",
    "mortalidad seguimiento",
    "último contacto",
    "tiempo sobrevida",
]

EXACT_ID_COLS = {
    "id",
    "id base",
    "id_paciente",
    "paciente_id",
    "identificacion",
    "identificación",
    "documento",
    "nombre",
}

ID_KEYWORDS = [
    "historia",
]

DATE_KEYWORDS = [
    "fecha",
    "date",
    "datetime",
]


# =====================================================
# UTILIDADES
# =====================================================

def normalize_col(col):
    col = str(col).strip().lower()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }

    for old, new in replacements.items():
        col = col.replace(old, new)

    col = re.sub(r"\s+", " ", col)
    col = col.strip()

    return col


def encode_binary_target(y):
    """
    Convierte el outcome a 0/1.
    Si ya viene como 0/1, no cambia nada.
    """

    y = y.copy()

    if pd.api.types.is_numeric_dtype(y):
        values = sorted(y.dropna().unique().tolist())

        if set(values).issubset({0, 1}):
            return y.astype(int)

        if len(values) == 2:
            mapping = {values[0]: 0, values[1]: 1}
            print("Mapping target:", mapping)
            return y.map(mapping).astype(int)

        raise ValueError(f"Target numérico no binario: {values}")

    y_str = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("í", "i", regex=False)
        .str.replace("sí", "si", regex=False)
    )

    mapping = {
        "1": 1,
        "si": 1,
        "yes": 1,
        "true": 1,
        "positivo": 1,
        "evento": 1,
        "desenlace": 1,
        "con desenlace": 1,
        "con evento": 1,

        "0": 0,
        "no": 0,
        "false": 0,
        "negativo": 0,
        "no evento": 0,
        "sin evento": 0,
        "no desenlace": 0,
        "sin desenlace": 0,
    }

    y_encoded = y_str.map(mapping)

    if y_encoded.isna().any():
        print("Valores no reconocidos del target:")
        print(y[y_encoded.isna()].value_counts(dropna=False))
        raise ValueError("No se pudo codificar el target a 0/1.")

    return y_encoded.astype(int)


def should_drop_column(col, series):
    """
    Decide si una columna debe eliminarse por leakage, ID o fecha.
    El TARGET_COL nunca se elimina.
    """

    col_norm = normalize_col(col)
    target_norm = normalize_col(TARGET_COL)

    # Nunca eliminar el outcome
    if col_norm == target_norm:
        return False

    # Columnas vacías típicas de Excel
    if col_norm.startswith("unnamed"):
        return True

    # Leakage explícito
    leakage_norm = {normalize_col(c) for c in LEAKAGE_COLS}

    if col_norm in leakage_norm:
        return True

    # IDs exactos
    exact_id_norm = {normalize_col(c) for c in EXACT_ID_COLS}

    if col_norm in exact_id_norm:
        return True

    # Historia clínica u otros identificadores por keyword específica
    for key in ID_KEYWORDS:
        if normalize_col(key) in col_norm:
            return True

    # Fechas
    for key in DATE_KEYWORDS:
        if normalize_col(key) in col_norm:
            return True

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    return False


def prepare_dataframe(df):
    """
    Limpieza básica:
    - limpia nombres de columnas
    - verifica target
    - elimina filas sin target
    - codifica target a 0/1
    - elimina columnas de leakage, ID, fecha, vacías y constantes
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    if TARGET_COL not in df.columns:
        related = [
            c for c in df.columns
            if any(
                k in normalize_col(c)
                for k in ["desenlace", "evento", "mort", "cardio"]
            )
        ]

        raise ValueError(
            f"No encontré TARGET_COL='{TARGET_COL}'. "
            f"Columnas relacionadas encontradas: {related}"
        )

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = encode_binary_target(df[TARGET_COL])

    drop_cols = []

    for col in df.columns:
        if should_drop_column(col, df[col]):
            drop_cols.append(col)

    data = df.drop(columns=drop_cols, errors="ignore")

    # Eliminar columnas completamente vacías
    empty_cols = data.columns[data.isna().all()].tolist()
    data = data.drop(columns=empty_cols, errors="ignore")

    # Eliminar columnas constantes excepto target
    constant_cols = [
        c for c in data.columns
        if c != TARGET_COL and data[c].nunique(dropna=True) <= 1
    ]
    data = data.drop(columns=constant_cols, errors="ignore")

    if TARGET_COL not in data.columns:
        raise RuntimeError(
            f"El target '{TARGET_COL}' fue eliminado accidentalmente. "
            "Revisa LEAKAGE_COLS y should_drop_column()."
        )

    predictors = [c for c in data.columns if c != TARGET_COL]

    prep_info = {
        "data_path": str(DATA_PATH),
        "target": TARGET_COL,
        "n_rows": int(data.shape[0]),
        "n_predictors": int(len(predictors)),
        "positive_count": int(data[TARGET_COL].sum()),
        "negative_count": int((data[TARGET_COL] == 0).sum()),
        "positive_rate": float(data[TARGET_COL].mean()),
        "dropped_cols": drop_cols,
        "empty_cols": empty_cols,
        "constant_cols": constant_cols,
        "predictors": predictors,
    }

    return data, predictors, prep_info


def make_oof_split(data):
    """
    Split estratificado para entrenamiento OOF:
    - train: 90%
    - test: 10%
    """

    train_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COL],
    )

    return train_df, test_df


def make_valid_split(data):
    """
    Split estratificado para validación fija:
    - train: 80%
    - valid: 10%
    - test: 10%
    """

    train_valid_df, test_df = train_test_split(
        data,
        test_size=VALID_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COL],
    )

    valid_fraction_within_train_valid = VALID_SIZE / (VALID_TRAIN_SIZE + VALID_SIZE)

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_fraction_within_train_valid,
        random_state=RANDOM_STATE,
        stratify=train_valid_df[TARGET_COL],
    )

    return train_df, valid_df, test_df


def make_split(data):
    if SPLIT_STRATEGY == "oof":
        train_df, test_df = make_oof_split(data)
        return {"train": train_df, "test": test_df}

    if SPLIT_STRATEGY == "valid":
        train_df, valid_df, test_df = make_valid_split(data)
        return {"train": train_df, "valid": valid_df, "test": test_df}

    raise ValueError(f"SPLIT_STRATEGY no soportado: {SPLIT_STRATEGY}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Leyendo dataset original:")
    print(DATA_PATH)

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    data, predictors, prep_info = prepare_dataframe(df)

    split_dfs = make_split(data)

    print("\nSplit final:")
    for split_name in SPLITS:
        split_df = split_dfs[split_name]
        print(
            f"{split_name}:",
            split_df.shape,
            split_df[TARGET_COL].value_counts().to_dict(),
        )

    requested_fractions = (
        {"train": TRAIN_SIZE, "test": TEST_SIZE}
        if SPLIT_STRATEGY == "oof"
        else {
            "train": VALID_TRAIN_SIZE,
            "valid": VALID_SIZE,
            "test": VALID_TEST_SIZE,
        }
    )

    split_info = {
        "split_strategy": SPLIT_STRATEGY,
        "requested_fractions": requested_fractions,
        "splits": {
            split_name: {
                "n": int(len(split_df)),
                "fraction_real": float(len(split_df) / len(data)),
                "positive_count": int(split_df[TARGET_COL].sum()),
                "negative_count": int((split_df[TARGET_COL] == 0).sum()),
                "positive_rate": float(split_df[TARGET_COL].mean()),
            }
            for split_name, split_df in split_dfs.items()
        },
    }

    for split_name, split_df in split_dfs.items():
        split_df.to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False)

    pd.Series(predictors).to_csv(
        OUTPUT_DIR / "feature_columns.csv",
        index=False,
        header=False,
    )

    with open(OUTPUT_DIR / "prep_info.json", "w", encoding="utf-8") as f:
        json.dump(prep_info, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / f"split_info_{SPLIT_STRATEGY}.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print("\nArchivos guardados en:", OUTPUT_DIR)
    print("Split real:", split_info)


if __name__ == "__main__":
    main()
