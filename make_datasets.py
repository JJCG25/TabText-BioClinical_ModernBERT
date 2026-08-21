# make_dataset.py

import json
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, EMB_DIR, TARGET_COL, SPLIT_STRATEGY, SPLITS


# =====================================================
# CONFIG
# =====================================================

MODEL_KEY = "bioclinical_modernbert_base"

OUT_DIR = OUTPUT_DIR / "tabular_longformer"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# UTILIDADES
# =====================================================

def load_split(split_name):
    tabular_path = OUTPUT_DIR / f"{split_name}.csv"
    emb_path = EMB_DIR / f"{MODEL_KEY}_{split_name}_embeddings.npy"
    labels_path = EMB_DIR / f"{MODEL_KEY}_{split_name}_labels.npy"

    if not tabular_path.exists():
        raise FileNotFoundError(f"No existe el CSV tabular: {tabular_path}")

    if not emb_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de embeddings: {emb_path}. "
            "Debes ejecutar extract_embeddings.py después de regenerar los textos."
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de labels: {labels_path}. "
            "Debes ejecutar extract_embeddings.py después de regenerar los textos."
        )

    df = pd.read_csv(tabular_path)
    embeddings = np.load(emb_path)
    labels = np.load(labels_path).astype(int)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"No encontré TARGET_COL='{TARGET_COL}' en {tabular_path}."
        )

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch en {split_name}: "
            f"df tiene {len(df)} filas, embeddings tienen {embeddings.shape[0]} filas. "
            "Probablemente los embeddings pertenecen a un split anterior."
        )

    if len(df) != len(labels):
        raise ValueError(
            f"Mismatch en {split_name}: "
            f"df tiene {len(df)} filas, labels tienen {len(labels)} filas."
        )

    return df, embeddings, labels


def make_embedding_dataframe(embeddings):
    emb_cols = [
        f"{MODEL_KEY}_emb_{i:03d}"
        for i in range(embeddings.shape[1])
    ]

    return pd.DataFrame(
        embeddings,
        columns=emb_cols,
    )


def make_tabular_plus_embeddings(split_name):
    df, embeddings, labels = load_split(split_name)

    y_from_csv = df[TARGET_COL].astype(int).values

    if not np.array_equal(y_from_csv, labels):
        raise ValueError(
            f"Las labels .npy no coinciden con el target del CSV en {split_name}. "
            "Regenera textos y embeddings usando el split actual."
        )

    tabular_df = df.drop(columns=[TARGET_COL]).copy()
    emb_df = make_embedding_dataframe(embeddings)

    out_df = pd.concat(
        [
            tabular_df.reset_index(drop=True),
            emb_df.reset_index(drop=True),
        ],
        axis=1,
    )

    out_df[TARGET_COL] = labels

    return out_df


def main():
    summary = {}

    for split_name in SPLITS:
        print("\n" + "=" * 80)
        print(f"Procesando split: {split_name}")
        print("=" * 80)

        out_df = make_tabular_plus_embeddings(split_name)

        out_path = OUT_DIR / f"tabular_longformer_{split_name}.csv"
        out_df.to_csv(out_path, index=False)

        embedding_cols = [
            c for c in out_df.columns
            if c.startswith(f"{MODEL_KEY}_emb_")
        ]

        summary[split_name] = {
            "path": str(out_path),
            "n_rows": int(out_df.shape[0]),
            "n_cols": int(out_df.shape[1]),
            "target": TARGET_COL,
            "positive_count": int(out_df[TARGET_COL].sum()),
            "negative_count": int((out_df[TARGET_COL] == 0).sum()),
            "positive_rate": float(out_df[TARGET_COL].mean()),
            "n_embedding_cols": int(len(embedding_cols)),
            "n_tabular_cols": int(out_df.shape[1] - 1 - len(embedding_cols)),
        }

        print("Guardado:", out_path)
        print("Shape:", out_df.shape)
        print("Distribución target:")
        print(out_df[TARGET_COL].value_counts().to_dict())

    with open(OUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_strategy": SPLIT_STRATEGY,
                "splits": SPLITS,
                "datasets": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDatasets tabular + Clinical-Longformer guardados en:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()