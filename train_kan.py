# train_kan.py
#
# Entrena un Kolmogorov-Arnold Network (KAN) simplificado -- basado en
# funciones de base radial por edge, estilo "FastKAN" -- sobre los 3
# datasets ya procesados por el pipeline (outputs_mice, outputs_mice_threshold,
# outputs_data_imputed_log), usando el mismo esquema de evaluacion que
# train.py (Optuna + threshold que maximiza F1 sobre valid + evaluacion
# final en test), para que los resultados sean directamente comparables
# contra los 11 modelos manuales.
#
# No depende de ninguna libreria externa de KAN (pykan, efficient-kan, etc.)
# -- la capa KAN esta implementada aqui mismo en torch puro.

import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from config import PROJECT_DIR, RANDOM_STATE, N_JOBS

warnings.filterwarnings("ignore")


# =====================================================
# CONFIG
# =====================================================

# (nombre de carpeta de salida del pipeline, columna target usada al generarla)
DATASETS = [
    ("outputs_mice", "Desenlace combinado"),
    ("outputs_mice_threshold", "Desenlace combinado"),
    ("outputs_data_imputed_log", "desenlace_combinado"),
]

EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 40
MAX_EPOCHS = 200
PATIENCE = 20

THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)

GRID_MIN = -3.0
GRID_MAX = 3.0


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if N_JOBS != -1:
    torch.set_num_threads(max(1, N_JOBS))


# =====================================================
# DATA
# =====================================================

def load_dataset_splits(output_dir_name, target_col):
    data_dir = PROJECT_DIR / output_dir_name / "tabular_longformer"

    train_csv = data_dir / "tabular_longformer_train.csv"
    valid_csv = data_dir / "tabular_longformer_valid.csv"
    test_csv = data_dir / "tabular_longformer_test.csv"

    for p in (train_csv, valid_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(f"No existe {p}")

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)

    for df, name in [(train_df, "train"), (valid_df, "valid"), (test_df, "test")]:
        if target_col not in df.columns:
            raise ValueError(
                f"No encontré TARGET_COL='{target_col}' en {name} de {output_dir_name}."
            )

    x_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[x_cols].copy()
    y_train = train_df[target_col].astype(int).values

    X_valid = valid_df[x_cols].copy()
    y_valid = valid_df[target_col].astype(int).values

    X_test = test_df[x_cols].copy()
    y_test = test_df[target_col].astype(int).values

    return X_train, y_train, X_valid, y_valid, X_test, y_test, x_cols


# =====================================================
# PREPROCESSING (igual que train.py, para comparar manzanas con manzanas)
# =====================================================

def make_preprocessor(X_train):
    numeric_cols = [
        c for c in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[c])
    ]

    categorical_cols = [
        c for c in X_train.columns
        if c not in numeric_cols
    ]

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", onehot),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0.0,
    )

    return preprocessor


# =====================================================
# METRICS (igual que train.py)
# =====================================================

def best_f1_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -np.inf

    y_prob = np.asarray(y_prob, dtype=float)

    for thr in THRESHOLD_GRID:
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_thr = thr

    return float(best_thr), float(best_f1)


def safe_roc_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_pr_auc(y_true, y_prob):
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return None


def compute_metrics(y_true, y_prob, threshold):
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp + 1e-12)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =====================================================
# KAN (RBF-basis por edge, estilo FastKAN) -- torch puro
# =====================================================

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min, grid_max, num_grids):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)
        self.denominator = (grid_max - grid_min) / max(num_grids - 1, 1)

    def forward(self, x):
        # x: (batch, features) -> (batch, features, num_grids)
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


class KANLayer(nn.Module):
    """
    Una capa KAN: por cada conexión (edge) entre input y output, aprende
    una función univariada no lineal (aproximada con una combinación
    lineal de funciones de base radial), en vez de un solo peso escalar
    como en una capa lineal tradicional.
    """

    def __init__(self, in_features, out_features, num_grids, grid_min, grid_max):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = nn.Linear(in_features * num_grids, out_features, bias=False)
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_norm = self.layernorm(x)
        basis = self.rbf(x_norm)
        spline_out = self.spline_linear(basis.reshape(basis.shape[0], -1))
        return spline_out + self.base_linear(x)


class KANClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers, num_grids, dropout,
                 grid_min=GRID_MIN, grid_max=GRID_MAX):
        super().__init__()

        layers = []
        dim_in = in_features

        for _ in range(num_layers):
            layers.append(KANLayer(dim_in, hidden_dim, num_grids, grid_min, grid_max))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim_in = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = KANLayer(dim_in, 1, num_grids, grid_min, grid_max)

    def forward(self, x):
        h = self.hidden(x)
        out = self.output_layer(h)
        return out.squeeze(-1)


# =====================================================
# TRAINING LOOP
# =====================================================

def suggest_params(trial):
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 8, 64, step=8),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "num_grids": trial.suggest_int("num_grids", 4, 16, step=2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


def train_kan_model(X_train_p, y_train, X_eval_p, y_eval, params,
                     max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=RANDOM_STATE):
    """
    Entrena un KANClassifier con early stopping sobre (X_eval_p, y_eval)
    usando F1 como criterio. Devuelve el modelo (mejores pesos) y sus
    probabilidades sobre X_eval_p.
    """

    device = get_device()
    torch.manual_seed(seed)

    X_train_t = torch.tensor(X_train_p, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_eval_t = torch.tensor(X_eval_p, dtype=torch.float32, device=device)

    model = KANClassifier(
        in_features=X_train_p.shape[1],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        num_grids=params["num_grids"],
        dropout=params["dropout"],
    ).to(device)

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    n = X_train_t.shape[0]
    batch_size = min(params["batch_size"], n)

    best_eval_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            optimizer.zero_grad()
            logits = model(X_train_t[idx])
            loss = criterion(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_prob = torch.sigmoid(model(X_eval_t)).cpu().numpy()

        _, eval_f1 = best_f1_threshold(y_eval, eval_prob)

        if eval_f1 > best_eval_f1:
            best_eval_f1 = eval_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_prob = torch.sigmoid(model(X_eval_t)).cpu().numpy()

    return model, final_prob


def predict(model, X_p):
    device = get_device()
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_p, dtype=torch.float32, device=device)
        prob = torch.sigmoid(model(X_t)).cpu().numpy()

    return prob


# =====================================================
# POR DATASET
# =====================================================

def run_dataset(output_dir_name, target_col, exp_root_dir):
    print("\n" + "#" * 90)
    print(f"Dataset: {output_dir_name}")
    print("#" * 90)

    data_dir = PROJECT_DIR / output_dir_name / "tabular_longformer"
    train_csv = data_dir / "tabular_longformer_train.csv"

    if not train_csv.exists():
        print(f"  SKIP: no existe {train_csv} (el pipeline de este dataset no ha terminado)")
        return None

    X_train, y_train, X_valid, y_valid, X_test, y_test, x_cols = load_dataset_splits(
        output_dir_name, target_col
    )

    print("Train:", X_train.shape, pd.Series(y_train).value_counts().to_dict())
    print("Valid:", X_valid.shape, pd.Series(y_valid).value_counts().to_dict())
    print("Test :", X_test.shape, pd.Series(y_test).value_counts().to_dict())

    preprocessor = make_preprocessor(X_train)
    X_train_p = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    X_valid_p = np.asarray(preprocessor.transform(X_valid), dtype=np.float32)
    X_test_p = np.asarray(preprocessor.transform(X_test), dtype=np.float32)

    device = get_device()
    print("Device:", device)

    def objective(trial):
        params = suggest_params(trial)
        _, prob = train_kan_model(X_train_p, y_train, X_valid_p, y_valid, params)
        _, f1 = best_f1_threshold(y_valid, prob)
        return f1

    study = optuna.create_study(
        direction="maximize",
        study_name=f"kan_{output_dir_name}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params

    print("\nBest VALID F1 from Optuna:", study.best_value)
    print("Best params:", best_params)

    print("\nRecalculando con mejores hiperparámetros (early stopping sobre valid)...")

    final_model, valid_prob = train_kan_model(X_train_p, y_train, X_valid_p, y_valid, best_params)
    selected_threshold, valid_f1 = best_f1_threshold(y_valid, valid_prob)
    valid_metrics = compute_metrics(y_valid, valid_prob, selected_threshold)

    test_prob = predict(final_model, X_test_p)
    test_metrics = compute_metrics(y_test, test_prob, selected_threshold)

    result = {
        "model": "kan",
        "dataset": output_dir_name,
        "objective": "maximize_valid_f1_on_holdout",
        "threshold_strategy": "single_threshold_maximizing_f1_on_validation_predictions",
        "best_valid_f1_from_optuna": float(study.best_value),
        "selected_threshold_on_valid": float(selected_threshold),
        "valid_f1_with_selected_threshold": float(valid_f1),
        "best_params": best_params,
        "n_trials": N_TRIALS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "n_features": len(x_cols),
        "valid": valid_metrics,
        "test": test_metrics,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = exp_root_dir / f"{output_dir_name}_kan_valid_f1_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    torch.save(final_model.state_dict(), model_dir / "kan_state_dict.pt")

    pd.DataFrame(
        {
            "y_true": y_valid,
            "valid_probability": valid_prob,
            "valid_prediction": (valid_prob >= selected_threshold).astype(int),
        }
    ).to_csv(exp_dir / "kan_valid_predictions.csv", index=False)

    pd.DataFrame(
        {
            "y_true": y_test,
            "test_probability": test_prob,
            "test_prediction": (test_prob >= selected_threshold).astype(int),
        }
    ).to_csv(exp_dir / "kan_test_predictions.csv", index=False)

    with open(exp_dir / "kan_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    study.trials_dataframe().to_csv(exp_dir / "kan_optuna_trials.csv", index=False)

    print("\nVALID:")
    print(valid_metrics)

    print("\nTEST:")
    print(test_metrics)

    print("\nGuardado en:", exp_dir)

    return result


# =====================================================
# MAIN
# =====================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root_dir = EXPERIMENTS_DIR

    print("Device global:", get_device())
    print("N_JOBS (threads CPU):", N_JOBS if N_JOBS != -1 else "sin límite")

    all_results = []

    for output_dir_name, target_col in DATASETS:
        try:
            result = run_dataset(output_dir_name, target_col, exp_root_dir)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\nERROR en dataset {output_dir_name}: {e}")

    rows = []

    for r in all_results:
        row = {
            "dataset": r["dataset"],
            "selected_threshold_on_valid": r["selected_threshold_on_valid"],
            "best_valid_f1_from_optuna": r["best_valid_f1_from_optuna"],
            "valid_f1_with_selected_threshold": r["valid_f1_with_selected_threshold"],
        }

        for split in ["valid", "test"]:
            for metric_name, value in r[split].items():
                row[f"{split}_{metric_name}"] = value

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values(by="test_f1_score", ascending=False)

    comparison_path = exp_root_dir / f"kan_comparison_across_datasets_{timestamp}.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 90)
    print("Comparación KAN entre los 3 datasets guardada en:")
    print(comparison_path)
    print("=" * 90)

    if len(comparison_df) > 0:
        print(comparison_df[["dataset", "test_f1_score", "test_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
