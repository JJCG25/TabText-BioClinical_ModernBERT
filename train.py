# train.py

import json
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
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

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import (
    OUTPUT_DIR,
    TARGET_COL,
    SPLIT_STRATEGY,
    RANDOM_STATE,
    VALID_TRAIN_SIZE,
    VALID_SIZE,
    VALID_TEST_SIZE,
)

warnings.filterwarnings("ignore")


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = OUTPUT_DIR / "tabular_longformer"

TRAIN_CSV = DATA_DIR / "tabular_longformer_train.csv"
VALID_CSV = DATA_DIR / "tabular_longformer_valid.csv"
TEST_CSV = DATA_DIR / "tabular_longformer_test.csv"

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 100
N_SPLITS = 5

THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)

MODELS_TO_RUN = [
    "lightgbm",
    "gradient_boosting",
    "xgboost",
    "catboost",
    "random_forest",
    "bagging_dt",
    "svm_rbf",
    "adaboost_dt",
    "decision_tree",
    "knn",
    "gaussian_nb",
]


# =====================================================
# DATA
# =====================================================

def load_data():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"No existe TRAIN_CSV: {TRAIN_CSV}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"No existe TEST_CSV: {TEST_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    valid_df = None

    if SPLIT_STRATEGY == "valid":
        if not VALID_CSV.exists():
            raise FileNotFoundError(f"No existe VALID_CSV: {VALID_CSV}")
        valid_df = pd.read_csv(VALID_CSV)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"No encontré TARGET_COL='{TARGET_COL}' en train.")

    if TARGET_COL not in test_df.columns:
        raise ValueError(f"No encontré TARGET_COL='{TARGET_COL}' en test.")

    if valid_df is not None and TARGET_COL not in valid_df.columns:
        raise ValueError(f"No encontré TARGET_COL='{TARGET_COL}' en valid.")

    x_cols = [c for c in train_df.columns if c != TARGET_COL]

    X_train = train_df[x_cols].copy()
    y_train = train_df[TARGET_COL].astype(int).values

    X_valid = None
    y_valid = None

    if valid_df is not None:
        X_valid = valid_df[x_cols].copy()
        y_valid = valid_df[TARGET_COL].astype(int).values

    X_test = test_df[x_cols].copy()
    y_test = test_df[TARGET_COL].astype(int).values

    return X_train, y_train, X_valid, y_valid, X_test, y_test, x_cols


# =====================================================
# PREPROCESSING
# =====================================================

def make_preprocessor(X_train):
    """
    Se usa salida densa para todos los modelos.
    Ventaja:
    - GaussianNB, SVM, KNN y CatBoost no fallan por matrices sparse.
    - En tu caso el tamaño es manejable porque tienes ~1000 pacientes.
    """

    numeric_cols = [
        c for c in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[c])
    ]

    categorical_cols = [
        c for c in X_train.columns
        if c not in numeric_cols
    ]

    try:
        onehot = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        onehot = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

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
# METRICS
# =====================================================

def best_f2_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f2 = -np.inf

    y_prob = np.asarray(y_prob, dtype=float)

    for thr in THRESHOLD_GRID:
        y_pred = (y_prob >= thr).astype(int)
        score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        if score > best_f2:
            best_f2 = score
            best_thr = thr

    return float(best_thr), float(best_f2)


def best_f1_threshold(y_true, y_prob):
    """Select the probability threshold that maximizes F1."""
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

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

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


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)

        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1]

        return prob.ravel()

    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


# =====================================================
# MODEL FACTORY
# =====================================================

def class_weight_scale(y_train):
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)

    return neg / pos if pos > 0 else 1.0


def build_model(model_name, trial, y_train):
    spw = class_weight_scale(y_train)

    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
            scale_pos_weight=trial.suggest_float(
                "scale_pos_weight",
                max(0.25, spw * 0.25),
                max(1.0, spw * 4.0),
                log=True,
            ),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 1000, step=50),
            depth=trial.suggest_int("depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            class_weights=[1.0, spw],
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 600, step=25),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "bagging_dt":
        base_max_depth = trial.suggest_int("base_max_depth", 1, 20)
        base_min_samples_split = trial.suggest_int("base_min_samples_split", 2, 30)
        base_min_samples_leaf = trial.suggest_int("base_min_samples_leaf", 1, 20)
        n_estimators = trial.suggest_int("n_estimators", 20, 400, step=20)
        max_samples = trial.suggest_float("max_samples", 0.4, 1.0)
        max_features = trial.suggest_float("max_features", 0.4, 1.0)

        base_tree = DecisionTreeClassifier(
            max_depth=base_max_depth,
            min_samples_split=base_min_samples_split,
            min_samples_leaf=base_min_samples_leaf,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        try:
            return BaggingClassifier(
                estimator=base_tree,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        except TypeError:
            return BaggingClassifier(
                base_estimator=base_tree,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

    if model_name == "svm_rbf":
        return SVC(
            C=trial.suggest_float("C", 1e-3, 100.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-5, 1.0, log=True),
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "adaboost_dt":
        base_max_depth = trial.suggest_int("base_max_depth", 1, 4)
        base_min_samples_leaf = trial.suggest_int("base_min_samples_leaf", 1, 20)
        n_estimators = trial.suggest_int("n_estimators", 25, 500, step=25)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 1.0, log=True)

        base_tree = DecisionTreeClassifier(
            max_depth=base_max_depth,
            min_samples_leaf=base_min_samples_leaf,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        try:
            return AdaBoostClassifier(
                estimator=base_tree,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=RANDOM_STATE,
            )
        except TypeError:
            return AdaBoostClassifier(
                base_estimator=base_tree,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=RANDOM_STATE,
            )

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 1, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 40),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 75),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2),
            leaf_size=trial.suggest_int("leaf_size", 10, 60),
            n_jobs=-1,
        )

    if model_name == "gaussian_nb":
        return GaussianNB(
            var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)
        )

    raise ValueError(f"Modelo no soportado: {model_name}")


# =====================================================
# OOF CV
# =====================================================

def get_oof_predictions(model_name, trial, X_train, y_train):
    """
    Genera predicciones out-of-fold.
    Importante:
    - El preprocessor se ajusta dentro de cada fold.
    - No se usa test.
    - El threshold se calcula sobre todas las predicciones OOF.
    """

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_prob = np.zeros(len(y_train), dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
        print(f"  Fold {fold_idx}/{N_SPLITS}")

        X_tr = X_train.iloc[tr_idx].copy()
        y_tr = y_train[tr_idx]

        X_va = X_train.iloc[va_idx].copy()

        preprocessor = make_preprocessor(X_tr)

        X_tr_p = preprocessor.fit_transform(X_tr)
        X_va_p = preprocessor.transform(X_va)

        X_tr_p = np.asarray(X_tr_p)
        X_va_p = np.asarray(X_va_p)

        model = build_model(model_name, trial, y_tr)
        model.fit(X_tr_p, y_tr)

        prob = get_probabilities(model, X_va_p)
        oof_prob[va_idx] = prob

    return oof_prob


def make_objective(model_name, X_train, y_train):
    def objective(trial):
        oof_prob = get_oof_predictions(
            model_name=model_name,
            trial=trial,
            X_train=X_train,
            y_train=y_train,
        )

        _, oof_f1 = best_f1_threshold(y_train, oof_prob)

        return oof_f1

    return objective


def fit_and_predict_holdout(model_name, trial, X_train, y_train, X_eval):
    preprocessor = make_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_eval_p = preprocessor.transform(X_eval)

    X_train_p = np.asarray(X_train_p)
    X_eval_p = np.asarray(X_eval_p)

    model = build_model(model_name, trial, y_train)
    model.fit(X_train_p, y_train)

    return get_probabilities(model, X_eval_p)


def make_valid_objective(model_name, X_train, y_train, X_valid, y_valid):
    def objective(trial):
        valid_prob = fit_and_predict_holdout(
            model_name=model_name,
            trial=trial,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_valid,
        )

        _, valid_f1 = best_f1_threshold(y_valid, valid_prob)

        return valid_f1

    return objective


# =====================================================
# FINAL TRAINING
# =====================================================

def train_final_model(model_name, best_trial, X_train, y_train, X_test):
    """
    Entrena el modelo final con TODO train.
    El test solo se transforma y predice al final.
    """

    preprocessor = make_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    X_train_p = np.asarray(X_train_p)
    X_test_p = np.asarray(X_test_p)

    model = build_model(model_name, best_trial, y_train)
    model.fit(X_train_p, y_train)

    test_prob = get_probabilities(model, X_test_p)

    return preprocessor, model, test_prob


def train_one_model_oof(model_name, X_train, y_train, X_test, y_test, exp_dir):
    print("\n" + "=" * 90)
    print(f"Entrenando modelo: {model_name}")
    print("=" * 90)

    study = optuna.create_study(
        direction="maximize",
        study_name=model_name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(
        make_objective(model_name, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_trial = study.best_trial

    print("\nBest OOF F1 from Optuna:", study.best_value)
    print("Best params:", study.best_params)

    print("\nRecalculando OOF con mejores hiperparámetros...")

    oof_prob = get_oof_predictions(
        model_name=model_name,
        trial=best_trial,
        X_train=X_train,
        y_train=y_train,
    )

    selected_threshold, oof_f1 = best_f1_threshold(y_train, oof_prob)

    oof_metrics = compute_metrics(
        y_true=y_train,
        y_prob=oof_prob,
        threshold=selected_threshold,
    )

    print("\nEntrenando modelo final con todo train...")

    preprocessor, final_model, test_prob = train_final_model(
        model_name=model_name,
        best_trial=best_trial,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )

    test_metrics = compute_metrics(
        y_true=y_test,
        y_prob=test_prob,
        threshold=selected_threshold,
    )

    result = {
        "model": model_name,
        "objective": "maximize_oof_f1_on_train_cv",
        "cv_strategy": {
            "type": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "threshold_strategy": "single_threshold_maximizing_f1_on_oof_predictions",
        "best_oof_f1_from_optuna": float(study.best_value),
        "selected_threshold_from_oof": float(selected_threshold),
        "oof_f1_with_selected_threshold": float(oof_f1),
        "best_params": study.best_params,
        "oof_train": oof_metrics,
        "test": test_metrics,
    }

    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": final_model,
            "selected_threshold_from_oof": selected_threshold,
            "result": result,
        },
        model_dir / f"{model_name}.joblib",
    )

    pd.DataFrame(
        {
            "y_true": y_train,
            "oof_probability": oof_prob,
            "oof_prediction": (oof_prob >= selected_threshold).astype(int),
        }
    ).to_csv(
        exp_dir / f"{model_name}_oof_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "y_true": y_test,
            "test_probability": test_prob,
            "test_prediction": (test_prob >= selected_threshold).astype(int),
        }
    ).to_csv(
        exp_dir / f"{model_name}_test_predictions.csv",
        index=False,
    )

    with open(exp_dir / f"{model_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    study.trials_dataframe().to_csv(
        exp_dir / f"{model_name}_optuna_trials.csv",
        index=False,
    )

    print("\nOOF TRAIN:")
    print(oof_metrics)

    print("\nTEST:")
    print(test_metrics)

    return result


def train_one_model_valid(model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, exp_dir):
    print("\n" + "=" * 90)
    print(f"Entrenando modelo: {model_name}")
    print("=" * 90)

    study = optuna.create_study(
        direction="maximize",
        study_name=model_name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(
        make_valid_objective(model_name, X_train, y_train, X_valid, y_valid),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_trial = study.best_trial

    print("\nBest VALID F1 from Optuna:", study.best_value)
    print("Best params:", study.best_params)

    print("\nRecalculando validación con mejores hiperparámetros...")

    valid_prob = fit_and_predict_holdout(
        model_name=model_name,
        trial=best_trial,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_valid,
    )

    selected_threshold, valid_f1 = best_f1_threshold(y_valid, valid_prob)

    valid_metrics = compute_metrics(
        y_true=y_valid,
        y_prob=valid_prob,
        threshold=selected_threshold,
    )

    print("\nEntrenando modelo final con train...")

    preprocessor, final_model, test_prob = train_final_model(
        model_name=model_name,
        best_trial=best_trial,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )

    test_metrics = compute_metrics(
        y_true=y_test,
        y_prob=test_prob,
        threshold=selected_threshold,
    )

    result = {
        "model": model_name,
        "objective": "maximize_valid_f1_on_holdout",
        "validation_strategy": {
            "type": "stratified_holdout",
            "train_fraction": VALID_TRAIN_SIZE,
            "valid_fraction": VALID_SIZE,
            "test_fraction": VALID_TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "threshold_strategy": "single_threshold_maximizing_f1_on_validation_predictions",
        "best_valid_f1_from_optuna": float(study.best_value),
        "selected_threshold_on_valid": float(selected_threshold),
        "valid_f1_with_selected_threshold": float(valid_f1),
        "best_params": study.best_params,
        "valid": valid_metrics,
        "test": test_metrics,
    }

    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": final_model,
            "selected_threshold_on_valid": selected_threshold,
            "result": result,
        },
        model_dir / f"{model_name}.joblib",
    )

    pd.DataFrame(
        {
            "y_true": y_valid,
            "valid_probability": valid_prob,
            "valid_prediction": (valid_prob >= selected_threshold).astype(int),
        }
    ).to_csv(
        exp_dir / f"{model_name}_valid_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "y_true": y_test,
            "test_probability": test_prob,
            "test_prediction": (test_prob >= selected_threshold).astype(int),
        }
    ).to_csv(
        exp_dir / f"{model_name}_test_predictions.csv",
        index=False,
    )

    with open(exp_dir / f"{model_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    study.trials_dataframe().to_csv(
        exp_dir / f"{model_name}_optuna_trials.csv",
        index=False,
    )

    print("\nVALID:")
    print(valid_metrics)

    print("\nTEST:")
    print(test_metrics)

    return result


def train_one_model(model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, exp_dir):
    if SPLIT_STRATEGY == "oof":
        return train_one_model_oof(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            exp_dir=exp_dir,
        )

    if SPLIT_STRATEGY == "valid":
        if X_valid is None or y_valid is None:
            raise ValueError("SPLIT_STRATEGY='valid' requiere X_valid/y_valid.")

        return train_one_model_valid(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            exp_dir=exp_dir,
        )

    raise ValueError(f"SPLIT_STRATEGY no soportado: {SPLIT_STRATEGY}")


# =====================================================
# MAIN
# =====================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = (
        EXPERIMENTS_DIR
        / f"manual_models_tabular_plus_bioclinical_modernbert_base_{SPLIT_STRATEGY}_f1_{timestamp}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Experimento:", exp_dir)

    X_train, y_train, X_valid, y_valid, X_test, y_test, x_cols = load_data()

    print("\nShapes:")
    print("Train:", X_train.shape, pd.Series(y_train).value_counts().to_dict())
    if SPLIT_STRATEGY == "valid":
        print("Valid:", X_valid.shape, pd.Series(y_valid).value_counts().to_dict())
    print("Test :", X_test.shape, pd.Series(y_test).value_counts().to_dict())

    all_results = []

    for model_name in MODELS_TO_RUN:
        try:
            result = train_one_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                exp_dir=exp_dir,
            )
            all_results.append(result)

        except Exception as e:
            print(f"\nERROR en {model_name}: {e}")

            error_info = {
                "model": model_name,
                "error": str(e),
            }

            with open(exp_dir / f"{model_name}_error.json", "w", encoding="utf-8") as f:
                json.dump(error_info, f, ensure_ascii=False, indent=2)

    rows = []

    for r in all_results:
        if SPLIT_STRATEGY == "oof":
            row = {
                "model": r["model"],
                "best_oof_f1_from_optuna": r["best_oof_f1_from_optuna"],
                "selected_threshold_from_oof": r["selected_threshold_from_oof"],
                "oof_f1_with_selected_threshold": r["oof_f1_with_selected_threshold"],
            }
            metric_splits = ["oof_train", "test"]
        else:
            row = {
                "model": r["model"],
                "selected_threshold_on_valid": r["selected_threshold_on_valid"],
                "best_valid_f1_from_optuna": r["best_valid_f1_from_optuna"],
                "valid_f1_with_selected_threshold": r["valid_f1_with_selected_threshold"],
            }
            metric_splits = ["valid", "test"]

        for split in metric_splits:
            for metric_name, value in r[split].items():
                row[f"{split}_{metric_name}"] = value

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values(
            by="test_f1_score",
            ascending=False,
        )

    comparison_path = exp_dir / "manual_models_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    config = {
        "target": TARGET_COL,
        "split_strategy": SPLIT_STRATEGY,
        "train_csv": str(TRAIN_CSV),
        "valid_csv": str(VALID_CSV) if SPLIT_STRATEGY == "valid" else None,
        "test_csv": str(TEST_CSV),
        "n_trials": N_TRIALS,
        "n_splits": N_SPLITS if SPLIT_STRATEGY == "oof" else None,
        "objective": (
            "maximize_oof_f1_on_train_cv"
            if SPLIT_STRATEGY == "oof"
            else "maximize_valid_f1_on_holdout"
        ),
        "threshold_selection": (
            "single_threshold_maximizing_f1_on_oof_predictions"
            if SPLIT_STRATEGY == "oof"
            else "single_threshold_maximizing_f1_on_validation_predictions"
        ),
        "models": MODELS_TO_RUN,
        "n_features": len(x_cols),
        "random_state": RANDOM_STATE,
    }

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nComparación final:")
    print(comparison_df)

    print("\nResultados guardados en:")
    print(exp_dir)
    print("Comparación:", comparison_path)


if __name__ == "__main__":
    main()
