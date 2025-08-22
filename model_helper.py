import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Optional dependencies
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna  # type: ignore
    from optuna import Trial
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    import category_encoders as ce  # type: ignore
    HAS_CAT_ENCODERS = True
except Exception:
    HAS_CAT_ENCODERS = False


@dataclass
class AutoMLResult:
    preprocess_object: Any
    model_object: Any
    preprocess_bytes: bytes
    model_bytes: bytes
    justification: Dict[str, Any]


def _detect_feature_types(df: pd.DataFrame, target_col: str) -> Dict[str, List[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "features": feature_cols,
    }


def _split_categorical_by_cardinality(df: pd.DataFrame, categorical_cols: List[str], threshold: int = 20) -> Tuple[List[str], List[str]]:
    low = []
    high = []
    for c in categorical_cols:
        card = int(df[c].nunique(dropna=True))
        if card <= threshold:
            low.append(c)
        else:
            high.append(c)
    return low, high


def _build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    selector_k: Optional[int] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    task = task.lower().strip()
    types = _detect_feature_types(df, target_col)
    numeric_cols = types["numeric"]
    categorical_cols = types["categorical"]
    low_card, high_card = _split_categorical_by_cardinality(df, categorical_cols, threshold=20)

    steps_justification: Dict[str, Any] = {
        "numeric_cols": numeric_cols,
        "categorical_low_cardinality_cols": low_card,
        "categorical_high_cardinality_cols": high_card,
        "encoders": {},
        "scaling": "standard",
        "imputation": {"numeric": "median", "categorical": "most_frequent"},
    }

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))

    # Helper to create OneHotEncoder across sklearn versions
    def _make_ohe(min_freq: Optional[float] = None) -> OneHotEncoder:
        # Prefer modern API
        try:
            if min_freq is not None:
                return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=min_freq)
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # Older API without sparse_output
            try:
                if min_freq is not None:
                    return OneHotEncoder(handle_unknown="ignore", sparse=False, min_frequency=min_freq)
                return OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                # Very old API without min_frequency
                try:
                    return OneHotEncoder(handle_unknown="ignore", sparse=False)
                except TypeError:
                    # Final fallback
                    return OneHotEncoder(handle_unknown="ignore")

    if low_card:
        steps_justification["encoders"]["low_cardinality"] = "one_hot"
        transformers.append(("cat_ohe", _make_ohe(), low_card))

    if high_card and HAS_CAT_ENCODERS:
        steps_justification["encoders"]["high_cardinality"] = "target_encoding"
        transformers.append(
            (
                "cat_te",
                ce.TargetEncoder(cols=high_card),  # y is passed during fit(X, y)
                high_card,
            )
        )
    elif high_card:
        # Fallback to one-hot even for high cardinality if category_encoders is missing
        steps_justification["encoders"]["high_cardinality"] = "one_hot_fallback"
        transformers.append(("cat_ohe_high", _make_ohe(min_freq=0.01), high_card))

    pre_ct = ColumnTransformer(transformers=transformers, remainder="drop")

    pre_steps: List[Tuple[str, Any]] = [("transform", pre_ct)]

    # Optional dimensionality reduction if many features
    if selector_k is not None and selector_k > 0:
        if task == "classification":
            selector = SelectKBest(score_func=mutual_info_classif, k=selector_k)
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=selector_k)
        pre_steps.append(("select", selector))
        steps_justification["feature_selection"] = {
            "method": "SelectKBest",
            "k": selector_k,
        }

    preprocessor = Pipeline(steps=pre_steps)
    return preprocessor, steps_justification


def _candidate_models(task: str) -> Dict[str, Any]:
    task = task.lower().strip()
    if task == "classification":
        models: Dict[str, Any] = {
            "logreg": LogisticRegression(max_iter=1000, n_jobs=None),
            "rf": RandomForestClassifier(n_estimators=300, random_state=42),
        }
        if HAS_XGB:
            models["xgb"] = XGBClassifier(
                n_estimators=400,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                tree_method="hist",
            )
    else:
        models = {
            "ridge": Ridge(alpha=1.0),
            "rf": RandomForestRegressor(n_estimators=400, random_state=42),
        }
        if HAS_XGB:
            models["xgb"] = XGBRegressor(
                n_estimators=500,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                tree_method="hist",
            )
    return models


def _metric_name_and_cv(task: str) -> Tuple[str, Any]:
    if task.lower() == "classification":
        return "f1_macro", StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return "r2", KFold(n_splits=5, shuffle=True, random_state=42)


def _evaluate_candidates(
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: Pipeline,
    task: str,
) -> Tuple[str, Dict[str, float]]:
    scoring, cv = _metric_name_and_cv(task)
    scores: Dict[str, float] = {}
    for name, model in _candidate_models(task).items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        scores[name] = float(np.mean(cv_scores))
    best_name = max(scores, key=scores.get)
    return best_name, scores


def _tune_model(
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: Pipeline,
    task: str,
    model_name: str,
    n_trials: int = 25,
    timeout: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any], float]:
    if not HAS_OPTUNA:
        # Return default model without tuning
        model = _candidate_models(task)[model_name]
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        scoring, cv = _metric_name_and_cv(task)
        score = float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)))
        return model, getattr(model, "get_params", lambda: {})(), score

    scoring, cv = _metric_name_and_cv(task)

    def build_model(trial: "Trial") -> Any:
        if task == "classification":
            if model_name == "logreg":
                C = trial.suggest_float("C", 1e-3, 100.0, log=True)
                penalty = trial.suggest_categorical("penalty", ["l2"])
                solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
                return LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
            if model_name == "rf":
                n_estimators = trial.suggest_int("n_estimators", 200, 800)
                max_depth = trial.suggest_int("max_depth", 3, 20)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                )
            if model_name == "xgb" and HAS_XGB:
                n_estimators = trial.suggest_int("n_estimators", 200, 800)
                max_depth = trial.suggest_int("max_depth", 3, 12)
                learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
                return XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    eval_metric="logloss",
                    random_state=42,
                    tree_method="hist",
                )
        else:  # regression
            if model_name == "ridge":
                alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
                return Ridge(alpha=alpha, random_state=42)
            if model_name == "rf":
                n_estimators = trial.suggest_int("n_estimators", 200, 800)
                max_depth = trial.suggest_int("max_depth", 3, 20)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
                return RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                )
            if model_name == "xgb" and HAS_XGB:
                n_estimators = trial.suggest_int("n_estimators", 200, 1000)
                max_depth = trial.suggest_int("max_depth", 3, 12)
                learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
                return XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    objective="reg:squarederror",
                    random_state=42,
                    tree_method="hist",
                )
        # Fallback (should not happen)
        return _candidate_models(task)[model_name]

    def objective(trial: "Trial") -> float:
        model = build_model(trial)
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best_params = study.best_params
    best_model = build_model(optuna.trial.FixedTrial(best_params))
    best_score = float(study.best_value)
    return best_model, best_params, best_score


def run_automl(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    n_trials: int = 25,
    selector_k_if_many: int = 50,
) -> AutoMLResult:
    task = task.lower().strip()
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")

    # Drop rows with missing target
    df_local = df.dropna(subset=[target_col]).copy()
    y = df_local[target_col]
    X = df_local.drop(columns=[target_col])

    y_encoder: Optional[LabelEncoder] = None
    if task == "classification":
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y.astype(str))

    # Feature selection decision
    selector_k: Optional[int] = None
    if X.shape[1] > 50:
        selector_k = selector_k_if_many

    preprocessor, pre_just = _build_preprocessor(df_local, target_col, task, selector_k=selector_k)

    # Evaluate candidates
    best_name, scores = _evaluate_candidates(X, y, preprocessor, task)

    # Hyperparameter tuning
    best_model, best_params, best_score = _tune_model(
        X=X,
        y=y,
        preprocessor=preprocessor,
        task=task,
        model_name=best_name,
        n_trials=n_trials,
    )

    # Fit final artifacts (separate preprocess and model)
    # Fit preprocessor on whole data
    preprocessor_fitted = preprocessor.fit(X, y)
    X_trans = preprocessor_fitted.transform(X)

    # Fit model on transformed features
    model_fitted = best_model.fit(X_trans, y)

    # Serialize artifacts
    preprocess_package = {
        "preprocessor": preprocessor_fitted,
        "y_encoder": y_encoder,
        "feature_names": list(X.columns),
        "task": task,
        "target_col": target_col,
    }
    preprocess_bytes = pickle.dumps(preprocess_package)
    model_bytes = pickle.dumps(model_fitted)

    # Build justification
    justification: Dict[str, Any] = {
        "feature_overview": {
            "n_rows": int(df_local.shape[0]),
            "n_features": int(X.shape[1]),
        },
        "preprocessing": pre_just,
        "candidate_scores": scores,
        "selected_model": {
            "name": best_name,
            "best_params": best_params,
            "cv_score": best_score,
            "metric": _metric_name_and_cv(task)[0],
        },
        "notes": "Pipelines include imputation, scaling, and encoding. Target encoded if high cardinality categorical features present.",
    }

    return AutoMLResult(
        preprocess_object=preprocess_package,
        model_object=model_fitted,
        preprocess_bytes=preprocess_bytes,
        model_bytes=model_bytes,
        justification=justification,
    )


