"""Model definitions (as sklearn Pipelines) and grouped cross-validation helpers.

Every model is a Pipeline, so imputation/scaling is refit inside each CV fold
and nothing learned from validation people leaks into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
N_SPLITS = 5
PRIMARY_KPI = "roc_auc"


def get_cv(n_splits: int = N_SPLITS, seed: int = SEED) -> StratifiedGroupKFold:
    """Outer splitter: folds are whole participants, stratified on the outcome."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def fixed_folds(X, y, groups, cv=None):
    """Materialize the folds once so every model is scored on identical splits."""
    cv = cv or get_cv()
    return list(cv.split(X, y, groups))


# ---------------------------------------------------------------- model zoo
def _impute(add_indicator=True):
    return SimpleImputer(strategy="median", add_indicator=add_indicator)


def make_dummy(strategy="most_frequent"):
    return Pipeline([("impute", _impute(False)),
                     ("model", DummyClassifier(strategy=strategy, random_state=SEED))])


def _sklearn_at_least(major, minor):
    import sklearn
    v = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    return v >= (major, minor)


def make_logistic(C=1.0, l1_ratio=0.0):
    """l1_ratio=0 -> ridge (L2) logistic; 0<l1_ratio<1 -> elastic net; 1 -> lasso.

    Works on old and new sklearn: `penalty=` was deprecated in sklearn 1.8.
    """
    solver = "lbfgs" if l1_ratio == 0 else "saga"
    kw = dict(C=C, solver=solver, class_weight="balanced", max_iter=5000, random_state=SEED)
    if _sklearn_at_least(1, 8):
        kw["l1_ratio"] = l1_ratio
    elif l1_ratio == 0:
        kw["penalty"] = "l2"
    else:
        kw.update(penalty="elasticnet", l1_ratio=l1_ratio)
    model = LogisticRegression(**kw)
    return Pipeline([("impute", _impute()), ("scale", StandardScaler()), ("model", model)])


def make_tree(max_depth=3, min_samples_leaf=50):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                   class_weight="balanced", random_state=SEED)
    return Pipeline([("impute", _impute()), ("model", model)])


def make_random_forest(**kw):
    params = dict(n_estimators=400, min_samples_leaf=20, max_features="sqrt",
                  class_weight="balanced", n_jobs=-1, random_state=SEED)
    params.update(kw)
    return Pipeline([("impute", _impute()), ("model", RandomForestClassifier(**params))])


def make_lightgbm(**kw):
    from lightgbm import LGBMClassifier
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  class_weight="balanced", random_state=SEED, verbose=-1)
    params.update(kw)
    # LightGBM handles NaN natively, so no imputer
    return Pipeline([("model", LGBMClassifier(**params))])


# ---------------------------------------------------------------- evaluation
def score_predictions(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    y_true = np.asarray(y_true)
    neg = y_true == 0
    return {
        # accuracy = team's primary KPI (kpis.md); always compare it with no_info_acc,
        # the accuracy of always predicting the majority class in the same fold
        "accuracy": accuracy_score(y_true, pred),
        "no_info_acc": max(y_true.mean(), 1 - y_true.mean()),
        "hit_rate": recall_score(y_true, pred, zero_division=0),          # sensitivity / TPR
        "false_alarm": pred[neg].mean() if neg.any() else np.nan,         # FPR = 1 - specificity
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "precision_bad": precision_score(y_true, pred, zero_division=0),
        "f1_bad": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
    }


def evaluate_cv(model, X, y, groups, folds, name: str):
    """Fit on each training fold, score on held-out people.

    Returns (per-fold DataFrame, out-of-fold predictions DataFrame).
    """
    from sklearn.base import clone
    rows, oof = [], np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(folds):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        rows.append({"model": name, "fold": k, "n_people_val": groups.iloc[va].nunique(),
                     "n_val": len(va), **score_predictions(y.iloc[va].values, p)})
    oof_df = pd.DataFrame({"id": groups.values, "y": y.values, "proba": oof, "model": name})
    return pd.DataFrame(rows), oof_df


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in fold_df.columns if c not in ("model", "fold", "n_people_val", "n_val")]
    g = fold_df.groupby("model", sort=False)[metrics]
    out = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
    return out.sort_values(PRIMARY_KPI, ascending=False)
