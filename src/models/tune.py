"""Nested, participant-grouped hyperparameter tuning.

Outer loop: the shared fixed folds (same ones every model is scored on).
Inner loop: StratifiedGroupKFold on the outer-training people only, so the
search never sees the people it is later scored on.

sklearn does not forward `groups` into a nested search automatically, which is
why the outer loop is written out by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from .train import SEED, PRIMARY_KPI, score_predictions

INNER_SPLITS = 3

SEARCH_SPACES = {
    "elasticnet_lr": {
        "model__C": loguniform(1e-3, 10),
        "model__l1_ratio": uniform(0, 1),
    },
    "random_forest": {
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": randint(5, 80),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "lightgbm": {
        "model__num_leaves": randint(4, 32),
        "model__min_child_samples": randint(20, 150),
        "model__learning_rate": loguniform(0.01, 0.1),
        "model__n_estimators": randint(100, 500),
        "model__reg_lambda": loguniform(1e-2, 10),
        "model__colsample_bytree": uniform(0.5, 0.5),
    },
}


def make_search(model, space, n_iter=20, seed=SEED):
    inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    return RandomizedSearchCV(model, space, n_iter=n_iter, scoring="roc_auc", cv=inner,
                              refit=True, n_jobs=-1, random_state=seed)


def nested_cv(model, space, X, y, groups, folds, name, n_iter=20):
    """Return (per-fold scores incl. chosen params, out-of-fold predictions)."""
    rows, oof = [], np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(folds):
        search = make_search(model, space, n_iter=n_iter)
        search.fit(X.iloc[tr], y.iloc[tr], groups=groups.iloc[tr])
        p = search.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        params = {k_.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                  for k_, v in search.best_params_.items()}
        rows.append({"model": name, "fold": k, "n_people_val": groups.iloc[va].nunique(),
                     "n_val": len(va), "inner_best_auc": round(search.best_score_, 3),
                     **score_predictions(y.iloc[va].values, p), "best_params": params})
    oof_df = pd.DataFrame({"id": groups.values, "y": y.values, "proba": oof, "model": name})
    return pd.DataFrame(rows), oof_df


def tune_final(model, space, X, y, groups, n_iter=30):
    """Tune on ALL training people (for the final refit/serialization step)."""
    search = make_search(model, space, n_iter=n_iter)
    search.fit(X, y, groups=groups)
    return search
