"""PCA dimensionality reduction: pick n_components by validation accuracy, then fit on full train."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.models.logistic_regression import LogisticRegressionModel
from src.utils import get_logger



def fit_best_pca_then_transform(
    X_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.integer[Any]],
    X_test: NDArray[np.floating[Any]],
    *,
    random_state: int = 42,
    val_fraction: float = 0.2,
    n_min: int = 2,
    n_max: Optional[int] = None,
    step: int = 5,
    probe_max_iter: int = 3000,
    probe_learning_rate: float = 0.05,
    target_variance: float = 0.90,
    logger: Optional[logging.Logger] = get_logger("pca"),
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], PCA, dict[str, Any]]:
    """
    Search n_components on a stratified train/val split (probe logistic regression),
    then fit PCA on the full training set and transform train and test.
    """
    
    n_features = int(X_train.shape[1])
    n_max_eff = min(n_max if n_max is not None else min(120, n_features), n_features)
    n_min_eff = min(n_min, n_features)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_train,
    )

    candidates = list(range(n_min_eff, n_max_eff + 1, step))
    if candidates[-1] != n_max_eff:
        candidates.append(n_max_eff)

    best_n: Optional[int] = None
    best_acc = -1.0
    for n in candidates:
        pca = PCA(n_components=n, random_state=random_state)
        X_tr_p = pca.fit_transform(X_tr)
        X_val_p = pca.transform(X_val)
        probe = LogisticRegressionModel(random_state=random_state, max_iter=probe_max_iter)
        probe.fit(X_tr_p, y_tr, learning_rate=probe_learning_rate)
        acc = float(accuracy_score(y_val, probe.predict(X_val_p)))
        var_exp = float(np.sum(pca.explained_variance_ratio_))
        logger.info("PCA search n_components=%d val_accuracy=%.4f variance_explained=%.4f", n, acc, var_exp)
        # Prefer the first n (smallest in the grid) that explains enough variance; else keep best val accuracy.
        if var_exp >= target_variance:
            best_acc = acc
            best_n = n
            break
        elif acc > best_acc:
            best_n, best_acc = n, acc

    
    logger.info("PCA chosen n_components=%d (val_accuracy=%.4f)", best_n, best_acc)

    pca_final = PCA(n_components=best_n, random_state=random_state)
    X_train_p = pca_final.fit_transform(X_train)
    X_test_p = pca_final.transform(X_test)
    meta = {
        "n_components": best_n,
        "val_accuracy_probe": best_acc,
        "random_state": random_state,
    }
    return X_train_p, X_test_p, pca_final, meta
