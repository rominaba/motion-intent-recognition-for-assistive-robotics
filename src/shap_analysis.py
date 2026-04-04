"""SHAP explanations for the custom multinomial LogisticRegressionModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from numpy.typing import NDArray

from src.models.logistic_regression import LogisticRegressionModel


class _SklearnLikeLogistic:
    """Sklearn-like coef_/intercept_/predict_proba for shap.LinearExplainer."""

    def __init__(self, model: LogisticRegressionModel) -> None:
        self._model = model
        self.coef_ = np.asarray(model.weight_matrix)
        self.intercept_ = np.asarray(model.bias_vector)

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return self._model.predict_proba(X)


def run_shap_analysis(
    model: LogisticRegressionModel,
    X_train: NDArray[np.floating[Any]],
    X_explain: NDArray[np.floating[Any]],
    feature_names: Sequence[str],
    class_names: Sequence[str],
    output_dir: str | Path,
    *,
    n_background: int = 200,
    random_state: int = 42,
    max_display: int = 20,
    logger: Any | None = None,
) -> Path:
    """
    Compute exact linear SHAP values (via LinearExplainer) and save summary plots.

    For multinomial logistic regression, SHAP values have shape
    (n_explain, n_features, n_classes).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_bg = min(n_background, X_train.shape[0])
    background = shap.sample(X_train, n_bg, random_state=random_state)
    wrapped = _SklearnLikeLogistic(model)
    explainer = shap.LinearExplainer(wrapped, background)
    shap_values = explainer.shap_values(X_explain)
    # (n_samples, n_features, n_classes)
    sv = np.asarray(shap_values)
    fn = list(feature_names)

    if logger is not None:
        mean_abs = np.abs(sv).mean(axis=(0, 2))
        top_global = np.argsort(mean_abs)[::-1][:15]
        logger.info("SHAP: top features by mean |value| (all classes): %s", [fn[i] for i in top_global])
        for c, name in enumerate(class_names):
            class_mean = np.abs(sv[:, :, c]).mean(axis=0)
            top_c = np.argsort(class_mean)[::-1][:10]
            logger.info("SHAP: top features for class %r: %s", name, [fn[i] for i in top_c])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv,
        X_explain,
        feature_names=fn,
        class_names=list(class_names),
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    summary_path = out / "shap_summary_multiclass.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_all = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(mean_abs_all)[::-1][:max_display]
    plt.figure(figsize=(8, max(4, max_display * 0.25)))
    y_pos = np.arange(len(order))
    plt.barh(y_pos, mean_abs_all[order][::-1], color="steelblue")
    plt.yticks(y_pos, [fn[i] for i in order[::-1]])
    plt.xlabel("mean |SHAP| (averaged over samples and classes)")
    plt.title("Global feature importance (multinomial logistic regression)")
    plt.tight_layout()
    bar_path = out / "shap_mean_abs_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    if logger is not None:
        logger.info("SHAP plots written to %s and %s", summary_path.resolve(), bar_path.resolve())

    return out
